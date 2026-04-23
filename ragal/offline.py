import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

from .data_loaders import Dataset
from .hf_model import HFLLM, HF_MODEL_DEFAULT

logger = logging.getLogger(__name__)

# Compute embeddings
def _compute_embeddings(dataset: Dataset, llm: HFLLM,
                        cache_path: str,
                        batch_size: int = 16,
                        max_length: int = 512) -> np.ndarray:
    cache_npy = Path(cache_path)
    cache_ids = cache_npy.with_suffix(".ids.json")
    item_ids = list(dataset.items.keys())

    if cache_npy.exists() and cache_ids.exists():
        with open(cache_ids) as f:
            cached_ids = json.load(f)
        if cached_ids == item_ids:
            embeddings = np.load(cache_npy)
            logger.info("Loaded cached embeddings from %s (%d items, dim=%d)",
                        cache_npy, embeddings.shape[0], embeddings.shape[1])
            return embeddings
        logger.info("Cache item IDs mismatch — recomputing embeddings")

    texts = [dataset.items[iid].text[:max_length] for iid in item_ids]
    logger.info("Embedding %d items via HF %s (batch=%d, max_length=%d)",
                len(texts), llm.model_name, batch_size, max_length)

    chunks = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embed {dataset.name}"):
        chunk = texts[i : i + batch_size]
        emb = llm.embed(chunk, batch_size=batch_size, max_length=max_length)
        chunks.append(emb)
    matrix = np.vstack(chunks).astype(np.float32)
    logger.info("Raw embeddings shape: %s", matrix.shape)

    cache_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_npy, matrix)
    with open(cache_ids, "w") as f:
        json.dump(item_ids, f)
    logger.info("Cached embeddings to %s", cache_npy)
    return matrix

# FAISS Index definition
@dataclass
class ItemIndex:
    item_ids: list[str]
    embeddings: np.ndarray
    faiss_index: faiss.Index
    _id_to_pos: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._id_to_pos = {iid: i for i, iid in enumerate(self.item_ids)}

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]

    def get_embedding(self, item_id: str) -> np.ndarray | None:
        pos = self._id_to_pos.get(item_id)
        if pos is None:
            return None
        return self.embeddings[pos]

    def query(self, vector: np.ndarray, k: int = 50,
              exclude_ids: set[str] | None = None) -> list[tuple[str, float]]:
        fetch_k = k + (len(exclude_ids) if exclude_ids else 0) + 10
        fetch_k = min(fetch_k, len(self.item_ids))
        q = vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.faiss_index.search(q, fetch_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            iid = self.item_ids[idx]
            if exclude_ids and iid in exclude_ids:
                continue
            results.append((iid, float(dist)))
            if len(results) >= k:
                break
        return results

    def save(self, path: str):
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.embeddings)
        with open(d / "item_ids.json", "w") as f:
            json.dump(self.item_ids, f)
        faiss.write_index(self.faiss_index, str(d / "index.faiss"))
        logger.info("ItemIndex saved to %s", d)

    @classmethod
    def load(cls, path: str) -> "ItemIndex":
        d = Path(path)
        embeddings = np.load(d / "embeddings.npy")
        with open(d / "item_ids.json") as f:
            item_ids = json.load(f)
        faiss_index = faiss.read_index(str(d / "index.faiss"))
        logger.info("ItemIndex loaded from %s (%d items, dim=%d)",
                     d, len(item_ids), embeddings.shape[1])
        return cls(item_ids=item_ids, embeddings=embeddings, faiss_index=faiss_index)

def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    n_items = embeddings.shape[0]
    if n_items < 4000:
        index = faiss.IndexFlatIP(dim)
        logger.info("FAISS: flat inner-product index (n=%d, dim=%d)", n_items, dim)
    else:
        n_clusters = min(int(np.sqrt(n_items)), 256)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = min(16, n_clusters)
        logger.info("FAISS: IVF index (n=%d, dim=%d, clusters=%d, nprobe=%d)",
                     n_items, dim, n_clusters, index.nprobe)
    index.add(embeddings)
    return index

def build_item_index(dataset: Dataset, llm: HFLLM,
                     cache_dir: str = "outputs/embedding_cache",
                     batch_size: int = 16,
                     max_length: int = 512) -> ItemIndex:
    cache_path = str(Path(cache_dir) / f"emb_{dataset.name}.npy")
    raw = _compute_embeddings(dataset, llm, cache_path,
                              batch_size=batch_size, max_length=max_length)
    embeddings = normalize(raw, norm="l2").astype(np.float32)
    index = _build_faiss_index(embeddings)
    item_ids = list(dataset.items.keys())
    item_index = ItemIndex(item_ids=item_ids, embeddings=embeddings, faiss_index=index)
    logger.info("ItemIndex built: %d items, dim=%d", len(item_ids), item_index.dim)
    return item_index

# Compute risk scores
RISK_PROMPT_TEMPLATE = (
    "You are assessing how risky it is to recommend the following item "
    "to a complete stranger with no established taste profile.\n\n"
    "An item is RISKY if it is niche, assumes significant prior context, "
    "or contains polarizing content.\n"
    "An item is SAFE if it has broad mainstream appeal and minimal polarizing content.\n\n"
    "Item: {item_description}\n\n"
    "Classify this item. Answer with a single word: RISKY or SAFE.\n"
    "Answer:"
)

RISK_WORD_POS = "RISKY"
RISK_WORD_NEG = "SAFE"

def compute_risk_scores(dataset: Dataset, llm: HFLLM,
                        cache_path: str) -> dict[str, float]:
    cache_file = Path(cache_path)
    cached = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        logger.info("Loaded %d cached risk scores from %s", len(cached), cache_path)

    risk_scores = {}
    items_to_score = []
    for iid, item in dataset.items.items():
        if iid in cached:
            risk_scores[iid] = cached[iid]
        else:
            items_to_score.append((iid, item))

    if items_to_score:
        logger.info("Scoring risk for %d items via HF %s (%d cached)",
                    len(items_to_score), llm.model_name, len(risk_scores))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        save_every = 50

        for i, (iid, item) in enumerate(tqdm(items_to_score, desc=f"Risk {dataset.name}")):
            prompt = RISK_PROMPT_TEMPLATE.format(item_description=item.text[:500])
            risk_scores[iid] = float(
                llm.contrast_logprob(prompt, RISK_WORD_POS, RISK_WORD_NEG)
            )
            if (i + 1) % save_every == 0:
                with open(cache_file, "w") as f:
                    json.dump(risk_scores, f)

        with open(cache_file, "w") as f:
            json.dump(risk_scores, f)
        logger.info("Saved %d risk scores to %s", len(risk_scores), cache_path)

    n_high = sum(1 for r in risk_scores.values() if r > 0.8)
    n_low = sum(1 for r in risk_scores.values() if r < 0.2)
    logger.info("Risk scores: %d items, %d high-risk (>0.8), %d low-risk (<0.2)",
                len(risk_scores), n_high, n_low)
    return risk_scores

# Catalog definition
@dataclass
class CatalogStore:
    dataset: Dataset
    item_index: ItemIndex
    risk_scores: dict[str, float]

    def get_risk(self, item_id: str) -> float:
        return self.risk_scores.get(item_id, 0.5)

    def save(self, path: str):
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        self.dataset.save(str(d / "dataset"))
        self.item_index.save(str(d / "item_index"))
        with open(d / "risk_scores.json", "w") as f:
            json.dump(self.risk_scores, f)
        with open(d / "manifest.json", "w") as f:
            json.dump({
                "dataset_name": self.dataset.name,
                "n_items": len(self.dataset.items),
                "embedding_dim": self.item_index.dim,
                "n_risk_scores": len(self.risk_scores),
            }, f, indent=2)
        logger.info("CatalogStore saved to %s", d)

    @classmethod
    def load(cls, path: str) -> "CatalogStore":
        d = Path(path)
        if not (d / "manifest.json").exists():
            raise FileNotFoundError(f"No catalog found at {d}")
        dataset = Dataset.load(str(d / "dataset"))
        item_index = ItemIndex.load(str(d / "item_index"))
        with open(d / "risk_scores.json") as f:
            risk_scores = json.load(f)
        logger.info("CatalogStore loaded from %s (%d items, dim=%d, %d risk scores)",
                     d, len(dataset.items), item_index.dim, len(risk_scores))
        return cls(dataset=dataset, item_index=item_index, risk_scores=risk_scores)

def build_catalog(dataset: Dataset, llm: HFLLM,
                  cache_dir: str = "outputs/cache",
                  save_dir: str | None = None,
                  batch_size: int = 16,
                  max_length: int = 512) -> CatalogStore:
    logger.info("=== Offline phase for %s (HF=%s) ===",
                dataset.name, llm.model_name)

    embedding_cache_dir = str(Path(cache_dir) / "embeddings")
    risk_cache_path = str(Path(cache_dir) / "risk" / f"risk_{dataset.name}.json")

    item_index = build_item_index(dataset, llm,
                                   cache_dir=embedding_cache_dir,
                                   batch_size=batch_size,
                                   max_length=max_length)
    risk_scores = compute_risk_scores(dataset, llm, cache_path=risk_cache_path)

    catalog = CatalogStore(dataset=dataset, item_index=item_index, risk_scores=risk_scores)

    if save_dir is None:
        save_dir = str(Path(cache_dir) / "catalogs" / dataset.name)
    catalog.save(save_dir)

    return catalog

# CLI execution
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run RAGAL offline phase")
    parser.add_argument("--dataset", choices=["movielens", "s2", "citation"], default="movielens")
    parser.add_argument("--model", default=HF_MODEL_DEFAULT, help="HuggingFace model id")
    parser.add_argument("--device", default="cuda", help='"cuda" or "cpu"')
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--load-in-4bit", action="store_true", help="Quantize the model to 4-bit")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Per-text truncation before embedding")
    parser.add_argument("--max-papers", type=int, default=10_000, help="Max papers for citation net")
    parser.add_argument("--cache-dir", default="outputs/cache", help="Cache directory")
    parser.add_argument("--load", metavar="PATH", help="Load a saved CatalogStore instead of building")
    args = parser.parse_args()

    if args.load:
        catalog = CatalogStore.load(args.load)
    else:
        from .data_loaders import load_movielens, load_semantic_scholar, load_citation_network

        if args.dataset == "movielens":
            ds = load_movielens()
        elif args.dataset == "s2":
            ds = load_semantic_scholar()
        elif args.dataset == "citation":
            ds = load_citation_network(max_papers=args.max_papers, fos_filter=["Computer science"])

        llm = HFLLM(model_name=args.model, device=args.device,
                    dtype=args.dtype, load_in_4bit=args.load_in_4bit)
        catalog = build_catalog(ds, llm=llm, cache_dir=args.cache_dir,
                                batch_size=args.batch_size, max_length=args.max_length)

    print(f"\n{catalog.dataset.summary()}")
    print(f"Embeddings: {catalog.item_index.embeddings.shape}")
    print(f"Risk scores: {len(catalog.risk_scores)} items")

    risks = sorted(catalog.risk_scores.items(), key=lambda x: x[1])
    print("\n5 Safest items:")
    for iid, r in risks[:5]:
        print(f"  R={r:.2f}  {catalog.dataset.items[iid].title[:70]}")
    print("\n5 Riskiest items:")
    for iid, r in risks[-5:]:
        print(f"  R={r:.2f}  {catalog.dataset.items[iid].title[:70]}")
