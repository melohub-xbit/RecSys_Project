"""
Offline phase for RACS — uses a dedicated encoder model (BGE-large by
default) for item embeddings and the LLM (HFLLM) for risk scoring.

  Step 1 — Embeddings  ->  HFEmbedder (encoder, CLS-pooled, L2-normalised)
  Step 2 — Risk R(a)   ->  HFLLM.contrast_logprob(prompt, "RISKY", "SAFE")

Caches
------
  <cache-dir>/embeddings/emb_<Dataset>__<embed-slug>.npy  + .ids.json
  <cache-dir>/risk/risk_<Dataset>.json
  <cache-dir>/catalogs/<Dataset>/                          (CatalogStore)
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import torch
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .data_loaders import Dataset, load_movielens, load_semantic_scholar
from .hf_model import HF_MODEL_DEFAULT, HFLLM

logger = logging.getLogger(__name__)

EMBED_MODEL_DEFAULT = "BAAI/bge-large-en-v1.5"


def _embed_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


# ---------------------------------------------------------------------------
# HFEmbedder — encoder-only model for item embeddings
# ---------------------------------------------------------------------------

class HFEmbedder:
    """
    Wraps an encoder-only embedding model (e.g. BGE) on top of HF transformers.
    Uses CLS pooling, then L2 normalisation, matching the published BGE recipe.
    """

    def __init__(self, model_name: str = EMBED_MODEL_DEFAULT,
                 device: str = "cuda",
                 dtype: str = "float16"):
        self.model_name = model_name
        logger.info("Loading HF embedder %s (dtype=%s)", model_name, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        dtype_map = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unknown dtype {dtype!r}; expected one of {list(dtype_map)}")

        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype_map[dtype]
        ).to(device).eval()
        self.device = next(self.model.parameters()).device
        logger.info("HF embedder ready on %s", self.device)

    @torch.no_grad()
    def embed(self, texts: list[str],
              batch_size: int = 32,
              max_length: int = 512) -> np.ndarray:
        all_emb: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            out = self.model(**inputs)
            # CLS-token pooling — standard for BGE/E5
            cls = out.last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            all_emb.append(cls.float().cpu().numpy())
        return np.vstack(all_emb)


# ---------------------------------------------------------------------------
# Step 1 — semantic embeddings
# ---------------------------------------------------------------------------

def _compute_embeddings(dataset: Dataset, embedder: HFEmbedder,
                        cache_path: str,
                        batch_size: int = 32,
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
    logger.info("Embedding %d items via encoder %s (batch=%d, max_length=%d)",
                len(texts), embedder.model_name, batch_size, max_length)

    chunks: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size),
                   desc=f"Embed {dataset.name}"):
        chunk = texts[i : i + batch_size]
        emb = embedder.embed(chunk, batch_size=batch_size, max_length=max_length)
        chunks.append(emb)
    matrix = np.vstack(chunks).astype(np.float32)
    logger.info("Raw embeddings shape: %s", matrix.shape)

    cache_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_npy, matrix)
    with open(cache_ids, "w") as f:
        json.dump(item_ids, f)
    logger.info("Cached embeddings to %s", cache_npy)
    return matrix


# ---------------------------------------------------------------------------
# FAISS Index
# ---------------------------------------------------------------------------

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


def build_item_index(dataset: Dataset, embedder: HFEmbedder,
                     cache_dir: str = "outputs/embedding_cache",
                     batch_size: int = 32,
                     max_length: int = 512) -> ItemIndex:
    slug = _embed_slug(embedder.model_name)
    cache_path = str(Path(cache_dir) / f"emb_{dataset.name}__{slug}.npy")
    raw = _compute_embeddings(dataset, embedder, cache_path,
                              batch_size=batch_size, max_length=max_length)
    embeddings = normalize(raw, norm="l2").astype(np.float32)
    index = _build_faiss_index(embeddings)
    item_ids = list(dataset.items.keys())
    item_index = ItemIndex(item_ids=item_ids, embeddings=embeddings, faiss_index=index)
    logger.info("ItemIndex built: %d items, dim=%d", len(item_ids), item_index.dim)
    return item_index


# ---------------------------------------------------------------------------
# Step 2 — risk scores via contrast logprob (LLM, unchanged from offline.py)
# ---------------------------------------------------------------------------

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

    risk_scores: dict[str, float] = {}
    items_to_score: list = []
    for iid, item in dataset.items.items():
        if iid in cached:
            risk_scores[iid] = cached[iid]
        else:
            items_to_score.append((iid, item))

    if items_to_score:
        logger.info("Scoring risk for %d items via HF %s  (%d cached)",
                    len(items_to_score), llm.model_name, len(risk_scores))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        save_every = 50

        for i, (iid, item) in enumerate(tqdm(items_to_score,
                                               desc=f"Risk {dataset.name}")):
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


# ---------------------------------------------------------------------------
# CatalogStore — no get_risk() fallback; callers read .risk_scores directly
# ---------------------------------------------------------------------------

@dataclass
class CatalogStore:
    dataset: Dataset
    item_index: ItemIndex
    risk_scores: dict[str, float]

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


def build_catalog(dataset: Dataset, embedder: HFEmbedder, llm: HFLLM,
                  cache_dir: str = "outputs/cache",
                  save_dir: str | None = None,
                  embed_batch_size: int = 32,
                  embed_max_length: int = 512) -> CatalogStore:
    logger.info("=== Offline phase (encoder embedder) for %s "
                "(embed=%s, llm=%s) ===",
                dataset.name, embedder.model_name, llm.model_name)

    embedding_cache_dir = str(Path(cache_dir) / "embeddings")
    risk_cache_path = str(Path(cache_dir) / "risk" / f"risk_{dataset.name}.json")

    item_index = build_item_index(dataset, embedder,
                                   cache_dir=embedding_cache_dir,
                                   batch_size=embed_batch_size,
                                   max_length=embed_max_length)
    risk_scores = compute_risk_scores(dataset, llm, cache_path=risk_cache_path)

    catalog = CatalogStore(dataset=dataset, item_index=item_index,
                            risk_scores=risk_scores)
    if save_dir is None:
        save_dir = str(Path(cache_dir) / "catalogs" / dataset.name)
    catalog.save(save_dir)
    return catalog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="RACS offline phase with a dedicated encoder embedder "
                    "and HFLLM-based risk scoring.")
    p.add_argument("--dataset", choices=["movielens", "s2"], required=False,
                   help="Kind of dataset to load.")
    p.add_argument("--data-path", metavar="PATH",
                   help="Path to the dataset root (ml-1m / ml-20m / ml-25m "
                        "directory) or to a Semantic Scholar JSONL file.")
    p.add_argument("--name", metavar="NAME",
                   help="Override Dataset.name (controls cache filenames).")
    p.add_argument("--embed-model", default=EMBED_MODEL_DEFAULT,
                   help="HuggingFace encoder model id used for item embeddings "
                        "(e.g. BAAI/bge-large-en-v1.5).")
    p.add_argument("--model", default=HF_MODEL_DEFAULT,
                   help="HuggingFace LLM id used for risk scoring "
                        "(e.g. meta-llama/Llama-3.1-8B-Instruct).")
    p.add_argument("--device", default="cuda",
                   help='"cuda", "cpu", or a specific device like "cuda:0".')
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Quantise the LLM to 4-bit via bitsandbytes "
                        "(embedder stays full precision).")
    p.add_argument("--embed-batch-size", type=int, default=32,
                   help="Embedding batch size.")
    p.add_argument("--embed-max-length", type=int, default=512,
                   help="Per-text truncation before embedding.")
    p.add_argument("--cache-dir", default="outputs/cache",
                   help="Cache directory (use a different one per HF model).")
    p.add_argument("--load", metavar="PATH",
                   help="Load a saved CatalogStore and print its summary, "
                        "then exit. No HF model is loaded.")
    args = p.parse_args()

    if args.load:
        catalog = CatalogStore.load(args.load)
        print(f"\n{catalog.dataset.summary()}")
        print(f"Embeddings: {catalog.item_index.embeddings.shape}")
        print(f"Risk scores: {len(catalog.risk_scores)} items")
        sys.exit(0)

    if not args.dataset or not args.data_path:
        p.error("Pass --dataset {movielens,s2} and --data-path PATH, "
                "or --load PATH to inspect an existing catalog.")

    embedder = HFEmbedder(model_name=args.embed_model,
                          device=args.device, dtype=args.dtype)
    llm = HFLLM(model_name=args.model, device=args.device,
                dtype=args.dtype, load_in_4bit=args.load_in_4bit)

    if args.dataset == "movielens":
        ds = load_movielens(args.data_path,
                            name=args.name or "MovieLens-1M")
    elif args.dataset == "s2":
        ds = load_semantic_scholar(args.data_path,
                                   name=args.name or "SemanticScholar")
    build_catalog(ds, embedder=embedder, llm=llm,
                  cache_dir=args.cache_dir,
                  embed_batch_size=args.embed_batch_size,
                  embed_max_length=args.embed_max_length)
