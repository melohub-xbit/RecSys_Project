"""
Offline phase for RAGAL.

Runs once per catalog to produce (per the paper, Section IV.C):

    Step 1 — Semantic Embeddings:
        "A frozen LLM or sentence encoder maps every item to a dense vector"
        Generated via Ollama's /api/embed endpoint. Cached to disk.

    Step 2 — Risk Weight Assignment:
        "The LLM is prompted once per item" with the risk prompt.
        Generated via Ollama's /api/generate endpoint. Cached to disk.

Both are indexed into a CatalogStore that the online phase consumes.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import requests
from sklearn.preprocessing import normalize
from tqdm import tqdm

from .data_loaders import Dataset

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _check_ollama(model: str):
    """Verify Ollama is running and the model is available."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.ConnectionError:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        )
    available = [m["name"] for m in r.json().get("models", [])]
    matched = any(model == m or model == m.split(":")[0] for m in available)
    if not matched:
        raise RuntimeError(
            f"Model '{model}' not found in Ollama. Available: {available}\n"
            f"Pull it with: ollama pull {model}"
        )


def _ollama_embed(texts: list[str], model: str) -> list[list[float]]:
    """
    Get embeddings from Ollama's /api/embed endpoint.

    Sends a batch of texts and returns a list of vectors.
    """
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": model, "input": texts},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def _ollama_generate(prompt: str, model: str, temperature: float = 0.0) -> str:
    """Send a prompt to Ollama and return the response text."""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


# ---------------------------------------------------------------------------
# Step 1: Semantic Embeddings via LLM
# ---------------------------------------------------------------------------

def _compute_embeddings(dataset: Dataset, model: str,
                        cache_path: str, batch_size: int = 64) -> np.ndarray:
    """
    Generate embeddings for all items using the LLM via Ollama /api/embed.

    Per the paper (Section IV.C.1):
        "A frozen LLM or sentence encoder maps every item to a dense vector:
         f_LLM : a -> e_a ∈ R^d"

    Embeddings are cached to a .npy file so re-runs skip computation.
    Item order file (.json) is saved alongside to map rows back to item IDs.
    """
    cache_npy = Path(cache_path)
    cache_ids = cache_npy.with_suffix(".ids.json")

    item_ids = list(dataset.items.keys())

    # check cache: valid only if same item IDs in same order
    if cache_npy.exists() and cache_ids.exists():
        with open(cache_ids) as f:
            cached_ids = json.load(f)
        if cached_ids == item_ids:
            embeddings = np.load(cache_npy)
            logger.info("Loaded cached embeddings from %s (%d items, dim=%d)",
                        cache_npy, embeddings.shape[0], embeddings.shape[1])
            return embeddings
        else:
            logger.info("Cache item IDs mismatch — recomputing embeddings")

    # compute embeddings in batches
    texts = [dataset.items[iid].text[:512] for iid in item_ids]
    all_embeddings = []

    logger.info("Computing embeddings for %d items via %s (batch_size=%d)...",
                len(texts), model, batch_size)

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        vectors = _ollama_embed(batch, model=model)
        all_embeddings.extend(vectors)

    matrix = np.array(all_embeddings, dtype=np.float32)
    logger.info("Raw embeddings shape: %s", matrix.shape)

    # save cache
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
    """Dense embeddings + FAISS index for a set of items."""
    item_ids: list[str]
    embeddings: np.ndarray               # (n_items, dim), L2-normalized
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
        """ANN search. Returns up to k (item_id, score) pairs."""
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
        """Save item index: embeddings (.npy), item IDs (.json), FAISS index (.faiss)."""
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.embeddings)
        with open(d / "item_ids.json", "w") as f:
            json.dump(self.item_ids, f)
        faiss.write_index(self.faiss_index, str(d / "index.faiss"))
        logger.info("ItemIndex saved to %s", d)

    @classmethod
    def load(cls, path: str) -> "ItemIndex":
        """Load item index from directory saved by save()."""
        d = Path(path)
        embeddings = np.load(d / "embeddings.npy")
        with open(d / "item_ids.json") as f:
            item_ids = json.load(f)
        faiss_index = faiss.read_index(str(d / "index.faiss"))
        logger.info("ItemIndex loaded from %s (%d items, dim=%d)",
                     d, len(item_ids), embeddings.shape[1])
        return cls(item_ids=item_ids, embeddings=embeddings, faiss_index=faiss_index)


def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build an appropriate FAISS index for the given embeddings."""
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


def build_item_index(dataset: Dataset, model: str,
                     cache_dir: str = "outputs/embedding_cache") -> ItemIndex:
    """
    Build dense LLM embeddings and FAISS index for all items in a dataset.
    """
    cache_path = str(Path(cache_dir) / f"emb_{dataset.name}.npy")

    _check_ollama(model)
    raw_embeddings = _compute_embeddings(dataset, model=model, cache_path=cache_path)
    embeddings = normalize(raw_embeddings, norm="l2").astype(np.float32)

    index = _build_faiss_index(embeddings)
    item_ids = list(dataset.items.keys())

    item_index = ItemIndex(item_ids=item_ids, embeddings=embeddings, faiss_index=index)
    logger.info("ItemIndex built: %d items, dim=%d", len(item_ids), item_index.dim)
    return item_index


# ---------------------------------------------------------------------------
# Step 2: Risk Weight Assignment via LLM
# ---------------------------------------------------------------------------

RISK_PROMPT_TEMPLATE = (
    "On a scale from 0 to 1, how risky is it to recommend the following item "
    "to a complete stranger with no established taste profile? "
    "Consider: how niche it is, how much prior context it assumes, "
    "how polarizing its content is.\n\n"
    "Item: {item_description}\n\n"
    "Output only a decimal number between 0 and 1, nothing else."
)


def _parse_risk_score(response: str) -> float:
    """Extract a float from the LLM response, clamping to [0, 1]."""
    text = response.strip().strip('"').strip("'")
    for token in text.split():
        token = token.strip(".,;:!?")
        try:
            val = float(token)
            return max(0.0, min(1.0, val))
        except ValueError:
            continue
    logger.warning("Could not parse risk score from: %r — defaulting to 0.5", response)
    return 0.5


def compute_risk_scores(dataset: Dataset, model: str,
                        cache_path: str) -> dict[str, float]:
    """
    Compute R(a) ∈ [0, 1] for every item by prompting the LLM once per item.

    Per the paper (Section IV.C.2):
        "The LLM is prompted once per item: 'On a scale from 0 to 1,
         how risky is it to recommend this item to a complete stranger...'"

    Results are cached to a JSON file so re-runs don't re-prompt.
    """
    _check_ollama(model)

    # load cache
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
        logger.info("Computing risk scores for %d items via %s (%d cached)...",
                     len(items_to_score), model, len(risk_scores))

        for iid, item in tqdm(items_to_score, desc="Risk scoring"):
            prompt = RISK_PROMPT_TEMPLATE.format(item_description=item.text[:500])
            try:
                response = _ollama_generate(prompt, model=model, temperature=0.0)
                score = _parse_risk_score(response)
            except Exception as e:
                logger.warning("Risk scoring failed for item %s: %s", iid, e)
                score = 0.5
            risk_scores[iid] = score

        # save cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(risk_scores, f)
        logger.info("Saved %d risk scores to %s", len(risk_scores), cache_path)

    n_high = sum(1 for r in risk_scores.values() if r > 0.8)
    n_low = sum(1 for r in risk_scores.values() if r < 0.2)
    logger.info("Risk scores: %d items, %d high-risk (>0.8), %d low-risk (<0.2)",
                len(risk_scores), n_high, n_low)
    return risk_scores


# ---------------------------------------------------------------------------
# CatalogStore — bundles everything the online phase needs
# ---------------------------------------------------------------------------

@dataclass
class CatalogStore:
    """All precomputed offline data for one dataset."""
    dataset: Dataset
    item_index: ItemIndex
    risk_scores: dict[str, float]

    def get_risk(self, item_id: str) -> float:
        return self.risk_scores.get(item_id, 0.5)

    def save(self, path: str):
        """
        Save the entire catalog to a directory.

        Layout:
            path/
              manifest.json       — metadata
              dataset/            — Dataset (items.json + interactions.npz)
              item_index/         — ItemIndex (embeddings.npy + item_ids.json + index.faiss)
              risk_scores.json    — R(a) lookup table
        """
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
        """Load a complete CatalogStore from a directory saved by save()."""
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


def build_catalog(dataset: Dataset,
                  ollama_model: str = "llama3.1:8b",
                  cache_dir: str = "outputs/cache",
                  save_dir: str | None = None) -> CatalogStore:
    """
    Run the full offline phase:
        1. Generate LLM embeddings for all items + build FAISS index
        2. Compute LLM-based risk scores R(a) for all items
    Both steps are cached to disk individually during computation.
    The final CatalogStore is also saved if save_dir is provided.
    """
    logger.info("=== Offline phase for %s ===", dataset.name)

    embedding_cache_dir = str(Path(cache_dir) / "embeddings")
    risk_cache_path = str(Path(cache_dir) / "risk" / f"risk_{dataset.name}.json")

    item_index = build_item_index(dataset, model=ollama_model, cache_dir=embedding_cache_dir)
    risk_scores = compute_risk_scores(dataset, model=ollama_model, cache_path=risk_cache_path)

    catalog = CatalogStore(dataset=dataset, item_index=item_index, risk_scores=risk_scores)

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

    parser = argparse.ArgumentParser(description="Run RAGAL offline phase")
    parser.add_argument("--dataset", choices=["movielens", "s2", "citation"], default="movielens")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model for embeddings + risk")
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

        catalog = build_catalog(ds, ollama_model=args.model, cache_dir=args.cache_dir)

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
