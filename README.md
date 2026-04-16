# RAGAL — Risk-Aware Generative Active Learning

Research implementation of a cold-start recommendation framework that combines LLM-based contrastive scoring, epistemic uncertainty, and risk-aware exploration.

Paper: *Explore Safely, Learn Fast: Risk-Aware Cold-Start Recommendation with LLM Contrastive Scoring*

## Setup

**Python environment:**
```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv\Scripts\activate on Windows cmd
pip install -r requirements.txt
```

**Ollama (required for offline phase):**
1. Install from https://ollama.com/download
2. Start the server: `ollama serve` (leave running in a separate terminal)
3. Pull a model: `ollama pull llama3.1:8b`

Other supported models (pass via `--model`):

| Model | Download size | RAM needed |
|-------|--------------|-----------|
| `llama3.1:8b` | ~4.7GB | ~8GB |
| `gemma3:4b` | ~3GB | ~6GB |
| `llama3.2:3b` | ~2GB | ~4GB |

---

## Data Loaders — `ragal/data_loaders.py`

Loads three datasets into a common format.

**Common output:** Every loader returns a `Dataset` with:
- `items` — dict mapping `item_id` to an `Item` (title, text for embedding, metadata)
- `interactions` — list of `(user_id, item_id, reward, timestamp)` where reward is binary (1=engaged, 0=skipped)
- `user_sequences` — per-user chronologically sorted `[(item_id, reward), ...]`

**Datasets:**

| Dataset | Loader | Items | Users | How engagement is modeled |
|---------|--------|-------|-------|--------------------------|
| MovieLens-1M | `load_movielens()` | 3,883 movies | 6,040 | Rating >= 4 = engaged |
| Semantic Scholar | `load_semantic_scholar()` | 5,647 papers | 1,458 authors | Author wrote paper = engaged, random papers = skipped (3:1) |
| Citation Network (DBLP v12) | `load_citation_network()` | Up to 100k (configurable) | Varies | Same as S2: author = user |

```python
from ragal.data_loaders import load_movielens, load_semantic_scholar, load_citation_network

ml = load_movielens()
s2 = load_semantic_scholar()
cn = load_citation_network(max_papers=50_000, fos_filter=["Computer science"])

print(ml.summary())
# Dataset(MovieLens-1M): 3883 items, 6040 users, 1000209 interactions (pos=575281, neg=424928)
```

**Test all loaders:**
```bash
python -m ragal.data_loaders
```

**Persistence:** Datasets can be saved/loaded independently:
```python
ml.save("outputs/datasets/movielens")
ml_loaded = Dataset.load("outputs/datasets/movielens")
```

---

## Offline Phase — `ragal/offline.py`

Runs once per catalog. Both steps use the LLM via Ollama — no fallbacks.

**Step 1 — Semantic Embeddings** (paper Section IV.C.1)

> *"A frozen LLM or sentence encoder maps every item to a dense vector"*

- Calls Ollama `/api/embed` in batches of 64
- Vectors are L2-normalized and indexed into FAISS (flat index for <4k items, IVF for larger)
- Cached to `outputs/cache/embeddings/emb_{dataset}.npy`

**Step 2 — Risk Scores R(a)** (paper Section IV.C.2)

> *"The LLM is prompted once per item: On a scale from 0 to 1, how risky is it to recommend this item to a complete stranger..."*

- Produces a static lookup table `R: item_id -> [0, 1]`
- R(a) ~ 0 = safe for anyone, R(a) ~ 1 = risky for strangers
- Cached to `outputs/cache/risk/risk_{dataset}.json`

**Output:** A `CatalogStore` containing the dataset, FAISS item index, and risk scores — all saved to disk automatically.

### Running

```bash
# Make sure Ollama is running with a model pulled

python -m ragal.offline --dataset movielens --model llama3.1:8b
python -m ragal.offline --dataset s2 --model llama3.1:8b
python -m ragal.offline --dataset citation --model llama3.1:8b --max-papers 50000

# Different model — use a separate cache dir
python -m ragal.offline --dataset movielens --model gemma3:4b --cache-dir outputs/cache_gemma
```

### Loading a saved catalog

```python
from ragal.offline import CatalogStore

# No Ollama needed — loads from disk
catalog = CatalogStore.load("outputs/cache/catalogs/MovieLens-1M")

catalog.dataset.summary()          # dataset stats
catalog.item_index.query(vec, k=5) # ANN search
catalog.get_risk("item_123")       # risk lookup
```

### What gets saved

```
outputs/cache/catalogs/{dataset_name}/
  manifest.json              # metadata
  dataset/
    items.json               # all items (title, text, metadata)
    interactions.npz         # (user_id, item_id, reward, timestamp) compressed
    manifest.json            # dataset stats
  item_index/
    embeddings.npy           # (n_items, dim) L2-normalized float32
    item_ids.json            # row index -> item_id mapping
    index.faiss              # native FAISS serialized index
  risk_scores.json           # {item_id: float} lookup table
```

### Caching behavior

- Embeddings: cached by dataset name. If item IDs change, recomputes automatically.
- Risk scores: cached per item. New items get scored, existing scores are reused. Interrupted runs resume from where they left off.
- Switching models does NOT auto-invalidate cache — use `--cache-dir` to keep them separate.

### Time estimates

Per item: embeddings are batched (fast), risk scoring is one LLM generation each (~1-3 sec). For MovieLens (3,883 items), risk scoring takes roughly 1-3 hours depending on hardware. Everything is cached, so re-runs are instant.

---

## Project Structure

```
ragal/
  __init__.py
  data_loaders.py     # Dataset, Item, load_movielens, load_semantic_scholar, load_citation_network
  offline.py          # ItemIndex, CatalogStore, build_catalog (LLM embeddings + FAISS + LLM risk)
data/
  movielens/ml-1m/    # ratings.dat, movies.dat, users.dat
  semantic_scholar/    # recsys_complete_dataset.jsonl (~5.6k papers)
  citation_network/   # dblp.v12.json (~12GB, ~4.9M papers)
outputs/
  cache/              # LLM embedding + risk score caches + saved catalogs
requirements.txt
```

## Documentation

- `framework_explanation.md` — full mathematical specification of the RAGAL framework with worked examples
- `IMT2023094_RecSys_Report.pdf` / `RecSys_Report_latest.pdf` — project report
- `references/` — academic PDFs (ACS, SND, harm mitigation research)
