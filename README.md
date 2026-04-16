# RAGAL — Implementation Guide

## What's built so far

### `ragal/data_loaders.py` — Data Loading

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
| Citation Network | `load_citation_network()` | Up to 100k (configurable) | Varies | Same as S2: author = user |

**Usage:**
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

---

### `ragal/offline.py` — Offline Phase

Runs once per catalog. Produces a `CatalogStore` containing everything the online phase needs.

**Step 1 — Semantic Embeddings** (paper Section IV.C.1)

A frozen LLM maps every item to a dense vector via Ollama's `/api/embed` endpoint.
Vectors are L2-normalized and indexed into FAISS for ANN retrieval.

- Batched (64 items per request)
- Cached to `outputs/cache/embeddings/emb_{dataset}.npy`
- Same model used across all datasets for consistent embedding space

**Step 2 — Risk Scores R(a)** (paper Section IV.C.2)

The LLM is prompted once per item:
> "On a scale from 0 to 1, how risky is it to recommend this item to a complete stranger with no established taste profile?"

- Produces a static lookup table `R: item_id -> [0, 1]`
- Cached to `outputs/cache/risk/risk_{dataset}.json`
- R(a) ≈ 0 means safe for anyone, R(a) ≈ 1 means risky for strangers

**Usage:**
```python
from ragal.data_loaders import load_movielens
from ragal.offline import build_catalog

ds = load_movielens()
catalog = build_catalog(ds, ollama_model="llama3.1:8b")

# catalog.item_index   — FAISS index + embeddings
# catalog.risk_scores  — {item_id: float}
# catalog.get_risk(item_id) — lookup helper
```

**CLI:**
```bash
# Requires Ollama running: ollama serve (in another terminal)
# Requires model pulled: ollama pull llama3.1:8b

python -m ragal.offline --dataset movielens --model llama3.1:8b
python -m ragal.offline --dataset s2 --model llama3.1:8b
python -m ragal.offline --dataset citation --model llama3.1:8b --max-papers 50000

# Different model — use separate cache to avoid overwriting
python -m ragal.offline --dataset movielens --model gemma3:4b --cache-dir outputs/cache_gemma
```

**Caching behavior:**
- Embeddings: cached by dataset name. If item IDs change, recomputes automatically.
- Risk scores: cached by dataset name. New items get scored, existing scores are reused.
- Switching models does NOT auto-invalidate cache — use `--cache-dir` to separate.

---

## What's next — Online Phase

The online phase runs per-user sequential replay. Three components to implement:

### 1. ACS Scorer (`scoring.py`)

Anchored Contrastive Scoring — the exploitation term.

For each candidate item:
- Find `x+` = closest liked item in user history (by embedding distance)
- Find `x-` = closest skipped item in user history
- Prompt the LLM: *"User engaged with [x+] but skipped [x-]. Is [candidate] more like what they engaged with or skipped?"*
- Extract `P_ACS` from the LLM response (ENGAGED vs SKIPPED)
- Cold-start fallback: when user has no positives or no negatives, ACS can't anchor contrastively

### 2. SND Score (`scoring.py`)

Semantic Neighborhood Disagreement — the exploration term.

For each candidate item:
- Retrieve k nearest already-rated items from the user's history
- Count positives (`n+`) and negatives (`n-`) among them
- `SND = min(n+, n-) / (max(n+, n-) + 1)`
- Add ε-floor (0.05) when fewer than 2 neighbors exist (cold-start)

No LLM needed — pure neighbor counting on the embedding space.

### 3. Policy Runner (`policy.py`)

Sequential offline replay loop:
```
for each user in test set:
    for t = 1, 2, ..., max_steps:
        1. Compute preference centroid from positive history
        2. ANN retrieve top-M candidates, excluding seen items
        3. Score each: Q = P_ACS + α·SND - λ·R(a)
        4. Select a* = argmax Q
        5. Observe reward from held-out ground truth
        6. Update history
```

### 4. LinUCB Baseline (`baselines.py`)

Standard contextual bandit baseline for comparison.

### 5. Evaluation & Grid Search

Metrics: `avg_reward`, `cumulative_reward`, `hitrate@10`, `ndcg@10`, `risky_exposure_rate`, `latency_ms`.

Grid search over `α` (exploration weight) and `λ` (risk penalty weight).

---

## Project structure

```
ragal/
  __init__.py
  data_loaders.py    # Dataset, Item, load_movielens, load_semantic_scholar, load_citation_network
  offline.py         # ItemIndex, CatalogStore, build_catalog (LLM embeddings + FAISS + LLM risk)
  scoring.py         # (TODO) ACSScorer, snd_score, compute_q_score
  policy.py          # (TODO) run_policy, sequential replay loop
  baselines.py       # (TODO) LinUCB
  evaluate.py        # (TODO) metrics, grid search, report generation
data/
  movielens/ml-1m/   # ratings.dat, movies.dat, users.dat
  semantic_scholar/   # recsys_complete_dataset.jsonl
  citation_network/   # dblp.v12.json (12GB)
outputs/
  cache/              # LLM embedding + risk score caches
```
