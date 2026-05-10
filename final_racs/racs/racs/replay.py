"""
Evaluation harness (sequential rollout + static top-K) for the HF online
path. Imports Dataset / CatalogStore / BaseRecommender from the local HF
modules.

Paper-grade upgrades over the previous version:

* NEG_SAMPLE bumped 200 → 1000 (closer to "real" full-catalog ranking).
* Per-user RNG seeded via hashlib.blake2b instead of Python's salted
  built-in `hash()` — runs are now reproducible across processes.
* Static eval also reports MRR, Novelty, Avg-R(a), Head-Recall,
  Tail-Recall per user, and Coverage / Gini aggregated over the test set.
* `DatasetStats` precomputes item popularity / head-tail buckets / novelty
  log-pop once per dataset.
* `split_users_by_first_interaction` provides a temporal cold-start
  partition (test = users whose first interaction is in the most recent
  20 % of the timeline) to complement the random split.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .data_loaders import Dataset
from .offline_emb import CatalogStore
from .online import BaseRecommender

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
NEG_SAMPLE = 1000
MAX_STEPS = 10
N_USERS_EVAL = 30
K_REVEAL_DEFAULTS = (3, 5, 10)
TOP_K_DEFAULT = 10
HEAD_FRAC = 0.20  # top-20 % of items by popularity = "head"


# ---------------------------------------------------------------------------
# Reproducible per-user RNG
# ---------------------------------------------------------------------------

def _user_rng(seed: int, uid: str) -> np.random.Generator:
    h = hashlib.blake2b(f"{seed}|{uid}".encode("utf-8"), digest_size=8).digest()
    user_seed = int.from_bytes(h, "little")
    return np.random.default_rng(user_seed)


# ---------------------------------------------------------------------------
# DatasetStats — precomputed per-dataset auxiliaries for paper metrics
# ---------------------------------------------------------------------------

@dataclass
class DatasetStats:
    popularity: dict[str, int]
    head_set: set[str]
    tail_set: set[str]
    novelty_log_pop: dict[str, float]   # −log2 P(item)
    n_total_items: int

    @classmethod
    def from_train(cls, train_interactions: list[tuple],
                   all_item_ids: list[str]) -> "DatasetStats":
        pop: Counter = Counter()
        for _, iid, r, _ in train_interactions:
            if r == 1:
                pop[iid] += 1

        n_total = len(all_item_ids)
        ranked = sorted(all_item_ids, key=lambda iid: -pop.get(iid, 0))
        head_n = max(1, int(HEAD_FRAC * n_total))
        head = set(ranked[:head_n])
        tail = set(ranked[head_n:])

        total_pop = sum(pop.values()) + n_total  # add-1 smoothing
        nov: dict[str, float] = {}
        for iid in all_item_ids:
            p = (pop.get(iid, 0) + 1) / total_pop
            nov[iid] = -math.log2(p)

        return cls(popularity=dict(pop), head_set=head, tail_set=tail,
                   novelty_log_pop=nov, n_total_items=n_total)


# ---------------------------------------------------------------------------
# User splits
# ---------------------------------------------------------------------------

@dataclass
class UserSplit:
    train_uids: set[str]
    test_uids: list[str]
    train_interactions: list[tuple]
    mode: str = "random"


def split_users(dataset: Dataset, train_frac: float = 0.8,
                n_test_users: int = N_USERS_EVAL,
                seed: int = RANDOM_SEED) -> UserSplit:
    rng = np.random.RandomState(seed)
    uids = np.array(list(dataset.user_sequences.keys()))
    rng.shuffle(uids)

    split = max(1, int(len(uids) * train_frac))
    train_uids = set(uids[:split].tolist())
    test_uids_all = uids[split:].tolist()

    if len(test_uids_all) > n_test_users > 0:
        chosen = rng.choice(len(test_uids_all), size=n_test_users, replace=False)
        test_uids = [test_uids_all[i] for i in chosen]
    else:
        test_uids = test_uids_all

    train_interactions = [
        (uid, iid, r, ts) for (uid, iid, r, ts) in dataset.interactions
        if uid in train_uids
    ]
    logger.info("UserSplit (random): %d train / %d test users (%d train interactions)",
                len(train_uids), len(test_uids), len(train_interactions))
    return UserSplit(train_uids=train_uids, test_uids=test_uids,
                     train_interactions=train_interactions, mode="random")


def split_users_by_first_interaction(dataset: Dataset,
                                     train_frac: float = 0.8,
                                     n_test_users: int = N_USERS_EVAL,
                                     seed: int = RANDOM_SEED) -> UserSplit:
    """Temporal cold-start split.

    Order users by the timestamp of their first interaction; the earliest
    `train_frac` are the train set ("legacy users"), the rest are the test
    set ("new arrivals"). All interactions of a train user feed the
    popularity / SASRec / LinUCB baselines; test users are evaluated as
    cold starts.
    """
    user_first_ts: dict[str, float] = {}
    for uid, _iid, _r, ts in dataset.interactions:
        if uid not in user_first_ts or ts < user_first_ts[uid]:
            user_first_ts[uid] = ts

    ordered = sorted(user_first_ts.items(), key=lambda kv: kv[1])
    split = max(1, int(len(ordered) * train_frac))
    train_uids = {uid for uid, _ in ordered[:split]}
    test_uids_all = [uid for uid, _ in ordered[split:]]

    rng = np.random.RandomState(seed)
    if len(test_uids_all) > n_test_users > 0:
        chosen = rng.choice(len(test_uids_all), size=n_test_users, replace=False)
        test_uids = [test_uids_all[i] for i in chosen]
    else:
        test_uids = test_uids_all

    train_interactions = [
        (uid, iid, r, ts) for (uid, iid, r, ts) in dataset.interactions
        if uid in train_uids
    ]
    logger.info("UserSplit (temporal): %d train / %d test users (%d train interactions)",
                len(train_uids), len(test_uids), len(train_interactions))
    return UserSplit(train_uids=train_uids, test_uids=test_uids,
                     train_interactions=train_interactions, mode="temporal")


# ---------------------------------------------------------------------------
# Per-user candidate pool
# ---------------------------------------------------------------------------

def _build_user_candidate_pool(user_seq: list[tuple[str, int]],
                               all_item_ids: list[str],
                               neg_sample: int,
                               rng: np.random.Generator) -> dict[str, int]:
    user_items = {iid: r for iid, r in user_seq}
    catalog_set = set(all_item_ids)
    user_set = set(user_items.keys())
    rest = list(catalog_set - user_set)
    n_neg = min(neg_sample, len(rest))
    if n_neg > 0:
        idx = rng.choice(len(rest), size=n_neg, replace=False)
        neg_ids = [rest[i] for i in idx]
    else:
        neg_ids = []
    pool = dict(user_items)
    for iid in neg_ids:
        pool[iid] = 0
    return pool


# ---------------------------------------------------------------------------
# Sequential rollout
# ---------------------------------------------------------------------------

@dataclass
class SequentialResult:
    per_user_rewards: list[list[int]] = field(default_factory=list)

    @property
    def cumulative_reward(self) -> float:
        return float(np.mean([sum(rs) for rs in self.per_user_rewards])) \
            if self.per_user_rewards else 0.0

    @property
    def precision_at_T(self) -> float:
        totals = [sum(rs) / max(1, len(rs)) for rs in self.per_user_rewards]
        return float(np.mean(totals)) if totals else 0.0

    @property
    def first_step_skip_rate(self) -> float:
        firsts = [rs[0] for rs in self.per_user_rewards if rs]
        return 1.0 - float(np.mean(firsts)) if firsts else 0.0

    def summary(self) -> dict:
        return {
            "cum_reward": round(self.cumulative_reward, 4),
            "precision_at_T": round(self.precision_at_T, 4),
            "first_step_skip_rate": round(self.first_step_skip_rate, 4),
            "n_users": len(self.per_user_rewards),
        }


def evaluate_sequential(model: BaseRecommender,
                        dataset: Dataset,
                        catalog: CatalogStore,
                        test_uids: list[str],
                        max_steps: int = MAX_STEPS,
                        neg_sample: int = NEG_SAMPLE,
                        seed: int = RANDOM_SEED) -> SequentialResult:
    all_ids = catalog.item_index.item_ids
    result = SequentialResult()

    for uid in test_uids:
        user_seq = dataset.user_sequences.get(uid, [])
        if len(user_seq) < 2:
            continue
        rng = _user_rng(seed, uid)
        pool = _build_user_candidate_pool(user_seq, all_ids, neg_sample, rng)

        history: list[tuple[str, int]] = []
        rewards: list[int] = []
        model.reset()

        for _ in range(max_steps):
            remaining = [iid for iid in pool.keys()
                         if iid not in {h[0] for h in history}]
            if not remaining:
                break
            top = model.recommend(history, remaining, top_k=1)
            if not top:
                break
            picked = top[0]
            r = pool[picked]
            rewards.append(int(r))
            history.append((picked, int(r)))

        result.per_user_rewards.append(rewards)

    return result


# ---------------------------------------------------------------------------
# Static top-K ranking — paper-grade metrics
# ---------------------------------------------------------------------------

def _recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for iid in ranked[:k] if iid in relevant) / len(relevant)


def _ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(r + 1)
              for r, iid in enumerate(ranked[:k], start=1) if iid in relevant)
    idcg = sum(1.0 / np.log2(r + 1)
               for r in range(1, min(len(relevant), k) + 1))
    return float(dcg / idcg) if idcg else 0.0


def _mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for r, iid in enumerate(ranked[:k], start=1):
        if iid in relevant:
            return 1.0 / r
    return 0.0


def _hr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if any(iid in relevant for iid in ranked[:k]) else 0.0


def _precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    return sum(1 for iid in ranked[:k] if iid in relevant) / k


def _intra_diversity(ranked: list[str], catalog: CatalogStore, k: int) -> float:
    """Mean pairwise (1 − cosine sim) across the top-K recommended items.

    Item embeddings in the catalog are already L2-normalised, so dot product
    equals cosine similarity. Returns 0 if fewer than 2 items have embeddings.
    """
    items = ranked[:k]
    embs = []
    for iid in items:
        e = catalog.item_index.get_embedding(iid)
        if e is not None:
            embs.append(e)
    if len(embs) < 2:
        return 0.0
    arr = np.stack(embs)
    sims = arr @ arr.T
    n = arr.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(1.0 - sims[iu].mean())


def _auc_full(ranked_full: list[str], all_candidates: list[str],
              relevant: set[str]) -> float:
    """ROC-AUC over the candidate pool.

    Items returned by `model.recommend(top_k=len(candidates))` get scores
    from −position; items not in the model's ranking are tied at the worst
    score. Returns NaN if the user has zero or all-positive candidates.
    """
    if not relevant or not all_candidates:
        return float("nan")
    rank_pos = {iid: i for i, iid in enumerate(ranked_full)}
    worst = len(ranked_full)
    y_true = [1 if iid in relevant else 0 for iid in all_candidates]
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return float("nan")
    scores = [-(rank_pos.get(iid, worst)) for iid in all_candidates]
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def _bucket_recall(ranked: list[str], relevant_bucket: set[str], k: int) -> Optional[float]:
    """Recall@k restricted to a bucket. Returns None if the bucket has no
    positives for this user (so the caller can skip the user from the mean)."""
    if not relevant_bucket:
        return None
    return sum(1 for iid in ranked[:k] if iid in relevant_bucket) / len(relevant_bucket)


@dataclass
class StaticResult:
    top_k: int = TOP_K_DEFAULT
    recalls: list[float] = field(default_factory=list)
    ndcgs: list[float] = field(default_factory=list)
    mrrs: list[float] = field(default_factory=list)
    hrs: list[float] = field(default_factory=list)
    precisions: list[float] = field(default_factory=list)
    intra_diversities: list[float] = field(default_factory=list)
    aucs: list[float] = field(default_factory=list)
    novelties: list[float] = field(default_factory=list)
    avg_risks: list[float] = field(default_factory=list)
    head_recalls: list[float] = field(default_factory=list)
    tail_recalls: list[float] = field(default_factory=list)
    ranked_items: list[list[str]] = field(default_factory=list)
    user_ids: list[str] = field(default_factory=list)

    @staticmethod
    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    @property
    def recall(self) -> float: return self._mean(self.recalls)

    @property
    def ndcg(self) -> float: return self._mean(self.ndcgs)

    @property
    def mrr(self) -> float: return self._mean(self.mrrs)

    @property
    def hr(self) -> float: return self._mean(self.hrs)

    @property
    def precision(self) -> float: return self._mean(self.precisions)

    @property
    def intra_diversity(self) -> float: return self._mean(self.intra_diversities)

    @property
    def auc(self) -> float:
        clean = [a for a in self.aucs if a == a]   # drop NaN
        return self._mean(clean)

    @property
    def novelty(self) -> float: return self._mean(self.novelties)

    @property
    def avg_risk(self) -> float: return self._mean(self.avg_risks)

    @property
    def head_recall(self) -> float: return self._mean(self.head_recalls)

    @property
    def tail_recall(self) -> float: return self._mean(self.tail_recalls)

    def coverage(self, n_total_items: int) -> float:
        if not self.ranked_items or n_total_items == 0:
            return 0.0
        seen: set[str] = set()
        for r in self.ranked_items:
            seen.update(r[:self.top_k])
        return len(seen) / n_total_items

    def coverage_relative(self, n_total_items: int) -> float:
        """Coverage normalised by the theoretical maximum.

        On sparse-catalog domains (e.g. SemanticScholar-bigger: 175k items,
        100 users × 10 recs) plain Coverage@K is upper-bounded by
        n_users·top_k / n_total_items ≪ 1, which makes it uninformative.
        Reporting `unique recs / min(catalog, n_users·top_k)` gives a value
        in [0, 1] regardless of catalog size and is the diversity definition
        used in Aggarwal's RecSys textbook.
        """
        if not self.ranked_items or n_total_items == 0:
            return 0.0
        seen: set[str] = set()
        for r in self.ranked_items:
            seen.update(r[:self.top_k])
        cap = min(n_total_items, len(self.ranked_items) * self.top_k)
        return len(seen) / cap if cap else 0.0

    def gini(self) -> float:
        if not self.ranked_items:
            return 0.0
        counts: Counter = Counter()
        for r in self.ranked_items:
            counts.update(r[:self.top_k])
        if not counts:
            return 0.0
        vals = sorted(counts.values())
        n = len(vals)
        cum = sum((i + 1) * v for i, v in enumerate(vals))
        s = sum(vals)
        if s == 0:
            return 0.0
        return (2.0 * cum) / (n * s) - (n + 1) / n

    def personalisation(self, n_pairs: int = 1000, seed: int = 42) -> float:
        """Mean pairwise (1 − Jaccard) between users' top-K lists. Aggregate
        metric: how distinct each user's recommendations are. Sampled to
        n_pairs random user pairs when there are too many to compute exactly.
        """
        if len(self.ranked_items) < 2:
            return 0.0
        topks = [set(r[:self.top_k]) for r in self.ranked_items]
        n = len(topks)
        rng = np.random.default_rng(seed)
        max_pairs = n * (n - 1) // 2
        if max_pairs <= n_pairs:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            seen_pairs: set[tuple[int, int]] = set()
            pairs = []
            while len(pairs) < n_pairs:
                i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                if (i, j) in seen_pairs:
                    continue
                seen_pairs.add((i, j))
                pairs.append((i, j))

        dists = []
        for i, j in pairs:
            a, b = topks[i], topks[j]
            u = a | b
            if not u:
                continue
            dists.append(1.0 - len(a & b) / len(u))
        return self._mean(dists)

    def summary(self, n_total_items: int = 0) -> dict:
        return {
            "recall": round(self.recall, 4),
            "ndcg": round(self.ndcg, 4),
            "mrr": round(self.mrr, 4),
            "hr": round(self.hr, 4),
            "precision": round(self.precision, 4),
            "intra_diversity": round(self.intra_diversity, 4),
            "auc": round(self.auc, 4),
            "novelty": round(self.novelty, 4),
            "avg_risk": round(self.avg_risk, 4),
            "head_recall": round(self.head_recall, 4),
            "tail_recall": round(self.tail_recall, 4),
            "coverage": round(self.coverage(n_total_items), 4) if n_total_items else 0.0,
            "coverage_relative": round(self.coverage_relative(n_total_items), 4) if n_total_items else 0.0,
            "gini": round(self.gini(), 4),
            "personalisation": round(self.personalisation(), 4),
            "n_users": len(self.recalls),
        }


def evaluate_static(model: BaseRecommender,
                    dataset: Dataset,
                    catalog: CatalogStore,
                    test_uids: list[str],
                    k_reveal: int,
                    top_k: int = TOP_K_DEFAULT,
                    neg_sample: int = NEG_SAMPLE,
                    seed: int = RANDOM_SEED,
                    stats: Optional[DatasetStats] = None) -> StaticResult:
    all_ids = catalog.item_index.item_ids
    result = StaticResult(top_k=top_k)

    for uid in test_uids:
        user_seq = dataset.user_sequences.get(uid, [])
        if len(user_seq) < k_reveal + 1:
            continue

        history = user_seq[:k_reveal]
        held = user_seq[k_reveal:]
        relevant = {iid for iid, r in held if r == 1}
        if not relevant:
            continue

        rng = _user_rng(seed, uid)
        pool = _build_user_candidate_pool(user_seq, all_ids, neg_sample, rng)
        seen = {iid for iid, _ in history}
        candidates = [iid for iid in pool.keys() if iid not in seen]

        model.reset()
        # Request the full ranking the model can produce so we can compute
        # AUC; slice to top_k for the K-bound metrics.
        ranked = model.recommend(list(history), candidates,
                                 top_k=len(candidates))

        result.recalls.append(_recall_at_k(ranked, relevant, top_k))
        result.ndcgs.append(_ndcg_at_k(ranked, relevant, top_k))
        result.mrrs.append(_mrr_at_k(ranked, relevant, top_k))
        result.hrs.append(_hr_at_k(ranked, relevant, top_k))
        result.precisions.append(_precision_at_k(ranked, relevant, top_k))
        result.intra_diversities.append(_intra_diversity(ranked, catalog, top_k))
        result.aucs.append(_auc_full(ranked, candidates, relevant))
        result.ranked_items.append(list(ranked))
        result.user_ids.append(uid)

        if stats is not None:
            top_ranked = ranked[:top_k]
            if top_ranked:
                novs = [stats.novelty_log_pop.get(iid, 0.0) for iid in top_ranked]
                result.novelties.append(float(np.mean(novs)))
            if catalog.risk_scores and top_ranked:
                risks = [catalog.risk_scores.get(iid, 0.0) for iid in top_ranked]
                result.avg_risks.append(float(np.mean(risks)))
            head_pos = relevant & stats.head_set
            tail_pos = relevant & stats.tail_set
            head_r = _bucket_recall(ranked, head_pos, top_k)
            tail_r = _bucket_recall(ranked, tail_pos, top_k)
            if head_r is not None:
                result.head_recalls.append(head_r)
            if tail_r is not None:
                result.tail_recalls.append(tail_r)

    return result
