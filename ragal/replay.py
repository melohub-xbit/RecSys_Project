import logging
from dataclasses import dataclass, field

import numpy as np

from .data_loaders import Dataset
from .offline import CatalogStore
from .online import BaseRecommender

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
NEG_SAMPLE = 200
MAX_STEPS = 10
N_USERS_EVAL = 30
K_REVEAL_DEFAULTS = (3, 5, 10)
TOP_K_DEFAULT = 10

# User split functions
@dataclass
class UserSplit:
    train_uids: set[str]
    test_uids: list[str]
    train_interactions: list[tuple]

def split_users(dataset: Dataset, train_frac: float = 0.8,
                n_test_users: int = N_USERS_EVAL,
                seed: int = RANDOM_SEED) -> UserSplit:
    rng = np.random.RandomState(seed)
    uids = np.array(list(dataset.user_sequences.keys()))
    rng.shuffle(uids)

    split = max(1, int(len(uids) * train_frac))
    train_uids = set(uids[:split].tolist())
    test_uids_all = uids[split:].tolist()

    if len(test_uids_all) > n_test_users:
        chosen = rng.choice(len(test_uids_all), size=n_test_users, replace=False)
        test_uids = [test_uids_all[i] for i in chosen]
    else:
        test_uids = test_uids_all

    train_interactions = [
        (uid, iid, r, ts) for (uid, iid, r, ts) in dataset.interactions
        if uid in train_uids
    ]
    logger.info("UserSplit: %d train / %d test users (%d train interactions)",
                len(train_uids), len(test_uids), len(train_interactions))
    return UserSplit(train_uids=train_uids,
                     test_uids=test_uids,
                     train_interactions=train_interactions)

# Candidate pool functions
def _build_user_candidate_pool(user_seq: list[tuple[str, int]],
                               all_item_ids: list[str],
                               neg_sample: int,
                               rng: np.random.RandomState) -> dict[str, int]:
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

# Sequential rollout
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
        rng = np.random.RandomState(seed + hash(uid) % (2**31))
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

# Static top-K ranking
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

@dataclass
class StaticResult:
    recalls: list[float] = field(default_factory=list)
    ndcgs: list[float] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return float(np.mean(self.recalls)) if self.recalls else 0.0

    @property
    def ndcg(self) -> float:
        return float(np.mean(self.ndcgs)) if self.ndcgs else 0.0

    def summary(self) -> dict:
        return {
            "recall": round(self.recall, 4),
            "ndcg": round(self.ndcg, 4),
            "n_users": len(self.recalls),
        }

def evaluate_static(model: BaseRecommender,
                    dataset: Dataset,
                    catalog: CatalogStore,
                    test_uids: list[str],
                    k_reveal: int,
                    top_k: int = TOP_K_DEFAULT,
                    neg_sample: int = NEG_SAMPLE,
                    seed: int = RANDOM_SEED) -> StaticResult:
    all_ids = catalog.item_index.item_ids
    result = StaticResult()

    for uid in test_uids:
        user_seq = dataset.user_sequences.get(uid, [])
        if len(user_seq) < k_reveal + 1:
            continue

        history = user_seq[:k_reveal]
        held = user_seq[k_reveal:]
        relevant = {iid for iid, r in held if r == 1}
        if not relevant:
            continue

        rng = np.random.RandomState(seed + hash(uid) % (2**31))
        pool = _build_user_candidate_pool(user_seq, all_ids, neg_sample, rng)
        seen = {iid for iid, _ in history}
        candidates = [iid for iid in pool.keys() if iid not in seen]

        model.reset()
        ranked = model.recommend(list(history), candidates, top_k=top_k)
        result.recalls.append(_recall_at_k(ranked, relevant, top_k))
        result.ndcgs.append(_ndcg_at_k(ranked, relevant, top_k))

    return result
