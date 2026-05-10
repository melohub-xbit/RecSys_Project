"""
EASE — Embarrassingly Shallow Auto-Encoder for Sparse Data (Steck, WWW 2019).

Closed-form linear item-item model:

    G   = X^T X        (item co-occurrence)
    G  += λI           (ridge regularisation)
    P   = G^{-1}
    B   = P / -diag(P)  (column-normalised)
    diag(B) = 0
    score(user) = X_user · B

Often beats SASRec on dense MovieLens-style datasets while having no training
loop and no hyper-parameter tuning beyond λ. Limited to catalogs small enough
for an n_items × n_items dense inverse to fit in RAM (default cap 50 000
items — beyond that, EASE is skipped with a warning).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import numpy as np

from .offline_emb import CatalogStore
from .online import BaseRecommender

logger = logging.getLogger(__name__)

EASE_LAMBDA_DEFAULT = 500.0
EASE_MAX_ITEMS = 50_000


class EASERecommender(BaseRecommender):
    name = "EASE"

    def __init__(self, catalog: CatalogStore, train_interactions: list[tuple],
                 lam: float = EASE_LAMBDA_DEFAULT,
                 max_items: int = EASE_MAX_ITEMS):
        item_ids = catalog.item_index.item_ids
        n_items = len(item_ids)

        if n_items > max_items:
            raise ValueError(
                f"EASE catalog too large: {n_items} > max_items={max_items}. "
                f"Dense {n_items}x{n_items} inverse would not fit in RAM. "
                f"Use --no-ease for this catalog or raise --ease-max-items."
            )

        self._iid_to_idx = {iid: i for i, iid in enumerate(item_ids)}
        self._n_items = n_items

        # Build sparse user-item matrix from positive train interactions.
        from scipy.sparse import csr_matrix
        user_pos: dict[str, set[int]] = defaultdict(set)
        for uid, iid, r, _ in train_interactions:
            if r == 1 and iid in self._iid_to_idx:
                user_pos[uid].add(self._iid_to_idx[iid])
        users = list(user_pos.keys())
        rows: list[int] = []
        cols: list[int] = []
        for u_i, uid in enumerate(users):
            for j in user_pos[uid]:
                rows.append(u_i)
                cols.append(j)
        if not rows:
            raise ValueError("EASE: no positive interactions in train")

        X = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(users), n_items), dtype=np.float32,
        )
        logger.info("EASE: training on %d users × %d items (λ=%.1f)",
                    len(users), n_items, lam)

        G = (X.T @ X).toarray().astype(np.float64)
        diag_idx = np.arange(n_items)
        G[diag_idx, diag_idx] += lam
        P = np.linalg.inv(G)
        diagP = np.diag(P).copy()
        # Avoid divide-by-zero — λI keeps diag strictly positive in practice.
        B = P / -diagP[None, :]
        B[diag_idx, diag_idx] = 0.0
        self._B = B.astype(np.float32)

        # Popularity fallback for users with no positives in revealed history.
        pop: Counter = Counter()
        for uid, iid, r, _ in train_interactions:
            if r == 1 and iid in self._iid_to_idx:
                pop[iid] += 1
        self._pop_ranking = [iid for iid, _ in pop.most_common()]

    def score_history(self, history: list[tuple[str, int]],
                      candidate_ids: list[str]) -> dict[str, float]:
        """Return EASE score for each candidate given the user's revealed
        history. Cold users (no positives in history) get popularity-rank
        scaled to [0, 1] as a fallback so the value is always defined.

        Used by both `recommend()` (rank candidates) and `RACSRecommender`
        (extra term in Q, or as a narrowing scorer).
        """
        pos_idx = [self._iid_to_idx[iid] for iid, r in history
                   if r == 1 and iid in self._iid_to_idx]
        if not pos_idx:
            n_pop = max(1, len(self._pop_ranking))
            pop_rank = {iid: (n_pop - i) / n_pop
                        for i, iid in enumerate(self._pop_ranking)}
            return {iid: pop_rank.get(iid, 0.0) for iid in candidate_ids}

        scores_all = self._B[pos_idx, :].sum(axis=0)             # (n_items,)
        out: dict[str, float] = {}
        for iid in candidate_ids:
            j = self._iid_to_idx.get(iid)
            out[iid] = float(scores_all[j]) if j is not None else 0.0
        return out

    def recommend(self, history, candidates, top_k=10):
        seen = {iid for iid, _ in history}
        unseen = [iid for iid in candidates if iid not in seen]
        if not unseen:
            return []

        pos_idx = [self._iid_to_idx[iid] for iid, r in history
                   if r == 1 and iid in self._iid_to_idx]
        if not pos_idx:
            unseen_set = set(unseen)
            ranked = [iid for iid in self._pop_ranking if iid in unseen_set]
            for iid in unseen:
                if iid not in ranked:
                    ranked.append(iid)
            return ranked[:top_k]

        scores = self.score_history(history, unseen)
        scored = sorted(scores.items(), key=lambda kv: -kv[1])
        return [iid for iid, _ in scored[:top_k]]
