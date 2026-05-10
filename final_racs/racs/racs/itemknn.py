"""
ItemKNN baseline (Deshpande & Karypis 2004, Sarwar et al. 2001).

Cosine-similarity item-based collaborative filtering using the pre-computed
BGE-large embeddings already present in the CatalogStore.  No model training
is required — scoring is done purely at inference time by comparing candidate
embeddings to the user's revealed positive history.

For a cold-start user with history H = {(item, reward)}, the score of a
candidate c is:

    score(c) = (1 / |H_pos|) · Σ_{i ∈ H_pos}  cos(emb_c, emb_i)

where H_pos = {i : (i, r) ∈ H, r = 1} are the positively-interacted items.
If the user has no positives yet, fall back to global popularity.

Design notes
------------
* Embeddings in the CatalogStore are already L2-normalised, so dot product
  equals cosine similarity.
* The recommender implements the same `BaseRecommender.recommend()` interface
  as LinUCB, SASRec, and the RACS variants, making it a drop-in baseline
  for the evaluation harness.
* No parameters are learned; the only "training" input is `train_interactions`
  used solely to build a popularity fallback for zero-positive users.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import numpy as np

from .offline_emb import CatalogStore
from .online import BaseRecommender

logger = logging.getLogger(__name__)


class ItemKNNRecommender(BaseRecommender):
    name = "ItemKNN"

    def __init__(self, catalog: CatalogStore,
                 train_interactions: list[tuple]):
        """
        Parameters
        ----------
        catalog : CatalogStore
            Pre-loaded catalog with L2-normalised item embeddings.
        train_interactions : list[tuple]
            (uid, iid, reward, ts) tuples from the train split.
            Used only to build a popularity fallback ranking.
        """
        self.catalog = catalog

        # Popularity fallback (same logic as PopularityRecommender)
        self._popularity: Counter = Counter()
        for _, iid, r, _ in train_interactions:
            if r == 1:
                self._popularity[iid] += 1
        self._pop_ranking = [iid for iid, _ in self._popularity.most_common()]

    def recommend(self, history: list[tuple[str, int]],
                  candidates: list[str], top_k: int = 10) -> list[str]:
        seen = {iid for iid, _ in history}
        unseen = [iid for iid in candidates if iid not in seen]
        if not unseen:
            return []

        # Collect positive history embeddings
        pos_embs = []
        for iid, r in history:
            if r == 1:
                emb = self.catalog.item_index.get_embedding(iid)
                if emb is not None:
                    pos_embs.append(emb)

        # Fallback to popularity if no positive embeddings
        if not pos_embs:
            unseen_set = set(unseen)
            ranked = [iid for iid in self._pop_ranking if iid in unseen_set]
            for iid in unseen:
                if iid not in ranked:
                    ranked.append(iid)
            return ranked[:top_k]

        # Stack positive embeddings: (n_pos, dim)
        pos_matrix = np.stack(pos_embs, axis=0)  # (n_pos, D)

        # Collect candidate embeddings
        cand_ids_valid = []
        cand_embs = []
        for iid in unseen:
            emb = self.catalog.item_index.get_embedding(iid)
            if emb is not None:
                cand_ids_valid.append(iid)
                cand_embs.append(emb)

        if not cand_embs:
            return unseen[:top_k]

        # Vectorised scoring: (n_cand, D) @ (D, n_pos) → (n_cand, n_pos)
        cand_matrix = np.stack(cand_embs, axis=0)  # (n_cand, D)
        sim_matrix = cand_matrix @ pos_matrix.T     # (n_cand, n_pos)
        scores = sim_matrix.mean(axis=1)             # (n_cand,)

        # Rank by descending score
        order = np.argsort(-scores)
        ranked = [cand_ids_valid[i] for i in order]
        return ranked[:top_k]
