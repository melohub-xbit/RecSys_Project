"""
Online phase for RACS — HuggingFace Transformers backend.

ACS is the literal Part-5 formulation:

    P_ACS = exp(logit[ENGAGED]) / [exp(logit[ENGAGED]) + exp(logit[SKIPPED])]

computed by `HFLLM.contrast_logprob` at the next-token position after the
prompt.  No verbalised-probability proxy, no parsing, no fallback.

Cold-start branch (no positives or no negatives in history): the same
two-token contrast against `ACS_ABSOLUTE_PROMPT` — an explicit branch per
spec Part 5, not a silent default.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD

from .hf_model import HFLLM
from .offline_emb import CatalogStore

logger = logging.getLogger(__name__)

ALPHA_DEFAULT = 2.0          # was 0.5 — ACS produced scores in [0,1] dwarfed SND in [0,0.5]
LAMBDA_DEFAULT = 0.3
LAMBDA_PAPERS_DEFAULT = 0.05  # niche papers were being penalised as "risky"; small λ for sparse domains
GAMMA_DEFAULT = 0.3           # weight on the EASE blend term
K_SND_DEFAULT = 5
CANDIDATE_SIZE_DEFAULT = 200  # was 40 — too aggressive narrowing crushed AUC and Recall room
EPSILON_FLOOR = 0.05
SND_MODE_DEFAULT = "bayesian"
NARROWING_DEFAULT = "centroid"   # alt: "ease" — uses EASE B-matrix scores instead of FAISS over centroid

ACS_WORD_POS = "ENGAGED"
ACS_WORD_NEG = "SKIPPED"


# ---------------------------------------------------------------------------
# Reward-weighted centroid (Part 4, stage 1)
# ---------------------------------------------------------------------------

def reward_weighted_centroid(history: list[tuple[str, int]],
                             catalog: CatalogStore) -> np.ndarray:
    dim = catalog.item_index.dim
    if not history:
        return np.zeros(dim, dtype=np.float32)
    num = np.zeros(dim, dtype=np.float32)
    denom = 0.0
    for iid, r in history:
        emb = catalog.item_index.get_embedding(iid)
        if emb is None:
            raise KeyError(f"History item {iid} has no embedding in catalog")
        num += r * emb
        denom += r
    return num / (denom + 1e-8)


# ---------------------------------------------------------------------------
# SND (Part 6)
# ---------------------------------------------------------------------------

def _snd_neighbors(cand_id: str,
                   history: list[tuple[str, int]],
                   catalog: CatalogStore,
                   k: int) -> tuple[int, int, int]:
    """Return (n_pos, n_neg, k_eff) for the k nearest neighbours of cand_id
    inside the user's revealed history."""
    cand_emb = catalog.item_index.get_embedding(cand_id)
    if cand_emb is None:
        raise KeyError(f"No embedding for candidate {cand_id}")
    sims: list[tuple[float, int]] = []
    for iid, r in history:
        emb = catalog.item_index.get_embedding(iid)
        if emb is None:
            raise KeyError(f"History item {iid} has no embedding in catalog")
        sims.append((float(np.dot(cand_emb, emb)), r))
    sims.sort(key=lambda x: -x[0])
    k_eff = min(k, len(sims))
    top = sims[:k_eff]
    n_pos = sum(1 for _, r in top if r == 1)
    n_neg = sum(1 for _, r in top if r == 0)
    return n_pos, n_neg, k_eff


def _snd_neighbors_weighted(cand_id: str,
                            history: list[tuple[str, int]],
                            catalog: CatalogStore,
                            k: int) -> tuple[float, float, int]:
    """Return (w_pos, w_neg, k_eff) with *similarity-weighted* soft counts.

    Unlike the unweighted version, this makes SND discriminative even when
    the history is very short (k_reveal ≤ 5): every candidate still queries
    the same history items, but the *weights* (cosine similarities) differ
    per candidate, breaking the degeneracy.
    """
    cand_emb = catalog.item_index.get_embedding(cand_id)
    if cand_emb is None:
        raise KeyError(f"No embedding for candidate {cand_id}")
    sims: list[tuple[float, int]] = []
    for iid, r in history:
        emb = catalog.item_index.get_embedding(iid)
        if emb is None:
            raise KeyError(f"History item {iid} has no embedding in catalog")
        sims.append((float(np.dot(cand_emb, emb)), r))
    sims.sort(key=lambda x: -x[0])
    k_eff = min(k, len(sims))
    top = sims[:k_eff]
    # Shift similarities to [0, 1] via (sim + 1) / 2 (cosine range is [-1, 1])
    w_pos = sum((s + 1.0) / 2.0 for s, r in top if r == 1)
    w_neg = sum((s + 1.0) / 2.0 for s, r in top if r == 0)
    return w_pos, w_neg, k_eff


def snd_score_heuristic(cand_id, history, catalog,
                        k: int = K_SND_DEFAULT,
                        epsilon: float = EPSILON_FLOOR) -> float:
    """Original heuristic from the paper: min/(max+1) ratio in [0, 0.5]."""
    if not history:
        return epsilon
    n_pos, n_neg, k_eff = _snd_neighbors(cand_id, history, catalog, k)
    snd = min(n_pos, n_neg) / (max(n_pos, n_neg) + 1)
    if k_eff < 2:
        snd += epsilon
    return snd


def snd_score_bayesian(cand_id, history, catalog,
                       k: int = K_SND_DEFAULT,
                       epsilon: float = EPSILON_FLOOR) -> float:
    """Similarity-weighted Bayesian SND.

    Uses cosine-similarity-weighted soft counts instead of hard neighbour
    counts. This makes the posterior *candidate-dependent* even when the
    history is very short (k_reveal ≤ k), fixing the degeneracy where all
    candidates receive an identical SND score.

    The posterior std of Beta(1 + w_pos, 1 + w_neg) is maximised at
    perfect disagreement and collapses smoothly to 0 at full agreement.
    Bounded in [0, 1/sqrt(12)] ≈ [0, 0.289] regardless of k, which lets
    us combine cleanly with ACS in [0, 1] under a single α.
    """
    if not history:
        return epsilon
    w_pos, w_neg, k_eff = _snd_neighbors_weighted(cand_id, history, catalog, k)
    a = 1.0 + w_pos
    b = 1.0 + w_neg
    var = (a * b) / ((a + b) ** 2 * (a + b + 1.0))
    snd = float(var ** 0.5)
    if k_eff < 2:
        snd += epsilon
    return snd


def snd_score(cand_id: str,
              history: list[tuple[str, int]],
              catalog: CatalogStore,
              k: int = K_SND_DEFAULT,
              epsilon: float = EPSILON_FLOOR,
              mode: str = SND_MODE_DEFAULT) -> float:
    """Dispatch on mode. Default is the Bayesian formulation."""
    if mode == "heuristic":
        return snd_score_heuristic(cand_id, history, catalog, k, epsilon)
    return snd_score_bayesian(cand_id, history, catalog, k, epsilon)


# ---------------------------------------------------------------------------
# ACS — contrastive logprob over an HF local model (Part 5)
# ---------------------------------------------------------------------------

ACS_CONTRASTIVE_PROMPT = """A user engaged with this item:
"{x_plus}"

But skipped this item:
"{x_minus}"

Now consider this new item:
"{candidate}"

Decide whether the user will engage with or skip the new item. Judge by whether it is closer in content and style to what they engaged with or to what they skipped. Answer with a single word: ENGAGED or SKIPPED.

Answer:"""

ACS_ABSOLUTE_PROMPT = """Consider this item:
"{candidate}"

With no information about any particular user's taste, decide whether a typical user will engage with or skip this item. Consider general appeal and mainstream relevance. Answer with a single word: ENGAGED or SKIPPED.

Answer:"""


class ACSScorer:
    """
    HFLLM-backed ACS scorer with an on-disk JSON cache.

    Cache key = (cand_id, x_plus_id, x_minus_id). x_plus/x_minus are None
    for the cold-start absolute branch.
    """

    def __init__(self, catalog: CatalogStore, llm: HFLLM,
                 cache_path: Optional[Path] = None,
                 batch_size: int = 16):
        self.catalog = catalog
        self.llm = llm
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, float] = {}
        self._dirty = False
        self.batch_size = batch_size

        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path) as f:
                self.cache = json.load(f)
            logger.info("ACS cache loaded: %d entries from %s",
                        len(self.cache), self.cache_path)

    @staticmethod
    def _key(cand_id: str, x_plus_id: Optional[str], x_minus_id: Optional[str]) -> str:
        return f"{cand_id}|{x_plus_id or ''}|{x_minus_id or ''}"

    def flush(self):
        if self.cache_path is None or not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)
        self._dirty = False
        logger.info("ACS cache flushed: %d entries → %s",
                    len(self.cache), self.cache_path)

    def _closest_by_cosine(self, cand_emb: np.ndarray, item_ids: list[str]) -> str:
        best_id = None
        best_sim = -1e9
        for iid in item_ids:
            emb = self.catalog.item_index.get_embedding(iid)
            if emb is None:
                raise KeyError(f"Anchor candidate {iid} has no embedding")
            sim = float(np.dot(cand_emb, emb))
            if sim > best_sim:
                best_sim = sim
                best_id = iid
        if best_id is None:
            raise RuntimeError(f"No anchor selected from {item_ids}")
        return best_id

    def _build_prompt(self, cand_id: str,
                      x_plus_id: Optional[str],
                      x_minus_id: Optional[str]) -> str:
        cand_text = self.catalog.dataset.items[cand_id].text[:400]
        if x_plus_id is None or x_minus_id is None:
            return ACS_ABSOLUTE_PROMPT.format(candidate=cand_text)
        xp_text = self.catalog.dataset.items[x_plus_id].text[:300]
        xm_text = self.catalog.dataset.items[x_minus_id].text[:300]
        return ACS_CONTRASTIVE_PROMPT.format(
            x_plus=xp_text, x_minus=xm_text, candidate=cand_text,
        )

    def score_many(self, cand_ids: list[str],
                   pos_history: list[str],
                   neg_history: list[str]) -> dict[str, float]:
        """Batched ACS: collect all cache misses, run them through one
        `contrast_logprob_batch` call (chunked at `self.batch_size`), then
        merge with cache hits.
        """
        results: dict[str, float] = {}
        miss_keys: list[str] = []
        miss_prompts: list[str] = []
        miss_cands: list[str] = []

        cold = (not pos_history) or (not neg_history)
        for cid in cand_ids:
            cand_emb = self.catalog.item_index.get_embedding(cid)
            if cand_emb is None:
                raise KeyError(f"No embedding for candidate {cid}")
            if cold:
                key = self._key(cid, None, None)
                xp_id = xm_id = None
            else:
                xp_id = self._closest_by_cosine(cand_emb, pos_history)
                xm_id = self._closest_by_cosine(cand_emb, neg_history)
                key = self._key(cid, xp_id, xm_id)
            if key in self.cache:
                results[cid] = self.cache[key]
            else:
                miss_keys.append(key)
                miss_prompts.append(self._build_prompt(cid, xp_id, xm_id))
                miss_cands.append(cid)

        if miss_prompts:
            scores = self.llm.contrast_logprob_batch(
                miss_prompts, ACS_WORD_POS, ACS_WORD_NEG,
                batch_size=self.batch_size,
            )
            for cid, key, p in zip(miss_cands, miss_keys, scores):
                results[cid] = float(p)
                self.cache[key] = float(p)
            self._dirty = True
        return results


# ---------------------------------------------------------------------------
# Recommenders
# ---------------------------------------------------------------------------

class BaseRecommender:
    name: str = "Base"

    def recommend(self, history: list[tuple[str, int]],
                  candidates: list[str], top_k: int = 10) -> list[str]:
        raise NotImplementedError

    def reset(self):
        pass


class RACSRecommender(BaseRecommender):
    def __init__(self, catalog: CatalogStore, acs_scorer: Optional[ACSScorer],
                 alpha: float = ALPHA_DEFAULT, lam: float = LAMBDA_DEFAULT,
                 k: int = K_SND_DEFAULT,
                 candidate_size: int = CANDIDATE_SIZE_DEFAULT,
                 use_acs: bool = True, use_snd: bool = True, use_risk: bool = True,
                 snd_mode: str = SND_MODE_DEFAULT,
                 ease_scorer=None,
                 gamma: float = GAMMA_DEFAULT,
                 use_ease: bool = False,
                 narrowing_mode: str = NARROWING_DEFAULT):
        self.catalog = catalog
        self.acs = acs_scorer
        self.alpha = alpha
        self.lam = lam
        self.k = k
        self.candidate_size = candidate_size
        self.use_acs = use_acs
        self.use_snd = use_snd
        self.use_risk = use_risk
        self.snd_mode = snd_mode
        self.ease_scorer = ease_scorer
        self.gamma = gamma
        self.use_ease = use_ease
        self.narrowing_mode = narrowing_mode

        if use_acs and acs_scorer is None:
            raise ValueError("use_acs=True requires an ACSScorer")
        if (use_ease or narrowing_mode == "ease") and ease_scorer is None:
            raise ValueError(
                "use_ease=True or narrowing_mode='ease' requires an EASE scorer "
                "(an instance of EASERecommender exposing score_history()).")

        parts = []
        if use_acs:  parts.append("ACS")
        if use_snd:  parts.append("SND")
        if use_risk: parts.append("R(a)")
        if use_ease: parts.append("EASE")
        base = "+".join(parts) if parts else "Zero"
        self.name = base + (" [EASE-nar]" if narrowing_mode == "ease" else "")

    def _narrow_candidates(self, history: list[tuple[str, int]],
                           full_candidates: list[str]) -> list[str]:
        M = self.candidate_size
        seen = {iid for iid, _ in history}
        unseen = [iid for iid in full_candidates if iid not in seen]

        if not history or len(unseen) <= M:
            return unseen[:M]

        # EASE-narrowing: rank candidates by the linear item-item model and
        # keep the top M. Falls back to centroid if the EASE scorer wasn't
        # supplied (constructor-checked, but defensive).
        if self.narrowing_mode == "ease" and self.ease_scorer is not None:
            scores = self.ease_scorer.score_history(history, unseen)
            return [iid for iid, _ in sorted(scores.items(),
                                             key=lambda kv: -kv[1])][:M]

        centroid = reward_weighted_centroid(history, self.catalog)
        if float(np.linalg.norm(centroid)) < 1e-9:
            return unseen[:M]

        centroid = (centroid / (np.linalg.norm(centroid) + 1e-9)).astype(np.float32)
        fetch = min(M * 4, len(self.catalog.item_index.item_ids))
        results = self.catalog.item_index.query(centroid, k=fetch, exclude_ids=seen)
        cand_set = set(unseen)
        narrowed = [iid for iid, _ in results if iid in cand_set]

        for iid in unseen:
            if len(narrowed) >= M:
                break
            if iid not in narrowed:
                narrowed.append(iid)
        return narrowed[:M]

    def recommend(self, history, candidates, top_k=10):
        narrowed = self._narrow_candidates(history, candidates)
        if not narrowed:
            return []

        pos_hist = [iid for iid, r in history if r == 1]
        neg_hist = [iid for iid, r in history if r == 0]

        acs_scores = (self.acs.score_many(narrowed, pos_hist, neg_hist)
                      if self.use_acs else {})
        ease_scores = (self.ease_scorer.score_history(history, narrowed)
                       if self.use_ease and self.ease_scorer is not None
                       else {})

        ranked = []
        for cid in narrowed:
            q = 0.0
            if self.use_acs:
                q += acs_scores[cid]
            if self.use_snd:
                q += self.alpha * snd_score(cid, history, self.catalog,
                                            k=self.k, mode=self.snd_mode)
            if self.use_risk:
                if cid not in self.catalog.risk_scores:
                    raise KeyError(f"Risk score missing for item {cid}")
                q -= self.lam * self.catalog.risk_scores[cid]
            if self.use_ease:
                q += self.gamma * ease_scores.get(cid, 0.0)
            ranked.append((cid, q))

        ranked.sort(key=lambda x: -x[1])
        return [iid for iid, _ in ranked[:top_k]]


class LinUCBRecommender(BaseRecommender):
    name = "LinUCB"

    def __init__(self, catalog: CatalogStore, alpha: float = 1.0,
                 feat_dim: int = 64, seed: int = 42):
        self.catalog = catalog
        self.alpha = alpha
        self.feat_dim = feat_dim

        emb = catalog.item_index.embeddings.astype(np.float32)
        if emb.shape[1] > feat_dim:
            svd = TruncatedSVD(n_components=feat_dim, random_state=seed)
            reduced = svd.fit_transform(emb).astype(np.float32)
        else:
            reduced = emb
        self.d = reduced.shape[1]
        self._feats = {iid: reduced[i]
                       for i, iid in enumerate(catalog.item_index.item_ids)}

    def _x(self, iid: str) -> np.ndarray:
        f = self._feats.get(iid)
        if f is None:
            raise KeyError(f"No feature vector for item {iid}")
        return f.astype(np.float64)

    def _fit(self, history):
        A = np.eye(self.d, dtype=np.float64)
        b = np.zeros(self.d, dtype=np.float64)
        for iid, click in history:
            x = self._x(iid)
            A += np.outer(x, x)
            b += click * x
        Ai = np.linalg.inv(A)
        theta = Ai @ b
        return Ai, theta

    def recommend(self, history, candidates, top_k=10):
        seen = {iid for iid, _ in history}
        unseen = [iid for iid in candidates if iid not in seen]
        if not unseen:
            return []
        Ai, theta = self._fit(history)
        scored = []
        for iid in unseen:
            x = self._x(iid)
            ucb = float(theta @ x + self.alpha * np.sqrt(x @ Ai @ x))
            scored.append((iid, ucb))
        scored.sort(key=lambda p: -p[1])
        return [iid for iid, _ in scored[:top_k]]


class PopularityRecommender(BaseRecommender):
    name = "Popularity-Safe"

    def __init__(self, catalog: CatalogStore, train_interactions: list[tuple]):
        self.catalog = catalog
        self.clicks: dict[str, int] = {}
        for _, iid, reward, _ in train_interactions:
            if reward == 1:
                self.clicks[iid] = self.clicks.get(iid, 0) + 1

    def recommend(self, history, candidates, top_k=10):
        seen = {iid for iid, _ in history}
        unseen = [iid for iid in candidates if iid not in seen]
        unseen.sort(key=lambda iid: -self.clicks.get(iid, 0))
        return unseen[:top_k]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@dataclass
class RACSConfig:
    alpha: float = ALPHA_DEFAULT
    lam: float = LAMBDA_DEFAULT
    gamma: float = GAMMA_DEFAULT
    k: int = K_SND_DEFAULT
    candidate_size: int = CANDIDATE_SIZE_DEFAULT
    linucb_alpha: float = 1.0
    linucb_feat_dim: int = 64
    snd_mode: str = SND_MODE_DEFAULT


def build_all_recommenders(catalog: CatalogStore,
                           train_interactions: list[tuple],
                           acs_scorer: ACSScorer,
                           cfg: RACSConfig = RACSConfig(),
                           extras: Optional[list[BaseRecommender]] = None,
                           ease_scorer=None,
                           include_ease_blend: bool = False,
                           include_ease_narrowing: bool = False,
                           ) -> list[BaseRecommender]:
    """Full ablation set: every {ACS, SND, R(a)} subset, plus Popularity,
    LinUCB, and any externally-trained baselines passed via `extras`
    (e.g. SASRec built once outside this factory).

    When `ease_scorer` is provided AND a flag is set, additional RACS
    variants are appended:
      * include_ease_blend     → ACS+SND+R(a)+EASE             (centroid narrow + γ·EASE term)
      * include_ease_narrowing → ACS+SND+R(a) [EASE-nar]       (EASE-driven narrowing, no blend)
      * both                   → ACS+SND+R(a)+EASE [EASE-nar]  (full upgrade)
    """
    def make(use_acs, use_snd, use_risk,
              use_ease=False, narrowing="centroid"):
        return RACSRecommender(
            catalog,
            acs_scorer=acs_scorer if use_acs else None,
            alpha=cfg.alpha, lam=cfg.lam, gamma=cfg.gamma, k=cfg.k,
            candidate_size=cfg.candidate_size,
            use_acs=use_acs, use_snd=use_snd, use_risk=use_risk,
            snd_mode=cfg.snd_mode,
            ease_scorer=ease_scorer,
            use_ease=use_ease, narrowing_mode=narrowing,
        )

    recs: list[BaseRecommender] = [
        PopularityRecommender(catalog, train_interactions),
        LinUCBRecommender(catalog, alpha=cfg.linucb_alpha,
                          feat_dim=cfg.linucb_feat_dim),
    ]
    if extras:
        recs.extend(extras)
    recs.extend([
        make(True,  False, False),  # ACS
        make(False, True,  False),  # SND
        make(False, False, True),   # R(a)
        make(True,  True,  False),  # ACS+SND
        make(True,  False, True),   # ACS+R(a)
        make(False, True,  True),   # SND+R(a)
        make(True,  True,  True),   # ACS+SND+R(a)
    ])

    if ease_scorer is not None:
        if include_ease_blend:
            recs.append(make(True, True, True, use_ease=True))            # +EASE
        if include_ease_narrowing:
            recs.append(make(True, True, True, narrowing="ease"))         # [EASE-nar]
        if include_ease_blend and include_ease_narrowing:
            recs.append(make(True, True, True, use_ease=True,
                              narrowing="ease"))                            # +EASE [EASE-nar]
    return recs
