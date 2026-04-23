import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD

from .hf_model import HFLLM
from .offline import CatalogStore

logger = logging.getLogger(__name__)

ALPHA_DEFAULT = 0.5
LAMBDA_DEFAULT = 0.3
K_SND_DEFAULT = 5
CANDIDATE_SIZE_DEFAULT = 40
EPSILON_FLOOR = 0.05

ACS_WORD_POS = "ENGAGED"
ACS_WORD_NEG = "SKIPPED"

# Reward-weighted centroid
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

# SND score
def snd_score(cand_id: str,
              history: list[tuple[str, int]],
              catalog: CatalogStore,
              k: int = K_SND_DEFAULT,
              epsilon: float = EPSILON_FLOOR) -> float:
    cand_emb = catalog.item_index.get_embedding(cand_id)
    if cand_emb is None:
        raise KeyError(f"No embedding for candidate {cand_id}")
    if not history:
        return epsilon

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

    snd = min(n_pos, n_neg) / (max(n_pos, n_neg) + 1)
    if k_eff < 2:
        snd += epsilon
    return snd

# ACS scoring logic
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
    def __init__(self, catalog: CatalogStore, llm: HFLLM,
                 cache_path: Optional[Path] = None):
        self.catalog = catalog
        self.llm = llm
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, float] = {}
        self._dirty = False

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

    def _score_one(self, cand_id: str,
                   pos_history: list[str], neg_history: list[str]) -> float:
        cand_emb = self.catalog.item_index.get_embedding(cand_id)
        if cand_emb is None:
            raise KeyError(f"No embedding for candidate {cand_id}")

        if not pos_history or not neg_history:
            key = self._key(cand_id, None, None)
            if key in self.cache:
                return self.cache[key]
            cand_text = self.catalog.dataset.items[cand_id].text[:400]
            prompt = ACS_ABSOLUTE_PROMPT.format(candidate=cand_text)
            p = self.llm.contrast_logprob(prompt, ACS_WORD_POS, ACS_WORD_NEG)
            self.cache[key] = p
            self._dirty = True
            return p

        x_plus_id = self._closest_by_cosine(cand_emb, pos_history)
        x_minus_id = self._closest_by_cosine(cand_emb, neg_history)
        key = self._key(cand_id, x_plus_id, x_minus_id)
        if key in self.cache:
            return self.cache[key]

        cand_text = self.catalog.dataset.items[cand_id].text[:400]
        xp_text = self.catalog.dataset.items[x_plus_id].text[:300]
        xm_text = self.catalog.dataset.items[x_minus_id].text[:300]
        prompt = ACS_CONTRASTIVE_PROMPT.format(
            x_plus=xp_text, x_minus=xm_text, candidate=cand_text,
        )
        p = self.llm.contrast_logprob(prompt, ACS_WORD_POS, ACS_WORD_NEG)
        self.cache[key] = p
        self._dirty = True
        return p

    def score_many(self, cand_ids: list[str],
                   pos_history: list[str], neg_history: list[str]) -> dict[str, float]:
        return {c: self._score_one(c, pos_history, neg_history) for c in cand_ids}

# Recommenders setup
class BaseRecommender:
    name: str = "Base"

    def recommend(self, history: list[tuple[str, int]],
                  candidates: list[str], top_k: int = 10) -> list[str]:
        raise NotImplementedError

    def reset(self):
        pass

class RAGALRecommender(BaseRecommender):
    def __init__(self, catalog: CatalogStore, acs_scorer: Optional[ACSScorer],
                 alpha: float = ALPHA_DEFAULT, lam: float = LAMBDA_DEFAULT,
                 k: int = K_SND_DEFAULT,
                 candidate_size: int = CANDIDATE_SIZE_DEFAULT,
                 use_acs: bool = True, use_snd: bool = True, use_risk: bool = True):
        self.catalog = catalog
        self.acs = acs_scorer
        self.alpha = alpha
        self.lam = lam
        self.k = k
        self.candidate_size = candidate_size
        self.use_acs = use_acs
        self.use_snd = use_snd
        self.use_risk = use_risk

        if use_acs and acs_scorer is None:
            raise ValueError("use_acs=True requires an ACSScorer")

        parts = []
        if use_acs:  parts.append("ACS")
        if use_snd:  parts.append("SND")
        if use_risk: parts.append("R(a)")
        self.name = "+".join(parts) if parts else "Zero"

    def _narrow_candidates(self, history: list[tuple[str, int]],
                           full_candidates: list[str]) -> list[str]:
        M = self.candidate_size
        seen = {iid for iid, _ in history}
        unseen = [iid for iid in full_candidates if iid not in seen]

        if not history or len(unseen) <= M:
            return unseen[:M]

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

        ranked = []
        for cid in narrowed:
            q = 0.0
            if self.use_acs:
                q += acs_scores[cid]
            if self.use_snd:
                q += self.alpha * snd_score(cid, history, self.catalog, k=self.k)
            if self.use_risk:
                if cid not in self.catalog.risk_scores:
                    raise KeyError(f"Risk score missing for item {cid}")
                q -= self.lam * self.catalog.risk_scores[cid]
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

# Factory config
@dataclass
class RAGALConfig:
    alpha: float = ALPHA_DEFAULT
    lam: float = LAMBDA_DEFAULT
    k: int = K_SND_DEFAULT
    candidate_size: int = CANDIDATE_SIZE_DEFAULT
    linucb_alpha: float = 1.0
    linucb_feat_dim: int = 64

def build_all_recommenders(catalog: CatalogStore,
                           train_interactions: list[tuple],
                           acs_scorer: ACSScorer,
                           cfg: RAGALConfig = RAGALConfig()) -> list[BaseRecommender]:
    return [
        PopularityRecommender(catalog, train_interactions),
        LinUCBRecommender(catalog, alpha=cfg.linucb_alpha,
                           feat_dim=cfg.linucb_feat_dim),
        RAGALRecommender(catalog, acs_scorer=None,
                          alpha=cfg.alpha, lam=cfg.lam, k=cfg.k,
                          candidate_size=cfg.candidate_size,
                          use_acs=False, use_snd=True, use_risk=False),
        RAGALRecommender(catalog, acs_scorer=acs_scorer,
                          alpha=cfg.alpha, lam=cfg.lam, k=cfg.k,
                          candidate_size=cfg.candidate_size,
                          use_acs=True, use_snd=True, use_risk=False),
        RAGALRecommender(catalog, acs_scorer=acs_scorer,
                          alpha=cfg.alpha, lam=cfg.lam, k=cfg.k,
                          candidate_size=cfg.candidate_size,
                          use_acs=True, use_snd=True, use_risk=True),
    ]
