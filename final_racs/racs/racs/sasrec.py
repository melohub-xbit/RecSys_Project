"""
Minimal SASRec baseline (Kang & McAuley 2018).

Causal self-attention over a user's recent positive interactions; trained
with BCE on (positive next item, random negative). Used as a non-cold
sequential baseline for RACS.

Design notes
------------
* Positive interactions only feed the sequence. Zero-reward interactions
  are dropped (they would otherwise make the next-item supervision noisy).
* Item indices are 1..N; index 0 is the pad token.
* `recommend(history, candidates)` ignores history rewards: the encoder
  only consumes positives. A user with no positives in `history` falls
  back to global popularity (matches paper convention for the first step).
* Default hyper-parameters are deliberately small so the baseline fits
  within the run-time budget of the RACS evaluation. Tune via the
  `SASRecConfig` dataclass.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .offline_emb import CatalogStore
from .online import BaseRecommender

logger = logging.getLogger(__name__)


@dataclass
class SASRecConfig:
    hidden: int = 64
    max_len: int = 50
    n_blocks: int = 2
    n_heads: int = 2
    dropout: float = 0.2
    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 42


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _CausalBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads,
                                          dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask, key_padding_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class _SASRecNet(nn.Module):
    def __init__(self, n_items: int, cfg: SASRecConfig):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(n_items + 1, cfg.hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.hidden)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            _CausalBlock(cfg.hidden, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_blocks)
        ])
        self.ln_f = nn.LayerNorm(cfg.hidden)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, L) int — 0 is pad
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.dropout(x)
        key_pad = (seq == 0)
        causal = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            x = blk(x, attn_mask=causal, key_padding_mask=key_pad)
        return self.ln_f(x)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class SASRecRecommender(BaseRecommender):
    name = "SASRec"

    def __init__(self, catalog: CatalogStore, train_interactions: list[tuple],
                 cfg: SASRecConfig = SASRecConfig(),
                 device: str = "cuda"):
        self.catalog = catalog
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        item_ids = catalog.item_index.item_ids
        self._iid_to_idx = {iid: i + 1 for i, iid in enumerate(item_ids)}  # 1..N
        self._idx_to_iid = {i + 1: iid for i, iid in enumerate(item_ids)}
        self.n_items = len(item_ids)

        self._popularity = Counter()
        for _, iid, r, _ in train_interactions:
            if r == 1 and iid in self._iid_to_idx:
                self._popularity[iid] += 1
        self._pop_ranking = [iid for iid, _ in self._popularity.most_common()]

        self.net = _SASRecNet(self.n_items, cfg).to(self.device)
        self._train(train_interactions)
        self.net.eval()

    # ------------------------------------------------------------------ train
    def _build_user_seqs(self, train_interactions) -> list[list[int]]:
        by_user: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for uid, iid, r, ts in train_interactions:
            if r != 1 or iid not in self._iid_to_idx:
                continue
            by_user[uid].append((self._iid_to_idx[iid], ts))
        seqs: list[list[int]] = []
        for uid, seq in by_user.items():
            seq.sort(key=lambda x: x[1])
            seqs.append([s[0] for s in seq])
        return [s for s in seqs if len(s) >= 2]

    def _make_batch(self, seqs: list[list[int]],
                    rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L = self.cfg.max_len
        B = len(seqs)
        x = np.zeros((B, L), dtype=np.int64)
        pos = np.zeros((B, L), dtype=np.int64)
        neg = np.zeros((B, L), dtype=np.int64)
        for b, s in enumerate(seqs):
            tail = s[-(L + 1):]
            inp = tail[:-1]
            tgt = tail[1:]
            n = len(inp)
            x[b, -n:] = inp
            pos[b, -n:] = tgt
            seen = set(s)
            for j in range(n):
                while True:
                    cand = int(rng.integers(1, self.n_items + 1))
                    if cand not in seen:
                        neg[b, L - n + j] = cand
                        break
        return (torch.from_numpy(x), torch.from_numpy(pos), torch.from_numpy(neg))

    def _train(self, train_interactions):
        cfg = self.cfg
        seqs = self._build_user_seqs(train_interactions)
        if not seqs:
            logger.warning("SASRec: no positive sequences in train; "
                           "model is randomly initialised.")
            return
        logger.info("SASRec: training on %d user sequences (n_items=%d, "
                    "hidden=%d, max_len=%d, epochs=%d, batch=%d)",
                    len(seqs), self.n_items, cfg.hidden, cfg.max_len,
                    cfg.epochs, cfg.batch_size)

        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        rng = np.random.default_rng(cfg.seed)
        self.net.train()

        for ep in range(cfg.epochs):
            t0 = time.time()
            order = rng.permutation(len(seqs))
            losses = []
            for i in range(0, len(seqs), cfg.batch_size):
                idx = order[i:i + cfg.batch_size]
                batch = [seqs[j] for j in idx]
                x, pos, neg = self._make_batch(batch, rng)
                x = x.to(self.device)
                pos = pos.to(self.device)
                neg = neg.to(self.device)

                h = self.net.encode(x)            # (B, L, D)
                pos_e = self.net.item_emb(pos)    # (B, L, D)
                neg_e = self.net.item_emb(neg)    # (B, L, D)
                pos_logit = (h * pos_e).sum(-1)
                neg_logit = (h * neg_e).sum(-1)
                mask = (pos != 0).float()
                loss_pos = F.binary_cross_entropy_with_logits(
                    pos_logit, torch.ones_like(pos_logit), reduction="none")
                loss_neg = F.binary_cross_entropy_with_logits(
                    neg_logit, torch.zeros_like(neg_logit), reduction="none")
                loss = ((loss_pos + loss_neg) * mask).sum() / mask.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            logger.info("  SASRec epoch %d/%d  loss=%.4f  (%.1fs)",
                        ep + 1, cfg.epochs, float(np.mean(losses)),
                        time.time() - t0)

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def recommend(self, history: list[tuple[str, int]],
                  candidates: list[str], top_k: int = 10) -> list[str]:
        seen = {iid for iid, _ in history}
        unseen = [iid for iid in candidates if iid not in seen]
        if not unseen:
            return []
        pos_hist = [iid for iid, r in history if r == 1 and iid in self._iid_to_idx]
        if not pos_hist:
            ranked = [iid for iid in self._pop_ranking if iid in set(unseen)]
            for iid in unseen:
                if iid not in ranked:
                    ranked.append(iid)
            return ranked[:top_k]

        L = self.cfg.max_len
        x = np.zeros((1, L), dtype=np.int64)
        seq_idx = [self._iid_to_idx[iid] for iid in pos_hist[-L:]]
        x[0, -len(seq_idx):] = seq_idx
        x_t = torch.from_numpy(x).to(self.device)

        h = self.net.encode(x_t)[0, -1, :]              # (D,)
        cand_idx = [self._iid_to_idx.get(iid, 0) for iid in unseen]
        cand_t = torch.tensor(cand_idx, device=self.device, dtype=torch.long)
        cand_e = self.net.item_emb(cand_t)              # (M, D)
        scores = (cand_e @ h)
        order = torch.argsort(scores, descending=True).cpu().numpy()
        return [unseen[i] for i in order[:top_k]]
