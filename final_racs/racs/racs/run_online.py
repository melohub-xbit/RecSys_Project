"""
RACS online phase — main experiment runner (HuggingFace backend).

Capabilities:

* Multi-seed evaluation: pass `--eval-seeds 0 1 2 3 4` to repeat every
  static evaluation under different negative-sampling seeds, then report
  mean +/- std + 95% bootstrap CI + paired Wilcoxon p-values.
* Full ablation set: every {ACS, SND, R(a)} subset is built and reported.
* Optional SASRec sequential baseline (`--include-sasrec`).
* Optional EASE baseline plus EASE-blend / EASE-narrowing variants.
* Random or temporal cold-start split (`--split-mode random|temporal`).
* Experiment 6: risk-recall trade-off (sweep lambda).

Typical invocation (offline phase + risk calibration already done):

    python -m racs.calibrate_risk --cache-dir <path/to/cache>
    python -m racs.run_online \\
        --cache-dir <path/to/cache> \\
        --results-dir <path/to/results> \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --device cuda --dtype float16 \\
        --eval-seeds 0 1 2 3 4 \\
        --include-sasrec \\
        --n-users 1000000

Pin to a specific GPU by setting CUDA_VISIBLE_DEVICES in the shell before
launching, e.g. on Windows PowerShell:

    $env:CUDA_VISIBLE_DEVICES = "0"
"""

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .hf_model import HF_MODEL_DEFAULT, HFLLM
from .offline_emb import CatalogStore
from .online import (
    ACSScorer, LinUCBRecommender, RACSConfig, RACSRecommender,
    SND_MODE_DEFAULT, LAMBDA_PAPERS_DEFAULT, ALPHA_DEFAULT, LAMBDA_DEFAULT,
    CANDIDATE_SIZE_DEFAULT, GAMMA_DEFAULT,
    build_all_recommenders,
)
from .replay import (
    DatasetStats, K_REVEAL_DEFAULTS, MAX_STEPS, NEG_SAMPLE, N_USERS_EVAL,
    TOP_K_DEFAULT, evaluate_sequential, evaluate_static, split_users,
    split_users_by_first_interaction,
)
from .stats import bootstrap_ci, mean_std, paired_wilcoxon

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR = ROOT / "results" / "figures"

MODEL_COLORS = {
    "Popularity-Safe":            "#aaaaaa",
    "LinUCB":                     "#e07b39",
    "SASRec":                     "#9b59b6",
    "EASE":                       "#2ecc71",
    "ACS":                        "#3498db",
    "SND":                        "#8fbf60",
    "R(a)":                       "#f1c40f",
    "ACS+SND":                    "#5b8db8",
    "ACS+R(a)":                   "#16a085",
    "SND+R(a)":                   "#d35400",
    "ACS+SND+R(a)":               "#c0392b",
    "ACS+SND+R(a)+EASE":          "#7d3c98",
    "ACS+SND+R(a) [EASE-nar]":    "#1abc9c",
    "ACS+SND+R(a)+EASE [EASE-nar]": "#34495e",
}

BASELINE_NAME = "Popularity-Safe"


# ---------------------------------------------------------------------------
# Catalog discovery
# ---------------------------------------------------------------------------

def _discover_catalogs(cache_dir: Path) -> list[str]:
    root = cache_dir / "catalogs"
    if not root.exists():
        raise FileNotFoundError(
            f"No catalogs directory at {root}. Run `python -m racs.offline_emb` "
            f"to populate it first.")
    found = [c.name for c in sorted(root.iterdir())
             if (c / "manifest.json").exists()]
    if not found:
        raise FileNotFoundError(f"No catalogs found under {root}")
    return found


def _load(ds_name: str, cache_dir: Path) -> CatalogStore:
    return CatalogStore.load(str(cache_dir / "catalogs" / ds_name))


def _fmt_name(catalog: CatalogStore) -> str:
    return catalog.dataset.name


# ---------------------------------------------------------------------------
# Experiment 1 — main cold-start table (multi-seed)
# ---------------------------------------------------------------------------

def exp1(catalogs, splits, ds_stats, recs_per_ds, args):
    raw_rows = []        # per-seed rows (all metrics)
    sig_rows = []        # paired Wilcoxon vs baseline (one row per dataset/model/k_reveal)
    per_user = defaultdict(dict)   # (dataset, k_reveal, seed) → {model: per-user recall}

    for ds_name, catalog in catalogs.items():
        split = splits[ds_name]
        stats = ds_stats[ds_name]
        recs = recs_per_ds[ds_name]
        for rec in recs:
            for k_rev in args.k_reveals:
                for eseed in args.eval_seeds:
                    t0 = time.time()
                    res = evaluate_static(
                        rec, catalog.dataset, catalog, split.test_uids,
                        k_reveal=k_rev, top_k=args.top_k,
                        neg_sample=args.neg_sample, seed=eseed, stats=stats,
                    )
                    dt = time.time() - t0
                    summ = res.summary(n_total_items=stats.n_total_items)
                    print(f"  {_fmt_name(catalog):22s} | {rec.name:18s} | "
                          f"k={k_rev:2d} s={eseed} "
                          f"R@{args.top_k}={summ['recall']:.4f} "
                          f"HR={summ['hr']:.4f} "
                          f"NDCG={summ['ndcg']:.4f} MRR={summ['mrr']:.4f} "
                          f"AUC={summ['auc']:.4f} "
                          f"CovR={summ['coverage_relative']:.4f} ({dt:.1f}s)")
                    raw_rows.append({
                        "dataset": _fmt_name(catalog),
                        "model": rec.name,
                        "k_reveal": k_rev,
                        "eval_seed": eseed,
                        **{k: v for k, v in summ.items() if k != "n_users"},
                        "n_users": summ["n_users"],
                    })
                    per_user[(ds_name, k_rev, eseed)][rec.name] = (
                        list(res.recalls), list(res.user_ids))

        # ACS cache flush per dataset (any RACS recommender shares it)
        for r in recs:
            if isinstance(r, RACSRecommender) and r.acs is not None:
                r.acs.flush()
                break

    # Significance test: per (dataset, k_reveal), compare each non-baseline
    # model to BASELINE_NAME using eval_seeds[0] per-user vectors.
    s0 = args.eval_seeds[0]
    for ds_name in catalogs:
        for k_rev in args.k_reveals:
            slot = per_user.get((ds_name, k_rev, s0), {})
            base = slot.get(BASELINE_NAME)
            if base is None:
                continue
            base_vec, base_uids = base
            for mname, (vec, uids) in slot.items():
                if mname == BASELINE_NAME:
                    continue
                if uids != base_uids:
                    continue
                p = paired_wilcoxon(vec, base_vec)
                sig_rows.append({
                    "dataset": ds_name, "model": mname, "k_reveal": k_rev,
                    "baseline": BASELINE_NAME,
                    "wilcoxon_p_recall": round(p, 6),
                })

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(TABLES_DIR / "exp1_main_table_per_seed.csv", index=False)

    # Aggregate across eval_seeds
    metric_cols = ["recall", "ndcg", "mrr", "hr", "precision",
                   "intra_diversity", "auc",
                   "novelty", "avg_risk", "head_recall", "tail_recall",
                   "coverage", "coverage_relative", "gini", "personalisation"]
    agg_rows = []
    for (ds, mdl, kr), grp in raw_df.groupby(["dataset", "model", "k_reveal"]):
        row = {"dataset": ds, "model": mdl, "k_reveal": kr,
               "n_users": int(grp["n_users"].iloc[0])}
        for col in metric_cols:
            vals = grp[col].tolist()
            mu, sd = mean_std(vals)
            lo, hi = bootstrap_ci(vals)
            row[col] = round(mu, 4)
            row[f"{col}_std"] = round(sd, 4)
            row[f"{col}_ci_lo"] = round(lo, 4)
            row[f"{col}_ci_hi"] = round(hi, 4)
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(TABLES_DIR / "exp1_main_table.csv", index=False)

    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(
            TABLES_DIR / "exp1_significance.csv", index=False)

    print(f"\nSaved → {TABLES_DIR/'exp1_main_table.csv'}")
    print(f"Saved → {TABLES_DIR/'exp1_main_table_per_seed.csv'}")
    if sig_rows:
        print(f"Saved → {TABLES_DIR/'exp1_significance.csv'}")


# ---------------------------------------------------------------------------
# Experiment 2 — learning curve (single seed)
# ---------------------------------------------------------------------------

def exp2(catalogs, splits, ds_stats, recs_per_ds, args):
    eseed = args.eval_seeds[0]
    fig, axes = plt.subplots(1, len(catalogs), figsize=(6 * len(catalogs), 5),
                             squeeze=False)
    rows = []

    for ax, (ds_name, catalog) in zip(axes[0], catalogs.items()):
        split = splits[ds_name]
        stats = ds_stats[ds_name]
        recs = recs_per_ds[ds_name]
        for rec in recs:
            curve_k = list(range(1, args.max_k_reveal + 1))
            vals = []
            for k_rev in curve_k:
                res = evaluate_static(
                    rec, catalog.dataset, catalog, split.test_uids,
                    k_reveal=k_rev, top_k=args.top_k,
                    neg_sample=args.neg_sample, seed=eseed, stats=stats)
                vals.append(res.recall)
                rows.append({"dataset": _fmt_name(catalog), "model": rec.name,
                             "k_reveal": k_rev,
                             f"Recall@{args.top_k}": round(res.recall, 4)})
                print(f"  {_fmt_name(catalog):22s} | {rec.name:18s} | "
                      f"k_rev={k_rev:2d}  R@{args.top_k}={res.recall:.4f}")
            ax.plot(curve_k, vals, marker="o", ms=3, lw=1.8,
                    color=MODEL_COLORS.get(rec.name, "#333"), label=rec.name)
        for r in recs:
            if isinstance(r, RACSRecommender) and r.acs is not None:
                r.acs.flush()
                break
        ax.set_title(_fmt_name(catalog), fontweight="bold")
        ax.set_xlabel("k_reveal")
        ax.set_ylabel(f"Recall@{args.top_k}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Exp 2 — Learning curve", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "exp2_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    pd.DataFrame(rows).to_csv(TABLES_DIR / "exp2_learning_curve.csv", index=False)
    print(f"Saved → {FIGS_DIR/'exp2_learning_curve.png'}")


# ---------------------------------------------------------------------------
# Experiment 3 — full component isolation (single seed)
# ---------------------------------------------------------------------------

def exp3(catalogs, splits, ds_stats, recs_per_ds, args):
    eseed = args.eval_seeds[0]
    rows = []
    iso_names = {"ACS", "SND", "R(a)", "ACS+SND", "ACS+R(a)", "SND+R(a)",
                 "ACS+SND+R(a)",
                 "ACS+SND+R(a)+EASE",
                 "ACS+SND+R(a) [EASE-nar]",
                 "ACS+SND+R(a)+EASE [EASE-nar]"}

    for ds_name, catalog in catalogs.items():
        split = splits[ds_name]
        stats = ds_stats[ds_name]
        recs = [r for r in recs_per_ds[ds_name] if r.name in iso_names]
        for rec in recs:
            res = evaluate_static(
                rec, catalog.dataset, catalog, split.test_uids,
                k_reveal=5, top_k=args.top_k,
                neg_sample=args.neg_sample, seed=eseed, stats=stats)
            print(f"  {_fmt_name(catalog):22s} | {rec.name:18s} | "
                  f"R@{args.top_k}={res.recall:.4f}")
            rows.append({"dataset": _fmt_name(catalog), "model": rec.name,
                         f"Recall@{args.top_k}": round(res.recall, 4)})
        for r in recs_per_ds[ds_name]:
            if isinstance(r, RACSRecommender) and r.acs is not None:
                r.acs.flush()
                break

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "exp3_component_isolation.csv", index=False)

    fig, axes = plt.subplots(1, len(catalogs), figsize=(5 * len(catalogs), 5),
                             squeeze=False)
    for ax, (ds_name, catalog) in zip(axes[0], catalogs.items()):
        sub = df[df["dataset"] == _fmt_name(catalog)]
        vals = sub[f"Recall@{args.top_k}"].tolist()
        names = sub["model"].tolist()
        bars = ax.bar(range(len(names)), vals,
                      color=[MODEL_COLORS.get(n, "#999") for n in names],
                      width=0.5, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                    f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"{_fmt_name(catalog)} (k_reveal=5)", fontweight="bold")
        ax.set_ylabel(f"Recall@{args.top_k}")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals + [0.01]) * 1.25)

    fig.suptitle("Exp 3 — Component isolation (full ablation)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "exp3_component_isolation.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"Saved → {FIGS_DIR/'exp3_component_isolation.png'}")


# ---------------------------------------------------------------------------
# Experiment 4 — risk penalty validation (sequential, single seed)
# ---------------------------------------------------------------------------

def exp4(catalogs, splits, ds_stats, recs_per_ds, args):
    eseed = args.eval_seeds[0]
    rows = []
    for ds_name, catalog in catalogs.items():
        split = splits[ds_name]
        full = next(r for r in recs_per_ds[ds_name] if r.name == "ACS+SND+R(a)")
        no_risk = next(r for r in recs_per_ds[ds_name] if r.name == "ACS+SND")

        res_with = evaluate_sequential(full, catalog.dataset, catalog,
                                       split.test_uids, max_steps=args.max_steps,
                                       neg_sample=args.neg_sample, seed=eseed)
        res_no = evaluate_sequential(no_risk, catalog.dataset, catalog,
                                     split.test_uids, max_steps=args.max_steps,
                                     neg_sample=args.neg_sample, seed=eseed)
        for r in recs_per_ds[ds_name]:
            if isinstance(r, RACSRecommender) and r.acs is not None:
                r.acs.flush()
                break

        sr_with = res_with.first_step_skip_rate
        sr_no = res_no.first_step_skip_rate
        delta = sr_no - sr_with
        print(f"  {_fmt_name(catalog):22s} | skip_with={sr_with:.4f}  "
              f"skip_no={sr_no:.4f}  Δ={delta:+.4f}")

        actual_lam = full.lam  # per-dataset λ (may differ from args.lam)
        rows.extend([
            {"dataset": _fmt_name(catalog),
             "model": f"With R(a) λ={actual_lam}",
             "first_step_skip_rate": round(sr_with, 4),
             "cum_reward": round(res_with.cumulative_reward, 4)},
            {"dataset": _fmt_name(catalog),
             "model": "Without R(a) λ=0",
             "first_step_skip_rate": round(sr_no, 4),
             "cum_reward": round(res_no.cumulative_reward, 4)},
        ])

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "exp4_risk_penalty.csv", index=False)

    fig, axes = plt.subplots(1, len(catalogs), figsize=(5 * len(catalogs), 5),
                             squeeze=False)
    for ax, (ds_name, catalog) in zip(axes[0], catalogs.items()):
        sub = df[df["dataset"] == _fmt_name(catalog)].reset_index(drop=True)
        full_rec = next(r for r in recs_per_ds[ds_name] if r.name == "ACS+SND+R(a)")
        actual_lam = full_rec.lam
        labels = [f"With R(a)\nλ={actual_lam}", "Without R(a)\nλ=0"]
        vals = sub["first_step_skip_rate"].tolist()
        bars = ax.bar(labels, vals, color=["#27ae60", "#e74c3c"], width=0.38,
                      edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004,
                    f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
        ax.set_title(f"{_fmt_name(catalog)} (T={args.max_steps})", fontweight="bold")
        ax.set_ylabel("First-step skip rate ↓")
        ax.set_ylim(0, max(vals + [0.01]) * 1.35)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Exp 4 — Risk penalty validation", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "exp4_risk_penalty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {FIGS_DIR/'exp4_risk_penalty.png'}")


# ---------------------------------------------------------------------------
# Experiment 5 — cross-domain consistency (single seed)
# ---------------------------------------------------------------------------

def exp5(catalogs, splits, ds_stats, recs_per_ds, args):
    eseed = args.eval_seeds[0]
    rows = []
    for ds_name, catalog in catalogs.items():
        split = splits[ds_name]
        stats = ds_stats[ds_name]
        lin = next(r for r in recs_per_ds[ds_name] if r.name == "LinUCB")
        full = next(r for r in recs_per_ds[ds_name] if r.name == "ACS+SND+R(a)")
        for rec in [lin, full]:
            res = evaluate_static(
                rec, catalog.dataset, catalog, split.test_uids,
                k_reveal=5, top_k=args.top_k,
                neg_sample=args.neg_sample, seed=eseed, stats=stats)
            print(f"  {_fmt_name(catalog):22s} | {rec.name:18s} | "
                  f"R@{args.top_k}={res.recall:.4f}  NDCG@{args.top_k}={res.ndcg:.4f}")
            rows.append({"dataset": _fmt_name(catalog), "model": rec.name,
                         f"Recall@{args.top_k}": round(res.recall, 4),
                         f"NDCG@{args.top_k}":   round(res.ndcg, 4)})
        for r in recs_per_ds[ds_name]:
            if isinstance(r, RACSRecommender) and r.acs is not None:
                r.acs.flush()
                break

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "exp5_cross_domain.csv", index=False)

    print("\n-- Relative margin (RACS vs LinUCB) --")
    for ds_name in df["dataset"].unique():
        sub = df[df["dataset"] == ds_name].set_index("model")
        lin_v = sub.loc["LinUCB", f"Recall@{args.top_k}"]
        our_v = sub.loc["ACS+SND+R(a)", f"Recall@{args.top_k}"]
        mg = 100 * (our_v - lin_v) / (lin_v + 1e-9)
        print(f"  {ds_name:22s}  LinUCB={lin_v:.4f}  Ours={our_v:.4f}  margin={mg:+.1f}%")

    dsets = df["dataset"].unique()
    x = np.arange(len(dsets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(dsets)), 5))
    for i, (mname, col) in enumerate([("LinUCB", "#e07b39"),
                                       ("ACS+SND+R(a)", "#c0392b")]):
        vals = [df[(df["dataset"] == d) & (df["model"] == mname)]
                [f"Recall@{args.top_k}"].values[0] for d in dsets]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=mname, color=col,
                      edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                    f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(dsets, rotation=15, ha="right")
    ax.set_ylabel(f"Recall@{args.top_k}")
    ax.set_title("Exp 5 — Cross-domain consistency (k_reveal=5)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "exp5_cross_domain.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {FIGS_DIR/'exp5_cross_domain.png'}")


# ---------------------------------------------------------------------------
# Experiment 6 — risk-recall trade-off curve (NEW)
# ---------------------------------------------------------------------------

def exp6(catalogs, splits, ds_stats, recs_per_ds, args):
    eseed = args.eval_seeds[0]
    rows = []

    fig, axes = plt.subplots(1, len(catalogs), figsize=(5 * len(catalogs), 5),
                             squeeze=False)
    for ax, (ds_name, catalog) in zip(axes[0], catalogs.items()):
        split = splits[ds_name]
        stats = ds_stats[ds_name]
        # Re-use the ACS scorer object embedded in the existing recommenders.
        ref = next(r for r in recs_per_ds[ds_name] if r.name == "ACS+SND+R(a)")
        acs = ref.acs

        recalls, risks, lams = [], [], []
        for lam in args.lam_sweep:
            rec = RACSRecommender(
                catalog, acs_scorer=acs,
                alpha=args.alpha, lam=lam, k=ref.k,
                candidate_size=ref.candidate_size,
                use_acs=True, use_snd=True, use_risk=True,
            )
            res = evaluate_static(
                rec, catalog.dataset, catalog, split.test_uids,
                k_reveal=5, top_k=args.top_k,
                neg_sample=args.neg_sample, seed=eseed, stats=stats)
            recalls.append(res.recall)
            risks.append(res.avg_risk)
            lams.append(lam)
            rows.append({"dataset": _fmt_name(catalog), "lambda": lam,
                         f"Recall@{args.top_k}": round(res.recall, 4),
                         "avg_risk": round(res.avg_risk, 4)})
            print(f"  {_fmt_name(catalog):22s} | λ={lam:.2f} | "
                  f"R@{args.top_k}={res.recall:.4f}  avg_risk={res.avg_risk:.4f}")
        if acs is not None:
            acs.flush()

        ax.plot(risks, recalls, "o-", color="#c0392b", lw=2)
        for r, rec_v, lam in zip(risks, recalls, lams):
            ax.annotate(f"λ={lam}", (r, rec_v), fontsize=7,
                        textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel("Avg risk of top-K recs")
        ax.set_ylabel(f"Recall@{args.top_k}")
        ax.set_title(_fmt_name(catalog), fontweight="bold")
        ax.grid(alpha=0.3)

    fig.suptitle("Exp 6 — Risk-Recall trade-off (k_reveal=5)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "exp6_risk_recall_curve.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    pd.DataFrame(rows).to_csv(TABLES_DIR / "exp6_risk_recall_curve.csv", index=False)
    print(f"Saved → {FIGS_DIR/'exp6_risk_recall_curve.png'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    1: ("Main cold-start table",        exp1),
    2: ("Learning curve",               exp2),
    3: ("Component isolation",          exp3),
    4: ("Risk penalty validation",      exp4),
    5: ("Cross-domain consistency",     exp5),
    6: ("Risk-recall trade-off",        exp6),
}


def main():
    # Force line-buffered stdout so `print()` flushes per line when stdout
    # is a file (otherwise progress lines sit in a 4-8 KB buffer and don't
    # appear in the log until much later).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="RACS online-phase experiments (HuggingFace backend).")
    p.add_argument("--datasets", nargs="*",
                   help="Catalog directory names under <cache-dir>/catalogs. "
                        "If omitted, every catalog under that directory is run.")
    p.add_argument("--experiments", type=int, nargs="*", default=list(EXPERIMENTS),
                   choices=list(EXPERIMENTS))
    p.add_argument("--cache-dir", default="outputs/cache")
    p.add_argument("--acs-cache-dir", default=None)
    p.add_argument("--results-dir", default=None)

    p.add_argument("--model", default=HF_MODEL_DEFAULT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--attn", default="sdpa",
                   choices=["eager", "sdpa", "flash_attention_2"],
                   help="HF attention kernel. sdpa is built-in (PyTorch 2+) "
                        "and ~1.5–2× faster than eager. flash_attention_2 is "
                        "fastest but needs `pip install flash-attn` and an "
                        "Ampere+ GPU.")
    p.add_argument("--acs-batch-size", type=int, default=16,
                   help="LLM batch size for ACS scoring. Larger = faster but "
                        "more GPU memory. Try 32 if you have headroom.")

    p.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                   help="SND weight. Default 2.0 — chosen so α·SND ∈ [0, 0.58] "
                        "is comparable to ACS ∈ [0, 1].")
    p.add_argument("--lam", type=float, default=LAMBDA_DEFAULT,
                   help="Risk penalty for movie-style domains. Default 0.3.")
    p.add_argument("--lam-papers", type=float, default=LAMBDA_PAPERS_DEFAULT,
                   help="Risk penalty for sparse paper domains "
                        "(SemanticScholar*). Default 0.05 — niche items "
                        "in academic data are typically the relevant ones, "
                        "so a strong R(a) penalty hurts. Pass --lam-papers "
                        "= --lam to disable per-domain handling.")
    p.add_argument("--snd-mode", choices=["bayesian", "heuristic"],
                   default=SND_MODE_DEFAULT,
                   help="Bayesian: posterior std of Beta(1+n_pos, 1+n_neg). "
                        "Heuristic: paper's original min/(max+1) ratio.")
    p.add_argument("--candidate-size", type=int, default=CANDIDATE_SIZE_DEFAULT,
                   help="Candidate narrowing pool size (FAISS retrieval) before "
                        "ACS/SND/R(a) scoring. Default 200 — bigger pool "
                        "gives the reranker more headroom and a meaningful AUC.")
    p.add_argument("--include-ease", action="store_true",
                   help="Train and evaluate an EASE item-item baseline. Skipped "
                        "automatically on catalogs > 50k items (RAM blow-up).")
    p.add_argument("--include-ease-blend", action="store_true",
                   help="Add an `ACS+SND+R(a)+EASE` ablation that adds a "
                        "gamma * EASE term to the score. Requires that EASE was "
                        "trainable on the catalog (auto-skipped otherwise).")
    p.add_argument("--include-ease-narrowing", action="store_true",
                   help="Add an `ACS+SND+R(a) [EASE-nar]` ablation where the "
                        "candidate-narrowing step uses EASE scores instead "
                        "of FAISS over the centroid.")
    p.add_argument("--gamma", type=float, default=GAMMA_DEFAULT,
                   help="Weight on the EASE blend term. Default 0.3.")
    p.add_argument("--lam-sweep", type=float, nargs="*",
                   default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
                   help="λ values for the risk-recall sweep (exp6).")
    p.add_argument("--n-users", type=int, default=N_USERS_EVAL)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--max-k-reveal", type=int, default=10)
    p.add_argument("--k-reveals", type=int, nargs="*", default=list(K_REVEAL_DEFAULTS))
    p.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    p.add_argument("--neg-sample", type=int, default=NEG_SAMPLE)
    p.add_argument("--linucb-feat-dim", type=int, default=64)

    p.add_argument("--split-seed", type=int, default=42,
                   help="Seed for the user-level train/test split (fixed across "
                        "eval seeds).")
    p.add_argument("--split-mode", choices=["random", "temporal"], default="random",
                   help="random: shuffle users 80/20.  temporal: order users by "
                        "first-interaction time, last 20%% are the test set.")
    p.add_argument("--eval-seeds", type=int, nargs="*", default=[0],
                   help="One or more seeds for negative sampling per evaluation. "
                        "Multi-seed enables mean ± std + bootstrap CI in exp1.")

    p.add_argument("--include-sasrec", action="store_true",
                   help="Train and evaluate a SASRec sequential baseline.")
    p.add_argument("--sasrec-epochs", type=int, default=5)
    p.add_argument("--sasrec-hidden", type=int, default=64)

    args = p.parse_args()

    global TABLES_DIR, FIGS_DIR
    if args.results_dir:
        results_root = Path(args.results_dir)
        TABLES_DIR = results_root / "tables"
        FIGS_DIR = results_root / "figures"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    acs_cache_dir = Path(args.acs_cache_dir) if args.acs_cache_dir \
        else cache_dir / "acs"
    acs_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        ds_names = args.datasets
    else:
        ds_names = _discover_catalogs(cache_dir)

    print("=" * 60)
    print("  RACS online-phase experiments (HF, paper-grade)")
    print(f"  catalogs    : {ds_names}")
    print(f"  experiments : {args.experiments}")
    print(f"  HF model    : {args.model}")
    print(f"  device/dtype: {args.device} / {args.dtype}"
          f"{' (4bit)' if args.load_in_4bit else ''}")
    print(f"  attn / batch: {args.attn} / acs_bs={args.acs_batch_size}")
    print(f"  alpha/lambda: {args.alpha} / {args.lam} (papers: λ={args.lam_papers})")
    print(f"  snd_mode    : {args.snd_mode}   cand_size: {args.candidate_size}")
    ease_modes = []
    if args.include_ease:           ease_modes.append("baseline")
    if args.include_ease_blend:     ease_modes.append(f"blend(γ={args.gamma})")
    if args.include_ease_narrowing: ease_modes.append("narrowing")
    print(f"  EASE        : {', '.join(ease_modes) if ease_modes else 'off'}")
    print(f"  split mode  : {args.split_mode} (split_seed={args.split_seed})")
    print(f"  eval seeds  : {args.eval_seeds}")
    print(f"  neg sample  : {args.neg_sample}")
    print(f"  SASRec      : {'on' if args.include_sasrec else 'off'}")
    print(f"  n-users     : {args.n_users}")
    print("=" * 60)

    llm = HFLLM(model_name=args.model, device=args.device,
                dtype=args.dtype, load_in_4bit=args.load_in_4bit,
                attn_impl=args.attn)

    catalogs: dict[str, CatalogStore] = {}
    splits: dict = {}
    ds_stats: dict[str, DatasetStats] = {}
    for name in ds_names:
        print(f"\nLoading catalog {name}...")
        cat = _load(name, cache_dir)
        catalogs[name] = cat
        if args.split_mode == "temporal":
            splits[name] = split_users_by_first_interaction(
                cat.dataset, n_test_users=args.n_users, seed=args.split_seed)
        else:
            splits[name] = split_users(
                cat.dataset, n_test_users=args.n_users, seed=args.split_seed)
        ds_stats[name] = DatasetStats.from_train(
            splits[name].train_interactions, cat.item_index.item_ids)

    def _pick_lam(ds_name: str) -> float:
        """Per-domain λ: paper-style sparse domains use a smaller penalty.

        Heuristic on dataset name — robust enough since the only paper
        catalogs follow the SemanticScholar-* convention.
        """
        n = ds_name.lower()
        if any(k in n for k in ("scholar", "s2", "paper", "academic")):
            return args.lam_papers
        return args.lam

    # Build per-dataset recommender lists once. ACS scorer + SASRec + EASE
    # are trained / loaded here, not inside experiment functions.
    recs_per_ds: dict[str, list] = {}
    for name in ds_names:
        cat = catalogs[name]
        split = splits[name]
        ds_lam = _pick_lam(name)
        if ds_lam != args.lam:
            print(f"  [{name}] using paper-domain λ={ds_lam}")

        acs = ACSScorer(cat, llm=llm,
                        cache_path=acs_cache_dir / f"acs_{_fmt_name(cat)}.json",
                        batch_size=args.acs_batch_size)
        extras = []
        if args.include_sasrec:
            from .sasrec import SASRecConfig, SASRecRecommender
            sas_cfg = SASRecConfig(hidden=args.sasrec_hidden,
                                   epochs=args.sasrec_epochs,
                                   seed=args.split_seed)
            print(f"\n  Training SASRec on {name}...")
            t0 = time.time()
            sas = SASRecRecommender(cat, split.train_interactions,
                                    cfg=sas_cfg, device=args.device)
            print(f"  SASRec trained in {time.time()-t0:.1f}s")
            extras.append(sas)
        ease_obj = None
        ease_wanted = (args.include_ease or args.include_ease_blend
                       or args.include_ease_narrowing)
        if ease_wanted:
            from .ease import EASERecommender
            try:
                print(f"\n  Training EASE on {name}...")
                t0 = time.time()
                ease_obj = EASERecommender(cat, split.train_interactions)
                print(f"  EASE trained in {time.time()-t0:.1f}s")
            except ValueError as e:
                logger.warning("EASE skipped for %s: %s", name, e)
        if ease_obj is not None and args.include_ease:
            extras.append(ease_obj)

        recs_per_ds[name] = build_all_recommenders(
            cat, split.train_interactions, acs,
            cfg=RACSConfig(alpha=args.alpha, lam=ds_lam, gamma=args.gamma,
                            candidate_size=args.candidate_size,
                            linucb_feat_dim=args.linucb_feat_dim,
                            snd_mode=args.snd_mode),
            extras=extras,
            ease_scorer=ease_obj,
            include_ease_blend=args.include_ease_blend and ease_obj is not None,
            include_ease_narrowing=args.include_ease_narrowing and ease_obj is not None,
        )

    t_total = time.time()
    for exp_id in args.experiments:
        label, fn = EXPERIMENTS[exp_id]
        print(f"\n{'─'*60}\n  Experiment {exp_id}: {label}\n{'─'*60}")
        t0 = time.time()
        fn(catalogs, splits, ds_stats, recs_per_ds, args)
        print(f"  ✓ exp{exp_id} done in {time.time()-t0:.1f}s")

    print(f"\n{'='*60}\n  All done in {(time.time()-t_total)/60:.1f} min")
    print(f"  Tables  → {TABLES_DIR}")
    print(f"  Figures → {FIGS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
