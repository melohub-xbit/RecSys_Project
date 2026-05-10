"""
Standalone runner for the ItemKNN baseline.

This script mirrors the evaluation harness of `run_online.py` but trains and
evaluates only the ItemKNN recommender.  It uses the same:

  * catalog discovery and loading logic
  * user split strategy (random / temporal)
  * evaluation protocol (evaluate_static with neg_sample, eval_seeds, k_reveals)
  * metric computation (Recall, NDCG, MRR, HR, Precision, Diversity, AUC,
    Novelty, Coverage, Gini, Personalisation, avg_risk, head/tail recall)
  * statistical aggregation (mean ± std, bootstrap CI, Wilcoxon vs Popularity)

Output tables and figures are saved alongside (but do NOT overwrite) the
existing experiment artefacts:

    <results-dir>/tables/exp1_itemknn.csv
    <results-dir>/tables/exp1_itemknn_per_seed.csv
    <results-dir>/figures/exp2_itemknn_learning_curve.png

Typical invocation:

    python -m racs.run_itemknn \\
        --cache-dir <path/to/cache> \\
        --results-dir <path/to/results> \\
        --n-users 100 \\
        --eval-seeds 0
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

from .offline_emb import CatalogStore
from .replay import (
    DatasetStats, K_REVEAL_DEFAULTS, N_USERS_EVAL, NEG_SAMPLE,
    TOP_K_DEFAULT, evaluate_static, split_users,
    split_users_by_first_interaction,
)
from .stats import bootstrap_ci, mean_std, paired_wilcoxon
from .itemknn import ItemKNNRecommender

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR = ROOT / "results" / "figures"

BASELINE_NAME = "Popularity-Safe"


# ---------------------------------------------------------------------------
# Catalog discovery (same as run_online.py)
# ---------------------------------------------------------------------------

def _discover_catalogs(cache_dir: Path) -> list[str]:
    root = cache_dir / "catalogs"
    if not root.exists():
        raise FileNotFoundError(
            f"No catalogs directory at {root}. Run the offline phase first.")
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
# Experiment 1 — main cold-start table (multi-seed), ItemKNN only
# ---------------------------------------------------------------------------

def exp1_itemknn(catalogs, splits, ds_stats, recs_per_ds, args):
    """Mirrors run_online.exp1 but only evaluates ItemKNN + Popularity."""
    raw_rows = []
    sig_rows = []
    per_user = defaultdict(dict)

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

    # Significance test: ItemKNN vs Popularity
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
    raw_df.to_csv(TABLES_DIR / "exp1_itemknn_per_seed.csv", index=False)

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
    agg_df.to_csv(TABLES_DIR / "exp1_itemknn.csv", index=False)

    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(
            TABLES_DIR / "exp1_itemknn_significance.csv", index=False)

    print(f"\nSaved → {TABLES_DIR/'exp1_itemknn.csv'}")
    print(f"Saved → {TABLES_DIR/'exp1_itemknn_per_seed.csv'}")
    if sig_rows:
        print(f"Saved → {TABLES_DIR/'exp1_itemknn_significance.csv'}")


# ---------------------------------------------------------------------------
# Experiment 2 — learning curve, ItemKNN only
# ---------------------------------------------------------------------------

def exp2_itemknn(catalogs, splits, ds_stats, recs_per_ds, args):
    eseed = args.eval_seeds[0]
    fig, axes = plt.subplots(1, len(catalogs), figsize=(6 * len(catalogs), 5),
                             squeeze=False)
    rows = []

    MODEL_COLORS = {
        "Popularity-Safe": "#aaaaaa",
        "ItemKNN":         "#2980b9",
    }

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
        ax.set_title(_fmt_name(catalog), fontweight="bold")
        ax.set_xlabel("k_reveal")
        ax.set_ylabel(f"Recall@{args.top_k}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("ItemKNN — Learning curve", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "exp2_itemknn_learning_curve.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    pd.DataFrame(rows).to_csv(TABLES_DIR / "exp2_itemknn_learning_curve.csv",
                              index=False)
    print(f"Saved → {FIGS_DIR/'exp2_itemknn_learning_curve.png'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    1: ("Main cold-start table (ItemKNN)", exp1_itemknn),
    2: ("Learning curve (ItemKNN)",        exp2_itemknn),
}


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="ItemKNN baseline — standalone evaluation runner.")
    p.add_argument("--datasets", nargs="*",
                   help="Catalog directory names under <cache-dir>/catalogs. "
                        "If omitted, every catalog found there is run.")
    p.add_argument("--experiments", type=int, nargs="*",
                   default=list(EXPERIMENTS),
                   choices=list(EXPERIMENTS))
    p.add_argument("--cache-dir", default="outputs/cache")
    p.add_argument("--results-dir", default=None)

    p.add_argument("--n-users", type=int, default=N_USERS_EVAL)
    p.add_argument("--k-reveals", type=int, nargs="*",
                   default=list(K_REVEAL_DEFAULTS))
    p.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    p.add_argument("--neg-sample", type=int, default=NEG_SAMPLE)
    p.add_argument("--max-k-reveal", type=int, default=10)

    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--split-mode", choices=["random", "temporal"],
                   default="random")
    p.add_argument("--eval-seeds", type=int, nargs="*", default=[0])

    args = p.parse_args()

    global TABLES_DIR, FIGS_DIR
    if args.results_dir:
        results_root = Path(args.results_dir)
        TABLES_DIR = results_root / "tables"
        FIGS_DIR = results_root / "figures"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    if args.datasets:
        ds_names = args.datasets
    else:
        ds_names = _discover_catalogs(cache_dir)

    print("=" * 60)
    print("  ItemKNN baseline evaluation")
    print(f"  catalogs    : {ds_names}")
    print(f"  experiments : {args.experiments}")
    print(f"  split mode  : {args.split_mode} (split_seed={args.split_seed})")
    print(f"  eval seeds  : {args.eval_seeds}")
    print(f"  k_reveals   : {args.k_reveals}")
    print(f"  neg sample  : {args.neg_sample}")
    print(f"  n-users     : {args.n_users}")
    print("=" * 60)

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

    # Build recommender list: Popularity (for comparison) + ItemKNN
    from .online import PopularityRecommender

    recs_per_ds: dict[str, list] = {}
    for name in ds_names:
        cat = catalogs[name]
        split = splits[name]
        pop = PopularityRecommender(cat, split.train_interactions)
        iknn = ItemKNNRecommender(cat, split.train_interactions)
        recs_per_ds[name] = [pop, iknn]
        print(f"  Built recommenders for {name}: "
              f"{[r.name for r in recs_per_ds[name]]}")

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
