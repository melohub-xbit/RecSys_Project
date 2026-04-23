import argparse
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .hf_model import HFLLM, HF_MODEL_DEFAULT
from .offline import CatalogStore
from .online import (
    ACSScorer, LinUCBRecommender, PopularityRecommender,
    RAGALRecommender, RAGALConfig, build_all_recommenders,
)
from .replay import (
    K_REVEAL_DEFAULTS, MAX_STEPS, NEG_SAMPLE, N_USERS_EVAL, TOP_K_DEFAULT,
    evaluate_sequential, evaluate_static, split_users,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR = ROOT / "results" / "figures"

# Catalog loading
CATALOG_NAMES = {
    "movielens": "MovieLens-1M",
    "s2":        "SemanticScholar",
}

def _load_catalog(dataset_key: str, cache_dir: Path) -> CatalogStore:
    name = CATALOG_NAMES[dataset_key]
    path = cache_dir / "catalogs" / name
    if not path.exists():
        raise FileNotFoundError(
            f"No CatalogStore at {path}. Run `python -m ragal.offline "
            f"--dataset {dataset_key}` first."
        )
    return CatalogStore.load(str(path))

# Plot styles
MODEL_COLORS = {
    "Popularity-Safe": "#aaaaaa",
    "LinUCB":          "#e07b39",
    "SND":             "#8fbf60",
    "ACS+SND":         "#5b8db8",
    "ACS+SND+R(a)":    "#c0392b",
}

# Experiment 1
def exp1(catalogs, splits, llm, acs_cache_dir, args):
    rows = []
    for ds_key, catalog in catalogs.items():
        split = splits[ds_key]
        acs = ACSScorer(catalog, llm=llm,
                        cache_path=acs_cache_dir / f"acs_{catalog.dataset.name}.json")
        recs = build_all_recommenders(catalog, split.train_interactions, acs,
                                       cfg=RAGALConfig(alpha=args.alpha, lam=args.lam))
        for rec in recs:
            for k_rev in args.k_reveals:
                t0 = time.time()
                res = evaluate_static(rec, catalog.dataset, catalog,
                                      split.test_uids, k_reveal=k_rev,
                                      top_k=args.top_k, neg_sample=args.neg_sample,
                                      seed=args.seed)
                dt = time.time() - t0
                print(f"  {catalog.dataset.name:16s} | {rec.name:18s} | "
                      f"k_rev={k_rev:2d}  R@{args.top_k}={res.recall:.4f}  "
                      f"NDCG@{args.top_k}={res.ndcg:.4f}  ({dt:.1f}s)")
                rows.append({
                    "dataset": catalog.dataset.name,
                    "model": rec.name,
                    "k_reveal": k_rev,
                    f"Recall@{args.top_k}": round(res.recall, 4),
                    f"NDCG@{args.top_k}":   round(res.ndcg,   4),
                    "n_users": res.summary()["n_users"],
                })
        acs.flush()

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "exp1_main_table.csv", index=False)
    print(f"\nSaved → {TABLES_DIR/'exp1_main_table.csv'}")

# Experiment 2
def exp2(catalogs, splits, llm, acs_cache_dir, args):
    fig, axes = plt.subplots(1, len(catalogs), figsize=(6 * len(catalogs), 5),
                              squeeze=False)
    rows = []

    for ax, (ds_key, catalog) in zip(axes[0], catalogs.items()):
        split = splits[ds_key]
        acs = ACSScorer(catalog, llm=llm,
                        cache_path=acs_cache_dir / f"acs_{catalog.dataset.name}.json")
        recs = build_all_recommenders(catalog, split.train_interactions, acs,
                                       cfg=RAGALConfig(alpha=args.alpha, lam=args.lam))
        for rec in recs:
            curve_k = list(range(1, args.max_k_reveal + 1))
            vals = []
            for k_rev in curve_k:
                res = evaluate_static(rec, catalog.dataset, catalog,
                                      split.test_uids, k_reveal=k_rev,
                                      top_k=args.top_k, neg_sample=args.neg_sample,
                                      seed=args.seed)
                vals.append(res.recall)
                rows.append({
                    "dataset": catalog.dataset.name,
                    "model": rec.name,
                    "k_reveal": k_rev,
                    f"Recall@{args.top_k}": round(res.recall, 4),
                })
                print(f"  {catalog.dataset.name:16s} | {rec.name:18s} | "
                      f"k_rev={k_rev:2d}  R@{args.top_k}={res.recall:.4f}")
            ax.plot(curve_k, vals, marker="o", ms=3, lw=1.8,
                    color=MODEL_COLORS.get(rec.name, "#333"), label=rec.name)
        acs.flush()
        ax.set_title(catalog.dataset.name, fontweight="bold")
        ax.set_xlabel("k_reveal")
        ax.set_ylabel(f"Recall@{args.top_k}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Exp 2 — Learning curve", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "exp2_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    pd.DataFrame(rows).to_csv(TABLES_DIR / "exp2_learning_curve.csv", index=False)
    print(f"Saved → {FIGS_DIR/'exp2_learning_curve.png'}")

# Experiment 3
def exp3(catalogs, splits, llm, acs_cache_dir, args):
    rows = []
    for ds_key, catalog in catalogs.items():
        split = splits[ds_key]
        acs = ACSScorer(catalog, llm=llm,
                        cache_path=acs_cache_dir / f"acs_{catalog.dataset.name}.json")
        variants = [
            RAGALRecommender(catalog, acs_scorer=None,
                              alpha=args.alpha, lam=args.lam,
                              use_acs=False, use_snd=True, use_risk=False),
            RAGALRecommender(catalog, acs_scorer=acs,
                              alpha=args.alpha, lam=args.lam,
                              use_acs=True, use_snd=True, use_risk=False),
            RAGALRecommender(catalog, acs_scorer=acs,
                              alpha=args.alpha, lam=args.lam,
                              use_acs=True, use_snd=True, use_risk=True),
        ]
        for rec in variants:
            res = evaluate_static(rec, catalog.dataset, catalog,
                                  split.test_uids, k_reveal=5,
                                  top_k=args.top_k, neg_sample=args.neg_sample,
                                  seed=args.seed)
            print(f"  {catalog.dataset.name:16s} | {rec.name:18s} | "
                  f"R@{args.top_k}={res.recall:.4f}")
            rows.append({"dataset": catalog.dataset.name, "model": rec.name,
                         f"Recall@{args.top_k}": round(res.recall, 4)})
        acs.flush()

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "exp3_snd_isolation.csv", index=False)

    fig, axes = plt.subplots(1, len(catalogs), figsize=(5 * len(catalogs), 5),
                              squeeze=False)
    for ax, (ds_key, catalog) in zip(axes[0], catalogs.items()):
        sub = df[df["dataset"] == catalog.dataset.name]
        vals = sub[f"Recall@{args.top_k}"].tolist()
        names = sub["model"].tolist()
        bars = ax.bar(range(len(names)), vals,
                      color=[MODEL_COLORS.get(n, "#999") for n in names],
                      width=0.5, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                    f"{v:.4f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_title(f"{catalog.dataset.name} (k_reveal=5)", fontweight="bold")
        ax.set_ylabel(f"Recall@{args.top_k}")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals + [0.01]) * 1.25)

    fig.suptitle("Exp 3 — SND contribution isolation", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "exp3_snd_isolation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {FIGS_DIR/'exp3_snd_isolation.png'}")

# Experiment 4
def exp4(catalogs, splits, llm, acs_cache_dir, args):
    if "movielens" not in catalogs:
        print("  [skip] MovieLens catalog not loaded.")
        return

    catalog = catalogs["movielens"]
    split = splits["movielens"]
    acs = ACSScorer(catalog, llm=llm,
                    cache_path=acs_cache_dir / f"acs_{catalog.dataset.name}.json")

    with_risk = RAGALRecommender(catalog, acs_scorer=acs,
                                   alpha=args.alpha, lam=args.lam,
                                   use_acs=True, use_snd=True, use_risk=True)
    no_risk = RAGALRecommender(catalog, acs_scorer=acs,
                                 alpha=args.alpha, lam=0.0,
                                 use_acs=True, use_snd=True, use_risk=False)

    res_with = evaluate_sequential(with_risk, catalog.dataset, catalog,
                                    split.test_uids, max_steps=args.max_steps,
                                    neg_sample=args.neg_sample, seed=args.seed)
    res_no = evaluate_sequential(no_risk, catalog.dataset, catalog,
                                  split.test_uids, max_steps=args.max_steps,
                                  neg_sample=args.neg_sample, seed=args.seed)
    acs.flush()

    sr_with = res_with.first_step_skip_rate
    sr_no = res_no.first_step_skip_rate
    delta = sr_no - sr_with

    print(f"  First-step skip rate  with R(a) λ={args.lam}: {sr_with:.4f}")
    print(f"  First-step skip rate  without R(a) λ=0     : {sr_no:.4f}")
    print(f"  Δ = {delta:+.4f}  ({'R(a) helps' if delta > 0 else 'no improvement'})")

    df = pd.DataFrame([
        {"model": f"With R(a) λ={args.lam}",  "first_step_skip_rate": round(sr_with, 4),
         "cum_reward": round(res_with.cumulative_reward, 4)},
        {"model": "Without R(a) λ=0",          "first_step_skip_rate": round(sr_no, 4),
         "cum_reward": round(res_no.cumulative_reward, 4)},
    ])
    df.to_csv(TABLES_DIR / "exp4_risk_penalty.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    labels = [f"With R(a)\nλ={args.lam}", "Without R(a)\nλ=0"]
    vals = [sr_with, sr_no]
    bars = ax.bar(labels, vals, color=["#27ae60", "#e74c3c"], width=0.38,
                  edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004,
                f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title(f"Exp 4 — Risk penalty validation (MovieLens-1M, T={args.max_steps})",
                 fontweight="bold")
    ax.set_ylabel("First-step skip rate ↓")
    ax.set_ylim(0, max(vals + [0.01]) * 1.35)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "exp4_risk_penalty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {FIGS_DIR/'exp4_risk_penalty.png'}")

# Experiment 5
def exp5(catalogs, splits, llm, acs_cache_dir, args):
    rows = []
    for ds_key, catalog in catalogs.items():
        split = splits[ds_key]
        acs = ACSScorer(catalog, llm=llm,
                        cache_path=acs_cache_dir / f"acs_{catalog.dataset.name}.json")
        lin = LinUCBRecommender(catalog)
        full = RAGALRecommender(catalog, acs_scorer=acs,
                                  alpha=args.alpha, lam=args.lam,
                                  use_acs=True, use_snd=True, use_risk=True)
        for rec in [lin, full]:
            res = evaluate_static(rec, catalog.dataset, catalog,
                                  split.test_uids, k_reveal=5,
                                  top_k=args.top_k, neg_sample=args.neg_sample,
                                  seed=args.seed)
            print(f"  {catalog.dataset.name:16s} | {rec.name:18s} | "
                  f"R@{args.top_k}={res.recall:.4f}  NDCG@{args.top_k}={res.ndcg:.4f}")
            rows.append({"dataset": catalog.dataset.name, "model": rec.name,
                         f"Recall@{args.top_k}": round(res.recall, 4),
                         f"NDCG@{args.top_k}":   round(res.ndcg, 4)})
        acs.flush()

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "exp5_cross_domain.csv", index=False)

    print("\n── Relative margin (RAGAL vs LinUCB) ──")
    for ds_name in df["dataset"].unique():
        sub = df[df["dataset"] == ds_name].set_index("model")
        lin_v = sub.loc["LinUCB", f"Recall@{args.top_k}"]
        our_v = sub.loc["ACS+SND+R(a)", f"Recall@{args.top_k}"]
        mg = 100 * (our_v - lin_v) / (lin_v + 1e-9)
        print(f"  {ds_name:16s}  LinUCB={lin_v:.4f}  Ours={our_v:.4f}  margin={mg:+.1f}%")

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
    ax.set_xticklabels(dsets)
    ax.set_ylabel(f"Recall@{args.top_k}")
    ax.set_title("Exp 5 — Cross-domain consistency (k_reveal=5)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "exp5_cross_domain.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {FIGS_DIR/'exp5_cross_domain.png'}")

# Entry point
EXPERIMENTS = {
    1: ("Main cold-start table",        exp1),
    2: ("Learning curve",                exp2),
    3: ("SND contribution isolation",    exp3),
    4: ("Risk penalty validation",       exp4),
    5: ("Cross-domain consistency",      exp5),
}

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Run RAGAL online-phase experiments.")
    p.add_argument("--experiments", type=int, nargs="*", default=list(EXPERIMENTS),
                   choices=list(EXPERIMENTS))
    p.add_argument("--datasets", nargs="*", default=["movielens", "s2"],
                   choices=list(CATALOG_NAMES))
    p.add_argument("--model", default=HF_MODEL_DEFAULT,
                   help="HuggingFace model id")
    p.add_argument("--device", default="cuda", help='"cuda" or "cpu"')
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--load-in-4bit", action="store_true", help="Quantize the model to 4-bit")
    p.add_argument("--cache-dir", default="outputs/cache")
    p.add_argument("--acs-cache-dir", default="outputs/cache/acs")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--lam", type=float, default=0.3)
    p.add_argument("--n-users", type=int, default=N_USERS_EVAL)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--max-k-reveal", type=int, default=10,
                   help="Upper bound for exp2 learning-curve sweep.")
    p.add_argument("--k-reveals", type=int, nargs="*", default=list(K_REVEAL_DEFAULTS))
    p.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    p.add_argument("--neg-sample", type=int, default=NEG_SAMPLE)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    acs_cache_dir = Path(args.acs_cache_dir)
    acs_cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  RAGAL online-phase experiments")
    print(f"  datasets    : {args.datasets}")
    print(f"  experiments : {args.experiments}")
    print(f"  model       : {args.model}")
    print(f"  alpha/lambda: {args.alpha} / {args.lam}")
    print(f"  n-users     : {args.n_users}")
    print("=" * 60)

    catalogs = {}
    splits = {}
    for ds_key in args.datasets:
        print(f"\nLoading catalog for {ds_key}...")
        catalog = _load_catalog(ds_key, cache_dir)
        catalogs[ds_key] = catalog
        splits[ds_key] = split_users(catalog.dataset,
                                      n_test_users=args.n_users,
                                      seed=args.seed)

    llm = HFLLM(model_name=args.model, device=args.device,
                dtype=args.dtype, load_in_4bit=args.load_in_4bit)

    t_total = time.time()
    for exp_id in args.experiments:
        name, fn = EXPERIMENTS[exp_id]
        print(f"\n{'─'*60}\n  Experiment {exp_id}: {name}\n{'─'*60}")
        t0 = time.time()
        fn(catalogs, splits, llm, acs_cache_dir, args)
        print(f"  ✓ exp{exp_id} done in {time.time()-t0:.1f}s")

    print(f"\n{'='*60}\n  All done in {(time.time()-t_total)/60:.1f} min")
    print(f"  Tables  → {TABLES_DIR}")
    print(f"  Figures → {FIGS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
