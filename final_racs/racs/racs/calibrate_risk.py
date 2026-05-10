"""
Quantile-rescale R(a) values inside one or more CatalogStore directories.

Why: the contrast-logprob risk scoring collapses onto SAFE — most items end
up with R≈0 and λ·R(a) is numerically inert during scoring. Ranking is
correct, magnitudes are useless. Replace each value with its rank percentile
in [0, 1] over the catalog's distribution. Order preserved, distribution
becomes uniform.

The original file is backed up to risk_scores.original.json on first run.
Re-running re-calibrates from the original backup, so the operation is
idempotent.

Usage:
    python -m racs.calibrate_risk --cache-dir <path/to/cache>
    python -m racs.calibrate_risk --catalog <path/to/catalogs/MovieLens-1M>
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def quantile_rescale(scores: dict[str, float]) -> dict[str, float]:
    items = list(scores.items())
    items.sort(key=lambda kv: kv[1])
    n = len(items)
    if n <= 1:
        return {iid: 0.5 for iid, _ in items}
    return {iid: rank / (n - 1) for rank, (iid, _) in enumerate(items)}


def calibrate_catalog(catalog_dir: Path) -> bool:
    risk_path = catalog_dir / "risk_scores.json"
    backup_path = catalog_dir / "risk_scores.original.json"
    if not risk_path.exists():
        logger.warning("No risk_scores.json at %s — skipping", catalog_dir)
        return False

    if backup_path.exists():
        with open(backup_path) as f:
            raw = json.load(f)
        logger.info("[%s] reloading raw scores from backup %s",
                    catalog_dir.name, backup_path.name)
    else:
        with open(risk_path) as f:
            raw = json.load(f)
        shutil.copyfile(risk_path, backup_path)
        logger.info("[%s] backed up original → %s",
                    catalog_dir.name, backup_path.name)

    rescaled = quantile_rescale(raw)
    with open(risk_path, "w") as f:
        json.dump(rescaled, f)

    arr_old = np.array(list(raw.values()), dtype=np.float64)
    arr_new = np.array(list(rescaled.values()), dtype=np.float64)
    logger.info(
        "[%s] %d items | raw: min=%.4f p50=%.4f p99=%.4f max=%.4f"
        " | calibrated: min=%.4f p50=%.4f p99=%.4f max=%.4f",
        catalog_dir.name, len(raw),
        arr_old.min(), np.percentile(arr_old, 50),
        np.percentile(arr_old, 99), arr_old.max(),
        arr_new.min(), np.percentile(arr_new, 50),
        np.percentile(arr_new, 99), arr_new.max(),
    )
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="Quantile-rescale R(a) values in CatalogStore directories.")
    p.add_argument("--cache-dir",
                   help="Cache root containing catalogs/ — calibrates every "
                        "catalog under it.")
    p.add_argument("--catalog",
                   help="Path to a single CatalogStore directory.")
    args = p.parse_args()

    if args.catalog:
        if not calibrate_catalog(Path(args.catalog)):
            raise SystemExit(1)
    elif args.cache_dir:
        root = Path(args.cache_dir) / "catalogs"
        if not root.exists():
            raise SystemExit(f"No catalogs/ under {args.cache_dir}")
        catalogs = [c for c in sorted(root.iterdir())
                    if (c / "manifest.json").exists()]
        if not catalogs:
            raise SystemExit(f"No catalogs found in {root}")
        n_done = 0
        for c in catalogs:
            if calibrate_catalog(c):
                n_done += 1
        logger.info("Calibrated %d/%d catalogs", n_done, len(catalogs))
    else:
        raise SystemExit("Pass --cache-dir or --catalog")


if __name__ == "__main__":
    main()
