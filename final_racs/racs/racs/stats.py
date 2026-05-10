"""
Tiny statistics helpers used by run_online.py for paper-grade reporting:
bootstrap CI over a vector of per-user metric values, and paired Wilcoxon
signed-rank between two per-user vectors aligned on the same user list.

Both functions tolerate small samples and return NaN when the test cannot
be computed (e.g. all-zero diff).
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 1000, ci: float = 0.95,
                 seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI over the *mean* of `values`.

    Returns (lo, hi). If `values` is empty returns (0.0, 0.0).
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lo, hi


def paired_wilcoxon(a, b) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value of (a − b).

    Returns NaN when scipy is missing, lengths differ, or all diffs are zero.
    """
    a = np.asarray(list(a), dtype=np.float64)
    b = np.asarray(list(b), dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return float("nan")
    diff = a - b
    if np.allclose(diff, 0.0):
        return 1.0
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return float("nan")
    try:
        _, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        return float(p)
    except ValueError:
        return float("nan")


def mean_std(values) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))
