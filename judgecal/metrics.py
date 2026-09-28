"""Discrimination and calibration metrics, with clustered bootstrap CIs.

All functions take p = P(success) in [0, 1] and y in {0, 1}.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import rankdata


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUROC with tie handling. NaN if only one class present."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def balanced_accuracy(p: np.ndarray, y: np.ndarray, thr: float = 0.5) -> float:
    pred = (np.asarray(p) >= thr).astype(int)
    y = np.asarray(y, int)
    tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else np.nan
    tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else np.nan
    return float(np.nanmean([tpr, tnr]))


def failure_recall(p: np.ndarray, y: np.ndarray, thr: float = 0.5) -> float:
    """Fraction of true failures the judge flags. The number a customer cares about most."""
    y = np.asarray(y, int)
    if not (y == 0).any():
        return float("nan")
    return float((np.asarray(p)[y == 0] < thr).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, int)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Top-label expected calibration error with equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Returns (mean predicted P(success), observed success rate, count) per bin."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return np.array(rows) if rows else np.zeros((0, 3))


METRICS: dict[str, Callable] = {
    "auroc": auroc,
    "bal_acc": balanced_accuracy,
    "fail_recall": failure_recall,
    "ece": ece,
    "brier": brier,
    "nll": nll,
}


def cluster_bootstrap_ci(
    fn: Callable,
    p: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """Point estimate and percentile CI, resampling whole groups (tasks).

    Episodes from the same task are correlated. Resampling episodes independently
    gives CIs that are too narrow. Returns (point, lo, hi, n_valid_resamples).
    Resamples where the metric is undefined (e.g. one class only) are dropped and
    counted, so a low count is visible rather than silently biasing the CI.
    """
    p, y, groups = np.asarray(p), np.asarray(y), np.asarray(groups)
    point = fn(p, y)
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        v = fn(p[idx], y[idx])
        if np.isfinite(v):
            stats.append(v)
    if len(stats) < 0.5 * n_boot:
        return point, float("nan"), float("nan"), len(stats)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi), len(stats)
