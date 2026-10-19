"""Evaluation metrics.

Two conventions matter throughout and are easy to get wrong:

1. Everything predictive is scored on **deltas against the matching control**, not on raw
   expression. Raw-expression correlations are dominated by the mean expression profile,
   which every model including "predict the control mean" gets right, so they flatter all
   models and compress the differences between them.
2. Top-DEG sets are chosen from the *observed* test delta for that perturbation. This is
   evaluation-time information about the test point, which is standard in this literature,
   but it means top-DEG metrics cannot be used for model selection.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist


def pearson_delta(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    """Pearson correlation between predicted and observed mean shift across genes."""
    p, t = np.asarray(pred_delta, float).ravel(), np.asarray(true_delta, float).ravel()
    if p.std() < 1e-12 or t.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def top_deg_indices(true_delta: np.ndarray, k: int = 20) -> np.ndarray:
    """The k genes most moved by this perturbation, by absolute observed delta."""
    d = np.abs(np.asarray(true_delta, float).ravel())
    k = min(k, d.size)
    return np.argpartition(-d, k - 1)[:k]


def mse_top_degs(pred_delta: np.ndarray, true_delta: np.ndarray, k: int = 20) -> float:
    idx = top_deg_indices(true_delta, k)
    return float(np.mean((np.asarray(pred_delta).ravel()[idx] - np.asarray(true_delta).ravel()[idx]) ** 2))


def mse(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred_delta).ravel() - np.asarray(true_delta).ravel()) ** 2))


def energy_distance(x: np.ndarray, y: np.ndarray, max_n: int = 400, seed: int = 0) -> float:
    """Multivariate energy distance between two samples. 0 iff the distributions match.

    Subsampled to `max_n` per side because the statistic is O(n^2) in pairwise distances.
    """
    rng = np.random.default_rng(seed)
    x, y = _subsample(np.atleast_2d(x), max_n, rng), _subsample(np.atleast_2d(y), max_n, rng)
    xy = cdist(x, y).mean()
    xx = cdist(x, x).sum() / max(len(x) * (len(x) - 1), 1)
    yy = cdist(y, y).sum() / max(len(y) * (len(y) - 1), 1)
    return float(max(2.0 * xy - xx - yy, 0.0))


def mmd_rbf(x: np.ndarray, y: np.ndarray, max_n: int = 400, seed: int = 0) -> float:
    """Squared MMD with an RBF kernel at the median heuristic bandwidth."""
    rng = np.random.default_rng(seed)
    x, y = _subsample(np.atleast_2d(x), max_n, rng), _subsample(np.atleast_2d(y), max_n, rng)
    pooled = np.vstack([x, y])
    med = np.median(cdist(pooled, pooled))
    gamma = 1.0 / (2.0 * max(med, 1e-8) ** 2)
    k = lambda a, b: np.exp(-gamma * cdist(a, b) ** 2)
    kxx, kyy = k(x, x), k(y, y)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    m, n = len(x), len(y)
    return float(
        max(kxx.sum() / max(m * (m - 1), 1) + kyy.sum() / max(n * (n - 1), 1) - 2.0 * k(x, y).mean(), 0.0)
    )


def _subsample(a: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if len(a) <= max_n:
        return a
    return a[rng.choice(len(a), max_n, replace=False)]


# --- reversal and ranking ---------------------------------------------------------------


def reversal_error(pred_r: np.ndarray, true_r: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred_r, float) - np.asarray(true_r, float))))


def rank_correlation(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman correlation of an intervention ranking against ground truth."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    if pred.size < 3 or np.allclose(pred, pred[0]) or np.allclose(true, true[0]):
        return float("nan")
    return float(stats.spearmanr(pred, true).statistic)


def top_k_hit_rate(pred: np.ndarray, true: np.ndarray, k: int = 10, top_frac: float = 0.1) -> float:
    """Fraction of the model's top-k picks that are truly in the top `top_frac` of sets."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    k = min(k, pred.size)
    n_true = max(1, int(round(top_frac * true.size)))
    truly_best = set(np.argsort(-true)[:n_true].tolist())
    picks = np.argsort(-pred)[:k]
    return float(np.mean([int(i in truly_best) for i in picks]))


def precision_at_k(pred: np.ndarray, true: np.ndarray, k: int = 10, threshold: float = 0.5) -> float:
    """Fraction of the top-k picks whose true reversal clears `threshold`."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    k = min(k, pred.size)
    return float(np.mean(true[np.argsort(-pred)[:k]] > threshold))


# --- uncertainty quality ----------------------------------------------------------------


def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y, lower, upper = map(lambda a: np.asarray(a, float), (y, lower, upper))
    return float(np.mean((y >= lower) & (y <= upper)))


def pit_values(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Probability integral transform under a Gaussian predictive distribution."""
    sigma = np.clip(np.asarray(sigma, float), 1e-8, None)
    return stats.norm.cdf((np.asarray(y, float) - np.asarray(mu, float)) / sigma)


def calibration_error(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, n_bins: int = 10) -> float:
    """ECE for a real-valued predictive distribution: |empirical CDF - uniform| over bins.

    A perfectly calibrated model has uniform PIT values, so this is the mean absolute gap
    between nominal and empirical quantile levels.
    """
    pit = pit_values(y, mu, sigma)
    levels = np.linspace(0.0, 1.0, n_bins + 1)[1:]
    return float(np.mean([abs((pit <= q).mean() - q) for q in levels]))


def risk_coverage_curve(error: np.ndarray, uncertainty: np.ndarray, n_points: int = 20):
    """Risk as a function of coverage when abstaining on the most uncertain points.

    Returns (coverage, risk). Low uncertainty is kept first, so a useful uncertainty
    estimate makes this curve increase.
    """
    error, uncertainty = np.asarray(error, float), np.asarray(uncertainty, float)
    order = np.argsort(uncertainty)
    sorted_err = error[order]
    coverages = np.linspace(1.0 / len(error), 1.0, min(n_points, len(error)))
    risks = [sorted_err[: max(1, int(round(c * len(error))))].mean() for c in coverages]
    return coverages, np.array(risks)


def risk_coverage_auc(error: np.ndarray, uncertainty: np.ndarray, **kw) -> float:
    """Area under the risk-coverage curve. Lower is better; compare against random order."""
    cov, risk = risk_coverage_curve(error, uncertainty, **kw)
    return float(np.trapezoid(risk, cov) / (cov[-1] - cov[0])) if len(cov) > 1 else float(risk[0])


def auroc_error_detection(error: np.ndarray, uncertainty: np.ndarray, quantile: float = 0.75) -> float:
    """AUROC for using uncertainty to flag 'this prediction's error is large'."""
    error, uncertainty = np.asarray(error, float), np.asarray(uncertainty, float)
    label = error > np.quantile(error, quantile)
    if label.all() or not label.any():
        return float("nan")
    return float(_auroc(label, uncertainty))


def _auroc(label: np.ndarray, score: np.ndarray) -> float:
    # Mann-Whitney U with average ranks for ties.
    rank = stats.rankdata(score)
    n_pos = int(label.sum())
    n_neg = label.size - n_pos
    return (rank[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# --- statistics -------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, statistic=np.mean
):
    """Percentile bootstrap CI over perturbations. Returns (point, lo, hi)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    draws = statistic(v[rng.integers(0, v.size, size=(n_boot, v.size))], axis=1)
    return float(statistic(v)), float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def hierarchical_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
):
    """Two-level bootstrap: resample groups, then observations within each drawn group.

    The data here are nested -- many perturbations inside each of a few seeds -- and the two
    obvious options both waste it. Pooling every row pretends perturbations are independent
    replicates and gives intervals that are far too tight. Collapsing to one mean per seed
    is honest but throws away all within-seed information and, at five seeds, leaves
    intervals so wide that nothing is distinguishable.

    Resampling seeds first and then perturbations within the drawn seeds propagates both
    variance components, so the interval reflects uncertainty about which seeds were drawn
    *and* which perturbations were measured. Returns (point, lo, hi).
    """
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")

    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    if len(unique) < 2:
        return bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(unique), size=len(unique))
        sampled = [
            idx[rng.integers(0, idx.size, size=idx.size)] for idx in (members[p] for p in picked)
        ]
        draws[b] = values[np.concatenate(sampled)].mean()
    # The point estimate averages group means, so one large group cannot dominate it.
    point = float(np.mean([values[idx].mean() for idx in members]))
    return point, float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def paired_test(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict:
    """Paired comparison of two models scored on the same perturbations.

    Reports the Wilcoxon signed-rank p-value (no normality assumption), a bootstrap CI on
    the mean paired difference, and Cohen's d_z as the effect size.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    if d.size < 2 or np.allclose(d, 0):
        return {"n": int(d.size), "mean_diff": float(d.mean()) if d.size else float("nan"),
                "p_value": float("nan"), "cohens_dz": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan")}
    _, lo, hi = bootstrap_ci(d, seed=seed)
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p = float("nan")
    return {
        "n": int(d.size),
        "mean_diff": float(d.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "p_value": p,
        "cohens_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"),
    }
