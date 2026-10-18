"""Split-conformal calibration with a normalised nonconformity score.

Score: `|y - yhat| / spread`, where `spread` is ensemble disagreement. Dividing by the
spread is what makes the interval *adaptive* -- a model that knows it is unsure gets a
wider interval for the same calibration set -- and it is why the ensemble is worth its
cost beyond simply averaging predictions.

**Exchangeability.** The guarantee is that a new point's score is exchangeable with the
calibration scores. Calibration is grouped by *perturbation*: a perturbation is either
entirely in calibration or entirely out, so scores are not correlated through a shared
perturbation. This buys exchangeability across perturbations drawn from the same
distribution. It is explicitly **violated** under cell-context shift, where the test
context is not exchangeable with the training contexts. Coverage is therefore reported
separately for the unseen-context split, and is expected to fall short of nominal there;
reporting it as if the guarantee held would be the fraud this module exists to avoid.

The finite-sample quantile is the standard one: with `n` calibration scores and target
miscoverage `alpha`, take the `ceil((n+1)(1-alpha))`-th smallest score. When that index
exceeds `n` the calibration set is too small to certify the requested level at all, and
the radius is infinite -- an honest "cannot say" rather than a silently invalid interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MIN_SPREAD = 1e-8


@dataclass
class SplitConformal:
    """Scalar split conformal, optionally normalised by a per-point uncertainty."""

    alpha: float = 0.1
    normalise: bool = True
    scores_: np.ndarray = field(default_factory=lambda: np.empty(0))

    def fit(self, residuals: np.ndarray, spreads: np.ndarray | None = None) -> "SplitConformal":
        residuals = np.abs(np.asarray(residuals, float).ravel())
        if self.normalise:
            if spreads is None:
                raise ValueError("normalised conformal needs per-point spreads")
            spreads = np.clip(np.asarray(spreads, float).ravel(), MIN_SPREAD, None)
            if spreads.shape != residuals.shape:
                raise ValueError("residuals and spreads must have the same shape")
            scores = residuals / spreads
        else:
            scores = residuals
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            raise ValueError("no finite calibration scores")
        self.scores_ = np.sort(scores)
        self.n_ = scores.size
        k = math.ceil((self.n_ + 1) * (1.0 - self.alpha))
        # k > n means the calibration set cannot certify this level: say so, don't fake it.
        self.quantile_ = float("inf") if k > self.n_ else float(self.scores_[k - 1])
        self.certifiable_ = k <= self.n_
        return self

    @property
    def min_calibration_size(self) -> int:
        """Smallest calibration set that can certify `1 - alpha` at all."""
        return math.ceil(1.0 / self.alpha) - 1

    def radius(self, spread: np.ndarray | float = 1.0) -> np.ndarray:
        s = np.clip(np.asarray(spread, float), MIN_SPREAD, None) if self.normalise else 1.0
        return self.quantile_ * s

    def interval(self, pred: np.ndarray, spread: np.ndarray | float = 1.0):
        pred = np.asarray(pred, float)
        r = self.radius(spread)
        return pred - r, pred + r

    def covers(self, y: np.ndarray, pred: np.ndarray, spread: np.ndarray | float = 1.0) -> np.ndarray:
        lo, hi = self.interval(pred, spread)
        y = np.asarray(y, float)
        return (y >= lo) & (y <= hi)

    def coverage(self, y, pred, spread=1.0) -> float:
        return float(np.mean(self.covers(y, pred, spread)))


@dataclass
class VectorConformal(SplitConformal):
    """Conformal ball on a vector prediction, using the error norm as the score.

    This is the form the brief asks for -- "error of predicted mean shift divided by
    ensemble spread" -- and certifies `||y - yhat|| <= q * spread`.
    """

    def fit_vectors(
        self, y: np.ndarray, pred: np.ndarray, spreads: np.ndarray | None = None
    ) -> "VectorConformal":
        y, pred = np.atleast_2d(y), np.atleast_2d(pred)
        if y.shape != pred.shape:
            raise ValueError("y and pred must have the same shape")
        return self.fit(np.linalg.norm(y - pred, axis=1), spreads)

    def covers_vectors(
        self, y: np.ndarray, pred: np.ndarray, spreads: np.ndarray | float = 1.0
    ) -> np.ndarray:
        err = np.linalg.norm(np.atleast_2d(y) - np.atleast_2d(pred), axis=1)
        return err <= self.radius(spreads)


# --- abstention -------------------------------------------------------------------------


def abstention_threshold(spreads: np.ndarray, keep_frac: float) -> float:
    """Uncertainty cut-off that keeps the `keep_frac` least-uncertain calibration points.

    Calibrated on the calibration split, then applied unchanged at test time. Choosing it
    on test data would make every risk-coverage number meaningless.
    """
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError("keep_frac must be in (0, 1]")
    s = np.asarray(spreads, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("no finite spreads to calibrate on")
    return float(np.quantile(s, keep_frac))


def selective_risk(
    error: np.ndarray, spreads: np.ndarray, threshold: float
) -> tuple[float, float]:
    """(coverage, mean error on the retained points) when abstaining above `threshold`."""
    error, spreads = np.asarray(error, float), np.asarray(spreads, float)
    keep = spreads <= threshold
    if not keep.any():
        return 0.0, float("nan")
    return float(keep.mean()), float(error[keep].mean())


def coverage_report(
    y: np.ndarray,
    pred: np.ndarray,
    spreads: np.ndarray,
    alpha: float = 0.1,
    normalise: bool = True,
    calibrator: SplitConformal | None = None,
) -> dict:
    """Empirical coverage and mean interval width against the nominal level."""
    cal = calibrator
    if cal is None:
        raise ValueError("pass a calibrator fitted on the calibration split")
    y = np.asarray(y, float).ravel()
    pred = np.asarray(pred, float).ravel()
    lo, hi = cal.interval(pred, spreads)
    return {
        "nominal_coverage": 1.0 - cal.alpha,
        "empirical_coverage": cal.coverage(y, pred, spreads),
        "mean_interval_width": float(np.mean(hi - lo)),
        "median_interval_width": float(np.median(hi - lo)),
        "n_test": int(y.size),
        "n_calibration": int(cal.n_),
        "certifiable": bool(cal.certifiable_),
        "conformal_quantile": cal.quantile_,
        "normalised": normalise,
    }
