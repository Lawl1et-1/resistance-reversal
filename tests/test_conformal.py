"""Conformal calibration tests, including the coverage guarantee on exchangeable data.

The coverage test is the one that matters: split conformal promises marginal coverage of at
least `1 - alpha` whenever the calibration and test scores are exchangeable, and it promises
nothing otherwise. Verifying it on data constructed to be exchangeable separates "the
implementation is correct" from "the assumption happens to hold on our data", which are
different claims and are reported separately in the results.
"""

import numpy as np
import pytest

from src.uncertainty.conformal import (
    SplitConformal,
    VectorConformal,
    abstention_threshold,
    coverage_report,
    selective_risk,
)


def test_finite_sample_quantile_index():
    """The quantile must be the ceil((n+1)(1-alpha))-th smallest score, exactly."""
    scores = np.arange(1.0, 101.0)  # n = 100
    cal = SplitConformal(alpha=0.1, normalise=False).fit(scores)
    # ceil(101 * 0.9) = 91 -> the 91st smallest score, which is 91.0
    assert cal.quantile_ == pytest.approx(91.0)
    assert cal.certifiable_

    cal05 = SplitConformal(alpha=0.05, normalise=False).fit(scores)
    assert cal05.quantile_ == pytest.approx(96.0)  # ceil(101 * 0.95) = 96


def test_too_few_calibration_points_is_not_certifiable():
    """With n < 1/alpha - 1 the level cannot be certified; say so rather than fake it."""
    cal = SplitConformal(alpha=0.1, normalise=False).fit(np.arange(5.0))
    assert not cal.certifiable_
    assert cal.quantile_ == float("inf")
    assert np.isinf(cal.radius(1.0))
    # An infinite interval covers everything -- vacuous, but never falsely narrow.
    assert cal.coverage(np.array([1e9]), np.array([0.0]), np.array([1.0])) == 1.0


def test_min_calibration_size():
    assert SplitConformal(alpha=0.1).min_calibration_size == 9
    assert SplitConformal(alpha=0.05).min_calibration_size == 19
    cal = SplitConformal(alpha=0.1, normalise=False).fit(np.arange(9.0))
    assert cal.certifiable_


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_coverage_guarantee_on_exchangeable_data(alpha):
    """Empirical coverage must reach the nominal level when exchangeability holds."""
    rng = np.random.default_rng(0)
    covered = []
    for _ in range(300):
        # Calibration and test points are i.i.d. from the same law: exchangeable.
        cal_res = np.abs(rng.normal(size=200))
        cal = SplitConformal(alpha=alpha, normalise=False).fit(cal_res)
        y = rng.normal(size=100)
        covered.append(cal.coverage(y, np.zeros(100), np.ones(100)))
    mean_cov = float(np.mean(covered))
    assert mean_cov >= 1 - alpha - 0.02, f"under-covered: {mean_cov:.3f} vs {1-alpha:.2f}"
    assert mean_cov <= 1 - alpha + 0.05, f"wildly conservative: {mean_cov:.3f}"


def test_normalised_coverage_on_heteroscedastic_data():
    """With spread-normalised scores, coverage holds when noise scales with the spread."""
    rng = np.random.default_rng(0)
    covered = []
    for _ in range(300):
        spread_cal = rng.uniform(0.2, 3.0, size=300)
        res_cal = np.abs(rng.normal(scale=spread_cal))
        cal = SplitConformal(alpha=0.1, normalise=True).fit(res_cal, spread_cal)
        spread_te = rng.uniform(0.2, 3.0, size=150)
        y = rng.normal(scale=spread_te)
        covered.append(cal.coverage(y, np.zeros(150), spread_te))
    mean_cov = float(np.mean(covered))
    assert mean_cov >= 0.88, f"under-covered: {mean_cov:.3f}"
    assert mean_cov <= 0.95


def test_normalisation_makes_intervals_adaptive():
    """A point the ensemble disagrees on must get a wider interval."""
    rng = np.random.default_rng(0)
    spread = rng.uniform(0.2, 3.0, size=300)
    cal = SplitConformal(alpha=0.1).fit(np.abs(rng.normal(scale=spread)), spread)
    narrow = cal.radius(0.2)
    wide = cal.radius(3.0)
    assert wide > 5 * narrow


def test_unnormalised_intervals_are_constant_width():
    cal = SplitConformal(alpha=0.1, normalise=False).fit(np.abs(np.random.default_rng(0).normal(size=200)))
    assert cal.radius(0.1) == cal.radius(10.0)


def test_normalised_fit_requires_spreads():
    with pytest.raises(ValueError, match="spreads"):
        SplitConformal(alpha=0.1, normalise=True).fit(np.ones(10))
    with pytest.raises(ValueError, match="same shape"):
        SplitConformal(alpha=0.1, normalise=True).fit(np.ones(10), np.ones(5))


def test_rejects_all_nonfinite_calibration():
    with pytest.raises(ValueError, match="finite"):
        SplitConformal(alpha=0.1, normalise=False).fit(np.array([np.nan, np.inf]))


def test_vector_conformal_covers_the_ball():
    rng = np.random.default_rng(0)
    dim = 8
    spread_cal = rng.uniform(0.5, 2.0, size=400)
    pred_cal = np.zeros((400, dim))
    y_cal = rng.normal(scale=spread_cal[:, None], size=(400, dim))
    cal = VectorConformal(alpha=0.1).fit_vectors(y_cal, pred_cal, spread_cal)

    spread_te = rng.uniform(0.5, 2.0, size=400)
    y_te = rng.normal(scale=spread_te[:, None], size=(400, dim))
    cov = cal.covers_vectors(y_te, np.zeros((400, dim)), spread_te).mean()
    assert 0.85 <= cov <= 0.96


def test_vector_conformal_shape_validation():
    with pytest.raises(ValueError, match="same shape"):
        VectorConformal(alpha=0.1).fit_vectors(np.zeros((4, 3)), np.zeros((4, 2)), np.ones(4))


def test_abstention_threshold_keeps_requested_fraction():
    spreads = np.linspace(0.0, 1.0, 1000)
    t = abstention_threshold(spreads, keep_frac=0.7)
    assert 0.65 <= (spreads <= t).mean() <= 0.75
    assert abstention_threshold(spreads, 1.0) >= spreads.max()


def test_abstention_threshold_validation():
    with pytest.raises(ValueError, match="keep_frac"):
        abstention_threshold(np.ones(5), 0.0)
    with pytest.raises(ValueError, match="keep_frac"):
        abstention_threshold(np.ones(5), 1.5)


def test_selective_risk_falls_when_uncertainty_is_informative():
    """Abstaining on the uncertain points must reduce error when uncertainty tracks it."""
    rng = np.random.default_rng(0)
    spreads = rng.uniform(0.1, 2.0, size=500)
    error = spreads * np.abs(rng.normal(size=500))  # error grows with spread
    _, risk_all = selective_risk(error, spreads, abstention_threshold(spreads, 1.0))
    cov_half, risk_half = selective_risk(error, spreads, abstention_threshold(spreads, 0.5))
    assert 0.45 <= cov_half <= 0.55
    assert risk_half < risk_all


def test_selective_risk_flat_when_uncertainty_is_uninformative():
    rng = np.random.default_rng(0)
    error = np.abs(rng.normal(size=2000))
    spreads = rng.uniform(0.0, 1.0, size=2000)  # unrelated to error
    _, risk_all = selective_risk(error, spreads, abstention_threshold(spreads, 1.0))
    _, risk_half = selective_risk(error, spreads, abstention_threshold(spreads, 0.5))
    assert abs(risk_half - risk_all) < 0.1 * risk_all


def test_selective_risk_with_no_retained_points():
    cov, risk = selective_risk(np.ones(5), np.ones(5), threshold=0.0)
    assert cov == 0.0
    assert np.isnan(risk)


def test_coverage_report_fields():
    rng = np.random.default_rng(0)
    spread = rng.uniform(0.5, 2.0, size=200)
    cal = SplitConformal(alpha=0.1).fit(np.abs(rng.normal(scale=spread)), spread)
    spread_te = rng.uniform(0.5, 2.0, size=80)
    rep = coverage_report(
        rng.normal(scale=spread_te), np.zeros(80), spread_te, calibrator=cal
    )
    assert rep["nominal_coverage"] == pytest.approx(0.9)
    assert 0.0 <= rep["empirical_coverage"] <= 1.0
    assert rep["mean_interval_width"] > 0
    assert rep["n_calibration"] == 200
    assert rep["n_test"] == 80
    assert rep["certifiable"]


def test_coverage_report_requires_a_calibrator():
    with pytest.raises(ValueError, match="calibrator"):
        coverage_report(np.ones(3), np.ones(3), np.ones(3))
