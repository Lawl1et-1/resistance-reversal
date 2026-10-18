"""Metric correctness tests: each metric checked against a case whose answer is known."""

import numpy as np
import pytest

from src.eval.metrics import (
    auroc_error_detection,
    bootstrap_ci,
    calibration_error,
    energy_distance,
    interval_coverage,
    mmd_rbf,
    mse,
    mse_top_degs,
    paired_test,
    pearson_delta,
    pit_values,
    precision_at_k,
    rank_correlation,
    risk_coverage_auc,
    risk_coverage_curve,
    top_deg_indices,
    top_k_hit_rate,
)


def test_pearson_delta_exact_cases():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson_delta(x, x) == pytest.approx(1.0)
    assert pearson_delta(x, -x) == pytest.approx(-1.0)
    assert pearson_delta(2 * x + 5, x) == pytest.approx(1.0), "must be scale/shift invariant"


def test_pearson_delta_constant_input_is_nan():
    assert np.isnan(pearson_delta(np.ones(5), np.arange(5.0)))


def test_top_deg_indices_picks_largest_absolute():
    d = np.array([0.0, -5.0, 0.1, 3.0, -0.2])
    assert set(top_deg_indices(d, k=2).tolist()) == {1, 3}


def test_mse_top_degs_only_scores_those_genes():
    true = np.array([0.0, 10.0, 0.0, -10.0])
    pred = np.array([99.0, 10.0, 99.0, -10.0])  # wrong everywhere except the top DEGs
    assert mse_top_degs(pred, true, k=2) == pytest.approx(0.0)
    assert mse(pred, true) > 1000.0


def test_energy_distance_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 5)), rng.normal(size=(300, 5))
    same = energy_distance(a, b)
    shifted = energy_distance(a, b + 3.0)
    assert same < 0.1
    assert shifted > same + 1.0


def test_energy_distance_is_nonnegative_and_symmetric():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(120, 4)), rng.normal(2.0, size=(120, 4))
    assert energy_distance(a, b) >= 0.0
    assert energy_distance(a, b) == pytest.approx(energy_distance(b, a), rel=1e-9)


def test_mmd_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 5)), rng.normal(size=(300, 5))
    assert mmd_rbf(a, b) < 0.02
    assert mmd_rbf(a, b + 3.0) > 0.1


def test_interval_coverage_counts_correctly():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert interval_coverage(y, np.full(4, -1.0), np.full(4, 10.0)) == pytest.approx(1.0)
    assert interval_coverage(y, np.full(4, 1.5), np.full(4, 10.0)) == pytest.approx(0.5)


def test_pit_is_uniform_when_calibrated():
    rng = np.random.default_rng(0)
    mu, sigma = np.zeros(4000), np.ones(4000)
    pit = pit_values(rng.normal(mu, sigma), mu, sigma)
    assert abs(pit.mean() - 0.5) < 0.02
    assert abs(pit.std() - 1 / np.sqrt(12)) < 0.02


def test_calibration_error_detects_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.normal(size=4000)
    calibrated = calibration_error(y, np.zeros(4000), np.ones(4000))
    overconfident = calibration_error(y, np.zeros(4000), np.full(4000, 0.25))
    underconfident = calibration_error(y, np.zeros(4000), np.full(4000, 4.0))
    assert calibrated < 0.02
    # Both failure directions must register, and far above the calibrated floor. The
    # ceiling of this ECE definition is 0.25 (PIT piled at 0 and 1), so ~0.15 is severe.
    assert overconfident > 0.1 and overconfident > 5 * calibrated
    assert underconfident > 0.1 and underconfident > 5 * calibrated


def test_risk_coverage_rewards_informative_uncertainty():
    rng = np.random.default_rng(0)
    error = rng.exponential(size=500)
    informative = error + 0.05 * rng.normal(size=500)  # tracks the error
    uninformative = rng.normal(size=500)  # unrelated
    assert risk_coverage_auc(error, informative) < risk_coverage_auc(error, uninformative)


def test_risk_coverage_curve_increases_with_perfect_uncertainty():
    error = np.linspace(0.0, 1.0, 50)
    cov, risk = risk_coverage_curve(error, error)
    assert np.all(np.diff(risk) >= -1e-9), "risk must rise as more uncertain points are kept"
    assert cov[-1] == pytest.approx(1.0)
    assert risk[-1] == pytest.approx(error.mean())


def test_auroc_error_detection_bounds():
    error = np.linspace(0, 1, 200)
    assert auroc_error_detection(error, error) == pytest.approx(1.0)
    assert auroc_error_detection(error, -error) == pytest.approx(0.0)


def test_rank_correlation_known_values():
    true = np.arange(10.0)
    assert rank_correlation(true, true) == pytest.approx(1.0)
    assert rank_correlation(-true, true) == pytest.approx(-1.0)


def test_top_k_hit_rate_and_precision():
    true = np.array([0.9, 0.8, 0.1, 0.0, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0])
    perfect = true.copy()
    assert top_k_hit_rate(perfect, true, k=1, top_frac=0.1) == pytest.approx(1.0)
    worst = -true
    assert top_k_hit_rate(worst, true, k=1, top_frac=0.1) == pytest.approx(0.0)
    assert precision_at_k(perfect, true, k=2, threshold=0.5) == pytest.approx(1.0)
    assert precision_at_k(worst, true, k=2, threshold=0.5) == pytest.approx(0.0)


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(5.0, 1.0, size=400)
    point, lo, hi = bootstrap_ci(v)
    assert lo < point < hi
    assert lo < 5.0 < hi


def test_bootstrap_ci_handles_degenerate_input():
    assert all(np.isnan(x) for x in bootstrap_ci(np.array([])))
    assert bootstrap_ci(np.array([3.0])) == (3.0, 3.0, 3.0)
    point, lo, hi = bootstrap_ci(np.array([1.0, np.nan, 3.0]))
    assert np.isfinite([point, lo, hi]).all()


def test_paired_test_detects_a_real_difference():
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    res = paired_test(base + 0.8, base)
    assert res["mean_diff"] == pytest.approx(0.8, abs=0.05)
    assert res["p_value"] < 1e-6
    assert res["ci_lo"] > 0
    assert res["cohens_dz"] > 3


def test_paired_test_on_identical_inputs_is_not_significant():
    v = np.linspace(0, 1, 50)
    res = paired_test(v, v)
    assert res["mean_diff"] == pytest.approx(0.0)
    assert np.isnan(res["p_value"])
