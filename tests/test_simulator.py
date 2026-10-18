"""Ground-truth recovery tests for the synthetic GRN simulator.

These assert the properties every downstream experiment relies on: the network is
bistable, reversal is achievable within k but not by accident, and the causal structure
built into the weights is the one the ground-truth oracle reports.
"""

import numpy as np
import pytest

from src.simulator.grn import (
    RESISTANCE,
    SENSITIVE,
    GRNConfig,
    build_grn,
    check_bistable,
    check_reversible,
    enumerate_perturbation_sets,
    ground_truth_reversal,
    observe,
    optimal_reversal_set,
    sample_cell_bias,
    seed_state,
    simulate,
    true_sensitivity,
)

SEEDS = [0, 1, 2, 3, 4]


def smoke_grn(seed: int = 0):
    return build_grn(GRNConfig(n_genes=60, n_regulators=12, n_program=15, seed=seed))


@pytest.mark.parametrize("seed", SEEDS)
def test_bistable_attractors(seed):
    """Sensitive and resistant seeds relax to well-separated attractors."""
    s_sens, s_res = check_bistable(smoke_grn(seed))
    assert s_sens > 0.85
    assert s_res < 0.15


@pytest.mark.parametrize("seed", SEEDS)
def test_attractors_are_stable(seed):
    """An attractor state stays put: relaxing again does not move it."""
    grn = smoke_grn(seed)
    rng = np.random.default_rng(0)
    x = simulate(grn, seed_state(grn, "resistant", 8, rng))
    again = simulate(grn, x)
    assert np.abs(again - x).max() < 0.02


@pytest.mark.parametrize("seed", SEEDS)
def test_reversal_achievable_within_k(seed):
    """Some set of size <= 2 reverses, so the search problem is not vacuous."""
    assert check_reversible(smoke_grn(seed), k=2, n_cells=8) > 0.6


@pytest.mark.parametrize("seed", SEEDS)
def test_most_sets_do_not_reverse(seed):
    """The median set barely moves the state: reversal is not handed out for free."""
    truth = ground_truth_reversal(smoke_grn(seed), k=2, n_cells=8)
    vals = [v for p, v in truth.items() if p]
    assert np.median(vals) < 0.25


def test_resistance_drivers_beat_bystanders():
    """The regulators wired as resistance drivers are the ones the oracle prefers.

    Layout truth: regulator i has mode i % 4; mode 0 is a resistance driver and mode 3
    is a bystander with negligible drive onto the switch.
    """
    grn = smoke_grn(0)
    truth = ground_truth_reversal(grn, k=2, n_cells=8)
    drivers = tuple(sorted(((int(grn.regulators[0]), -1), (int(grn.regulators[4]), -1))))
    bystanders = tuple(sorted(((int(grn.regulators[3]), -1), (int(grn.regulators[7]), -1))))
    assert truth[drivers] > truth[bystanders] + 0.3


def test_knockdown_suppresses_target_gene():
    """A knockdown actually drives its target gene's expression down."""
    grn = smoke_grn(0)
    rng = np.random.default_rng(0)
    x0 = seed_state(grn, "resistant", 8, rng)
    gene = int(grn.regulators[0])
    base = simulate(grn, x0)[:, gene].mean()
    hit = simulate(grn, x0, ((gene, -1),))[:, gene].mean()
    assert hit < 0.1
    assert hit < base - 0.2


def test_reversal_moves_both_programs():
    """A reversing set raises the sensitive program and lowers the resistance program."""
    grn = smoke_grn(0)
    rng = np.random.default_rng(0)
    x0 = seed_state(grn, "resistant", 16, rng)
    best, score = optimal_reversal_set(grn, k=2, n_cells=8)
    assert score > 0.6
    before, after = simulate(grn, x0), simulate(grn, x0, best)
    assert after[:, grn.program_sensitive].mean() > before[:, grn.program_sensitive].mean() + 0.2
    assert after[:, grn.program_resistance].mean() < before[:, grn.program_resistance].mean() - 0.2
    assert after[:, SENSITIVE].mean() > after[:, RESISTANCE].mean()


def test_true_sensitivity_bounds():
    grn = smoke_grn(0)
    rng = np.random.default_rng(0)
    for ph in ("sensitive", "resistant"):
        s = true_sensitivity(grn, simulate(grn, seed_state(grn, ph, 8, rng)))
        assert np.all((s >= 0.0) & (s <= 1.0))


def test_simulation_is_deterministic_without_noise():
    """No rng passed means no process noise, so runs are bit-reproducible."""
    grn = smoke_grn(0)
    x0 = seed_state(grn, "resistant", 4, np.random.default_rng(0))
    assert np.array_equal(simulate(grn, x0), simulate(grn, x0))


def test_same_seed_same_network():
    a, b = smoke_grn(3), smoke_grn(3)
    assert np.array_equal(a.W, b.W)
    assert np.array_equal(a.b, b.b)
    assert not np.array_equal(smoke_grn(3).W, smoke_grn(4).W)


def test_enumerate_sets_are_canonical_and_unique():
    grn = smoke_grn(0)
    sets = enumerate_perturbation_sets(grn, k=3)
    assert len(sets) == len(set(sets)), "duplicate perturbation sets"
    for s in sets:
        assert list(s) == sorted(s), "sets must be canonically ordered"
        genes = [g for g, _ in s]
        assert len(genes) == len(set(genes)), "a gene cannot appear twice in one set"
    n = len(grn.regulators)
    assert len(sets) == n + n * (n - 1) // 2 + n * (n - 1) * (n - 2) // 6


def test_observation_layer_adds_noise_and_dropout():
    grn = build_grn(GRNConfig(n_genes=60, n_regulators=12, n_program=15, dropout=0.5, seed=0))
    rng = np.random.default_rng(0)
    x = simulate(grn, seed_state(grn, "resistant", 64, rng))
    y = observe(grn, x, rng)
    assert y.shape == x.shape
    assert np.all(y >= 0.0)
    assert 0.3 < (y == 0).mean() < 0.7, "dropout rate not in the expected range"


def test_config_rejects_too_small_network():
    with pytest.raises(ValueError):
        GRNConfig(n_genes=10, n_regulators=20, n_program=40)


def test_module_members_coexpress():
    """Regulators in a module co-express; regulators in different modules do not.

    Without this the unseen-gene splits are unsolvable in principle, because co-expression
    is the only channel through which a model could learn anything about a gene it never
    perturbed. See LOG.md Failure 5.
    """
    grn = smoke_grn(0)
    rng = np.random.default_rng(0)
    x = simulate(
        grn,
        seed_state(grn, "resistant", 400, rng),
        rng=rng,
        cell_bias=sample_cell_bias(grn, 400, rng),
    )
    R = grn.regulators
    corr = np.abs(np.corrcoef(observe(grn, x, rng)[:, R].T))
    same = grn.reg_module[:, None] == grn.reg_module[None, :]
    off_diag = ~np.eye(len(R), dtype=bool)
    within = corr[same & off_diag].mean()
    across = corr[~same].mean()
    assert within > 0.4, f"module members barely co-express ({within:.3f})"
    assert within > across + 0.3, f"within {within:.3f} vs across {across:.3f}"


def test_coexpression_is_informative_about_effect():
    """Co-expression distance must track true effect distance, or transfer is impossible."""
    grn = smoke_grn(0)
    rng = np.random.default_rng(0)
    x = simulate(
        grn,
        seed_state(grn, "resistant", 400, rng),
        rng=rng,
        cell_bias=sample_cell_bias(grn, 400, rng),
    )
    R = grn.regulators
    corr = np.corrcoef(observe(grn, x, rng)[:, R].T)
    effect = grn.reg_effect[:, 1] - grn.reg_effect[:, 0]
    iu = np.triu_indices(len(R), 1)
    # Similar co-expression (high corr) should mean similar effect (low distance).
    d_effect = np.abs(effect[:, None] - effect[None, :])
    assert np.corrcoef(corr[iu], d_effect[iu])[0, 1] < -0.3


def test_cell_bias_persists_through_relaxation():
    """A per-cell drive must survive relaxation; initial-state jitter would not."""
    grn = smoke_grn(0)
    rng = np.random.default_rng(0)
    x0 = seed_state(grn, "resistant", 64, rng)
    bias = sample_cell_bias(grn, 64, rng)
    plain = simulate(grn, x0)
    biased = simulate(grn, x0, cell_bias=bias)
    spread = lambda a: a[:, grn.regulators].std(axis=0).mean()
    assert spread(biased) > spread(plain) + 0.05


def test_cell_bias_is_off_when_disabled():
    grn = build_grn(GRNConfig(n_genes=60, n_regulators=12, n_program=15, module_coexpr=0.0, seed=0))
    bias = sample_cell_bias(grn, 16, np.random.default_rng(0))
    assert not bias.any()
