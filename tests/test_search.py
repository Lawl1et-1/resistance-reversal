"""Search and objective tests.

Deliberately uses stub models with analytically known optima, so a failure here means the
search or the objective is wrong rather than that PertFlow fit badly. Model quality is
tested in test_pertflow.py; this file tests the optimiser.
"""

import numpy as np
import pytest

from src.models.latent import LatentEncoder
from src.search.objective import ReversalObjective
from src.search.search import (
    STRATEGIES,
    beam_search,
    evolutionary,
    exhaustive,
    greedy_forward,
    search_regret,
)

LATENT_DIM = 4


class StubModel:
    """Each gene contributes a fixed latent shift and a fixed uncertainty, additively."""

    def __init__(self, effects: dict[int, float], uncertainties: dict[int, float] | None = None):
        self.effects = effects
        self.uncertainties = uncertainties or {}

    def predict_latent_delta(self, pert, context):
        d = np.zeros(LATENT_DIM)
        for gene, direction in pert:
            d[0] += direction * self.effects.get(gene, 0.0)
        return d

    def epistemic_uncertainty(self, pert, context):
        return float(sum(self.uncertainties.get(g, 0.0) for g, _ in pert))


def stub_sensitivity(Z):
    """Monotone in the first latent coordinate, squashed into [0, 1]."""
    return 1.0 / (1.0 + np.exp(-np.atleast_2d(Z)[:, 0]))


def make_objective(effects, uncertainties=None, lam=0.0, abstain_above=None):
    encoder = LatentEncoder(latent_dim=LATENT_DIM).fit(
        np.random.default_rng(0).normal(size=(40, LATENT_DIM))
    )
    return ReversalObjective(
        model=StubModel(effects, uncertainties),
        sensitivity=stub_sensitivity,
        encoder=encoder,
        z0=np.zeros((16, LATENT_DIM)),
        context=0,
        lam=lam,
        abstain_above=abstain_above,
    )


ATOMS = [(g, -1) for g in range(6)]
# Knocking down gene g with a negative effect raises the first coordinate, so more
# negative effects mean more reversal. Gene 0 is the strongest single driver.
EFFECTS = {0: -3.0, 1: -2.0, 2: -1.0, 3: 0.0, 4: 1.0, 5: 2.0}


def test_baseline_sensitivity_is_the_control_mean():
    obj = make_objective(EFFECTS)
    assert obj.baseline_s_ == pytest.approx(0.5)  # sigmoid(0)
    assert obj.score(())["reversal"] == pytest.approx(0.0)


def test_reversal_increases_with_a_stronger_driver():
    obj = make_objective(EFFECTS)
    r = [obj.score((((g), -1),))["reversal"] for g in (0, 1, 2, 3)]
    assert r[0] > r[1] > r[2] > r[3]
    assert r[3] == pytest.approx(0.0)  # gene 3 has no effect


def test_exhaustive_enumerates_every_valid_set():
    obj = make_objective(EFFECTS)
    res = exhaustive(obj, ATOMS, k=3)
    n = len(ATOMS)
    assert len(res.ranked) == n + n * (n - 1) // 2 + n * (n - 1) * (n - 2) // 6
    for rec in res.ranked:
        genes = [g for g, _ in rec["pert"]]
        assert len(genes) == len(set(genes)), "a gene must not appear twice in a set"
        assert len(genes) <= 3


def test_exhaustive_finds_the_known_optimum():
    """With additive effects the best triple is the three most negative genes."""
    obj = make_objective(EFFECTS)
    assert exhaustive(obj, ATOMS, k=3).best_pert == ((0, -1), (1, -1), (2, -1))


def test_greedy_matches_exhaustive_on_an_additive_objective():
    obj = make_objective(EFFECTS)
    assert greedy_forward(obj, ATOMS, k=3).best_pert == exhaustive(
        make_objective(EFFECTS), ATOMS, k=3
    ).best_pert


def test_greedy_stops_early_when_nothing_improves():
    """Adding a harmful gene must not happen just to fill the budget."""
    effects = {0: -3.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
    best = greedy_forward(make_objective(effects), ATOMS, k=3).best_pert
    assert best == ((0, -1),), f"greedy padded the set: {best}"


def test_greedy_respects_the_budget():
    obj = make_objective({g: -float(6 - g) for g in range(6)})
    for k in (1, 2, 3):
        assert len(greedy_forward(obj, ATOMS, k=k).best_pert) <= k


@pytest.mark.parametrize("width", [1, 3, 5])
def test_beam_search_finds_the_optimum(width):
    obj = make_objective(EFFECTS)
    assert beam_search(obj, ATOMS, k=3, beam_width=width).best_pert == (
        (0, -1),
        (1, -1),
        (2, -1),
    )


def test_evolutionary_finds_the_optimum():
    obj = make_objective(EFFECTS)
    res = evolutionary(obj, ATOMS, k=3, population=16, generations=10, seed=0)
    assert res.best_pert == ((0, -1), (1, -1), (2, -1))


def test_all_strategies_produce_valid_sets():
    for name, fn in STRATEGIES.items():
        res = fn(make_objective(EFFECTS), ATOMS, k=3)
        assert res.ranked, name
        genes = [g for g, _ in res.best_pert]
        assert len(genes) == len(set(genes)), name
        assert len(genes) <= 3, name


def test_uncertainty_penalty_changes_the_recommendation():
    """H3's mechanism: a slightly weaker but confident set should overtake a risky one."""
    effects = {0: -3.0, 1: -2.5, 2: -2.5, 3: 0.0, 4: 0.0, 5: 0.0}
    uncertainties = {0: 10.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    blind = exhaustive(make_objective(effects, uncertainties, lam=0.0), ATOMS, k=2)
    penalised = exhaustive(make_objective(effects, uncertainties, lam=0.1), ATOMS, k=2)
    assert 0 in {g for g, _ in blind.best_pert}, "uncertainty-blind search should take gene 0"
    assert 0 not in {g for g, _ in penalised.best_pert}, "penalty did not deter gene 0"


def test_abstained_sets_are_never_selected():
    effects = {0: -5.0, 1: -1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    uncertainties = {0: 9.0}
    obj = make_objective(effects, uncertainties, abstain_above=1.0)
    res = exhaustive(obj, ATOMS, k=1)
    assert res.best_pert == ((1, -1),), "an abstained set was selected"
    assert obj.score(((0, -1),))["abstain"] is True
    assert obj.objective(((0, -1),)) == -np.inf
    # Abstained sets still appear in the ranking, but at the bottom.
    assert res.ranked[-1]["abstain"] is True


def test_scores_are_cached_so_evaluations_are_not_repeated():
    obj = make_objective(EFFECTS)
    obj.score(((0, -1),))
    obj.score(((0, -1),))
    obj.score(((0, -1),))
    assert obj.n_evaluations == 1
    greedy_forward(obj, ATOMS, k=3)
    assert obj.n_evaluations == len(obj.cache_)


def test_shared_cache_does_not_leak_a_stale_penalty():
    """A lambda view reuses the cache but must recombine the penalty at its own lambda."""
    uncertainties = {g: 1.0 for g in range(6)}
    base = make_objective(EFFECTS, uncertainties, lam=0.0)
    pert = ((0, -1),)
    first = base.score(pert)
    view = base.with_penalty(lam=0.5)
    second = view.score(pert)

    assert view.cache_ is base.cache_, "the cache must be shared"
    assert base.n_evaluations == 1, "the model must not be re-evaluated for a new lambda"
    # Same underlying quantities, different penalty.
    assert second["reversal"] == pytest.approx(first["reversal"])
    assert second["uncertainty"] == pytest.approx(first["uncertainty"])
    assert second["penalised"] == pytest.approx(first["reversal"] - 0.5 * first["uncertainty"])
    assert second["penalised"] != pytest.approx(first["penalised"])
    # The original must be unaffected by the view.
    assert base.score(pert)["penalised"] == pytest.approx(first["penalised"])


def test_shared_cache_applies_each_views_abstention_rule():
    base = make_objective(EFFECTS, {0: 9.0}, lam=0.0)
    pert = ((0, -1),)
    assert base.score(pert)["abstain"] is False
    strict = base.with_penalty(lam=0.0, abstain_above=1.0)
    assert strict.score(pert)["abstain"] is True
    assert base.score(pert)["abstain"] is False, "the view must not mutate the original"


def test_lambda_sweep_costs_one_evaluation_per_set():
    base = make_objective(EFFECTS, {g: 1.0 for g in range(6)}, lam=0.0)
    for lam in (0.0, 0.5, 1.0, 2.0):
        view = base if lam == 0.0 else base.with_penalty(lam)
        exhaustive(view, ATOMS, k=2)
    assert base.n_evaluations == len(base.cache_)


def test_point_predictor_preserves_population_spread():
    """A deterministic model must shift the population, not collapse it to a point."""
    obj = make_objective(EFFECTS)
    rng = np.random.default_rng(0)
    obj.z0 = rng.normal(size=(64, LATENT_DIM))
    obj.__post_init__()
    pop = obj.predict_population(((0, -1),))
    assert pop.shape == obj.z0.shape
    assert pop.std(axis=0).mean() == pytest.approx(obj.z0.std(axis=0).mean(), rel=1e-9)


def test_model_without_uncertainty_scores_as_zero():
    class Bare:
        def predict_latent_delta(self, pert, context):
            return np.zeros(LATENT_DIM)

    encoder = LatentEncoder(latent_dim=LATENT_DIM).fit(
        np.random.default_rng(0).normal(size=(40, LATENT_DIM))
    )
    obj = ReversalObjective(
        model=Bare(),
        sensitivity=stub_sensitivity,
        encoder=encoder,
        z0=np.zeros((8, LATENT_DIM)),
        context=0,
        lam=1.0,
    )
    assert obj.uncertainty(((0, -1),)) == 0.0
    assert obj.score(((0, -1),))["penalised"] == pytest.approx(0.0)


def test_search_regret_separates_search_error_from_model_error():
    truth = {((0, -1),): 0.9, ((1, -1),): 0.4}
    truth_fn = lambda p: truth.get(tuple(sorted(p)), 0.0)

    class FakeResult:
        def __init__(self, pert, strategy):
            self.ranked = [{"pert": pert}]
            self.n_evaluations = 1
            self.strategy = strategy

        best_pert = property(lambda self: self.ranked[0]["pert"])

    rep = search_regret(FakeResult(((1, -1),), "greedy"), FakeResult(((0, -1),), "exhaustive"), truth_fn)
    assert rep["true_reversal"] == pytest.approx(0.4)
    assert rep["reference_true_reversal"] == pytest.approx(0.9)
    assert rep["optimisation_regret"] == pytest.approx(0.5)


def test_energy_to_sensitive_is_reported_when_a_reference_is_given():
    rng = np.random.default_rng(0)
    obj = make_objective(EFFECTS)
    obj.sensitive_reference = rng.normal(loc=3.0, size=(64, LATENT_DIM))
    rec = obj.score(((0, -1),))
    assert "energy_to_sensitive" in rec
    assert rec["energy_to_sensitive"] >= 0.0
