"""The reversal objective the search maximises.

`R(z0, S) = E[s(z')] - E[s(z0)]`, and the penalised objective
`J(S) = E[R(z0, S)] - lambda * U(S)` with `U` the model's epistemic uncertainty.

The point of the penalty is the optimiser's curse: a search over many candidate sets will
preferentially surface sets whose reversal the model *overestimates*, because an optimistic
error and a genuinely good intervention look identical from the inside. Penalising
uncertainty biases the search toward sets the model is confident about, trading some
predicted reversal for predictions more likely to survive verification. H3 is the claim that
this trade is worth making.

Scoring is cached per perturbation set. Greedy, beam and evolutionary search all revisit the
same sets repeatedly, and one PertFlow evaluation integrates an ODE for every ensemble
member over a population of cells, so the cache is what makes the comparison affordable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.eval.metrics import energy_distance
from src.models.latent import LatentEncoder


@dataclass
class ReversalObjective:
    """Scores a perturbation set by its predicted shift toward the sensitive state.

    `sensitivity` maps latent states to [0, 1]. `z0` is the control population in latent
    space -- a population, not a point, because `s` is averaged over cells and a single
    mean state would hide how much of the population actually crosses over.
    """

    model: object
    sensitivity: object
    encoder: LatentEncoder
    z0: np.ndarray
    context: int
    lam: float = 0.0
    n_samples: int = 128
    sensitive_reference: np.ndarray | None = None
    abstain_above: float | None = None
    cache_: dict = field(default_factory=dict, repr=False)
    n_evaluations: int = 0

    def __post_init__(self) -> None:
        self.z0 = np.atleast_2d(self.z0)
        self.baseline_s_ = float(np.mean(self.sensitivity(self.z0)))

    def predict_population(self, pert: tuple) -> np.ndarray:
        """Predicted post-perturbation latent population."""
        if hasattr(self.model, "sample_latent"):
            return self.model.sample_latent(pert, self.context, self.z0, self.n_samples)
        # Point predictors shift every control cell by the same predicted delta, which
        # preserves the control population's spread instead of collapsing it to a point.
        delta = np.asarray(self.model.predict_latent_delta(pert, self.context), float).ravel()
        return self.z0 + delta

    def uncertainty(self, pert: tuple) -> float:
        fn = getattr(self.model, "epistemic_uncertainty", None)
        return float(fn(pert, self.context)) if fn is not None else 0.0

    def score(self, pert: tuple) -> dict:
        key = tuple(sorted(pert))
        if key in self.cache_:
            return self.cache_[key]
        self.n_evaluations += 1

        z_pred = self.predict_population(pert)
        s_pred = float(np.mean(self.sensitivity(z_pred)))
        u = self.uncertainty(pert)
        out = {
            "pert": key,
            "size": len(key),
            "s_pred": s_pred,
            "s_baseline": self.baseline_s_,
            "reversal": s_pred - self.baseline_s_,
            "uncertainty": u,
            "penalised": (s_pred - self.baseline_s_) - self.lam * u,
            "abstain": bool(self.abstain_above is not None and u > self.abstain_above),
        }
        if self.sensitive_reference is not None:
            out["energy_to_sensitive"] = energy_distance(z_pred, self.sensitive_reference)
        self.cache_[key] = out
        return out

    def objective(self, pert: tuple) -> float:
        """What the search maximises. An abstained set is never selectable."""
        rec = self.score(pert)
        return -np.inf if rec["abstain"] else rec["penalised"]

    def score_many(self, perts: list[tuple]) -> list[dict]:
        return [self.score(p) for p in perts]


def true_reversal_scorer(grn, k: int = 3, n_cells: int = 32, seed: int | None = None):
    """Ground-truth reversal for every set of size <= k, from the simulator itself.

    This is the oracle that search results are verified against. It is the simulator's exact
    answer and is available only for synthetic data.
    """
    from src.simulator.grn import ground_truth_reversal

    truth = ground_truth_reversal(grn, k=k, n_cells=n_cells, seed=seed)
    return lambda pert: truth.get(tuple(sorted(pert)), float("nan")), truth
