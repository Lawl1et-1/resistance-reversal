"""Perturbation-response datasets drawn from the synthetic simulator.

Mirrors how real single-cell perturbation data actually arrives: unpaired populations of
control cells and perturbed cells per (context, perturbation), never matched pre/post
pairs for the same cell. Everything the models see goes through `observe()`, so they get
noisy, optionally dropped-out expression rather than the latent simulator state.

Two datasets are generated, and they are deliberately disjoint:

- `make_response_data`   -- perturbed *resistant* cells. Trains the response model `f`.
- `make_sensitivity_data` -- unperturbed sensitive vs resistant cells, different simulation
  seed and contexts. Trains the sensitivity functional `s`.

Keeping them disjoint is what stops `s` from being a readout of `f`'s own training signal.
The price is a domain shift between the two, which is exactly the caveat the paper has to
state rather than hide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.simulator.grn import (
    GRN,
    enumerate_perturbation_sets,
    make_context_offsets,
    observe,
    seed_state,
    simulate,
    true_sensitivity,
    with_context,
)

CONTROL: tuple = ()


@dataclass
class PerturbData:
    """Observed expression with (context, perturbation) labels.

    `X` is observed expression. `latent_state` is the noise-free simulator state, kept for
    diagnostics and for the oracle only -- no model may read it.
    """

    X: np.ndarray
    pert_ids: np.ndarray
    context_ids: np.ndarray
    perts: list[tuple]
    n_contexts: int
    latent_state: np.ndarray | None = None
    true_s: np.ndarray | None = None
    provenance: str = "synthetic"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.X)
        if not (len(self.pert_ids) == len(self.context_ids) == n):
            raise ValueError("label arrays must match X")
        self.pert_index = {p: i for i, p in enumerate(self.perts)}

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]

    def mask(self, pert: tuple | None = None, context: int | None = None) -> np.ndarray:
        m = np.ones(len(self.X), dtype=bool)
        if pert is not None:
            m &= self.pert_ids == self.pert_index[pert]
        if context is not None:
            m &= self.context_ids == context
        return m

    def cells(self, pert: tuple | None = None, context: int | None = None) -> np.ndarray:
        return self.X[self.mask(pert, context)]

    def control_mean(self, context: int) -> np.ndarray:
        cells = self.cells(CONTROL, context)
        if not len(cells):
            raise KeyError(f"no control cells for context {context}")
        return cells.mean(axis=0)

    def delta(self, pert: tuple, context: int) -> np.ndarray:
        """Mean shift of a perturbation against its own context's control."""
        return self.cells(pert, context).mean(axis=0) - self.control_mean(context)

    def observed_perts(self, context: int | None = None) -> list[tuple]:
        ids = np.unique(self.pert_ids if context is None else self.pert_ids[self.context_ids == context])
        return [self.perts[i] for i in ids]

    def subset(self, mask: np.ndarray) -> "PerturbData":
        keep = np.flatnonzero(mask)
        return PerturbData(
            X=self.X[keep],
            pert_ids=self.pert_ids[keep],
            context_ids=self.context_ids[keep],
            perts=self.perts,
            n_contexts=self.n_contexts,
            latent_state=None if self.latent_state is None else self.latent_state[keep],
            true_s=None if self.true_s is None else self.true_s[keep],
            provenance=self.provenance,
            meta=dict(self.meta),
        )


def sample_perturbations(
    grn: GRN,
    k: int = 2,
    n_singles: int | None = None,
    n_pairs: int | None = None,
    n_triples: int | None = None,
    seed: int = 0,
) -> list[tuple]:
    """A perturbation library: all singles plus a random sample of higher-order sets.

    Exhaustive pairs and triples would be most of the simulator's state space; real
    screens are sparse in combination space, and the sampled library keeps that shape.
    """
    rng = np.random.default_rng(seed)
    by_size: dict[int, list[tuple]] = {1: [], 2: [], 3: []}
    for p in enumerate_perturbation_sets(grn, k=k):
        by_size[len(p)].append(p)

    chosen = list(by_size[1]) if n_singles is None else _pick(by_size[1], n_singles, rng)
    for size, n in ((2, n_pairs), (3, n_triples)):
        if size > k:
            continue
        pool = by_size[size]
        chosen += pool if n is None else _pick(pool, n, rng)
    return chosen


def _pick(pool: list[tuple], n: int, rng: np.random.Generator) -> list[tuple]:
    if n >= len(pool):
        return list(pool)
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(idx)]


def make_response_data(
    grn: GRN,
    perts: list[tuple],
    n_cells: int = 64,
    n_control_cells: int = 256,
    n_contexts: int = 3,
    context_strength: float = 0.25,
    phenotype: str = "resistant",
    batch_shift: float = 0.0,
    seed: int = 0,
) -> PerturbData:
    """Perturbed cells across contexts, plus controls. Trains and evaluates `f`."""
    rng = np.random.default_rng(seed)
    offsets = make_context_offsets(grn, n_contexts, context_strength, rng)
    all_perts = [CONTROL] + [p for p in perts if p != CONTROL]
    pert_index = {p: i for i, p in enumerate(all_perts)}

    X, pids, cids, latent = [], [], [], []
    for ctx_id, offset in enumerate(offsets):
        ctx = with_context(grn, offset)
        # One batch offset per context, so batch and context are confounded exactly as
        # they usually are in practice.
        batch = batch_shift
        for pert in all_perts:
            n = n_control_cells if pert == CONTROL else n_cells
            x0 = seed_state(ctx, phenotype, n, rng)
            state = simulate(ctx, x0, pert, rng=rng)
            X.append(observe(ctx, state, rng, batch=batch))
            latent.append(state)
            pids.append(np.full(n, pert_index[pert]))
            cids.append(np.full(n, ctx_id))

    X = np.concatenate(X)
    latent = np.concatenate(latent)
    return PerturbData(
        X=X,
        pert_ids=np.concatenate(pids),
        context_ids=np.concatenate(cids),
        perts=all_perts,
        n_contexts=n_contexts,
        latent_state=latent,
        true_s=true_sensitivity(grn, latent),
        meta={
            "kind": "response",
            "phenotype": phenotype,
            "seed": seed,
            "context_strength": context_strength,
            "batch_shift": batch_shift,
            "n_cells": n_cells,
            "n_control_cells": n_control_cells,
        },
    )


def make_sensitivity_data(
    grn: GRN,
    n_cells: int = 512,
    n_contexts: int = 3,
    context_strength: float = 0.25,
    batch_shift: float = 0.0,
    seed: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unperturbed sensitive vs resistant cells for training `s`.

    Separate simulation seed and no perturbations at all, so this shares no cells with the
    response data. Returns (X, label, context_id) with label 1 = sensitive.
    """
    rng = np.random.default_rng(seed)
    offsets = make_context_offsets(grn, n_contexts, context_strength, rng)
    X, y, cids = [], [], []
    for ctx_id, offset in enumerate(offsets):
        ctx = with_context(grn, offset)
        for label, phenotype in ((1, "sensitive"), (0, "resistant")):
            state = simulate(ctx, seed_state(ctx, phenotype, n_cells, rng), rng=rng)
            X.append(observe(ctx, state, rng, batch=batch_shift))
            y.append(np.full(n_cells, label))
            cids.append(np.full(n_cells, ctx_id))
    return np.concatenate(X), np.concatenate(y), np.concatenate(cids)
