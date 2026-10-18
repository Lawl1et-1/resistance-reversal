"""Intervention search over perturbation sets of size <= k.

Four strategies, all maximising `objective.objective(S)`:

- `exhaustive`  -- every set of size <= k. Only tractable for small atom pools, but it gives
  the *true argmax of the model's own objective*, which separates two failure modes that
  are otherwise confounded: a bad recommendation because the search missed the optimum,
  versus because the model's objective is wrong. Every other strategy is reported as a
  regret against this.
- `greedy_forward` -- add the best atom at each step.
- `beam_search`    -- greedy with a frontier of `beam_width`.
- `evolutionary`   -- mutation/crossover over sets, for when the pool is large.

All return a ranked list of records, so top-k hit rate and rank correlation against ground
truth can be computed from the same output.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class SearchResult:
    ranked: list[dict]
    n_evaluations: int
    strategy: str

    @property
    def best(self) -> dict | None:
        return self.ranked[0] if self.ranked else None

    @property
    def best_pert(self) -> tuple:
        return self.ranked[0]["pert"] if self.ranked else ()

    def top(self, n: int) -> list[tuple]:
        return [r["pert"] for r in self.ranked[:n]]


def _rank(objective, perts: list[tuple], strategy: str, start_evals: int) -> SearchResult:
    seen, records = set(), []
    for pert in perts:
        key = tuple(sorted(pert))
        if key in seen:
            continue
        seen.add(key)
        records.append(objective.score(key))
    # Abstained sets sort last whatever their predicted reversal.
    records.sort(key=lambda r: (-np.inf if r["abstain"] else r["penalised"]), reverse=True)
    return SearchResult(records, objective.n_evaluations - start_evals, strategy)


def exhaustive(objective, atoms: list[tuple], k: int = 3) -> SearchResult:
    start = objective.n_evaluations
    perts = []
    for size in range(1, k + 1):
        for combo in combinations(atoms, size):
            if len({g for g, _ in combo}) == size:  # one direction per gene
                perts.append(tuple(sorted(combo)))
    return _rank(objective, perts, "exhaustive", start)


def greedy_forward(objective, atoms: list[tuple], k: int = 3) -> SearchResult:
    """Add the single best atom at a time, stopping early if nothing improves."""
    start = objective.n_evaluations
    current: tuple = ()
    best_value = -np.inf
    visited: list[tuple] = []
    for _ in range(k):
        used = {g for g, _ in current}
        candidates = [
            tuple(sorted(current + (a,))) for a in atoms if a[0] not in used
        ]
        if not candidates:
            break
        values = [objective.objective(c) for c in candidates]
        visited.extend(candidates)
        i = int(np.argmax(values))
        if values[i] <= best_value:
            break  # no atom improves the set; adding more only costs perturbations
        best_value, current = values[i], candidates[i]
    return _rank(objective, visited or [()], "greedy", start)


def beam_search(objective, atoms: list[tuple], k: int = 3, beam_width: int = 5) -> SearchResult:
    start = objective.n_evaluations
    beam: list[tuple] = [()]
    visited: list[tuple] = []
    for _ in range(k):
        candidates = set()
        for pert in beam:
            used = {g for g, _ in pert}
            for a in atoms:
                if a[0] not in used:
                    candidates.add(tuple(sorted(pert + (a,))))
        if not candidates:
            break
        scored = sorted(candidates, key=objective.objective, reverse=True)
        visited.extend(scored)
        beam = scored[:beam_width]
    return _rank(objective, visited or [()], f"beam{beam_width}", start)


def evolutionary(
    objective,
    atoms: list[tuple],
    k: int = 3,
    population: int = 24,
    generations: int = 12,
    elite: int = 6,
    seed: int = 0,
) -> SearchResult:
    """Mutation and crossover over perturbation sets, with elitism."""
    start = objective.n_evaluations
    rng = np.random.default_rng(seed)
    atom_list = list(atoms)

    def random_set() -> tuple:
        size = int(rng.integers(1, k + 1))
        picked: list[tuple] = []
        used: set[int] = set()
        for idx in rng.permutation(len(atom_list)):
            a = atom_list[idx]
            if a[0] not in used:
                picked.append(a)
                used.add(a[0])
            if len(picked) == size:
                break
        return tuple(sorted(picked))

    def mutate(pert: tuple) -> tuple:
        genes = {g for g, _ in pert}
        options = [a for a in atom_list if a[0] not in genes]
        choice = rng.random()
        if choice < 0.4 and len(pert) > 1:  # drop
            keep = list(pert)
            keep.pop(int(rng.integers(len(keep))))
            return tuple(sorted(keep))
        if choice < 0.8 and len(pert) < k and options:  # add
            return tuple(sorted(pert + (options[int(rng.integers(len(options)))],)))
        if options and pert:  # swap
            keep = list(pert)
            keep[int(rng.integers(len(keep)))] = options[int(rng.integers(len(options)))]
            return tuple(sorted(keep))
        return pert

    def crossover(a: tuple, b: tuple) -> tuple:
        pool, used, out = list(a) + list(b), set(), []
        for atom in sorted(pool, key=lambda _: rng.random()):
            if atom[0] not in used:
                out.append(atom)
                used.add(atom[0])
            if len(out) == k:
                break
        return tuple(sorted(out)) if out else a

    pop = {random_set() for _ in range(population)}
    visited: set[tuple] = set(pop)
    for _ in range(generations):
        ranked = sorted(pop, key=objective.objective, reverse=True)[:elite]
        children = set(ranked)
        while len(children) < population:
            if rng.random() < 0.5 and len(ranked) >= 2:
                i, j = rng.choice(len(ranked), 2, replace=False)
                children.add(crossover(ranked[i], ranked[j]))
            else:
                children.add(mutate(ranked[int(rng.integers(len(ranked)))]))
        pop = children
        visited |= pop
    return _rank(objective, sorted(visited), "evolutionary", start)


STRATEGIES = {
    "exhaustive": exhaustive,
    "greedy": greedy_forward,
    "beam": beam_search,
    "evolutionary": evolutionary,
}


def search_regret(result: SearchResult, reference: SearchResult, truth_fn) -> dict:
    """How much true reversal a strategy gives up against the model's own optimum.

    Separates search error from model error: `optimisation_regret` is what a better search
    would recover, `model_regret` is what no search could recover because the objective
    itself is wrong.
    """
    best_true = truth_fn(result.best_pert)
    ref_true = truth_fn(reference.best_pert)
    return {
        "strategy": result.strategy,
        "n_evaluations": result.n_evaluations,
        "selected": result.best_pert,
        "true_reversal": best_true,
        "reference_selected": reference.best_pert,
        "reference_true_reversal": ref_true,
        "optimisation_regret": float(ref_true - best_true),
    }
