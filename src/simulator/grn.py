"""Gene-regulatory-network simulator with two attractors and known causal structure.

Provenance: everything produced here is tagged `synthetic`. It exists to unit-test the
pipeline against ground truth that is exactly knowable -- the true ranking of every
perturbation set of size <= k -- and is never biological evidence.

Core motif: a toggle switch between a sensitive master regulator (gene 0) and a
resistance master regulator (gene 1), each self-activating and mutually inhibiting.
Upstream regulators feed signed input into the switch; downstream genes read it out.
Bistability means a perturbation that shifts the switch's basin flips the whole
transcriptional program, which is the phenomenon the task is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

SENSITIVE = 0
RESISTANCE = 1


@dataclass
class GRNConfig:
    n_genes: int = 200
    n_regulators: int = 20
    n_program: int = 40  # readout genes per program (sensitive / resistance)
    n_reg_modules: int = 5  # pathway-like groups of regulators
    # Per-cell shared variation within a module. This trades two requirements against each
    # other: co-expression has to be strong enough that a model can transfer to a held-out
    # gene (LOG.md Failure 5), but a large per-cell drive on the regulators pushes cells
    # across the separatrix, so the phenotype label stops matching the attractor the cell
    # actually reaches. Measured label noise (fraction of cells on the wrong side of
    # s = 0.5): 0.0 at coexpr 0.0, 0.10 at 0.3, 0.26 at 0.6, 0.34 at 0.9. At 0.3,
    # within-module |r| is 0.53 against 0.04 across modules, so the pathway signal is
    # unambiguous while the sensitivity labels stay mostly correct.
    module_coexpr: float = 0.3
    w_self: float = 3.5
    w_cross: float = 3.5
    w_program: float = 3.0
    w_regulator: float = 2.5
    w_noise: float = 0.15  # sparse off-target cross-talk
    alpha: float = 0.35  # relaxation rate
    pert_strength: float = 6.0
    n_steps: int = 120
    state_noise: float = 0.02  # per-step process noise
    init_jitter: float = 0.08
    obs_noise: float = 0.05
    dropout: float = 0.0
    batch_shift: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        used = 2 + self.n_regulators + 2 * self.n_program
        if used > self.n_genes:
            raise ValueError(f"n_genes={self.n_genes} too small for layout (needs {used})")


@dataclass
class GRN:
    """A fixed network instance. `W`, `b` and the index layout are the causal truth."""

    cfg: GRNConfig
    W: np.ndarray
    b: np.ndarray
    regulators: np.ndarray  # gene indices that may be perturbed
    reg_effect: np.ndarray  # (n_regulators, 2) signed drive onto [sensitive, resistance]
    reg_module: np.ndarray  # (n_regulators,) pathway membership
    program_sensitive: np.ndarray
    program_resistance: np.ndarray
    _truth_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_genes(self) -> int:
        return self.cfg.n_genes


def build_grn(cfg: GRNConfig) -> GRN:
    rng = np.random.default_rng(cfg.seed)
    g = cfg.n_genes
    W = np.zeros((g, g))

    # toggle switch
    W[SENSITIVE, SENSITIVE] = cfg.w_self
    W[RESISTANCE, RESISTANCE] = cfg.w_self
    W[SENSITIVE, RESISTANCE] = -cfg.w_cross
    W[RESISTANCE, SENSITIVE] = -cfg.w_cross

    regulators = np.arange(2, 2 + cfg.n_regulators)
    start = 2 + cfg.n_regulators
    program_sensitive = np.arange(start, start + cfg.n_program)
    program_resistance = np.arange(start + cfg.n_program, start + 2 * cfg.n_program)

    # Regulators drive the switch antagonistically: a resistance driver activates the
    # resistance master and represses the sensitive one, so knocking it down shifts the
    # separatrix toward sensitivity. Magnitudes are set so single knockdowns usually
    # cannot clear the hysteresis barrier but the right pair or triple can -- the
    # synergy regime that makes combinatorial generalisation the interesting part.
    #
    # Regulators are grouped into pathway-like modules. Members of a module share an
    # effect type and (via sample_cell_bias) co-express. Without this, upstream regulators
    # are mutually independent, co-expression carries no information about what a gene
    # does, and generalising to a *held-out gene* is impossible for any model -- the
    # unseen-single split would measure nothing. Modules are what make that split a real
    # test: the information needed to transfer exists, and whether a model exploits it is
    # then an empirical question.
    reg_module = np.arange(cfg.n_regulators) % cfg.n_reg_modules
    module_mode = {m: int(m % 4) for m in range(cfg.n_reg_modules)}
    module_mag = {
        m: cfg.w_regulator * (rng.uniform(0.85, 1.0) if module_mode[m] == 0 else rng.uniform(0.45, 1.0))
        for m in range(cfg.n_reg_modules)
    }

    reg_effect = np.zeros((cfg.n_regulators, 2))
    for i in range(cfg.n_regulators):
        mode = module_mode[int(reg_module[i])]
        # Shared module magnitude with a little per-gene spread, so module membership is
        # informative about effect without making members interchangeable.
        mag = module_mag[int(reg_module[i])] * rng.uniform(0.85, 1.15)
        if mode == 0:  # resistance driver
            # Near full strength so that *some* set of size <= k can always clear the
            # barrier. With a wide magnitude range, unlucky seeds produced networks where
            # nothing reversed at all and that seed's experiment was vacuous; see
            # check_reversible().
            reg_effect[i] = (-0.5 * mag, mag)
        elif mode == 1:  # sensitive driver
            reg_effect[i] = (mag, -0.5 * mag)
        elif mode == 2:  # weak resistance driver, useful mainly in combination
            reg_effect[i] = (-0.25 * mag, 0.5 * mag)
        else:  # bystander regulator, negligible effect on the switch
            reg_effect[i] = (0.05 * mag, 0.05 * mag)
        W[SENSITIVE, regulators[i]] = reg_effect[i, 0]
        W[RESISTANCE, regulators[i]] = reg_effect[i, 1]

    # Programs read out the switch.
    W[np.ix_(program_sensitive, [SENSITIVE])] = cfg.w_program
    W[np.ix_(program_sensitive, [RESISTANCE])] = -0.5 * cfg.w_program
    W[np.ix_(program_resistance, [RESISTANCE])] = cfg.w_program
    W[np.ix_(program_resistance, [SENSITIVE])] = -0.5 * cfg.w_program

    # Sparse off-target cross-talk everywhere else, so the problem is not rank-2.
    mask = rng.random((g, g)) < 0.03
    np.fill_diagonal(mask, False)
    mask[:2, :] = False  # keep the switch clean
    W = W + mask * rng.normal(0.0, cfg.w_noise, size=(g, g))

    b = rng.normal(-0.4, 0.3, size=g)
    # Centre the switch: at the symmetric point (both masters at 0.5) with regulators at
    # their resting level the net drive on each master must be ~0, otherwise the regulator
    # input saturates both masters high and the network is no longer bistable.
    resting = _sigmoid(b[regulators])
    for master, col in ((SENSITIVE, 0), (RESISTANCE, 1)):
        b[master] = -0.5 * (cfg.w_self - cfg.w_cross) - float(reg_effect[:, col] @ resting)

    return GRN(cfg, W, b, regulators, reg_effect, reg_module, program_sensitive, program_resistance)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def with_context(grn: GRN, b_offset: np.ndarray) -> GRN:
    """A cell context: same regulatory wiring, shifted basal expression.

    This is the stand-in for a different cell line -- the biology (`W`) is shared, the
    basal state is not -- and it is what the unseen-context split holds out.
    """
    return GRN(
        cfg=grn.cfg,
        W=grn.W,
        b=grn.b + b_offset,
        regulators=grn.regulators,
        reg_effect=grn.reg_effect,
        reg_module=grn.reg_module,
        program_sensitive=grn.program_sensitive,
        program_resistance=grn.program_resistance,
    )


def make_context_offsets(
    grn: GRN, n_contexts: int, strength: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """Basal-expression offsets, one per context. Context 0 is the reference (no shift).

    The masters are shifted only weakly: a large offset on the switch itself would move
    contexts into different attractors outright, which is a different experiment than
    the one we mean to run.
    """
    offsets = [np.zeros(grn.n_genes)]
    for _ in range(n_contexts - 1):
        off = rng.normal(0.0, strength, size=grn.n_genes)
        off[[SENSITIVE, RESISTANCE]] *= 0.25
        offsets.append(off)
    return offsets


def pert_vector(grn: GRN, pert: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Additive input for a perturbation set. Each item is (gene_index, direction)."""
    u = np.zeros(grn.n_genes)
    for gene, direction in pert:
        u[gene] += direction * grn.cfg.pert_strength
    return u


def simulate(
    grn: GRN,
    x0: np.ndarray,
    pert: tuple[tuple[int, int], ...] = (),
    rng: np.random.Generator | None = None,
    n_steps: int | None = None,
    cell_bias: np.ndarray | None = None,
) -> np.ndarray:
    """Relax states to the attractor under `pert`. x0 is (n_cells, n_genes).

    `cell_bias` is a persistent per-cell drive (see `sample_cell_bias`). It must persist
    through relaxation rather than being initial-state jitter, which decays away.
    """
    cfg = grn.cfg
    u = pert_vector(grn, pert)
    if cell_bias is not None:
        u = u + cell_bias
    x = np.atleast_2d(x0).copy()
    steps = cfg.n_steps if n_steps is None else n_steps
    a = cfg.alpha
    for _ in range(steps):
        drive = x @ grn.W.T + grn.b + u
        x = (1 - a) * x + a * _sigmoid(drive)
        if cfg.state_noise and rng is not None:
            x = np.clip(x + rng.normal(0.0, cfg.state_noise, size=x.shape), 0.0, 1.0)
    return x


def sample_cell_bias(grn: GRN, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    """Per-cell drive making module members co-express.

    One latent factor per regulator module, shared by its members, plus independent
    per-gene variation. This is the only reason co-expression carries information about a
    gene's function here, and therefore the only reason transfer to a held-out gene is
    possible in principle.
    """
    cfg = grn.cfg
    bias = np.zeros((n_cells, grn.n_genes))
    if not cfg.module_coexpr:
        return bias
    factors = rng.normal(0.0, cfg.module_coexpr, size=(n_cells, cfg.n_reg_modules))
    for i, gene in enumerate(grn.regulators):
        shared = factors[:, int(grn.reg_module[i])]
        private = rng.normal(0.0, 0.35 * cfg.module_coexpr, size=n_cells)
        bias[:, gene] = shared + private
    return bias


def seed_state(grn: GRN, phenotype: str, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    """Initial states inside the basin of the sensitive or the resistance attractor."""
    if phenotype not in ("sensitive", "resistant"):
        raise ValueError(phenotype)
    x = rng.uniform(0.25, 0.45, size=(n_cells, grn.n_genes))
    hi, lo = (SENSITIVE, RESISTANCE) if phenotype == "sensitive" else (RESISTANCE, SENSITIVE)
    x[:, hi] = rng.uniform(0.80, 0.95, size=n_cells)
    x[:, lo] = rng.uniform(0.05, 0.20, size=n_cells)
    x += rng.normal(0.0, grn.cfg.init_jitter, size=x.shape)
    return np.clip(x, 0.0, 1.0)


def observe(grn: GRN, x: np.ndarray, rng: np.random.Generator, batch: float = 0.0) -> np.ndarray:
    """Technical layer: observation noise, optional dropout and a batch offset."""
    cfg = grn.cfg
    y = x + rng.normal(0.0, cfg.obs_noise, size=x.shape)
    if batch:
        y = y + batch * rng.normal(0.0, 1.0, size=(1, x.shape[1]))
    if cfg.dropout:
        y = y * (rng.random(x.shape) > cfg.dropout)
    return np.clip(y, 0.0, None)


# --- ground truth -----------------------------------------------------------------


def true_sensitivity(grn: GRN, x: np.ndarray) -> np.ndarray:
    """Causal sensitivity in [0,1] read straight off the switch. Simulator-only truth."""
    x = np.atleast_2d(x)
    return np.clip(0.5 * (1.0 + x[:, SENSITIVE] - x[:, RESISTANCE]), 0.0, 1.0)


def enumerate_perturbation_sets(grn: GRN, k: int = 3, directions=(-1,)) -> list[tuple]:
    """All perturbation sets of size 1..k over regulators, canonically ordered."""
    atoms = [(int(gene), int(d)) for gene in grn.regulators for d in directions]
    sets: list[tuple] = []
    for size in range(1, k + 1):
        for combo in combinations(atoms, size):
            if len({gene for gene, _ in combo}) == size:  # one direction per gene
                sets.append(tuple(sorted(combo)))
    return sets


def ground_truth_reversal(
    grn: GRN,
    k: int = 3,
    n_cells: int = 32,
    directions=(-1,),
    seed: int | None = None,
) -> dict[tuple, float]:
    """True reversal score R = E[s(z')] - E[s(z0)] for every set of size <= k.

    Noise-free relaxation from the resistant attractor, so this is the simulator's
    exact answer and the oracle every search method is scored against.
    """
    key = (k, n_cells, directions, seed)
    if key in grn._truth_cache:
        return grn._truth_cache[key]
    rng = np.random.default_rng(grn.cfg.seed if seed is None else seed)
    x0 = seed_state(grn, "resistant", n_cells, rng)
    base = float(true_sensitivity(grn, simulate(grn, x0)).mean())
    truth = {(): 0.0}
    for pert in enumerate_perturbation_sets(grn, k, directions):
        final = simulate(grn, x0, pert)
        truth[pert] = float(true_sensitivity(grn, final).mean()) - base
    grn._truth_cache[key] = truth
    return truth


def optimal_reversal_set(grn: GRN, k: int = 3, **kw) -> tuple[tuple, float]:
    truth = ground_truth_reversal(grn, k=k, **kw)
    best = max((p for p in truth if p), key=lambda p: truth[p])
    return best, truth[best]


def check_reversible(grn: GRN, k: int = 2, min_best: float = 0.6, **kw) -> float:
    """Raise if no set of size <= k reverses. Guards against vacuous network instances."""
    _, best = optimal_reversal_set(grn, k=k, **kw)
    if best < min_best:
        raise ValueError(
            f"network (seed={grn.cfg.seed}) is not reversible within k={k}: "
            f"best reversal {best:.3f} < {min_best}"
        )
    return best


def check_bistable(grn: GRN, n_cells: int = 16, margin: float = 0.7, seed: int = 0) -> tuple[float, float]:
    """Raise unless the two attractors are well separated in true sensitivity."""
    rng = np.random.default_rng(seed)
    s = {}
    for ph in ("sensitive", "resistant"):
        s[ph] = float(true_sensitivity(grn, simulate(grn, seed_state(grn, ph, n_cells, rng))).mean())
    if s["sensitive"] - s["resistant"] < margin:
        raise ValueError(
            f"network (seed={grn.cfg.seed}) is not bistable: "
            f"s_sensitive={s['sensitive']:.3f} s_resistant={s['resistant']:.3f}"
        )
    return s["sensitive"], s["resistant"]
