"""Shared setup for every experiment: one place where a config becomes fitted objects.

Having a single `build()` matters for leakage discipline. The order is fixed and the
dependencies are explicit: splits are defined before anything is fitted, the encoder sees
only training cells, the sensitivity functional is fitted on a separate simulation seed, and
nothing downstream can reach the held-out data except through the evaluation helpers.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from src.data.dataset import (
    make_response_data,
    make_sensitivity_data,
    sample_perturbations,
)
from src.data.splits import make_all_splits, save_splits
from src.models.latent import fit_encoder
from src.models.sensitivity import fit_sensitivity
from src.simulator.grn import (
    GRNConfig,
    build_grn,
    check_bistable,
    check_reversible,
    observe,
    sample_cell_bias,
    seed_state,
    simulate,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_path"] = str(path)
    return cfg


def environment() -> dict:
    """Recorded with every result so a number can be traced to the run that made it."""
    try:
        import torch

        torch_version, cuda = torch.__version__, torch.cuda.is_available()
    except Exception:  # torch is optional for the baseline-only experiments
        torch_version, cuda = None, False
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        commit = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch_version,
        "cuda_available": cuda,
        "git_commit": commit,
    }


@dataclass
class Setup:
    """Everything an experiment needs for one (config, seed), already fitted."""

    cfg: dict
    seed: int
    grn: object
    perts: list
    data: object
    splits: dict
    encoders: dict = field(default_factory=dict)
    sensitivity: dict = field(default_factory=dict)
    sensitive_reference: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    @property
    def provenance(self) -> str:
        return self.cfg.get("provenance", "synthetic")

    def atoms(self) -> list[tuple]:
        return [(int(g), -1) for g in self.grn.regulators]


def build(cfg: dict, seed: int, save_split_dir: Path | None = None) -> Setup:
    sim, dat = cfg["simulator"], cfg["data"]
    grn = build_grn(GRNConfig(seed=seed, **sim))

    # Fail loudly on a degenerate network instead of producing a vacuous experiment.
    s_sens, s_res = check_bistable(grn)
    best_reversal = check_reversible(grn, k=2, n_cells=dat.get("oracle_cells", 16))
    checks = {
        "s_sensitive": s_sens,
        "s_resistant": s_res,
        "best_reversal_k2": best_reversal,
    }

    perts = sample_perturbations(grn, k=dat["k"], n_pairs=dat.get("n_pairs"), seed=seed)
    data = make_response_data(
        grn,
        perts,
        n_cells=dat["n_cells"],
        n_control_cells=dat["n_control_cells"],
        n_contexts=dat["n_contexts"],
        context_strength=dat["context_strength"],
        seed=seed,
    )
    splits = make_all_splits(perts, n_contexts=dat["n_contexts"], seed=seed)
    if save_split_dir is not None:
        save_splits(splits, Path(save_split_dir) / f"splits_seed{seed}.json")

    setup = Setup(cfg=cfg, seed=seed, grn=grn, perts=perts, data=data, splits=splits, checks=checks)

    # Sensitivity data uses a disjoint simulation seed so it shares no cells with `data`.
    Xs, ys, _ = make_sensitivity_data(
        grn,
        n_cells=dat["sensitivity_cells"],
        n_contexts=dat["n_contexts"],
        context_strength=dat["context_strength"],
        seed=10_000 + seed,
    )
    enc_cfg = cfg["encoder"]
    for name, split in splits.items():
        encoder = fit_encoder(
            data, split, latent_dim=enc_cfg["latent_dim"], n_hvg=enc_cfg.get("n_hvg"), seed=seed
        )
        setup.encoders[name] = encoder
        setup.sensitivity[name] = fit_sensitivity(Xs, ys, encoder)
        setup.sensitive_reference[name] = encoder.transform(Xs[ys == 1])
    return setup


def resistant_control_latents(setup: Setup, split_name: str, n_cells: int = 256) -> np.ndarray:
    """A fresh resistant control population to search from.

    Drawn with its own rng so the search starting point is not the same cells the model was
    fitted on, which would flatter the reversal estimate.
    """
    grn, encoder = setup.grn, setup.encoders[split_name]
    rng = np.random.default_rng(50_000 + setup.seed)
    state = simulate(
        grn,
        seed_state(grn, "resistant", n_cells, rng),
        rng=rng,
        cell_bias=sample_cell_bias(grn, n_cells, rng),
    )
    return encoder.transform(observe(grn, state, rng))


def write_json(obj, path: str | Path, cfg: dict | None = None) -> Path:
    """Every results file carries its provenance tag and environment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": (cfg or {}).get("provenance", "synthetic"),
        "profile": (cfg or {}).get("profile"),
        "config_path": (cfg or {}).get("_path"),
        "environment": environment(),
        "result": obj,
    }
    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
    return path


def write_table(df, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def pert_label(pert) -> str:
    """Readable, stable name for a perturbation set, for tables and JSON keys."""
    if not pert:
        return "control"
    return "+".join(f"{'KD' if d < 0 else 'OE'}g{g}" for g, d in sorted(pert))
