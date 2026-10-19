"""Evaluation harness: score a response model on a split, one record per test point.

Scoring happens per (perturbation, context) so that bootstrap CIs are taken over
perturbations -- the unit of generalisation -- rather than over cells, which would give
absurdly tight intervals by counting correlated cells as independent evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset import CONTROL, PerturbData
from src.data.splits import Split
from src.eval.metrics import (
    bootstrap_ci,
    hierarchical_bootstrap_ci,
    energy_distance,
    mse,
    mse_top_degs,
    pearson_delta,
)
from src.models.latent import LatentEncoder

POINT_METRICS = ("pearson_delta", "mse_top20", "latent_mse", "gene_mse")


def observed_delta(
    data: PerturbData, encoder: LatentEncoder, pert: tuple, context: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """(latent delta, gene-space delta over the encoder's gene subset), or None if absent."""
    cells = data.cells(pert, context)
    control = data.cells(CONTROL, context)
    if not len(cells) or not len(control):
        return None
    dz = encoder.transform(cells).mean(axis=0) - encoder.transform(control).mean(axis=0)
    dx = encoder.genes(cells).mean(axis=0) - encoder.genes(control).mean(axis=0)
    return dz, dx


def evaluate_point(
    model,
    data: PerturbData,
    split: Split,
    encoder: LatentEncoder,
    perts: list[tuple] | None = None,
    contexts: list[int] | None = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """Per-test-point point-prediction metrics."""
    perts = split.test if perts is None else perts
    contexts = split.test_contexts if contexts is None else contexts
    rows = []
    for pert in perts:
        if pert not in data.pert_index:
            continue
        for ctx in contexts:
            truth = observed_delta(data, encoder, pert, ctx)
            if truth is None:
                continue
            true_dz, true_dx = truth
            pred_dz = np.asarray(model.predict_latent_delta(pert, ctx), float).ravel()
            pred_dx = encoder.decode_delta(pred_dz).ravel()
            rows.append(
                {
                    "model": getattr(model, "name", type(model).__name__),
                    "split": split.name,
                    "pert": pert,
                    "size": len(pert),
                    "context": ctx,
                    "pearson_delta": pearson_delta(pred_dx, true_dx),
                    "mse_top20": mse_top_degs(pred_dx, true_dx, k=top_k),
                    "gene_mse": mse(pred_dx, true_dx),
                    "latent_mse": mse(pred_dz, true_dz),
                }
            )
    return pd.DataFrame(rows)


def evaluate_distribution(
    model,
    data: PerturbData,
    split: Split,
    encoder: LatentEncoder,
    n_samples: int = 128,
    perts: list[tuple] | None = None,
    contexts: list[int] | None = None,
) -> pd.DataFrame:
    """Energy distance between the predicted and observed post-perturbation populations.

    Only for models that expose `sample_latent`; point predictors have no distribution to
    score and are skipped rather than scored on a degenerate point mass.
    """
    if not hasattr(model, "sample_latent"):
        return pd.DataFrame()
    perts = split.test if perts is None else perts
    contexts = split.test_contexts if contexts is None else contexts
    rows = []
    for pert in perts:
        if pert not in data.pert_index:
            continue
        for ctx in contexts:
            cells = data.cells(pert, ctx)
            control = data.cells(CONTROL, ctx)
            if not len(cells) or not len(control):
                continue
            z_true = encoder.transform(cells)
            z0 = encoder.transform(control)
            z_pred = model.sample_latent(pert, ctx, z0, n_samples)
            rows.append(
                {
                    "model": getattr(model, "name", type(model).__name__),
                    "split": split.name,
                    "pert": pert,
                    "size": len(pert),
                    "context": ctx,
                    "energy_distance": energy_distance(z_pred, z_true),
                    "energy_distance_control": energy_distance(z0, z_true),
                }
            )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, metrics=POINT_METRICS, by=("model", "split"), seed: int = 0) -> pd.DataFrame:
    """Point estimate and bootstrap 95% CI per model, over perturbations."""
    if df.empty:
        return pd.DataFrame()
    present = [m for m in metrics if m in df.columns]
    rows = []
    for keys, g in df.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        row["n"] = len(g)
        # Rows are nested inside seeds. Pooling them treats every perturbation as an
        # independent replicate and gives intervals far tighter than the evidence supports;
        # the hierarchical bootstrap resamples seeds and then perturbations within them, so
        # both variance components reach the interval.
        nested = "seed" in g.columns and g["seed"].nunique() > 1
        row["n_seeds"] = int(g["seed"].nunique()) if "seed" in g.columns else 1
        row["ci_method"] = "hierarchical" if nested else "pooled"
        for m in present:
            point, lo, hi = (
                hierarchical_bootstrap_ci(g[m].to_numpy(), g["seed"].to_numpy(), seed=seed)
                if nested
                else bootstrap_ci(g[m].to_numpy(), seed=seed)
            )
            row[m] = point
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(by)).reset_index(drop=True)
