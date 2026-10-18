"""Latent encoder, fitted on training cells only.

Every response model in this project predicts a *latent* delta, so the comparison between
baselines and the proposed model is apples-to-apples and the reversal functional `s` has a
single input space. The cost is a reconstruction floor: no model can beat what the latent
basis can represent. That floor is measured and reported as a control row
(`oracle_latent`) rather than left implicit.

Leakage discipline: HVG selection, the scaler and the PCA basis are all fitted on cells
whose perturbation is in `split.train` (plus controls from training contexts). Cells from
test or calibration perturbations never touch the fit. `training_cell_mask` is the single
place that decision is made, so it cannot drift between call sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.dataset import CONTROL, PerturbData
from src.data.splits import Split


def training_cell_mask(data: PerturbData, split: Split) -> np.ndarray:
    """Cells the model is allowed to fit on: training perturbations and their controls."""
    allowed = {data.pert_index[p] for p in split.train if p in data.pert_index}
    allowed.add(data.pert_index[CONTROL])
    mask = np.isin(data.pert_ids, list(allowed))
    return mask & np.isin(data.context_ids, list(split.train_contexts))


@dataclass
class LatentEncoder:
    """HVG selection -> standardisation -> PCA. Linear, so the decoder is exact.

    Implemented directly on numpy's SVD rather than scikit-learn. Two reasons: PCA and
    standardisation are a handful of lines here and the delta helpers below already depend
    on the transform being exactly linear, and importing scikit-learn alongside torch
    segfaults under pytest in this environment (see LOG.md Failure 8), which would have
    meant no test coverage for any model that uses both.
    """

    latent_dim: int = 32
    n_hvg: int | None = None
    seed: int = 0

    def fit(self, X: np.ndarray) -> "LatentEncoder":
        X = np.asarray(X, float)
        if self.n_hvg is None or self.n_hvg >= X.shape[1]:
            self.gene_index = np.arange(X.shape[1])
        else:
            # Highly variable genes by raw variance on the training cells only.
            self.gene_index = np.sort(np.argsort(-X.var(axis=0))[: self.n_hvg])
        Xh = X[:, self.gene_index]

        # Standardise. Genes with no variance would divide by zero; they carry no signal,
        # so their scale is pinned to 1 and they end up contributing nothing.
        self.mean_ = Xh.mean(axis=0)
        sd = Xh.std(axis=0)
        self.scale_ = np.where(sd > 1e-8, sd, 1.0)
        Xs = (Xh - self.mean_) / self.scale_

        dim = min(self.latent_dim, Xh.shape[1], len(Xh))
        self.pca_mean_ = Xs.mean(axis=0)  # ~0 by construction; kept for exactness
        _, s, vt = np.linalg.svd(Xs - self.pca_mean_, full_matrices=False)
        self.components_ = vt[:dim]
        var = s**2
        self.explained_ = float(var[:dim].sum() / var.sum()) if var.sum() > 0 else 0.0
        self.latent_dim_fitted_ = dim
        return self

    # --- state <-> latent ---------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.atleast_2d(X)[:, self.gene_index] - self.mean_) / self.scale_
        return (Xs - self.pca_mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Back to HVG expression space."""
        Xs = np.atleast_2d(Z) @ self.components_ + self.pca_mean_
        return Xs * self.scale_ + self.mean_

    # --- deltas ------------------------------------------------------------------------
    # A delta is a difference of states, so every offset (the gene mean and the PCA mean)
    # cancels and only the scale applies. Passing a delta through inverse_transform would
    # wrongly add those offsets back in -- a silent and large error -- hence these helpers.

    def decode_delta(self, dZ: np.ndarray) -> np.ndarray:
        """Latent delta -> gene-space delta over `gene_index`."""
        return np.atleast_2d(dZ) @ self.components_ * self.scale_

    def encode_delta(self, dX: np.ndarray) -> np.ndarray:
        """Gene-space delta over `gene_index` -> latent delta."""
        return (np.atleast_2d(dX) / self.scale_) @ self.components_.T

    def genes(self, X: np.ndarray) -> np.ndarray:
        """Restrict a gene-space array to the HVG subset the encoder was fitted on."""
        return np.atleast_2d(X)[:, self.gene_index]

    @property
    def explained_variance(self) -> float:
        return self.explained_


def fit_encoder(
    data: PerturbData, split: Split, latent_dim: int = 32, n_hvg: int | None = None, seed: int = 0
) -> LatentEncoder:
    mask = training_cell_mask(data, split)
    if not mask.any():
        raise ValueError("no training cells available for the encoder")
    return LatentEncoder(latent_dim=latent_dim, n_hvg=n_hvg, seed=seed).fit(data.X[mask])
