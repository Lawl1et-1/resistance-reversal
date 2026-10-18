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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    """HVG selection -> standardisation -> PCA. Linear, so the decoder is exact."""

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
        self.scaler_ = StandardScaler().fit(Xh)
        dim = min(self.latent_dim, Xh.shape[1], len(Xh))
        self.pca_ = PCA(n_components=dim, random_state=self.seed).fit(self.scaler_.transform(Xh))
        self.latent_dim_fitted_ = dim
        return self

    # --- state <-> latent ---------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca_.transform(self.scaler_.transform(np.atleast_2d(X)[:, self.gene_index]))

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Back to (standardised-then-inverted) HVG expression space."""
        return self.scaler_.inverse_transform(self.pca_.inverse_transform(np.atleast_2d(Z)))

    # --- deltas ------------------------------------------------------------------------
    # A delta is a difference of states, so the scaler's *offset* cancels and only its
    # scale applies. Using inverse_transform on a delta directly would wrongly add the
    # mean back in, which is a silent and large error, hence these two helpers.

    def decode_delta(self, dZ: np.ndarray) -> np.ndarray:
        """Latent delta -> gene-space delta over `gene_index`."""
        return np.atleast_2d(dZ) @ self.pca_.components_ * self.scaler_.scale_

    def encode_delta(self, dX: np.ndarray) -> np.ndarray:
        """Gene-space delta over `gene_index` -> latent delta."""
        return (np.atleast_2d(dX) / self.scaler_.scale_) @ self.pca_.components_.T

    def genes(self, X: np.ndarray) -> np.ndarray:
        """Restrict a gene-space array to the HVG subset the encoder was fitted on."""
        return np.atleast_2d(X)[:, self.gene_index]

    @property
    def explained_variance(self) -> float:
        return float(self.pca_.explained_variance_ratio_.sum())


def fit_encoder(
    data: PerturbData, split: Split, latent_dim: int = 32, n_hvg: int | None = None, seed: int = 0
) -> LatentEncoder:
    mask = training_cell_mask(data, split)
    if not mask.any():
        raise ValueError("no training cells available for the encoder")
    return LatentEncoder(latent_dim=latent_dim, n_hvg=n_hvg, seed=seed).fit(data.X[mask])
