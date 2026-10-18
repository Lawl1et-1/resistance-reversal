"""Baseline response models, plus the latent-reconstruction control.

All models share one interface and one target: the latent mean shift of a perturbation
against its own context's control. Training targets are built from training perturbations
in training contexts only.

Honesty note on the two weakest baselines: `control_mean` predicts no change at all, and
`additive` can say nothing about a perturbation whose genes it never saw. Both return a
zero delta in that case rather than being quietly excluded from the comparison, because
"this model cannot address this split" is itself a result about the split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data.dataset import CONTROL, PerturbData
from src.data.splits import Split
from src.models.latent import LatentEncoder, training_cell_mask


def build_targets(
    data: PerturbData, split: Split, encoder: LatentEncoder, perts: list[tuple] | None = None
) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """Observed latent deltas for (perturbation, context) pairs.

    Returns (keys, Y, contexts) where keys[i] = (pert, context) and Y[i] is the latent
    mean shift against that context's control.
    """
    perts = split.train if perts is None else perts
    control_z = {
        c: encoder.transform(data.cells(CONTROL, c)).mean(axis=0) for c in split.train_contexts
    }
    keys, rows = [], []
    for pert in perts:
        if pert not in data.pert_index:
            continue
        for ctx in split.train_contexts:
            cells = data.cells(pert, ctx)
            if not len(cells):
                continue
            keys.append((pert, ctx))
            rows.append(encoder.transform(cells).mean(axis=0) - control_z[ctx])
    if not rows:
        raise ValueError("no training targets could be built")
    return keys, np.vstack(rows), np.array([c for _, c in keys])


def gene_embeddings(
    data: PerturbData, split: Split, dim: int = 16, seed: int = 0
) -> np.ndarray:
    """Per-gene embeddings from co-expression on training cells only.

    This is what lets a linear model say anything at all about a gene it never perturbed:
    an unseen gene inherits a prediction from genes with similar co-expression profiles.
    Computed over all genes (not the HVG subset) because the perturbable regulators are
    not guaranteed to survive HVG selection.
    """
    X = data.X[training_cell_mask(data, split)]
    X = X - X.mean(axis=0)
    sd = X.std(axis=0)
    corr = (X / np.clip(sd, 1e-8, None)).T @ (X / np.clip(sd, 1e-8, None)) / max(len(X) - 1, 1)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr[:, sd < 1e-8] = 0.0
    corr[sd < 1e-8, :] = 0.0

    # Truncated SVD of the (symmetric) co-expression matrix.
    u, s, _ = np.linalg.svd(corr, full_matrices=False)
    d = min(dim, u.shape[1])
    emb = u[:, :d] * s[:d]
    scale = np.abs(emb).max()
    return emb / scale if scale > 0 else emb


def set_features(
    pert: tuple, emb: np.ndarray, n_contexts: int, context: int, use_context: bool = True
) -> np.ndarray:
    """Sum-pooled gene embeddings for the set, optionally with a context one-hot.

    Sum pooling is permutation-invariant and defined for any set size, which is what makes
    an unseen combination representable at all.
    """
    vec = emb[[g for g, _ in pert]].sum(axis=0) if pert else np.zeros(emb.shape[1])
    direction = np.array([float(sum(d for _, d in pert)), float(len(pert))])
    parts = [vec, direction]
    if use_context:
        onehot = np.zeros(n_contexts)
        onehot[context] = 1.0
        parts.append(onehot)
    return np.concatenate(parts)


@dataclass
class ResponseModel:
    """Interface: fit on a split, then predict a latent delta for (pert, context)."""

    name: str = "base"

    def fit(self, data: PerturbData, split: Split, encoder: LatentEncoder) -> "ResponseModel":
        raise NotImplementedError

    def predict_latent_delta(self, pert: tuple, context: int) -> np.ndarray:
        raise NotImplementedError

    def predict_many(self, keys: list[tuple]) -> np.ndarray:
        return np.vstack([self.predict_latent_delta(p, c) for p, c in keys])


@dataclass
class ControlMean(ResponseModel):
    """Predict no change. The floor every other model must clear."""

    name: str = "control_mean"

    def fit(self, data, split, encoder):
        self.dim_ = encoder.latent_dim_fitted_
        return self

    def predict_latent_delta(self, pert, context):
        return np.zeros(self.dim_)


@dataclass
class Additive(ResponseModel):
    """Sum of the observed single-gene deltas. The standard non-interacting model."""

    name: str = "additive"
    singles_: dict = field(default_factory=dict)

    def fit(self, data, split, encoder):
        self.dim_ = encoder.latent_dim_fitted_
        keys, Y, _ = build_targets(data, split, encoder)
        acc: dict[tuple, list] = {}
        for (pert, _), y in zip(keys, Y):
            if len(pert) == 1:
                acc.setdefault(pert[0], []).append(y)
        self.singles_ = {atom: np.mean(v, axis=0) for atom, v in acc.items()}
        return self

    def predict_latent_delta(self, pert, context):
        known = [self.singles_[a] for a in pert if a in self.singles_]
        return np.sum(known, axis=0) if known else np.zeros(self.dim_)


@dataclass
class RidgeEmbedding(ResponseModel):
    """Ridge regression from sum-pooled gene embeddings to the latent delta."""

    name: str = "ridge_embedding"
    alpha: float = 1.0
    emb_dim: int = 16
    use_context: bool = True

    def fit(self, data, split, encoder):
        self.dim_ = encoder.latent_dim_fitted_
        self.n_contexts_ = data.n_contexts
        self.emb_ = gene_embeddings(data, split, dim=self.emb_dim)
        keys, Y, _ = build_targets(data, split, encoder)
        X = np.vstack([self._feat(p, c) for p, c in keys])
        self.mu_x_, self.mu_y_ = X.mean(axis=0), Y.mean(axis=0)
        Xc, Yc = X - self.mu_x_, Y - self.mu_y_
        gram = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
        self.W_ = np.linalg.solve(gram, Xc.T @ Yc)
        return self

    def _feat(self, pert, context):
        return set_features(pert, self.emb_, self.n_contexts_, context, self.use_context)

    def predict_latent_delta(self, pert, context):
        return (self._feat(pert, context) - self.mu_x_) @ self.W_ + self.mu_y_


@dataclass
class LowRankLinear(RidgeEmbedding):
    """Reduced-rank linear model: the ridge solution truncated to rank `rank`.

    The 2025 benchmark paper reports that a low-rank linear model is a surprisingly strong
    baseline for perturbation response prediction (`TODO[verify]` -- citation and the exact
    formulation must be checked before this is described as a reproduction). This is a
    reduced-rank ridge in the same spirit, not a reimplementation of their code.
    """

    name: str = "low_rank_linear"
    rank: int = 8

    def fit(self, data, split, encoder):
        super().fit(data, split, encoder)
        u, s, vt = np.linalg.svd(self.W_, full_matrices=False)
        r = min(self.rank, len(s))
        self.W_ = (u[:, :r] * s[:r]) @ vt[:r]
        return self


@dataclass
class OracleLatent(ResponseModel):
    """Not a model: the latent-reconstruction ceiling.

    Reads the observed test delta, projects it into the latent basis and back. Any gap
    between this and a perfect score is the cost of the latent bottleneck, not a modelling
    failure, so it bounds what every latent-space model here could possibly achieve.
    """

    name: str = "oracle_latent"

    def fit(self, data, split, encoder):
        self.data_, self.encoder_ = data, encoder
        self.dim_ = encoder.latent_dim_fitted_
        return self

    def predict_latent_delta(self, pert, context):
        cells = self.data_.cells(pert, context)
        if not len(cells):
            return np.zeros(self.dim_)
        z = self.encoder_.transform(cells).mean(axis=0)
        z0 = self.encoder_.transform(self.data_.cells(CONTROL, context)).mean(axis=0)
        return z - z0


@dataclass
class ShuffledLabels(RidgeEmbedding):
    """Negative control: the ridge model fitted with perturbation labels shuffled.

    Any metric on which this scores well is a metric that does not require knowing which
    perturbation was applied.
    """

    name: str = "shuffled_labels"
    seed: int = 0

    def fit(self, data, split, encoder):
        self.dim_ = encoder.latent_dim_fitted_
        self.n_contexts_ = data.n_contexts
        self.emb_ = gene_embeddings(data, split, dim=self.emb_dim)
        keys, Y, _ = build_targets(data, split, encoder)
        rng = np.random.default_rng(self.seed)
        Y = Y[rng.permutation(len(Y))]
        X = np.vstack([self._feat(p, c) for p, c in keys])
        self.mu_x_, self.mu_y_ = X.mean(axis=0), Y.mean(axis=0)
        Xc, Yc = X - self.mu_x_, Y - self.mu_y_
        gram = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
        self.W_ = np.linalg.solve(gram, Xc.T @ Yc)
        return self


BASELINES = {
    m.name: m
    for m in (ControlMean(), Additive(), RidgeEmbedding(), LowRankLinear(), ShuffledLabels())
}
