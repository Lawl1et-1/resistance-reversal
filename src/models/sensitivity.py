"""The sensitivity functional `s(z)` in [0, 1], and a second, independent readout.

`s` is trained on the sensitivity dataset -- unperturbed sensitive vs resistant cells -- which
shares no cells with the perturbation-response data used to fit `f`. That disjointness is
the point: if `s` were trained on the same experiment it would partly be reading back `f`'s
own training signal, and a high reversal score would be self-confirming.

The price of disjointness is **domain shift**: `s` is fitted on unperturbed populations and
then evaluated on predicted post-perturbation states, which are off its training
distribution. Nothing in this module fixes that. What it does instead is provide two
readouts that depend on different features, so their disagreement is measurable:

- `LatentClassifier` -- logistic regression on the full latent state. Analogous to a
  single-cell classifier.
- `SignatureReadout` -- logistic regression on a single up-minus-down signature score,
  built from differential expression on the sensitivity training data. Analogous to a bulk
  pharmacogenomic signature, and deliberately much lower-capacity.

Agreement between two readouts is weak evidence; disagreement is a warning that the
reversal score is readout-dependent and must be reported as such.

Logistic regression is fitted with scipy's L-BFGS rather than scikit-learn (see LOG.md
Failure 8 -- scikit-learn segfaults alongside torch under pytest here).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.models.latent import LatentEncoder


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression. Returns (weights, intercept).

    The intercept is deliberately left unpenalised, so regularisation does not drag the
    predicted probabilities toward 0.5 when the classes are imbalanced.
    """
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(y, float).ravel()
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if not np.isin(np.unique(y), (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    n, d = X.shape

    def objective(theta):
        w, b = theta[:d], theta[d]
        z = X @ w + b
        # Numerically stable log-loss via logaddexp.
        loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * (w @ w) / n
        resid = (_sigmoid(z) - y) / n
        grad = np.concatenate([X.T @ resid + l2 * w / n, [resid.sum()]])
        return loss, grad

    res = minimize(objective, np.zeros(d + 1), jac=True, method="L-BFGS-B")
    return res.x[:d], float(res.x[d])


@dataclass
class LatentClassifier:
    """s(z) from logistic regression on the latent state. Label 1 = sensitive."""

    name: str = "sc_classifier"
    l2: float = 1.0

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "LatentClassifier":
        self.w_, self.b_ = fit_logistic(Z, y, self.l2)
        return self

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return _sigmoid(np.atleast_2d(np.asarray(Z, float)) @ self.w_ + self.b_)


@dataclass
class SignatureReadout:
    """s(z) from a one-dimensional up-minus-down signature score on gene expression.

    The gene sets come from differential expression on the sensitivity training data only.
    Operates in gene space, so latent inputs are decoded first; the encoder is linear, so
    that decoding is exact up to the latent truncation.
    """

    name: str = "bulk_signature"
    n_genes: int = 25
    l2: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, encoder: LatentEncoder) -> "SignatureReadout":
        self.encoder_ = encoder
        Xg = encoder.genes(np.asarray(X, float))
        y = np.asarray(y).ravel()
        diff = Xg[y == 1].mean(axis=0) - Xg[y == 0].mean(axis=0)
        k = min(self.n_genes, diff.size // 2)
        order = np.argsort(diff)
        self.down_, self.up_ = order[:k], order[-k:]
        self.w_, self.b_ = fit_logistic(self._score(Xg), y, self.l2)
        return self

    def _score(self, Xg: np.ndarray) -> np.ndarray:
        Xg = np.atleast_2d(Xg)
        return (Xg[:, self.up_].mean(axis=1) - Xg[:, self.down_].mean(axis=1)).reshape(-1, 1)

    def from_genes(self, Xg: np.ndarray) -> np.ndarray:
        return _sigmoid(self._score(Xg) @ self.w_ + self.b_).ravel()

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return self.from_genes(self.encoder_.inverse_transform(np.atleast_2d(Z)))


@dataclass
class SensitivityEnsemble:
    """Both readouts together, reporting their disagreement rather than hiding it."""

    primary: LatentClassifier
    secondary: SignatureReadout

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return self.primary(Z)

    def both(self, Z: np.ndarray) -> dict[str, np.ndarray]:
        return {self.primary.name: self.primary(Z), self.secondary.name: self.secondary(Z)}

    def disagreement(self, Z: np.ndarray) -> float:
        a, b = self.primary(Z), self.secondary(Z)
        return float(np.mean(np.abs(a - b)))


def fit_sensitivity(
    X: np.ndarray, y: np.ndarray, encoder: LatentEncoder, l2: float = 1.0, n_sig_genes: int = 25
) -> SensitivityEnsemble:
    """Fit both readouts on the sensitivity dataset, in the response encoder's latent space.

    The encoder is passed in already fitted (on response *training* cells), so the
    sensitivity data never influences the latent basis and the two experiments stay
    separable.
    """
    Z = encoder.transform(X)
    return SensitivityEnsemble(
        primary=LatentClassifier(l2=l2).fit(Z, y),
        secondary=SignatureReadout(n_genes=n_sig_genes, l2=l2).fit(X, y, encoder),
    )
