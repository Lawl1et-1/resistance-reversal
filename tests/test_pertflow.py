"""Tests for PertFlow, including regression guards for two bugs that were costly to find.

See LOG.md Failures 6 and 7. Both produced plausible-looking point metrics while being
badly wrong, so each has an explicit test here:

- the flow must transport a *population*, not collapse it onto a conditional mean;
- the conditioning must actually be used, i.e. predictions must differ across
  perturbations rather than defaulting to one average shift.
"""

import numpy as np
import pytest
import torch

from src.data.dataset import CONTROL, make_response_data, sample_perturbations
from src.data.splits import make_split
from src.eval.metrics import energy_distance
from src.models.baselines import gene_embeddings
from src.models.latent import fit_encoder
from src.models.pertflow import PertFlow, SetEncoder
from src.simulator.grn import GRNConfig, build_grn

torch.set_num_threads(2)


@pytest.fixture(scope="module")
def setup():
    grn = build_grn(GRNConfig(n_genes=60, n_regulators=12, n_program=15, n_reg_modules=4, seed=0))
    perts = sample_perturbations(grn, k=2, n_pairs=30, seed=0)
    data = make_response_data(grn, perts, n_cells=32, n_control_cells=96, n_contexts=2, seed=0)
    split = make_split(perts, "unseen_combo_seen_genes", n_contexts=2, seed=0)
    encoder = fit_encoder(data, split, latent_dim=16)
    return grn, data, split, encoder


@pytest.fixture(scope="module")
def fitted(setup):
    _, data, split, encoder = setup
    model = PertFlow(steps=600, hidden=64, set_dim=16, n_ensemble=2, seed=0)
    return model.fit(data, split, encoder)


def test_gene_embeddings_are_standardised(setup):
    """Guards Failure 7: tiny embeddings make the network ignore the perturbation."""
    _, data, split, _ = setup
    emb = gene_embeddings(data, split, dim=16)
    per_dim = emb.std(axis=0)
    assert 0.5 < per_dim.mean() < 2.0, f"embedding scale off ({per_dim.mean():.3f})"


def test_set_encoder_is_permutation_invariant():
    """Sum pooling must not depend on the order genes are listed in."""
    torch.manual_seed(0)
    enc = SetEncoder(feat_dim=5, hidden=16, out_dim=8).eval()
    feats = torch.randn(1, 3, 5)
    mask = torch.ones(1, 3)
    with torch.no_grad():
        a = enc(feats, mask)
        b = enc(feats[:, [2, 0, 1]], mask)
    assert torch.allclose(a, b, atol=1e-6)


def test_set_encoder_ignores_masked_atoms():
    """Padding must not change the encoding, or set size would leak through the mask."""
    torch.manual_seed(0)
    enc = SetEncoder(feat_dim=5, hidden=16, out_dim=8).eval()
    feats = torch.randn(1, 2, 5)
    padded = torch.cat([feats, torch.randn(1, 1, 5)], dim=1)
    with torch.no_grad():
        a = enc(feats, torch.ones(1, 2))
        b = enc(padded, torch.tensor([[1.0, 1.0, 0.0]]))
    assert torch.allclose(a, b, atol=1e-6)


def test_flow_does_not_collapse_the_population(setup, fitted):
    """Guards Failure 6: a z0-conditioned field shrank spread to ~1/3 of the truth."""
    _, data, split, encoder = setup
    ctx = split.test_contexts[0]
    z0 = encoder.transform(data.cells(CONTROL, ctx))
    pert = split.train[0]
    z_true = encoder.transform(data.cells(pert, ctx))
    z_pred = fitted.sample_latent(pert, ctx, z0, 96)
    spread = lambda a: float(a.std(axis=0).mean())
    assert spread(z_pred) > 0.6 * spread(z_true), (
        f"predicted population collapsed: {spread(z_pred):.3f} vs true {spread(z_true):.3f}"
    )


def test_predictions_differ_across_perturbations(setup, fitted):
    """Guards Failure 7: the model must not emit one average shift for everything."""
    _, data, split, encoder = setup
    ctx = split.test_contexts[0]
    preds = np.vstack([fitted.predict_latent_delta(p, ctx) for p in split.train[:8]])
    norms = np.linalg.norm(preds, axis=1)
    assert preds.std(axis=0).mean() > 0.05, "predictions barely vary across perturbations"
    assert norms.std() / max(norms.mean(), 1e-9) > 0.05, "predicted magnitudes are near-constant"


def test_predicted_direction_correlates_with_truth(setup, fitted):
    """A weak but positive check that the model learned something, not just scale."""
    _, data, split, encoder = setup
    ctx = split.test_contexts[0]
    z0_mean = encoder.transform(data.cells(CONTROL, ctx)).mean(axis=0)
    cosines = []
    for pert in split.train[:8]:
        true = encoder.transform(data.cells(pert, ctx)).mean(axis=0) - z0_mean
        pred = fitted.predict_latent_delta(pert, ctx)
        cosines.append(pred @ true / (np.linalg.norm(pred) * np.linalg.norm(true)))
    assert np.mean(cosines) > 0.4, f"mean cosine to true shift only {np.mean(cosines):.3f}"


def test_beats_do_nothing_on_energy_distance(setup, fitted):
    """The predicted population must be closer to the truth than the control cells are."""
    _, data, split, encoder = setup
    ctx = split.test_contexts[0]
    z0 = encoder.transform(data.cells(CONTROL, ctx))
    wins = 0
    for pert in split.train[:6]:
        z_true = encoder.transform(data.cells(pert, ctx))
        pred = energy_distance(fitted.sample_latent(pert, ctx, z0, 96), z_true)
        control = energy_distance(z0, z_true)
        wins += pred < control
    assert wins >= 4, f"only beat the do-nothing control on {wins}/6 perturbations"


def test_ensemble_disagreement_is_positive(fitted, setup):
    _, _, split, _ = setup
    u = fitted.epistemic_uncertainty(split.test[0], split.test_contexts[0])
    assert u > 0.0
    mean, spread = fitted.predict_with_uncertainty(split.test[0], split.test_contexts[0])
    assert mean.shape == (fitted.dim_,)
    assert spread == pytest.approx(u)


def test_single_member_has_no_disagreement(setup):
    _, data, split, encoder = setup
    solo = PertFlow(steps=100, hidden=32, set_dim=16, n_ensemble=1, seed=0).fit(data, split, encoder)
    assert solo.epistemic_uncertainty(split.test[0], split.test_contexts[0]) == 0.0


def test_mlp_mode_runs_and_predicts(setup):
    """The flow-vs-regressor ablation path must work too."""
    _, data, split, encoder = setup
    mlp = PertFlow(
        name="pertflow_mlp", mode="mlp", steps=200, hidden=32, set_dim=16, n_ensemble=2, seed=0
    ).fit(data, split, encoder)
    pred = mlp.predict_latent_delta(split.test[0], split.test_contexts[0])
    assert pred.shape == (mlp.dim_,)
    assert np.isfinite(pred).all()


def test_unseen_context_falls_back_without_crashing(setup, fitted):
    """An unseen context has no control cells; the model must degrade, not raise."""
    _, data, split, _ = setup
    pred = fitted.predict_latent_delta(split.train[0], data.n_contexts - 1)
    assert np.isfinite(pred).all()
