"""Latent encoder tests: the linear identities the delta helpers rely on, and no leakage."""

import numpy as np
import pytest

from src.data.dataset import CONTROL, make_response_data, sample_perturbations
from src.data.splits import make_split
from src.models.latent import LatentEncoder, fit_encoder, training_cell_mask
from src.simulator.grn import GRNConfig, build_grn


@pytest.fixture(scope="module")
def setup():
    grn = build_grn(GRNConfig(n_genes=60, n_regulators=12, n_program=15, n_reg_modules=4, seed=0))
    perts = sample_perturbations(grn, k=2, n_pairs=30, seed=0)
    data = make_response_data(grn, perts, n_cells=24, n_control_cells=64, n_contexts=2, seed=0)
    split = make_split(perts, "unseen_single", n_contexts=2, seed=0)
    return grn, data, split


def test_delta_encoding_matches_difference_of_transforms(setup):
    """encode_delta(x1 - x2) must equal transform(x1) - transform(x2) exactly."""
    _, data, split = setup
    enc = fit_encoder(data, split, latent_dim=16)
    x1, x2 = data.X[:5], data.X[5:10]
    lhs = enc.encode_delta(enc.genes(x1) - enc.genes(x2))
    rhs = enc.transform(x1) - enc.transform(x2)
    assert np.allclose(lhs, rhs, atol=1e-9)


def test_delta_decoding_matches_difference_of_inverses(setup):
    """decode_delta(z1 - z2) must equal inverse_transform(z1) - inverse_transform(z2)."""
    _, data, split = setup
    enc = fit_encoder(data, split, latent_dim=16)
    z1, z2 = enc.transform(data.X[:5]), enc.transform(data.X[5:10])
    lhs = enc.decode_delta(z1 - z2)
    rhs = enc.inverse_transform(z1) - enc.inverse_transform(z2)
    assert np.allclose(lhs, rhs, atol=1e-9)


def test_inverse_transform_on_a_delta_is_wrong(setup):
    """Documents the trap the delta helpers exist to avoid.

    Passing a delta through inverse_transform adds the gene means back in, which is a
    silent error of roughly the size of the expression profile itself.
    """
    _, data, split = setup
    enc = fit_encoder(data, split, latent_dim=16)
    dz = enc.transform(data.X[:5]) - enc.transform(data.X[5:10])
    correct = enc.decode_delta(dz)
    naive = enc.inverse_transform(dz)
    assert not np.allclose(correct, naive, atol=1e-3)
    assert np.abs(naive - correct).max() > 0.1


def test_full_rank_round_trip_is_lossless(setup):
    """With as many components as genes, encode/decode must be the identity."""
    _, data, split = setup
    enc = fit_encoder(data, split, latent_dim=data.n_genes)
    x = data.X[:20]
    assert np.allclose(enc.inverse_transform(enc.transform(x)), enc.genes(x), atol=1e-8)
    dx = enc.genes(data.X[:20]) - enc.genes(data.X[20:40])
    assert np.allclose(enc.decode_delta(enc.encode_delta(dx)), dx, atol=1e-8)


def test_truncated_encoder_loses_little_variance(setup):
    _, data, split = setup
    enc = fit_encoder(data, split, latent_dim=16)
    assert 0.0 < enc.explained_variance <= 1.0
    assert enc.explained_variance > 0.8


def test_latent_dim_is_capped_by_available_dimensions(setup):
    _, data, split = setup
    enc = fit_encoder(data, split, latent_dim=10_000)
    assert enc.latent_dim_fitted_ == min(data.n_genes, int(training_cell_mask(data, split).sum()))


def test_hvg_selection_keeps_requested_count(setup):
    _, data, split = setup
    enc = fit_encoder(data, split, latent_dim=8, n_hvg=20)
    assert len(enc.gene_index) == 20
    assert enc.transform(data.X[:3]).shape == (3, 8)


def test_zero_variance_genes_do_not_produce_nans():
    X = np.random.default_rng(0).normal(size=(50, 8))
    X[:, 3] = 2.0  # constant gene
    enc = LatentEncoder(latent_dim=4).fit(X)
    z = enc.transform(X)
    assert np.isfinite(z).all()
    assert np.isfinite(enc.inverse_transform(z)).all()


def test_training_mask_excludes_test_and_calibration_cells(setup):
    """The leakage guard: no held-out perturbation may reach the encoder's fit."""
    _, data, split = setup
    mask = training_cell_mask(data, split)
    seen = {data.perts[i] for i in np.unique(data.pert_ids[mask])}
    assert seen <= set(split.train) | {CONTROL}
    assert not seen & set(split.test)
    assert not seen & set(split.cal)
    assert set(np.unique(data.context_ids[mask])) <= set(split.train_contexts)


def test_encoder_is_deterministic(setup):
    _, data, split = setup
    a, b = fit_encoder(data, split, latent_dim=16), fit_encoder(data, split, latent_dim=16)
    assert np.allclose(a.transform(data.X[:5]), b.transform(data.X[:5]))


def test_fit_encoder_rejects_empty_training_set(setup):
    _, data, split = setup
    import copy

    empty = copy.deepcopy(split)
    empty.train = []
    empty.train_contexts = [99]  # no such context
    with pytest.raises(ValueError):
        fit_encoder(data, empty, latent_dim=4)
