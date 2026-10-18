"""No-leakage tests for the perturbation splits.

The assertions here are the ones that make an "unseen perturbation" claim honest. They are
tests rather than review comments because leakage is silent: a leaked split produces
better-looking numbers and no error.
"""

import numpy as np
import pytest

from src.data.dataset import sample_perturbations
from src.data.splits import (
    KINDS,
    Split,
    load_splits,
    make_all_splits,
    make_split,
    save_splits,
)
from src.simulator.grn import GRNConfig, build_grn


@pytest.fixture(scope="module")
def perts():
    grn = build_grn(GRNConfig(n_genes=60, n_regulators=12, n_program=15, seed=0))
    return sample_perturbations(grn, k=2, n_pairs=40, seed=0)


PERT_AXIS_KINDS = [k for k in KINDS if k != "unseen_context"]


@pytest.mark.parametrize("kind", KINDS)
def test_every_split_is_populated_and_calibration_is_held_out(perts, kind):
    s = make_split(perts, kind, n_contexts=3, seed=0)
    s.check_no_leakage()  # redundant with make_split, but this is the property under test
    assert s.test and s.cal and s.train
    # Required for every split: conformal residuals must come from unfitted perturbations.
    assert set(s.train).isdisjoint(s.cal)


@pytest.mark.parametrize("kind", PERT_AXIS_KINDS)
def test_perturbation_splits_are_mutually_disjoint(perts, kind):
    s = make_split(perts, kind, n_contexts=3, seed=0)
    assert set(s.train).isdisjoint(s.test)
    assert set(s.cal).isdisjoint(s.test)


@pytest.mark.parametrize("kind", ["unseen_single", "unseen_combo_unseen_genes"])
def test_held_out_genes_never_appear_in_training(perts, kind):
    """A held-out gene must not appear in training even inside a larger set."""
    s = make_split(perts, kind, n_contexts=3, seed=0)
    held = set(s.meta["held_out_genes"])
    assert held
    for p in s.train:
        assert not ({g for g, _ in p} & held), f"training set {p} touches held-out gene"
    assert s.genes("test") <= held
    assert s.genes("cal") <= held
    assert s.genes("test").isdisjoint(s.genes("cal"))


def test_unseen_single_tests_only_singles(perts):
    s = make_split(perts, "unseen_single", n_contexts=3, seed=0)
    assert all(len(p) == 1 for p in s.test)


def test_unseen_combo_unseen_genes_tests_only_combinations(perts):
    s = make_split(perts, "unseen_combo_unseen_genes", n_contexts=3, seed=0)
    assert all(len(p) > 1 for p in s.test)


def test_unseen_combo_seen_genes_keeps_every_test_gene_in_training(perts):
    """The composition split: the pair is new, both its genes are not."""
    s = make_split(perts, "unseen_combo_seen_genes", n_contexts=3, seed=0)
    assert all(len(p) > 1 for p in s.test)
    assert s.genes("test") <= s.genes("train")
    assert s.genes("cal") <= s.genes("train")


def test_unseen_context_holds_out_a_context_not_perturbations(perts):
    """Here the held-out axis is the context, so sharing perturbations is the design."""
    s = make_split(perts, "unseen_context", n_contexts=3, seed=0)
    assert set(s.train_contexts).isdisjoint(s.test_contexts)
    assert s.test_contexts == [2]
    assert s.meta["held_out_axis"] == "context"
    assert set(s.train) & set(s.test), "the same perturbations must be probed in the new context"
    # Calibration still uses held-out perturbations inside the training contexts.
    assert set(s.cal).isdisjoint(s.train)


def test_context_leakage_check_fires_on_shared_context():
    leaked = Split(
        name="unseen_context",
        train=[((2, -1),)],
        cal=[((3, -1),)],
        test=[((2, -1),)],
        train_contexts=[0, 1],
        test_contexts=[1],
        meta={"held_out_axis": "context"},
    )
    with pytest.raises(AssertionError, match="contexts overlap"):
        leaked.check_no_leakage()


def test_unseen_context_requires_two_contexts(perts):
    with pytest.raises(ValueError):
        make_split(perts, "unseen_context", n_contexts=1, seed=0)


def test_control_is_never_a_target(perts):
    for kind in KINDS:
        s = make_split(perts, kind, n_contexts=3, seed=0)
        assert () not in s.test
        assert () not in s.cal
        assert () not in s.train


def test_leakage_check_actually_fires():
    """A deliberately leaked split must raise, or the other tests prove nothing."""
    leaked = Split(
        name="unseen_single",
        train=[((2, -1),)],
        cal=[((3, -1),)],
        test=[((2, -1),)],
        train_contexts=[0],
        test_contexts=[0],
    )
    with pytest.raises(AssertionError, match="train/test"):
        leaked.check_no_leakage()

    gene_leak = Split(
        name="unseen_single",
        train=[((2, -1), (5, -1))],
        cal=[((3, -1),)],
        test=[((5, -1),)],
        train_contexts=[0],
        test_contexts=[0],
        meta={"held_out_genes": [5]},
    )
    with pytest.raises(AssertionError, match="held-out genes"):
        gene_leak.check_no_leakage()

    composition_leak = Split(
        name="unseen_combo_seen_genes",
        train=[((2, -1),)],
        cal=[((2, -1), (3, -1))],
        test=[((2, -1), (9, -1))],  # gene 9 never seen in training
        train_contexts=[0],
        test_contexts=[0],
        meta={"require_test_genes_seen": True},
    )
    with pytest.raises(AssertionError, match="both genes seen"):
        composition_leak.check_no_leakage()


def test_splits_round_trip_through_disk(perts, tmp_path):
    splits = make_all_splits(perts, n_contexts=3, seed=0)
    path = save_splits(splits, tmp_path / "splits.json")
    loaded = load_splits(path)
    assert set(loaded) == set(splits)
    for name, s in splits.items():
        assert loaded[name].train == s.train
        assert loaded[name].cal == s.cal
        assert loaded[name].test == s.test
        assert loaded[name].test_contexts == s.test_contexts


def test_splits_are_deterministic_given_a_seed(perts):
    a = make_split(perts, "unseen_single", n_contexts=3, seed=7)
    b = make_split(perts, "unseen_single", n_contexts=3, seed=7)
    c = make_split(perts, "unseen_single", n_contexts=3, seed=8)
    assert a.test == b.test
    assert a.test != c.test


def test_make_all_splits_covers_every_kind(perts):
    splits = make_all_splits(perts, n_contexts=3, seed=0)
    assert set(splits) == set(KINDS)
