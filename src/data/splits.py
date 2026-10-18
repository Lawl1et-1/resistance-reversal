"""Perturbation splits, defined and saved before any modelling.

Rule: a test perturbation must not appear in training *in any form*. For a held-out single
knockdown that means the gene may not appear inside a training pair either -- otherwise the
model has already seen that gene's effect and "unseen single" is a lie. So the single and
unseen-gene splits hold out *genes*, not perturbation labels.

The calibration group is always disjoint from both train and test and is grouped by
perturbation, because split-conformal coverage on unseen perturbations is only meaningful
if the calibration residuals come from perturbations the model never fit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

KINDS = ("unseen_single", "unseen_combo_seen_genes", "unseen_combo_unseen_genes", "unseen_context")


@dataclass
class Split:
    name: str
    train: list[tuple]
    cal: list[tuple]
    test: list[tuple]
    train_contexts: list[int]
    test_contexts: list[int]
    meta: dict = field(default_factory=dict)

    def genes(self, which: str) -> set[int]:
        return {g for p in getattr(self, which) for g, _ in p}

    def check_no_leakage(self) -> None:
        """Raise on any overlap the split's own definition forbids.

        Which overlap counts as leakage depends on the held-out axis. When the axis is the
        perturbation, train/cal/test must be mutually disjoint. When the axis is the cell
        context, the *same* perturbations appear in train and test by design -- only the
        context is withheld -- so demanding perturbation disjointness there would be
        testing the wrong thing. Train/cal disjointness is required either way, because
        conformal residuals must come from perturbations the model did not fit.
        """
        axis = self.meta.get("held_out_axis", "perturbation")
        tr, ca, te = set(self.train), set(self.cal), set(self.test)

        pairs = [(tr, ca, "train/cal")]
        if axis == "perturbation":
            pairs += [(tr, te, "train/test"), (ca, te, "cal/test")]
        for a, b, label in pairs:
            if a & b:
                raise AssertionError(f"{self.name}: {label} perturbation overlap: {sorted(a & b)}")

        if axis == "context":
            shared = set(self.train_contexts) & set(self.test_contexts)
            if shared:
                raise AssertionError(f"{self.name}: train and test contexts overlap: {sorted(shared)}")
            if not self.test_contexts:
                raise AssertionError(f"{self.name}: no held-out context")

        if () in te or () in ca:
            raise AssertionError(f"{self.name}: control must not be a test or calibration target")

        held = self.meta.get("held_out_genes")
        if held:
            leaked = self.genes("train") & set(held)
            if leaked:
                raise AssertionError(
                    f"{self.name}: held-out genes {sorted(leaked)} appear in training perturbations"
                )
        if self.meta.get("require_test_genes_seen"):
            unseen = self.genes("test") - self.genes("train")
            if unseen:
                raise AssertionError(
                    f"{self.name}: split requires both genes seen, but {sorted(unseen)} are not"
                )


def _by_size(perts: list[tuple]) -> dict[int, list[tuple]]:
    out: dict[int, list[tuple]] = {}
    for p in perts:
        if p:
            out.setdefault(len(p), []).append(p)
    return out


def _free_of(perts: list[tuple], genes: set[int]) -> list[tuple]:
    return [p for p in perts if p and not ({g for g, _ in p} & genes)]


def make_split(
    perts: list[tuple],
    kind: str,
    n_contexts: int = 3,
    test_frac: float = 0.25,
    cal_frac: float = 0.25,
    seed: int = 0,
) -> Split:
    if kind not in KINDS:
        raise ValueError(f"unknown split kind {kind!r}; expected one of {KINDS}")
    rng = np.random.default_rng(seed)
    sizes = _by_size(perts)
    all_genes = sorted({g for p in perts for g, _ in p})
    contexts = list(range(n_contexts))

    if kind == "unseen_single":
        test_genes, cal_genes = _three_way_genes(all_genes, test_frac, cal_frac, rng)
        held = test_genes | cal_genes
        train = _free_of(perts, held)
        test = [p for p in sizes.get(1, []) if {g for g, _ in p} <= test_genes]
        cal = [p for p in sizes.get(1, []) if {g for g, _ in p} <= cal_genes]
        meta = {"held_out_genes": sorted(held)}

    elif kind == "unseen_combo_unseen_genes":
        test_genes, cal_genes = _three_way_genes(all_genes, test_frac, cal_frac, rng)
        held = test_genes | cal_genes
        train = _free_of(perts, held)
        test = [p for p in perts if len(p) > 1 and {g for g, _ in p} <= test_genes]
        cal = [p for p in perts if len(p) > 1 and {g for g, _ in p} <= cal_genes]
        meta = {"held_out_genes": sorted(held)}

    elif kind == "unseen_combo_seen_genes":
        # Hold out whole combinations while keeping every one of their genes in training as
        # a single. This is the split that asks whether a model composes.
        combos = [p for p in perts if len(p) > 1]
        singles = sizes.get(1, [])
        single_genes = {g for p in singles for g, _ in p}
        eligible = [p for p in combos if {g for g, _ in p} <= single_genes]
        test, cal, train_combos = _three_way(eligible, test_frac, cal_frac, rng)
        train = singles + train_combos + [p for p in combos if p not in set(eligible)]
        meta = {"require_test_genes_seen": True}

    else:  # unseen_context
        if n_contexts < 2:
            raise ValueError("unseen_context split needs at least 2 contexts")
        test_ctx = [contexts[-1]]
        train_ctx = contexts[:-1]
        # Same perturbations throughout; only the context is held out. Calibration still
        # uses held-out *perturbations* inside the training contexts, so coverage under
        # context shift is measured against a calibration set that never saw the shift.
        non_control = [p for p in perts if p]
        cal, _, train = _three_way(non_control, cal_frac, 0.0, rng)
        return _finish(
            Split(
                name=kind,
                train=train,
                cal=cal,
                test=non_control,
                train_contexts=train_ctx,
                test_contexts=test_ctx,
                meta={"shift": "cell_context", "held_out_axis": "context"},
            )
        )

    return _finish(
        Split(
            name=kind,
            train=train,
            cal=cal,
            test=test,
            train_contexts=contexts,
            test_contexts=contexts,
            meta=meta,
        )
    )


def _three_way_genes(
    genes: list[int], test_frac: float, cal_frac: float, rng: np.random.Generator
) -> tuple[set[int], set[int]]:
    order = rng.permutation(genes)
    n_test = max(1, int(round(test_frac * len(genes))))
    n_cal = max(1, int(round(cal_frac * len(genes))))
    if n_test + n_cal >= len(genes):
        raise ValueError("test_frac + cal_frac leaves no genes for training")
    return set(map(int, order[:n_test])), set(map(int, order[n_test : n_test + n_cal]))


def _three_way(items, test_frac, cal_frac, rng):
    order = rng.permutation(len(items))
    n_test = max(1, int(round(test_frac * len(items)))) if test_frac else 0
    n_cal = max(1, int(round(cal_frac * len(items)))) if cal_frac else 0
    if n_test + n_cal >= len(items):
        raise ValueError("test_frac + cal_frac leaves nothing for training")
    pick = lambda idx: [items[i] for i in sorted(idx)]
    return pick(order[:n_test]), pick(order[n_test : n_test + n_cal]), pick(order[n_test + n_cal :])


def _finish(split: Split) -> Split:
    split.train = [p for p in split.train if p]  # control is implicit, never a target
    split.meta.setdefault("n_train", len(split.train))
    split.meta.setdefault("n_cal", len(split.cal))
    split.meta.setdefault("n_test", len(split.test))
    split.check_no_leakage()
    if not split.test:
        raise ValueError(f"{split.name}: split produced an empty test set")
    if not split.cal:
        raise ValueError(f"{split.name}: split produced an empty calibration set")
    return split


def make_all_splits(perts: list[tuple], n_contexts: int = 3, seed: int = 0, **kw) -> dict[str, Split]:
    out = {}
    for kind in KINDS:
        try:
            out[kind] = make_split(perts, kind, n_contexts=n_contexts, seed=seed, **kw)
        except ValueError as exc:  # a library too small for this split is recorded, not fatal
            out[kind] = None
            print(f"[splits] skipping {kind}: {exc}")
    return {k: v for k, v in out.items() if v is not None}


# --- persistence ------------------------------------------------------------------------
# JSON cannot hold tuples, so perturbations round-trip as lists of [gene, direction].


def _enc(perts: list[tuple]) -> list:
    return [[[int(g), int(d)] for g, d in p] for p in perts]


def _dec(raw: list) -> list[tuple]:
    return [tuple(sorted((int(g), int(d)) for g, d in p)) for p in raw]


def save_splits(splits: dict[str, Split], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {}
    for name, s in splits.items():
        d = asdict(s)
        for key in ("train", "cal", "test"):
            d[key] = _enc(d[key])
        blob[name] = d
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return path


def load_splits(path: str | Path) -> dict[str, Split]:
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, d in blob.items():
        s = Split(
            name=d["name"],
            train=_dec(d["train"]),
            cal=_dec(d["cal"]),
            test=_dec(d["test"]),
            train_contexts=d["train_contexts"],
            test_contexts=d["test_contexts"],
            meta=d.get("meta", {}),
        )
        s.check_no_leakage()
        out[name] = s
    return out
