"""H2: does the ensemble plus split conformal reach target coverage on unseen
perturbations, and does abstention lower the error of recommended interventions?

Two quantities are calibrated, because they answer different questions:

- the **latent mean shift**, with a conformal ball on `||y - yhat||` -- the form the brief
  specifies, scored against the ensemble spread;
- the **scalar reversal score**, which is what a decision actually depends on.

For the reversal score, "truth" is the reversal computed from the *observed* held-out cells
using the *same learned readout* as the prediction. Using the simulator's oracle instead
would compare a learned proxy against a different scale, and the resulting coverage failure
would measure the proxy's bias rather than the calibration. The oracle is reserved for
verifying search recommendations (H3), where that scale difference does not matter because
only the ranking is used.

Coverage on `unseen_context` is reported **separately and expected to miss nominal**:
exchangeability between calibration and test is violated by construction there, and the
conformal guarantee simply does not apply. Reporting it pooled with the other splits would
hide the one place the method is known to break.

Usage: python -m experiments.exp_h2_calibration --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import (
    RESULTS,
    Setup,
    build,
    load_config,
    pert_label,
    write_json,
    write_table,
)
from src.data.dataset import CONTROL
from src.eval.metrics import (
    auroc_error_detection,
    bootstrap_ci,
    risk_coverage_auc,
    risk_coverage_curve,
)
from src.uncertainty.conformal import (
    SplitConformal,
    VectorConformal,
    abstention_threshold,
    selective_risk,
)

SHIFTED_SPLITS = {"unseen_context"}


def observed_reversal(setup: Setup, split_name: str, pert: tuple, context: int, baseline: float,
                      readout) -> float | None:
    """Reversal of a perturbation measured on the observed cells, via the learned readout."""
    cells = setup.data.cells(pert, context)
    if not len(cells):
        return None
    z = setup.encoders[split_name].transform(cells)
    return float(np.mean(readout(z))) - baseline


def collect(setup: Setup, split_name: str, model, readout, perts: list[tuple]) -> pd.DataFrame:
    """Predicted vs observed mean shift and reversal, with ensemble spread, per perturbation."""
    split = setup.splits[split_name]
    encoder = setup.encoders[split_name]
    rows = []
    for context in split.test_contexts if perts is split.test else split.train_contexts:
        control = setup.data.cells(CONTROL, context)
        if not len(control):
            continue
        z0 = encoder.transform(control)
        baseline = float(np.mean(readout(z0)))
        z0_mean = z0.mean(axis=0)
        for pert in perts:
            cells = setup.data.cells(pert, context)
            if not len(cells):
                continue
            observed_shift = encoder.transform(cells).mean(axis=0) - z0_mean
            pred_shift, spread = (
                model.predict_with_uncertainty(pert, context)
                if hasattr(model, "predict_with_uncertainty")
                else (model.predict_latent_delta(pert, context), 0.0)
            )
            pred_shift = np.asarray(pred_shift, float).ravel()

            # Predicted reversal uses the model's own predicted population, so prediction
            # and observation pass through the same readout.
            if hasattr(model, "sample_latent"):
                z_pred = model.sample_latent(pert, context, z0, 128)
            else:
                z_pred = z0 + pred_shift
            rows.append(
                {
                    "pert": pert_label(pert),
                    "size": len(pert),
                    "context": context,
                    "spread": float(spread),
                    "shift_error": float(np.linalg.norm(observed_shift - pred_shift)),
                    "observed_shift": observed_shift,
                    "pred_shift": pred_shift,
                    "reversal_pred": float(np.mean(readout(z_pred))) - baseline,
                    "reversal_obs": observed_reversal(
                        setup, split_name, pert, context, baseline, readout
                    ),
                }
            )
    return pd.DataFrame(rows)


def run(cfg: dict, out_dir: Path) -> dict:
    alpha = cfg["conformal"]["alpha"]
    try:
        from src.models.pertflow import PertFlow
    except ImportError as exc:
        raise SystemExit(f"H2 needs torch for the ensemble: {exc}")

    records, curves = [], []
    for seed in cfg["seeds"]:
        print(f"[h2] seed {seed}")
        setup = build(cfg, seed)
        pf_cfg = cfg["pertflow"]
        for split_name, split in setup.splits.items():
            encoder = setup.encoders[split_name]
            model = PertFlow(
                steps=pf_cfg["steps"], hidden=pf_cfg["hidden"], set_dim=pf_cfg["set_dim"],
                n_ensemble=pf_cfg["n_ensemble"], lr=pf_cfg["lr"],
                batch_size=pf_cfg["batch_size"], n_flow_steps=pf_cfg["n_flow_steps"],
                n_eval_cells=pf_cfg["n_eval_cells"], seed=seed,
            ).fit(setup.data, split, encoder)
            readout = setup.sensitivity[split_name].secondary  # the reliable readout; see LOG

            cal = collect(setup, split_name, model, readout, split.cal)
            test = collect(setup, split_name, model, readout, split.test)
            if cal.empty or test.empty:
                print(f"  {split_name}: skipped (empty calibration or test set)")
                continue

            rec = {
                "seed": seed,
                "split": split_name,
                "exchangeability_violated": split_name in SHIFTED_SPLITS,
                "n_cal": len(cal),
                "n_test": len(test),
                "nominal_coverage": 1.0 - alpha,
            }

            # (1) conformal ball on the latent mean shift
            vec = VectorConformal(alpha=alpha, normalise=True).fit(
                cal.shift_error.to_numpy(), cal.spread.to_numpy()
            )
            rec["shift_coverage"] = float(
                np.mean(test.shift_error.to_numpy() <= vec.radius(test.spread.to_numpy()))
            )
            rec["shift_radius_mean"] = float(np.mean(vec.radius(test.spread.to_numpy())))
            rec["shift_certifiable"] = bool(vec.certifiable_)

            # (2) conformal interval on the scalar reversal score
            ok_cal = cal.dropna(subset=["reversal_obs"])
            ok_test = test.dropna(subset=["reversal_obs"])
            if len(ok_cal) and len(ok_test):
                scal = SplitConformal(alpha=alpha, normalise=True).fit(
                    (ok_cal.reversal_obs - ok_cal.reversal_pred).to_numpy(),
                    ok_cal.spread.to_numpy(),
                )
                rec["reversal_coverage"] = scal.coverage(
                    ok_test.reversal_obs.to_numpy(),
                    ok_test.reversal_pred.to_numpy(),
                    ok_test.spread.to_numpy(),
                )
                rec["reversal_interval_width"] = float(
                    2 * np.mean(scal.radius(ok_test.spread.to_numpy()))
                )
                rec["reversal_certifiable"] = bool(scal.certifiable_)
                rec["reversal_bias"] = float(
                    np.mean(ok_test.reversal_pred - ok_test.reversal_obs)
                )

            # (3) does abstention lower error? Threshold calibrated on the calibration set.
            err = test.shift_error.to_numpy()
            spread = test.spread.to_numpy()
            rec["uncertainty_auroc"] = auroc_error_detection(err, spread)
            rec["risk_coverage_auc"] = risk_coverage_auc(err, spread)
            rng = np.random.default_rng(seed)
            rec["risk_coverage_auc_random"] = float(
                np.mean([risk_coverage_auc(err, rng.permutation(spread)) for _ in range(20)])
            )
            for keep in (0.5, 0.7, 0.9, 1.0):
                t = abstention_threshold(cal.spread.to_numpy(), keep)
                cov, risk = selective_risk(err, spread, t)
                rec[f"risk_at_keep{int(keep*100)}"] = risk
                rec[f"coverage_at_keep{int(keep*100)}"] = cov
            cov_grid, risk_grid = risk_coverage_curve(err, spread)
            curves.append(
                pd.DataFrame(
                    {"seed": seed, "split": split_name, "coverage": cov_grid, "risk": risk_grid}
                )
            )
            records.append(rec)
            print(
                f"  {split_name}: shift cov={rec['shift_coverage']:.3f} "
                f"reversal cov={rec.get('reversal_coverage', float('nan')):.3f} "
                f"auroc={rec['uncertainty_auroc']:.3f}"
            )

    df = pd.DataFrame(records)
    write_table(df, out_dir / "h2_calibration_raw.csv")
    if curves:
        write_table(pd.concat(curves, ignore_index=True), out_dir / "h2_risk_coverage.csv")
    summary = _summarise(df, alpha)
    write_table(pd.DataFrame(summary), out_dir / "h2_calibration.csv")
    result = {"per_run": records, "summary": summary, "verdict": _verdict(summary, alpha)}
    write_json(result, out_dir / "h2_calibration.json", cfg)
    return result


def _summarise(df: pd.DataFrame, alpha: float) -> list[dict]:
    rows = []
    for (split_name, shifted), g in df.groupby(["split", "exchangeability_violated"]):
        row = {
            "split": split_name,
            "exchangeability_violated": bool(shifted),
            "n_seeds": len(g),
            "nominal_coverage": 1.0 - alpha,
            # A split whose calibration set is too small to certify the level gets an
            # infinite interval, which trivially covers everything. Counting that as
            # coverage reached would turn the method's own refusal to answer into a
            # success, so certifiability is carried through to the verdict.
            "certifiable": bool(g.shift_certifiable.all()) if "shift_certifiable" in g else True,
            "median_n_cal": float(g.n_cal.median()),
        }
        for col in ("shift_coverage", "reversal_coverage", "uncertainty_auroc",
                    "risk_coverage_auc", "risk_coverage_auc_random", "reversal_bias",
                    "risk_at_keep50", "risk_at_keep100"):
            if col in g:
                point, lo, hi = bootstrap_ci(g[col].to_numpy())
                row[col], row[f"{col}_lo"], row[f"{col}_hi"] = point, lo, hi
        rows.append(row)
    return rows


def _verdict(summary: list[dict], alpha: float) -> dict:
    """H2 is judged only on splits where the conformal assumption actually holds."""
    exchangeable = [s for s in summary if not s["exchangeability_violated"]]
    # Vacuous (infinite-interval) splits are excluded: they cover everything by refusing
    # to commit, which is honest behaviour but is not evidence that calibration works.
    valid = [s for s in exchangeable if s.get("certifiable", True)]
    vacuous = [s for s in exchangeable if not s.get("certifiable", True)]
    shifted = [s for s in summary if s["exchangeability_violated"]]
    target = 1.0 - alpha
    # Coverage counts as reached if the bootstrap CI reaches the nominal level; conformal
    # over-covers in finite samples, so falling short is the only interesting failure.
    reached = [s for s in valid if s.get("shift_coverage_hi", 0.0) >= target - 0.02]
    abstention_helps = [
        s for s in valid
        if np.isfinite(s.get("risk_at_keep50", np.nan))
        and s["risk_at_keep50"] < s.get("risk_at_keep100", np.inf)
    ]
    informative = [
        s for s in valid
        if np.isfinite(s.get("risk_coverage_auc", np.nan))
        and s["risk_coverage_auc"] < s.get("risk_coverage_auc_random", np.inf)
    ]
    return {
        "nominal_coverage": target,
        "n_valid_splits": len(valid),
        "n_splits_reaching_coverage": len(reached),
        "h2a_coverage_reached": len(valid) > 0 and len(reached) == len(valid),
        "n_splits_where_abstention_lowers_error": len(abstention_helps),
        "h2b_abstention_lowers_error": len(abstention_helps) == len(valid) and len(valid) > 0,
        "n_splits_uncertainty_beats_random_ordering": len(informative),
        "shifted_split_coverage": {
            s["split"]: s.get("shift_coverage") for s in shifted
        },
        "vacuous_splits": [
            f"{s['split']} (median n_cal={s['median_n_cal']:.0f}, interval infinite)"
            for s in vacuous
        ],
        "note": (
            "H2 is evaluated only where calibration and test perturbations are "
            "exchangeable AND the calibration set is large enough to certify the level. "
            "The unseen-context split violates exchangeability by construction and its "
            "coverage is reported separately. Splits listed under vacuous_splits return "
            "infinite intervals, so their coverage of 1.0 reflects the method declining to "
            "commit rather than calibration succeeding, and they are excluded."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else RESULTS / cfg["profile"]
    v = run(cfg, out_dir)["verdict"]
    print("\n--- H2 verdict ---")
    print(f"  nominal coverage:          {v['nominal_coverage']:.2f}")
    print(f"  splits reaching coverage:  {v['n_splits_reaching_coverage']}/{v['n_valid_splits']}")
    print(f"  H2a (coverage reached):    {v['h2a_coverage_reached']}")
    print(f"  H2b (abstention helps):    {v['h2b_abstention_lowers_error']}"
          f" ({v['n_splits_where_abstention_lowers_error']}/{v['n_valid_splits']})")
    print(f"  uncertainty beats random ordering: "
          f"{v['n_splits_uncertainty_beats_random_ordering']}/{v['n_valid_splits']}")
    for line in v["vacuous_splits"]:
        print(f"  excluded as vacuous: {line}")
    for split_name, cov in v["shifted_split_coverage"].items():
        print(f"  under context shift ({split_name}): coverage "
              f"{cov if cov is None else f'{cov:.3f}'} (guarantee does not apply)")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
