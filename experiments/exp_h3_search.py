"""H3: does uncertainty-penalised search recommend interventions with higher
held-out-verified reversal than uncertainty-blind search?

The mechanism under test is the optimiser's curse. A search over many candidate sets does
not return the best set; it returns the set with the highest *predicted* score, and that
selection is biased toward sets whose score the model overestimates. Two sets look identical
from the inside -- one genuinely good, one optimistically mis-scored -- so maximising a noisy
prediction systematically over-selects the second kind. Penalising uncertainty gives up some
predicted reversal in exchange for recommendations more likely to hold up.

What is measured:

- **verified reversal** of each recommendation, scored by the simulator oracle. This is the
  only number that settles H3; predicted reversal cannot, since that is the quantity being
  gamed.
- **optimism gap** = predicted minus verified reversal of the selected set. The curse
  predicts this is positive, and larger at lambda = 0.
- **optimisation regret** against exhaustive search, which separates "the search missed the
  model's own optimum" from "the model's objective is wrong".

Comparisons across lambda are paired on the same seed, network and fitted model, so the only
difference is the penalty.

Usage: python -m experiments.exp_h3_search --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import (
    RESULTS,
    Setup,
    build,
    load_config,
    pert_label,
    resistant_control_latents,
    write_json,
    write_table,
)
from src.data.dataset import CONTROL
from src.eval.metrics import bootstrap_ci, paired_test
from src.search.objective import ReversalObjective
from src.search.search import STRATEGIES, exhaustive
from src.simulator.grn import ground_truth_reversal
from src.uncertainty.conformal import abstention_threshold

# Exhaustive search is the reference, but it costs one model evaluation per set and a
# PertFlow evaluation integrates an ODE per ensemble member. Above this many sets the
# reference falls back to wide beam search, and the substitution is recorded in the output.
EXHAUSTIVE_MAX_SETS = 400


def n_sets(n_atoms: int, k: int) -> int:
    return sum(len(list(combinations(range(n_atoms), s))) for s in range(1, k + 1))


def build_objective(setup: Setup, split_name: str, model, lam: float, n_samples: int,
                    abstain_above: float | None = None) -> ReversalObjective:
    split = setup.splits[split_name]
    return ReversalObjective(
        model=model,
        sensitivity=setup.sensitivity[split_name].secondary,  # the reliable readout; see LOG
        encoder=setup.encoders[split_name],
        z0=resistant_control_latents(setup, split_name),
        context=split.test_contexts[0],
        lam=lam,
        n_samples=n_samples,
        sensitive_reference=setup.sensitive_reference[split_name],
        abstain_above=abstain_above,
    )


def calibration_spreads(setup: Setup, split_name: str, model) -> np.ndarray:
    """Ensemble spreads on calibration perturbations, for the abstention threshold."""
    split = setup.splits[split_name]
    out = []
    for pert in split.cal:
        for context in split.train_contexts:
            if len(setup.data.cells(pert, context)):
                out.append(model.epistemic_uncertainty(pert, context))
    return np.array(out) if out else np.array([0.0])


def run(cfg: dict, out_dir: Path) -> dict:
    try:
        from src.models.pertflow import PertFlow
    except ImportError as exc:
        raise SystemExit(f"H3 needs torch for the ensemble: {exc}")

    k = cfg["search"]["k"]
    lambdas = cfg["search"]["lambdas"]
    strategies = cfg["search"]["strategies"]
    split_name = "unseen_combo_seen_genes"  # composition split: where search is meaningful
    rows, notes = [], []

    for seed in cfg["seeds"]:
        print(f"[h3] seed {seed}")
        setup = build(cfg, seed)
        if split_name not in setup.splits:
            notes.append(f"seed {seed}: split {split_name} unavailable")
            continue
        split = setup.splits[split_name]
        pf = cfg["pertflow"]
        model = PertFlow(
            steps=pf["steps"], hidden=pf["hidden"], set_dim=pf["set_dim"],
            n_ensemble=pf["n_ensemble"], lr=pf["lr"], batch_size=pf["batch_size"],
            n_flow_steps=pf["n_flow_steps"], n_eval_cells=pf["n_eval_cells"], seed=seed,
        ).fit(setup.data, split, setup.encoders[split_name])

        atoms = setup.atoms()
        truth = ground_truth_reversal(
            setup.grn, k=k, n_cells=cfg["search"]["oracle_cells"], seed=seed
        )
        truth_fn = lambda p: float(truth.get(tuple(sorted(p)), float("nan")))
        oracle_best = max((p for p in truth if p), key=lambda p: truth[p])
        oracle_value = truth[oracle_best]

        # One abstention threshold per (seed, model), calibrated on held-out perturbations
        # and then held fixed across every lambda and strategy.
        spreads = calibration_spreads(setup, split_name, model)
        threshold = abstention_threshold(spreads, keep_frac=0.9)

        use_exhaustive = n_sets(len(atoms), k) <= EXHAUSTIVE_MAX_SETS
        if not use_exhaustive:
            notes.append(
                f"seed {seed}: {n_sets(len(atoms), k)} sets exceeds {EXHAUSTIVE_MAX_SETS}; "
                "reference is beam(width=10) rather than exhaustive"
            )

        # One objective holds the score cache for the whole lambda sweep; the per-lambda
        # views reuse it, so each candidate set is evaluated by the model exactly once.
        shared = build_objective(setup, split_name, model, lambdas[0], pf["n_eval_cells"])
        for lam in lambdas:
            obj = shared if lam == lambdas[0] else shared.with_penalty(lam)
            reference = (
                exhaustive(obj, atoms, k=k)
                if use_exhaustive
                else STRATEGIES["beam"](obj, atoms, k=k, beam_width=10)
            )
            for strategy in strategies:
                if strategy == "exhaustive" and not use_exhaustive:
                    continue
                res = STRATEGIES[strategy](obj, atoms, k=k)
                sel = res.best_pert
                rec = obj.score(sel)
                verified = truth_fn(sel)
                rows.append(
                    {
                        "seed": seed,
                        "split": split_name,
                        "lambda": lam,
                        "strategy": res.strategy,
                        "abstention": False,
                        "selected": pert_label(sel),
                        "selected_size": len(sel),
                        "reversal_pred": rec["reversal"],
                        "reversal_verified": verified,
                        "optimism_gap": rec["reversal"] - verified,
                        "uncertainty": rec["uncertainty"],
                        "oracle_best": pert_label(oracle_best),
                        "oracle_value": oracle_value,
                        "regret_vs_oracle": oracle_value - verified,
                        "regret_vs_reference": truth_fn(reference.best_pert) - verified,
                        "n_evaluations": res.n_evaluations,
                        "top5_mean_verified": float(
                            np.nanmean([truth_fn(p) for p in res.top(5)])
                        ),
                    }
                )

            # Same search, now allowed to abstain on high-uncertainty sets.
            obj_abs = shared.with_penalty(lam, abstain_above=threshold)
            res = STRATEGIES["greedy"](obj_abs, atoms, k=k)
            sel = res.best_pert
            rec = obj_abs.score(sel)
            verified = truth_fn(sel)
            rows.append(
                {
                    "seed": seed, "split": split_name, "lambda": lam, "strategy": "greedy",
                    "abstention": True, "selected": pert_label(sel), "selected_size": len(sel),
                    "reversal_pred": rec["reversal"], "reversal_verified": verified,
                    "optimism_gap": rec["reversal"] - verified,
                    "uncertainty": rec["uncertainty"], "oracle_best": pert_label(oracle_best),
                    "oracle_value": oracle_value, "regret_vs_oracle": oracle_value - verified,
                    "regret_vs_reference": float("nan"), "n_evaluations": res.n_evaluations,
                    "top5_mean_verified": float(np.nanmean([truth_fn(p) for p in res.top(5)])),
                }
            )
            print(f"  lambda={lam}: verified={verified:.3f} oracle={oracle_value:.3f}")

    df = pd.DataFrame(rows)
    write_table(df, out_dir / "h3_search_raw.csv")
    summary = _summarise(df)
    write_table(pd.DataFrame(summary), out_dir / "h3_search.csv")
    result = {
        "per_run": df.to_dict(orient="records"),
        "summary": summary,
        "notes": notes,
        "verdict": _verdict(df),
    }
    write_json(result, out_dir / "h3_search.json", cfg)
    return result


def _summarise(df: pd.DataFrame) -> list[dict]:
    rows = []
    if df.empty:
        return rows
    for (lam, strategy, abst), g in df.groupby(["lambda", "strategy", "abstention"]):
        row = {"lambda": lam, "strategy": strategy, "abstention": bool(abst), "n_seeds": len(g)}
        for col in ("reversal_verified", "reversal_pred", "optimism_gap",
                    "regret_vs_oracle", "regret_vs_reference", "top5_mean_verified",
                    "n_evaluations"):
            point, lo, hi = bootstrap_ci(g[col].to_numpy())
            row[col], row[f"{col}_lo"], row[f"{col}_hi"] = point, lo, hi
        rows.append(row)
    return rows


def _verdict(df: pd.DataFrame) -> dict:
    """H3's verdict: paired comparison of lambda > 0 against lambda = 0."""
    if df.empty:
        return {"error": "no runs"}
    blind = df[(df["lambda"] == 0) & (~df.abstention) & (df.strategy == "greedy")]
    out = {"comparisons": [], "optimism_gap_at_lambda0": None}
    if not blind.empty:
        point, lo, hi = bootstrap_ci(blind.optimism_gap.to_numpy())
        out["optimism_gap_at_lambda0"] = {"mean": point, "ci_lo": lo, "ci_hi": hi}
        out["optimisers_curse_present"] = bool(lo > 0)

    for lam in sorted(set(df["lambda"]) - {0.0}):
        pen = df[(df["lambda"] == lam) & (~df.abstention) & (df.strategy == "greedy")]
        shared = sorted(set(blind.seed) & set(pen.seed))
        if len(shared) < 2:
            continue
        a = pen.set_index("seed").loc[shared, "reversal_verified"].to_numpy()
        b = blind.set_index("seed").loc[shared, "reversal_verified"].to_numpy()
        res = paired_test(a, b)
        out["comparisons"].append({"lambda": lam, "n_seeds": len(shared), **res})

    # If the search already reaches the oracle optimum, there is no headroom for any
    # penalty to improve the recommendation and H3 cannot be tested in this regime. That
    # is a statement about the regime, not evidence against H3, so it is recorded rather
    # than reported as a negative result.
    greedy = df[(~df.abstention) & (df.strategy == "greedy")]
    # Saturation must be judged at lambda = 0, because that is the baseline H3 proposes to
    # improve on. Averaging regret across every lambda mixes in the large-lambda runs,
    # where regret is high precisely *because* the penalty over-corrected, and so hides the
    # fact that the uncertainty-blind search had no headroom to begin with. Regret below 1%
    # of the oracle leaves nothing for a penalty to recover.
    blind_greedy = greedy[greedy["lambda"] == 0]
    blind_regret = float(blind_greedy.regret_vs_oracle.mean()) if len(blind_greedy) else float("nan")
    mean_regret = float(greedy.regret_vs_oracle.mean()) if len(greedy) else float("nan")
    mean_oracle = float(greedy.oracle_value.mean()) if len(greedy) else float("nan")
    saturated = bool(
        len(blind_greedy) > 0
        and np.isfinite(blind_regret)
        and blind_regret <= 0.01 * max(mean_oracle, 1e-9)
    )
    out["blind_regret_vs_oracle"] = blind_regret
    out["search_saturated_at_oracle"] = saturated
    out["saturation_threshold"] = 0.01 * mean_oracle if np.isfinite(mean_oracle) else None
    out["mean_regret_vs_oracle"] = mean_regret

    helps = [c for c in out["comparisons"] if c["ci_lo"] > 0]
    hurts = [c for c in out["comparisons"] if c["ci_hi"] < 0]
    out.update(
        {
            "n_lambdas_better_than_blind": len(helps),
            "n_lambdas_worse_than_blind": len(hurts),
            "h3_testable": not saturated,
            "h3_penalty_improves_verified_reversal": (
                None if saturated else (len(helps) > 0 and not hurts)
            ),
            "note": (
                "Verified reversal is scored by the simulator oracle, never by the model "
                "whose predictions the search maximised. Comparisons are paired by seed, so "
                "network, data and fitted model are identical and only lambda differs. "
                "When search_saturated_at_oracle is true the search already attains the "
                "best possible set, leaving no headroom for the penalty, and H3 is "
                "untestable in that regime rather than refuted. A negative optimism gap "
                "means the model underestimates reversal, which is the expected effect of "
                "its magnitude shrinkage and masks the optimiser's curse."
            ),
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else RESULTS / cfg["profile"]
    result = run(cfg, out_dir)
    v = result["verdict"]
    print("\n--- H3 verdict ---")
    if v.get("optimism_gap_at_lambda0"):
        g = v["optimism_gap_at_lambda0"]
        print(f"  optimism gap at lambda=0: {g['mean']:.3f} [{g['ci_lo']:.3f}, {g['ci_hi']:.3f}]")
        print(f"  optimiser's curse present: {v.get('optimisers_curse_present')}")
    for c in v.get("comparisons", []):
        print(f"  lambda={c['lambda']}: verified reversal vs blind "
              f"{c['mean_diff']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] "
              f"(n={c['n']}, p={c['p_value']:.3g})")
    if v.get("search_saturated_at_oracle"):
        print("  SEARCH SATURATED: greedy already attains the oracle optimum, so there is")
        print("  no headroom for the penalty. H3 is UNTESTABLE in this regime, not refuted.")
    print(f"  mean regret vs oracle: {v.get('mean_regret_vs_oracle', float('nan')):.4f}")
    print(f"  H3 (penalty improves verified reversal): "
          f"{v.get('h3_penalty_improves_verified_reversal')}")
    for note in result["notes"]:
        print(f"  note: {note}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
