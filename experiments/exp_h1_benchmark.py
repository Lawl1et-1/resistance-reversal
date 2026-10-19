"""H1: do deep perturbation models beat strong linear baselines, and does a
reversal-aware evaluation reorder the models relative to Pearson-delta?

Two separate claims, reported separately:

(a) **Does the deep model win?** Models are compared on the same splits and the same test
    perturbations, so comparisons are paired (Wilcoxon signed-rank, bootstrap CI on the
    mean paired difference, Cohen's d_z).

(b) **Does the evaluation change the ranking?** Each model is ranked twice -- once by
    Pearson-delta on top-DEG deltas, the conventional metric, and once by how well it
    ranks interventions by their true reversal. If the two orderings disagree, the
    conventional metric is not measuring what an intervention-selection task needs.

`shuffled_labels` is reported in every table. It is the model fitted with perturbation
labels permuted, so whatever score it reaches is obtainable without knowing which
perturbation was applied, and that is the real zero point for each metric.

Usage: python -m experiments.exp_h1_benchmark --config configs/smoke.yaml
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
    resistant_control_latents,
    write_json,
    write_table,
)
from src.eval.evaluate import evaluate_point, summarise
from src.eval.metrics import bootstrap_ci, paired_test, rank_correlation
from src.models.baselines import (
    Additive,
    ControlMean,
    LowRankLinear,
    OracleLatent,
    RidgeEmbedding,
    ShuffledLabels,
)
from src.search.objective import ReversalObjective
from src.simulator.grn import ground_truth_reversal

LINEAR = ("additive", "ridge_embedding", "low_rank_linear")
DEEP = ("pertflow", "pertflow_mlp")

# A Spearman correlation over three or four points takes only a handful of discrete values
# and carries essentially no information. Gene-holdout splits are inherently small -- the
# number of test perturbations is bounded by the number of perturbable genes -- so splits
# below this many test points get a NaN rank correlation and are excluded from the
# ranking-shift verdict, with the exclusion recorded rather than passed over.
MIN_PERTS_FOR_RANKING = 5


def make_models(cfg: dict, seed: int) -> list:
    pf = cfg["pertflow"]
    models = [
        ControlMean(),
        Additive(),
        RidgeEmbedding(),
        LowRankLinear(),
        ShuffledLabels(seed=seed),
    ]
    try:
        from src.models.pertflow import PertFlow

        common = dict(
            steps=pf["steps"],
            hidden=pf["hidden"],
            set_dim=pf["set_dim"],
            n_ensemble=pf["n_ensemble"],
            lr=pf["lr"],
            batch_size=pf["batch_size"],
            n_flow_steps=pf["n_flow_steps"],
            n_eval_cells=pf["n_eval_cells"],
            seed=seed,
        )
        models.append(PertFlow(**common))
        models.append(PertFlow(name="pertflow_mlp", mode="mlp", **common))
    except ImportError as exc:  # recorded, not silently skipped
        print(f"[h1] torch unavailable, skipping deep models: {exc}")
    models.append(OracleLatent())
    return models


def reversal_table(setup: Setup, split_name: str, models: list, cfg: dict) -> pd.DataFrame:
    """Predicted vs true reversal for every test perturbation, per model.

    True reversal comes from the simulator oracle (noise-free relaxation, exact by
    construction); predicted reversal uses the learned sensitivity readouts, so the two
    are on different scales and are compared by *ranking* and by error, never equated.
    """
    split = setup.splits[split_name]
    encoder = setup.encoders[split_name]
    sens = setup.sensitivity[split_name]
    truth = ground_truth_reversal(
        setup.grn, k=cfg["data"]["k"], n_cells=cfg["search"]["oracle_cells"], seed=setup.seed
    )
    z0 = resistant_control_latents(setup, split_name)
    testable = [p for p in split.test if tuple(sorted(p)) in truth]
    # Scoring reversal costs one model evaluation per (model, perturbation), and the
    # unseen-context split has every library perturbation in its test set. The cap keeps the
    # smoke profile inside its time budget; it is recorded in the output and the full
    # profile leaves it unset.
    cap = cfg["data"].get("max_reversal_perts")
    if cap and len(testable) > cap:
        stride = max(1, len(testable) // cap)
        testable = testable[::stride][:cap]
    if not testable:
        return pd.DataFrame()

    readouts = [(sens.primary.name, sens.primary), (sens.secondary.name, sens.secondary)]
    baselines = {name: float(np.mean(r(z0))) for name, r in readouts}
    context = split.test_contexts[0]

    rows = []
    for model in models:
        # The predicted population does not depend on the readout, so it is computed once
        # per (model, perturbation) and both readouts are applied to it. Scoring it per
        # readout instead doubles the cost, and each PertFlow prediction integrates an ODE
        # for every ensemble member over a cell population.
        obj = ReversalObjective(
            model=model,
            sensitivity=sens.primary,
            encoder=encoder,
            z0=z0,
            context=context,
            n_samples=cfg["pertflow"]["n_eval_cells"],
        )
        for pert in testable:
            z_pred = obj.predict_population(pert)
            for readout_name, readout in readouts:
                rows.append(
                    {
                        "model": getattr(model, "name", type(model).__name__),
                        "readout": readout_name,
                        "split": split_name,
                        "seed": setup.seed,
                        "pert": pert_label(pert),
                        "size": len(pert),
                        "reversal_pred": float(np.mean(readout(z_pred))) - baselines[readout_name],
                        "reversal_true": truth[tuple(sorted(pert))],
                    }
                )
    return pd.DataFrame(rows)


def run(cfg: dict, out_dir: Path) -> dict:
    point_frames, reversal_frames, checks = [], [], []
    for seed in cfg["seeds"]:
        print(f"[h1] seed {seed}")
        setup = build(cfg, seed, save_split_dir=out_dir / "splits")
        checks.append({"seed": seed, **setup.checks})
        for split_name, split in setup.splits.items():
            encoder = setup.encoders[split_name]
            models = make_models(cfg, seed)
            for model in models:
                model.fit(setup.data, split, encoder)
            for model in models:
                df = evaluate_point(model, setup.data, split, encoder)
                if not df.empty:
                    df["seed"] = seed
                    point_frames.append(df)
            rev = reversal_table(setup, split_name, models, cfg)
            if not rev.empty:
                reversal_frames.append(rev)
            print(f"  {split_name}: {len(split.test)} test perturbations")

    point = pd.concat(point_frames, ignore_index=True)
    reversal = pd.concat(reversal_frames, ignore_index=True) if reversal_frames else pd.DataFrame()

    write_table(point, out_dir / "h1_point_metrics_raw.csv")
    summary = summarise(point, by=("model", "split"))
    write_table(summary, out_dir / "h1_point_metrics.csv")

    result = {
        "network_checks": checks,
        "point_summary": summary.to_dict(orient="records"),
        "paired_tests": _paired(point),
    }
    if not reversal.empty:
        write_table(reversal, out_dir / "h1_reversal_raw.csv")
        rev_summary = _reversal_summary(reversal)
        write_table(rev_summary, out_dir / "h1_reversal_ranking.csv")
        result["reversal_ranking"] = rev_summary.to_dict(orient="records")
        result["readout_sanity"] = _readout_sanity(rev_summary)
        result["ranking_shift"] = _ranking_shift(summary, rev_summary)
        result["excluded_splits"] = sorted(
            set(rev_summary[~rev_summary.ranking_usable].split.tolist())
        )
    result["verdict"] = _verdict(result)
    write_json(result, out_dir / "h1_benchmark.json", cfg)
    return result


def _paired(point: pd.DataFrame) -> list[dict]:
    """Deep vs linear, paired at the level of independent replicates.

    The unit of independence is the **seed**, not the test perturbation. Every
    perturbation within a seed is scored by the same fitted model on the same network, so
    those rows are correlated; pairing on (seed, perturbation, context) treats 720
    correlated rows as 720 independent observations and reports p ~ 0 for differences of
    0.006, while the bootstrap CI on the same data spans zero. Aggregating to one mean per
    seed first gives n = 5 honest replicates, wide intervals, and far fewer significant
    results -- which is the correct answer.
    """
    out = []
    for split_name, g in point.groupby("split"):
        for deep in DEEP:
            for lin in LINEAR:
                for metric in ("pearson_delta", "mse_top20", "latent_mse"):
                    a = g[g.model == deep].groupby("seed")[metric].mean()
                    b = g[g.model == lin].groupby("seed")[metric].mean()
                    shared = a.index.intersection(b.index)
                    if len(shared) < 3:
                        continue
                    res = paired_test(a.loc[shared].to_numpy(), b.loc[shared].to_numpy())
                    out.append(
                        {
                            "split": split_name,
                            "metric": metric,
                            "deep": deep,
                            "linear": lin,
                            "unit": "seed",
                            **res,
                        }
                    )
    return out


def _reversal_summary(reversal: pd.DataFrame) -> pd.DataFrame:
    """How well each model ranks interventions by their true reversal."""
    rows = []
    for (model, readout, split_name), g in reversal.groupby(["model", "readout", "split"]):
        per_seed_rho, per_seed_mae, n_perts = [], [], []
        for _, gs in g.groupby("seed"):
            n_perts.append(gs.pert.nunique())
            per_seed_rho.append(
                rank_correlation(gs.reversal_pred.to_numpy(), gs.reversal_true.to_numpy())
                if gs.pert.nunique() >= MIN_PERTS_FOR_RANKING
                else float("nan")
            )
            per_seed_mae.append(
                float(np.mean(np.abs(gs.reversal_pred.to_numpy() - gs.reversal_true.to_numpy())))
            )
        rho, rho_lo, rho_hi = bootstrap_ci(np.array(per_seed_rho))
        mae, mae_lo, mae_hi = bootstrap_ci(np.array(per_seed_mae))
        rows.append(
            {
                "model": model,
                "readout": readout,
                "split": split_name,
                "n_seeds": len(per_seed_rho),
                "median_test_perts": float(np.median(n_perts)),
                "ranking_usable": bool(np.median(n_perts) >= MIN_PERTS_FOR_RANKING),
                "reversal_rank_rho": rho,
                "reversal_rank_rho_lo": rho_lo,
                "reversal_rank_rho_hi": rho_hi,
                "reversal_mae": mae,
                "reversal_mae_lo": mae_lo,
                "reversal_mae_hi": mae_hi,
            }
        )
    return pd.DataFrame(rows)


def _readout_sanity(rev_summary: pd.DataFrame) -> list[dict]:
    """Is each sensitivity readout fit to score reversal at all?

    `oracle_latent` reproduces the observed delta exactly in latent space, so under a
    well-behaved readout it should rank interventions at least as well as any model. When a
    real model beats the oracle, the readout -- not the model -- is the thing failing: it is
    being evaluated on predicted post-perturbation states that lie outside the unperturbed
    cells it was fitted on. This check is what turns that from an oddity someone might
    notice into a recorded property of the run.
    """
    out = []
    for (readout, split_name), g in rev_summary.groupby(["readout", "split"]):
        if not bool(g.ranking_usable.iloc[0]):
            continue
        g = g.dropna(subset=["reversal_rank_rho"])
        if "oracle_latent" not in set(g.model) or len(g) < 3:
            continue
        ranked = g.sort_values("reversal_rank_rho", ascending=False).model.tolist()
        oracle = g.set_index("model").loc["oracle_latent"]
        oracle_rank = ranked.index("oracle_latent") + 1
        oracle_rho = float(oracle["reversal_rank_rho"])
        oracle_hi = float(oracle.get("reversal_rank_rho_hi", oracle_rho))
        best_rho = float(g.reversal_rank_rho.max())

        # Flag only when the oracle is beaten by more than sampling noise. A bare rank
        # threshold fires whenever a model edges ahead by 0.02 on ten test perturbations
        # across two seeds, which is how this check previously condemned readouts that
        # were fine. The oracle counts as reliable if it ranks near the top *or* its
        # confidence interval still reaches the best model's point estimate.
        beaten_significantly = oracle_hi < best_rho
        out.append(
            {
                "readout": readout,
                "split": split_name,
                "oracle_rank": oracle_rank,
                "n_models": len(ranked),
                "oracle_rho": oracle_rho,
                "oracle_rho_hi": oracle_hi,
                "best_rho": best_rho,
                "best_model": ranked[0],
                "oracle_beaten_by": float(best_rho - oracle_rho),
                # The interval is bootstrapped over seeds, so with very few seeds a flag
                # means "not enough power to tell", not "readout broken". Recorded so a
                # reader can see which case they are looking at.
                "n_seeds": int(g.n_seeds.iloc[0]),
                "underpowered": bool(g.n_seeds.iloc[0] < 3),
                "readout_reliable": bool(oracle_rank <= 2 or not beaten_significantly),
            }
        )
    return out


def _ranking_shift(summary: pd.DataFrame, rev_summary: pd.DataFrame) -> list[dict]:
    """Does the reversal-aware ranking of models differ from the Pearson-delta ranking?"""
    out = []
    excluded = {"oracle_latent", "shuffled_labels", "control_mean"}
    usable = rev_summary[rev_summary.ranking_usable]
    for readout, rg in usable.groupby("readout"):
        for split_name in sorted(set(rg.split) & set(summary.split)):
            pear = summary[summary.split == split_name].dropna(subset=["pearson_delta"])
            rev = rg[rg.split == split_name].dropna(subset=["reversal_rank_rho"])
            models = sorted((set(pear.model) & set(rev.model)) - excluded)
            if len(models) < 3:
                continue
            p_order = pear.set_index("model").loc[models, "pearson_delta"]
            r_order = rev.set_index("model").loc[models, "reversal_rank_rho"]
            out.append(
                {
                    "split": split_name,
                    "readout": readout,
                    "models": models,
                    "pearson_best": str(p_order.idxmax()),
                    "reversal_best": str(r_order.idxmax()),
                    "same_winner": bool(p_order.idxmax() == r_order.idxmax()),
                    "rank_agreement_rho": rank_correlation(p_order.to_numpy(), r_order.to_numpy()),
                }
            )
    return out


def _verdict(result: dict) -> dict:
    """H1's verdict, stated from the numbers rather than asserted."""
    paired = [
        r for r in result.get("paired_tests", [])
        if r["metric"] == "pearson_delta" and np.isfinite(r.get("mean_diff", np.nan))
    ]
    # With 5 seeds the Wilcoxon signed-rank test cannot produce p < 0.0625 whatever the
    # data, so "significant at p < 0.05" is unreachable by construction and claiming it
    # would be meaningless. A difference counts as decisive only when the bootstrap CI on
    # the mean paired difference excludes zero *and* the effect is large (|d_z| > 1),
    # which is a descriptive claim about magnitude and consistency rather than a p-value.
    LARGE_EFFECT = 1.0
    decisive = lambda r: abs(r.get("cohens_dz") or 0.0) > LARGE_EFFECT
    deep_wins = [r for r in paired if r["ci_lo"] > 0 and decisive(r)]
    deep_losses = [r for r in paired if r["ci_hi"] < 0 and decisive(r)]
    shifts = result.get("ranking_shift", [])
    disagreements = [s for s in shifts if not s["same_winner"]]
    sanity = result.get("readout_sanity", [])
    unreliable = [s for s in sanity if not s["readout_reliable"]]
    return {
        "n_paired_comparisons": len(paired),
        "pairing_unit": "seed",
        "min_achievable_p_at_n5": 0.0625,
        "decisive_criterion": "bootstrap CI on mean paired difference excludes 0 and |d_z| > 1",
        "deep_beats_linear_significantly": len(deep_wins),
        "linear_beats_deep_significantly": len(deep_losses),
        "largest_deep_deficit": (
            min((r["mean_diff"] for r in paired), default=None)
        ),
        "largest_deep_advantage": (
            max((r["mean_diff"] for r in paired), default=None)
        ),
        # "Reliably beats" means wins decisively and does not decisively lose. Winning 4 of
        # 20 comparisons while losing 3 -- including one where the deficit is six times the
        # largest advantage -- is not reliability, so a bare win/loss count is not enough.
        "h1a_deep_fails_to_reliably_beat_linear": not (
            len(deep_wins) > 0 and len(deep_losses) == 0
        ),
        "n_ranking_comparisons": len(shifts),
        "n_ranking_winner_changes": len(disagreements),
        "h1b_evaluation_changes_ranking": len(disagreements) > 0,
        "splits_excluded_for_small_test_sets": result.get("excluded_splits", []),
        "unreliable_readouts": [
            f"{s['readout']}/{s['split']} (oracle ranked {s['oracle_rank']}/{s['n_models']},"
            f" beaten by {s['oracle_beaten_by']:.3f}"
            + (f", UNDERPOWERED: only {s['n_seeds']} seeds)" if s["underpowered"] else ")")
            for s in unreliable
        ],
        "note": (
            "Comparisons are paired at the seed level, because perturbations within a seed "
            "share one fitted model and are not independent. With 5 seeds no comparison can "
            "reach p < 0.05, so a difference is called decisive only when its CI excludes "
            "zero and the effect size is large; everything else is reported as no reliable "
            "difference rather than as a null result. Rank correlations are "
            f"computed only where a split has at least {MIN_PERTS_FOR_RANKING} test "
            "perturbations. Any readout listed under unreliable_readouts is one where the "
            "latent oracle did not rank in the top two, meaning that readout cannot score "
            "reversal faithfully and its ranking numbers should not be used as evidence."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else RESULTS / cfg["profile"]
    result = run(cfg, out_dir)
    v = result["verdict"]
    print("\n--- H1 verdict ---")
    print(f"  paired comparisons:        {v['n_paired_comparisons']}")
    print(f"  deep significantly better: {v['deep_beats_linear_significantly']}")
    print(f"  deep significantly worse:  {v['linear_beats_deep_significantly']}")
    print(f"  H1a (deep does not reliably win): {v['h1a_deep_fails_to_reliably_beat_linear']}")
    print(f"  H1b (evaluation changes ranking): {v['h1b_evaluation_changes_ranking']}"
          f" ({v['n_ranking_winner_changes']}/{v['n_ranking_comparisons']} winner changes)")
    if v["splits_excluded_for_small_test_sets"]:
        print(f"  excluded (test set < {MIN_PERTS_FOR_RANKING} perturbations): "
              f"{', '.join(v['splits_excluded_for_small_test_sets'])}")
    for line in v["unreliable_readouts"]:
        print(f"  UNRELIABLE READOUT: {line}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
