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
    """Deep vs linear on identical test points, within each split."""
    out = []
    for split_name, g in point.groupby("split"):
        pivot_keys = ["seed", "pert", "context"]
        for deep in DEEP:
            for lin in LINEAR:
                a = g[g.model == deep].set_index(pivot_keys)
                b = g[g.model == lin].set_index(pivot_keys)
                shared = a.index.intersection(b.index)
                if len(shared) < 3:
                    continue
                for metric in ("pearson_delta", "mse_top20", "latent_mse"):
                    res = paired_test(
                        a.loc[shared, metric].to_numpy(), b.loc[shared, metric].to_numpy()
                    )
                    out.append(
                        {"split": split_name, "metric": metric, "deep": deep, "linear": lin, **res}
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
        oracle_rank = ranked.index("oracle_latent") + 1
        oracle_rho = float(g.set_index("model").loc["oracle_latent", "reversal_rank_rho"])
        best_rho = float(g.reversal_rank_rho.max())
        out.append(
            {
                "readout": readout,
                "split": split_name,
                "oracle_rank": oracle_rank,
                "n_models": len(ranked),
                "oracle_rho": oracle_rho,
                "best_rho": best_rho,
                "best_model": ranked[0],
                "oracle_beaten_by": float(best_rho - oracle_rho),
                "readout_reliable": bool(oracle_rank <= 2),
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
    paired = [r for r in result.get("paired_tests", []) if r["metric"] == "pearson_delta"]
    deep_wins = [r for r in paired if r["ci_lo"] > 0]
    deep_losses = [r for r in paired if r["ci_hi"] < 0]
    shifts = result.get("ranking_shift", [])
    disagreements = [s for s in shifts if not s["same_winner"]]
    sanity = result.get("readout_sanity", [])
    unreliable = [s for s in sanity if not s["readout_reliable"]]
    return {
        "n_paired_comparisons": len(paired),
        "deep_beats_linear_significantly": len(deep_wins),
        "linear_beats_deep_significantly": len(deep_losses),
        "h1a_deep_fails_to_reliably_beat_linear": len(deep_wins) == 0 or len(deep_losses) >= len(deep_wins),
        "n_ranking_comparisons": len(shifts),
        "n_ranking_winner_changes": len(disagreements),
        "h1b_evaluation_changes_ranking": len(disagreements) > 0,
        "splits_excluded_for_small_test_sets": result.get("excluded_splits", []),
        "unreliable_readouts": [
            f"{s['readout']}/{s['split']} (oracle ranked {s['oracle_rank']}/{s['n_models']},"
            f" beaten by {s['oracle_beaten_by']:.3f})"
            for s in unreliable
        ],
        "note": (
            "Verdicts are read from paired CIs on identical test points. A CI on the mean "
            "paired difference that excludes zero counts as a significant win or loss; "
            "everything else is reported as no reliable difference. Rank correlations are "
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
