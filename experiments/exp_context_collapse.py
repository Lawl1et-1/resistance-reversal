"""Why does PertFlow collapse on unseen cell context? Two checks, on that split only.

The full run found PertFlow at 0.229 Pearson-delta on unseen context where ridge reaches
0.834 and PertFlow's own MLP ablation reaches 0.830. Two explanations were left open, and
the headline H1 claim depends on which is right:

**(a) Compounding.** The flow applies its context conditioning at every ODE integration
step, so an untrained context embedding compounds; the regressor applies it once. This
predicts that *reducing the number of integration steps at inference time should recover
performance*, with a single Euler step behaving like the regressor. `n_flow_steps` is an
inference parameter, so this is tested on an already-fitted model at no training cost.

**(b) Undertraining.** The full profile ran 4 000 steps for compute reasons where 12 000
was measurably better. If the collapse is a training artefact it should shrink or disappear
when retrained. This costs a retrain, so it runs on two seeds only.

The two make opposite predictions, and (a) is nearly free, so both are worth running before
the H1 wording is settled either way.

Usage: python -m experiments.exp_context_collapse --config configs/full.yaml --seeds 0,1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from experiments.common import RESULTS, build, load_config, write_json, write_table
from src.eval.evaluate import evaluate_point
from src.models.baselines import RidgeEmbedding

SPLIT = "unseen_context"
FLOW_STEPS = (1, 2, 5, 10, 20)


def run(cfg: dict, seeds: list[int], train_steps: list[int], out_dir: Path) -> dict:
    from src.models.pertflow import PertFlow

    pf = cfg["pertflow"]
    rows = []
    for seed in seeds:
        print(f"[ctx] seed {seed}")
        setup = build(cfg, seed)
        split = setup.splits[SPLIT]
        encoder = setup.encoders[SPLIT]

        ridge = RidgeEmbedding().fit(setup.data, split, encoder)
        rows.append(
            {"seed": seed, "model": "ridge_embedding", "train_steps": None, "flow_steps": None,
             "pearson_delta": evaluate_point(ridge, setup.data, split, encoder).pearson_delta.mean()}
        )

        common = dict(hidden=pf["hidden"], set_dim=pf["set_dim"], n_ensemble=pf["n_ensemble"],
                      lr=pf["lr"], batch_size=pf["batch_size"], n_eval_cells=pf["n_eval_cells"],
                      seed=seed)
        mlp = PertFlow(name="pertflow_mlp", mode="mlp", steps=train_steps[0], **common)
        mlp.fit(setup.data, split, encoder)
        rows.append(
            {"seed": seed, "model": "pertflow_mlp", "train_steps": train_steps[0],
             "flow_steps": None,
             "pearson_delta": evaluate_point(mlp, setup.data, split, encoder).pearson_delta.mean()}
        )

        for steps in train_steps:
            t0 = time.time()
            model = PertFlow(steps=steps, n_flow_steps=max(FLOW_STEPS), **common)
            model.fit(setup.data, split, encoder)
            print(f"  fitted flow @ {steps} steps in {time.time()-t0:.0f}s")
            # Same fitted weights, different integration budget at inference.
            for n in FLOW_STEPS:
                model.n_flow_steps = n
                df = evaluate_point(model, setup.data, split, encoder)
                rows.append(
                    {"seed": seed, "model": "pertflow", "train_steps": steps, "flow_steps": n,
                     "pearson_delta": df.pearson_delta.mean()}
                )
                print(f"    flow_steps={n:3d}  pearson_delta={df.pearson_delta.mean():.3f}")

    df = pd.DataFrame(rows)
    write_table(df, out_dir / "context_collapse_raw.csv")
    result = {"per_run": df.to_dict(orient="records"), "verdict": _verdict(df)}
    write_json(result, out_dir / "context_collapse.json", cfg)
    return result


def _verdict(df: pd.DataFrame) -> dict:
    flow = df[df.model == "pertflow"]
    ridge = float(df[df.model == "ridge_embedding"].pearson_delta.mean())
    mlp = float(df[df.model == "pertflow_mlp"].pearson_delta.mean())
    base_steps = int(flow.train_steps.min())
    long_steps = int(flow.train_steps.max())

    by_flow = flow[flow.train_steps == base_steps].groupby("flow_steps").pearson_delta.mean()
    at_1 = float(by_flow.get(1, float("nan")))
    at_max = float(by_flow.get(max(FLOW_STEPS), float("nan")))
    by_train = flow[flow.flow_steps == max(FLOW_STEPS)].groupby("train_steps").pearson_delta.mean()

    # (a) is supported if cutting the integration budget recovers most of the gap to the MLP.
    recovered = (at_1 - at_max) / max(mlp - at_max, 1e-9)
    longer = float(by_train.get(long_steps, float("nan")))
    training_recovered = (longer - at_max) / max(mlp - at_max, 1e-9)

    # The cleanest evidence for compounding is not the size of the recovery but its shape:
    # accuracy should fall as the integration budget rises, because each step reapplies the
    # same faulty conditioning. A ratio can exceed 100% when one step beats the regressor,
    # so the monotone trend is reported alongside it.
    ordered = by_flow.sort_index()
    monotone = bool((ordered.diff().dropna() <= 1e-6).all()) if len(ordered) > 2 else False
    return {
        "pearson_delta_by_flow_steps": {int(k): float(v) for k, v in ordered.items()},
        "monotone_decreasing_in_integration_steps": monotone,
        "ridge": ridge,
        "mlp_reference": mlp,
        "flow_at_1_step": at_1,
        "flow_at_max_steps": at_max,
        f"flow_trained_{long_steps}_at_max_steps": longer,
        "fraction_of_gap_recovered_by_fewer_integration_steps": recovered,
        "fraction_of_gap_recovered_by_longer_training": training_recovered,
        "compounding_supported": bool(recovered > 0.5 and monotone),
        "undertraining_supported": bool(training_recovered > 0.5),
        "note": (
            "Gap is measured against the MLP ablation, which shares the architecture and "
            "differs only in integrating an ODE. A fraction near 1 means that explanation "
            "accounts for the collapse; near 0 means it does not."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--train-steps", default=None, help="default: config steps plus 12000")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = [int(s) for s in args.seeds.split(",")]
    steps = (
        [int(s) for s in args.train_steps.split(",")]
        if args.train_steps
        else sorted({cfg["pertflow"]["steps"], 12000})
    )
    out_dir = Path(args.out) if args.out else RESULTS / cfg["profile"]
    v = run(cfg, seeds, steps, out_dir)["verdict"]
    print("\n--- unseen-context collapse ---")
    print(f"  ridge baseline                     {v['ridge']:.3f}")
    print(f"  MLP ablation (no integration)      {v['mlp_reference']:.3f}")
    print(f"  flow, {max(FLOW_STEPS)} integration steps      {v['flow_at_max_steps']:.3f}")
    print(f"  flow, 1 integration step           {v['flow_at_1_step']:.3f}")
    trend = "  ".join(f"{k}:{val:.3f}" for k, val in v["pearson_delta_by_flow_steps"].items())
    print(f"  by integration steps:              {trend}")
    print(f"  monotone decreasing:               {v['monotone_decreasing_in_integration_steps']}")
    print(f"  gap recovered by fewer steps:      {v['fraction_of_gap_recovered_by_fewer_integration_steps']:.1%}"
          f"   -> compounding supported: {v['compounding_supported']}")
    print(f"  gap recovered by longer training:  {v['fraction_of_gap_recovered_by_longer_training']:.1%}"
          f"   -> undertraining supported: {v['undertraining_supported']}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
