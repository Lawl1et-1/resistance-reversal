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

import numpy as np
import pandas as pd

from experiments.common import RESULTS, build, load_config, write_json, write_table
from src.eval.evaluate import evaluate_point
from src.models.baselines import RidgeEmbedding

# The shifted split and a control split whose test contexts were all seen in training.
# Without the control, a monotone decline with integration steps proves nothing about
# shift: a flow that simply integrates badly would show the same curve everywhere. The
# claim is only about shift if the control stays flat (or improves) while the shifted
# split degrades.
SHIFTED_SPLIT = "unseen_context"
CONTROL_SPLIT = "unseen_combo_seen_genes"
SPLITS = (SHIFTED_SPLIT, CONTROL_SPLIT)
FLOW_STEPS = (1, 2, 5, 10, 20)


def run(cfg: dict, seeds: list[int], train_steps: list[int], out_dir: Path) -> dict:
    from src.models.pertflow import PertFlow

    pf = cfg["pertflow"]
    rows = []
    for seed in seeds:
        print(f"[ctx] seed {seed}")
        setup = build(cfg, seed)
        common = dict(hidden=pf["hidden"], set_dim=pf["set_dim"], n_ensemble=pf["n_ensemble"],
                      lr=pf["lr"], batch_size=pf["batch_size"], n_eval_cells=pf["n_eval_cells"],
                      seed=seed)

        for split_name in SPLITS:
            if split_name not in setup.splits:
                print(f"  {split_name}: unavailable, skipped")
                continue
            shifted = split_name == SHIFTED_SPLIT
            split, encoder = setup.splits[split_name], setup.encoders[split_name]
            print(f"  --- {split_name} ({'context shift' if shifted else 'control, no shift'}) ---")

            ridge = RidgeEmbedding().fit(setup.data, split, encoder)
            rows.append(
                {"seed": seed, "split": split_name, "shifted": shifted,
                 "model": "ridge_embedding", "train_steps": None, "flow_steps": None,
                 "pearson_delta": evaluate_point(ridge, setup.data, split, encoder).pearson_delta.mean()}
            )

            mlp = PertFlow(name="pertflow_mlp", mode="mlp", steps=train_steps[0], **common)
            mlp.fit(setup.data, split, encoder)
            rows.append(
                {"seed": seed, "split": split_name, "shifted": shifted,
                 "model": "pertflow_mlp", "train_steps": train_steps[0], "flow_steps": None,
                 "pearson_delta": evaluate_point(mlp, setup.data, split, encoder).pearson_delta.mean()}
            )

            for steps in train_steps:
                t0 = time.time()
                model = PertFlow(steps=steps, n_flow_steps=max(FLOW_STEPS), **common)
                model.fit(setup.data, split, encoder)
                print(f"    fitted flow @ {steps} steps in {time.time()-t0:.0f}s")
                # Same fitted weights, different integration budget at inference.
                for n in FLOW_STEPS:
                    model.n_flow_steps = n
                    df = evaluate_point(model, setup.data, split, encoder)
                    rows.append(
                        {"seed": seed, "split": split_name, "shifted": shifted,
                         "model": "pertflow", "train_steps": steps, "flow_steps": n,
                         "pearson_delta": df.pearson_delta.mean()}
                    )
                    print(f"      flow_steps={n:3d}  pearson_delta={df.pearson_delta.mean():.3f}")

    df = pd.DataFrame(rows)
    write_table(df, out_dir / "context_collapse_raw.csv")
    result = {"per_run": df.to_dict(orient="records"), "verdict": _verdict(df)}
    write_json(result, out_dir / "context_collapse.json", cfg)
    return result


def _verdict(df: pd.DataFrame) -> dict:
    """Is the degradation specific to context shift, or generic to integration?

    Deliberately does **not** name a cause. The original hypothesis was an untrained
    context embedding, but that predicts longer training changes nothing, whereas longer
    training measurably *hurts*. Something else is involved -- overfitting to the training
    contexts, or the velocity field leaving the data manifold for inputs it never saw.
    The claim supported by this design is the narrower one: error compounds over
    integration under context shift.
    """
    per_split = {}
    for split_name, g in df[df.model == "pertflow"].groupby("split"):
        base_steps = int(g.train_steps.min())
        long_steps = int(g.train_steps.max())
        by_flow = g[g.train_steps == base_steps].groupby("flow_steps").pearson_delta.mean()
        ordered = by_flow.sort_index()
        at_1, at_max = float(ordered.iloc[0]), float(ordered.iloc[-1])
        by_train = g[g.flow_steps == max(FLOW_STEPS)].groupby("train_steps").pearson_delta.mean()
        ref = df[(df.split == split_name) & (df.model == "pertflow_mlp")].pearson_delta.mean()
        per_split[split_name] = {
            "shifted": bool(g.shifted.iloc[0]),
            "pearson_delta_by_flow_steps": {int(k): float(v) for k, v in ordered.items()},
            "degradation_1_to_max_steps": at_1 - at_max,
            "monotone_decreasing": bool((ordered.diff().dropna() <= 1e-6).all()),
            "mlp_reference": float(ref),
            "ridge_reference": float(
                df[(df.split == split_name) & (df.model == "ridge_embedding")].pearson_delta.mean()
            ),
            "at_longer_training": float(by_train.get(long_steps, float("nan"))),
            "longer_training_helps": bool(by_train.get(long_steps, -np.inf) > at_max),
        }

    shifted = next((v for v in per_split.values() if v["shifted"]), None)
    control = next((v for v in per_split.values() if not v["shifted"]), None)
    out = {"per_split": per_split}
    if shifted and control:
        ds, dc = shifted["degradation_1_to_max_steps"], control["degradation_1_to_max_steps"]
        out.update(
            {
                "degradation_shifted": ds,
                "degradation_control": dc,
                # Shift-specific only if the shifted split degrades materially and the
                # control split does not degrade anywhere near as much.
                "compounding_is_shift_specific": bool(ds > 0.05 and ds > 3 * max(dc, 0.0)),
                "generic_integration_problem": bool(dc > 0.05 and ds <= 3 * max(dc, 0.0)),
            }
        )
    out["note"] = (
        "A monotone decline with integration steps on the shifted split alone supports "
        "'error compounds over integration under context shift'. The same decline on the "
        "unshifted control instead indicates a generic integration defect and says nothing "
        "about shift. No causal mechanism is claimed either way."
    )
    return out


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
    print("\n--- integration budget vs context shift ---")
    for split_name, s in v["per_split"].items():
        tag = "SHIFTED" if s["shifted"] else "control"
        trend = "  ".join(f"{k}:{val:.3f}" for k, val in s["pearson_delta_by_flow_steps"].items())
        print(f"  [{tag}] {split_name}")
        print(f"      ridge={s['ridge_reference']:.3f}  mlp={s['mlp_reference']:.3f}")
        print(f"      by integration steps: {trend}")
        print(f"      degradation(1 -> max)={s['degradation_1_to_max_steps']:+.3f}"
              f"  monotone={s['monotone_decreasing']}"
              f"  longer training helps={s['longer_training_helps']}")
    if "compounding_is_shift_specific" in v:
        print(f"  shifted degradation {v['degradation_shifted']:+.3f} vs "
              f"control {v['degradation_control']:+.3f}")
        print(f"  compounding is SHIFT-SPECIFIC: {v['compounding_is_shift_specific']}")
        print(f"  generic integration problem:   {v['generic_integration_problem']}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
