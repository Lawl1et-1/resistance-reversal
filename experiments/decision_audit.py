"""Recompute every verdict under both the original and the revised decision rules.

Three analysis rules were changed *after* the full-profile numbers came back, and all
three revisions moved a verdict toward a cleaner story. Each revision is defensible on its
own; together they are a forking path, and a reader has no way to judge them without seeing
what the original rules would have concluded on the same data.

This script answers that mechanically rather than from memory: it applies both rule sets to
the saved raw rows and prints the verdict each produces. `DECISIONS.md` carries the
narrative; this is the executable version, so the comparison cannot drift from the code.

Usage: python -m experiments.decision_audit --profile full
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiments.common import RESULTS
from src.eval.metrics import paired_test

DEEP = ("pertflow", "pertflow_mlp")
LINEAR = ("additive", "ridge_embedding", "low_rank_linear")


# --- H1: pairing unit, significance criterion, and the "reliably wins" rule -------------


def h1_paired(point: pd.DataFrame, unit: str) -> pd.DataFrame:
    """unit='row' pairs on (seed, pert, context); unit='seed' aggregates per seed first."""
    out = []
    for split_name, g in point.groupby("split"):
        for deep in DEEP:
            for lin in LINEAR:
                if unit == "row":
                    keys = ["seed", "pert", "context"]
                    a = g[g.model == deep].set_index(keys)["pearson_delta"]
                    b = g[g.model == lin].set_index(keys)["pearson_delta"]
                    shared = a.index.intersection(b.index)
                    av, bv = a.loc[shared].to_numpy(), b.loc[shared].to_numpy()
                else:
                    a = g[g.model == deep].groupby("seed")["pearson_delta"].mean()
                    b = g[g.model == lin].groupby("seed")["pearson_delta"].mean()
                    shared = a.index.intersection(b.index)
                    av, bv = a.loc[shared].to_numpy(), b.loc[shared].to_numpy()
                if len(shared) < 3:
                    continue
                res = paired_test(av, bv)
                if np.isfinite(res.get("mean_diff", np.nan)):
                    out.append({"split": split_name, "deep": deep, "linear": lin, **res})
    return pd.DataFrame(out)


def h1_counts(paired: pd.DataFrame, criterion: str) -> tuple[int, int]:
    """criterion='ci' uses the CI alone; 'ci_effect' also demands |d_z| > 1."""
    if paired.empty:
        return 0, 0
    wins = paired.ci_lo > 0
    losses = paired.ci_hi < 0
    if criterion == "ci_effect":
        large = paired.cohens_dz.abs() > 1.0
        wins, losses = wins & large, losses & large
    return int(wins.sum()), int(losses.sum())


def h1_rule(wins: int, losses: int, rule: str) -> bool:
    """Returns H1a: 'deep fails to reliably beat linear'."""
    if rule == "original":  # wins==0 or losses>=wins
        return wins == 0 or losses >= wins
    return not (wins > 0 and losses == 0)  # revised


# --- H3: saturation rule and the optimism-gap scale -------------------------------------


def h3_saturation(df: pd.DataFrame, rule: str) -> tuple[bool, float, float]:
    greedy = df[(~df.abstention) & (df.strategy == "greedy")]
    oracle = float(greedy.oracle_value.mean())
    threshold = 0.01 * oracle
    regret = (
        float(greedy.regret_vs_oracle.mean())
        if rule == "original"
        else float(greedy[greedy["lambda"] == 0].regret_vs_oracle.mean())
    )
    return bool(regret <= threshold), regret, threshold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    args = ap.parse_args()
    res = RESULTS / args.profile

    print("=" * 78)
    print("H1a - 'deep models do not reliably beat linear baselines'")
    print("=" * 78)
    point = pd.read_csv(res / "h1_point_metrics_raw.csv")
    point["pert"] = point["pert"].astype(str)
    for unit in ("row", "seed"):
        paired = h1_paired(point, unit)
        for criterion in ("ci", "ci_effect"):
            w, l = h1_counts(paired, criterion)
            for rule in ("original", "revised"):
                verdict = h1_rule(w, l, rule)
                tag = "  <= AS REPORTED" if (unit, criterion, rule) == ("seed", "ci_effect", "revised") else ""
                print(f"  pairing={unit:4s} criterion={criterion:9s} rule={rule:8s} "
                      f"-> wins={w} losses={l}  H1a={verdict}{tag}")
    print("\n  n paired comparisons: "
          f"row-level={len(h1_paired(point,'row'))}, seed-level={len(h1_paired(point,'seed'))}")
    print("  (row-level pairing treats ~720 correlated rows per comparison as independent)")

    print()
    print("=" * 78)
    print("H3 - saturation rule and the optimism gap")
    print("=" * 78)
    df = pd.read_csv(res / "h3_search_raw.csv")
    for rule in ("original", "revised"):
        sat, regret, thr = h3_saturation(df, rule)
        verdict = "UNTESTABLE (None)" if sat else "testable -> reports True/False"
        label = "mean over all lambdas" if rule == "original" else "lambda=0 only"
        print(f"  rule={rule:8s} ({label:21s}) regret={regret:.4f} vs threshold={thr:.4f}"
              f" -> saturated={sat}  H3={verdict}")

    pre = res / "pre_revision" / "h3_search_raw_prefix_optimism.csv"
    if pre.exists():
        old = pd.read_csv(pre)
        og = old[(~old.abstention) & (old.strategy == "greedy") & (old["lambda"] == 0)]
        print(f"\n  optimism gap as originally computed (predicted - ORACLE scale): "
              f"{og.optimism_gap.mean():+.3f}")
    if "reversal_observed_same_readout" in df.columns:
        g = df[(~df.abstention) & (df.strategy == "greedy") & (df["lambda"] == 0)]
        print(f"  optimism gap corrected (predicted - observed, SAME readout):   "
              f"{g.optimism_gap.mean():+.3f}")
        print(f"  pure scale offset (learned readout - oracle):                  "
              f"{g.scale_gap_readout_vs_oracle.mean():+.3f}")
        print("  The original figure was mostly this scale offset, not optimism.")


if __name__ == "__main__":
    main()
