"""Rebuild the H1 summary, paired tests and verdict from the saved raw CSVs.

The raw per-test-point scores are everything the summary is derived from, so a change to
how results are *aggregated* does not require re-running the models. Re-fitting to change
an aggregation would also silently change the numbers, since the fit is stochastic; reusing
the saved rows keeps the correction auditable.

Usage: python -m experiments.recompute_h1 --profile full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.common import RESULTS, load_config, write_json, write_table
from experiments.exp_h1_benchmark import (
    _paired,
    _ranking_shift,
    _readout_sanity,
    _reversal_summary,
    _verdict,
)
from src.eval.evaluate import summarise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    out_dir = RESULTS / args.profile
    cfg = load_config(args.config or f"configs/{args.profile}.yaml")

    point = pd.read_csv(out_dir / "h1_point_metrics_raw.csv")
    point["pert"] = point["pert"].astype(str)
    summary = summarise(point, by=("model", "split"))
    write_table(summary, out_dir / "h1_point_metrics.csv")

    result = {
        "point_summary": summary.to_dict(orient="records"),
        "paired_tests": _paired(point),
        "recomputed_from": "h1_point_metrics_raw.csv",
    }

    rev_path = out_dir / "h1_reversal_raw.csv"
    if rev_path.exists():
        reversal = pd.read_csv(rev_path)
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

    v = result["verdict"]
    print("--- H1 verdict (recomputed, paired at the seed level) ---")
    print(f"  paired comparisons:        {v['n_paired_comparisons']}")
    print(f"  deep significantly better: {v['deep_beats_linear_significantly']}")
    print(f"  deep significantly worse:  {v['linear_beats_deep_significantly']}")
    print(f"  H1a (deep does not reliably win): {v['h1a_deep_fails_to_reliably_beat_linear']}")
    print(f"  H1b (evaluation changes ranking): {v['h1b_evaluation_changes_ranking']}"
          f" ({v['n_ranking_winner_changes']}/{v['n_ranking_comparisons']} winner changes)")
    for line in v["unreliable_readouts"]:
        print(f"  UNRELIABLE READOUT: {line}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
