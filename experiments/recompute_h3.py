"""Rebuild the H3 summary and verdict from the saved raw CSV.

Same rationale as `recompute_h1`: the per-run rows are everything the verdict is derived
from, so a change to how they are judged must not require re-running a stochastic fit.

Usage: python -m experiments.recompute_h3 --profile full
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiments.common import RESULTS, load_config, write_json, write_table
from experiments.exp_h3_search import _summarise, _verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    out_dir = RESULTS / args.profile
    cfg = load_config(args.config or f"configs/{args.profile}.yaml")

    df = pd.read_csv(out_dir / "h3_search_raw.csv")
    summary = _summarise(df)
    write_table(pd.DataFrame(summary), out_dir / "h3_search.csv")
    result = {
        "per_run": df.to_dict(orient="records"),
        "summary": summary,
        "notes": ["recomputed from h3_search_raw.csv"],
        "verdict": _verdict(df),
    }
    write_json(result, out_dir / "h3_search.json", cfg)

    v = result["verdict"]
    print("--- H3 verdict (recomputed) ---")
    g = v.get("optimism_gap_at_lambda0")
    if g:
        print(f"  optimism gap at lambda=0: {g['mean']:.3f} [{g['ci_lo']:.3f}, {g['ci_hi']:.3f}]")
        print(f"  optimiser's curse present: {v.get('optimisers_curse_present')}")
    print(f"  regret of uncertainty-blind search: {v.get('blind_regret_vs_oracle'):.4f}")
    print(f"  saturation threshold (1% of oracle): {v.get('saturation_threshold'):.4f}")
    if v.get("search_saturated_at_oracle"):
        print("  SEARCH SATURATED at lambda=0: no headroom for any penalty.")
        print("  H3 is UNTESTABLE in this regime, not refuted.")
        print("  NOTE: the optimism-gap scale fix does NOT change this. Saturation and the")
        print("        scale bug were independent problems; only the latter was fixed.")
    for c in v.get("comparisons", []):
        print(f"  lambda={c['lambda']}: vs blind {c['mean_diff']:+.4f} "
              f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] (n={c['n']})")
    print(f"  H3 (penalty improves verified reversal): "
          f"{v.get('h3_penalty_improves_verified_reversal')}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
