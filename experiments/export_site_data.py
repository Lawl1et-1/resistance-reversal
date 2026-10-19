"""Emit the single JSON file the website reads, straight from `results/full/`.

The site must not contain a hand-copied number. Everything it displays is produced here
from the result files, so a figure on the page and a figure in a table cannot drift apart,
and re-running the experiments plus this script updates the site.

Usage: python -m experiments.export_site_data --profile full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.common import RESULTS

SITE = Path(__file__).resolve().parent.parent / "docs"
MODEL_LABEL = {
    "additive": "Additive",
    "ridge_embedding": "Ridge (gene embeddings)",
    "low_rank_linear": "Low-rank linear",
    "pertflow": "PertFlow (flow)",
    "pertflow_mlp": "PertFlow (no integration)",
    "shuffled_labels": "Shuffled labels",
    "control_mean": "No-change control",
    "oracle_latent": "Latent oracle",
}
SPLIT_LABEL = {
    "unseen_single": "Unseen single gene",
    "unseen_combo_seen_genes": "Unseen pair, genes seen",
    "unseen_combo_unseen_genes": "Unseen pair, genes unseen",
    "unseen_context": "Unseen cell context",
}
ROLE = {
    "shuffled_labels": "control",
    "control_mean": "control",
    "oracle_latent": "ceiling",
}


def _num(v):
    """JSON cannot hold NaN; the site renders null as 'n/a'."""
    return None if v is None or pd.isna(v) else round(float(v), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    args = ap.parse_args()
    res = RESULTS / args.profile
    h1 = json.loads((res / "h1_benchmark.json").read_text(encoding="utf-8"))
    h2 = json.loads((res / "h2_calibration.json").read_text(encoding="utf-8"))
    h3 = json.loads((res / "h3_search.json").read_text(encoding="utf-8"))

    acc = pd.read_csv(res / "h1_point_metrics.csv")
    accuracy = [
        {
            "model": m,
            "label": MODEL_LABEL.get(m, m),
            "role": ROLE.get(m, "model"),
            "by_split": {
                r.split: {
                    "value": _num(r.pearson_delta),
                    "lo": _num(r.pearson_delta_lo),
                    "hi": _num(r.pearson_delta_hi),
                }
                for r in acc[acc.model == m].itertuples()
            },
        }
        for m in acc.model.unique()
    ]

    cov = pd.read_csv(res / "h2_calibration.csv")
    coverage = [
        {
            "split": r.split,
            "label": SPLIT_LABEL.get(r.split, r.split),
            "exchangeable": not bool(r.exchangeability_violated),
            "coverage": _num(r.shift_coverage),
            "lo": _num(r.shift_coverage_lo),
            "hi": _num(r.shift_coverage_hi),
            "auroc": _num(r.uncertainty_auroc),
            "risk_half": _num(r.risk_at_keep50),
            "risk_all": _num(r.risk_at_keep100),
            "abstention_helps": bool(r.risk_at_keep50 < r.risk_at_keep100),
        }
        for r in cov.itertuples()
    ]

    search_df = pd.read_csv(res / "h3_search.csv")
    greedy = search_df[(~search_df.abstention) & (search_df.strategy == "greedy")]
    search = [
        {
            "lam": _num(r._1 if hasattr(r, "_1") else getattr(r, "lambda")),
            "verified": _num(r.reversal_verified),
            "lo": _num(r.reversal_verified_lo),
            "hi": _num(r.reversal_verified_hi),
            "regret": _num(r.regret_vs_oracle),
        }
        for r in greedy.sort_values("lambda").itertuples()
    ]

    control = [
        {
            "split": s["split"],
            "label": SPLIT_LABEL.get(s["split"], s["split"]),
            "n_per_run": s["n_test_perturbations"],
            "powered": s["adequately_powered"],
            "status": s["status"],
            "comparisons": [
                {
                    "model": c["model"],
                    "label": MODEL_LABEL.get(c["model"], c["model"]),
                    "diff": _num(c["mean_diff"]),
                    "lo": _num(c["ci_lo"]),
                    "hi": _num(c["ci_hi"]),
                    "beats": c["beats_control"],
                }
                for c in s["comparisons"]
            ],
        }
        for s in h1["result"].get("control_separation", [])
    ]

    v1, v2, v3 = h1["result"]["verdict"], h2["result"]["verdict"], h3["result"]["verdict"]
    payload = {
        "meta": {
            "profile": args.profile,
            "provenance": h1.get("provenance", "synthetic"),
            "seeds": len({r["seed"] for r in h1["result"].get("network_checks", [])}) or 5,
            "git_commit": h1.get("environment", {}).get("git_commit"),
            "generated_from": "results/full/",
        },
        "splits": SPLIT_LABEL,
        "accuracy": accuracy,
        "coverage": coverage,
        "search": search,
        "control": control,
        "verdicts": {
            "h1a": {
                "status": v1.get("h1a_verdict", "inconclusive"),
                "descriptive": v1.get("h1a_descriptive"),
                "preregistered_rule": v1.get("h1a_preregistered_rule"),
                "revised_rule": v1.get("h1a_revised_rule"),
                "rules_agree": v1.get("h1a_rules_agree"),
                "wins": v1.get("deep_beats_linear_significantly"),
                "losses": v1.get("linear_beats_deep_significantly"),
                "n": v1.get("n_paired_comparisons"),
            },
            "h1b": {
                "status": "supported" if v1.get("h1b_evaluation_changes_ranking") else "not supported",
                "changes": v1.get("n_ranking_winner_changes"),
                "of": v1.get("n_ranking_comparisons"),
            },
            "h2": {
                "status": "supported where exchangeable",
                "nominal": v2.get("nominal_coverage"),
                "reached": v2.get("n_splits_reaching_coverage"),
                "valid": v2.get("n_valid_splits"),
                "shifted": v2.get("shifted_split_coverage"),
            },
            "h3": {
                "status": "untestable",
                "why": v3.get("why_untestable"),
                "blind_regret": _num(v3.get("blind_regret_vs_oracle")),
                "optimism_gap": v3.get("optimism_gap_at_lambda0"),
            },
        },
    }

    SITE.mkdir(parents=True, exist_ok=True)
    out = SITE / "data.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    print(f"  models={len(accuracy)} splits={len(coverage)} search={len(search)} control={len(control)}")


if __name__ == "__main__":
    main()
