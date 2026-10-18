"""Rebuild every figure from the CSVs in `results/`.

Figures are generated *only* from files on disk, never from numbers recomputed here, so a
figure cannot disagree with the table it came from. A missing results file is reported and
skipped rather than filled in with anything, which is why running this before the
experiments produces a short list of skips instead of a plausible-looking plot.

Usage: python -m experiments.make_figures [--profile smoke|full]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on the dev machine, and none needed
import matplotlib.pyplot as plt
import pandas as pd

from experiments.common import FIGURES, RESULTS

# Colour-blind-safe; the shuffled-label control is grey on purpose, since it is a floor
# rather than a competitor.
PALETTE = {
    "control_mean": "#9e9e9e",
    "shuffled_labels": "#616161",
    "additive": "#0072B2",
    "ridge_embedding": "#56B4E9",
    "low_rank_linear": "#009E73",
    "pertflow": "#D55E00",
    "pertflow_mlp": "#E69F00",
    "oracle_latent": "#000000",
}
SPLIT_LABEL = {
    "unseen_single": "unseen\nsingle",
    "unseen_combo_seen_genes": "unseen combo\n(genes seen)",
    "unseen_combo_unseen_genes": "unseen combo\n(genes unseen)",
    "unseen_context": "unseen\ncontext",
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  skip: {path.name} not found")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"  skip: {path.name} is empty")
        return None
    return df


def fig_model_comparison(res: Path, out: Path) -> None:
    """Pearson-delta per model and split, with the shuffled-label floor marked."""
    df = _read(res / "h1_point_metrics.csv")
    if df is None:
        return
    splits = [s for s in SPLIT_LABEL if s in set(df.split)]
    models = [m for m in PALETTE if m in set(df.model)]
    fig, ax = plt.subplots(figsize=(1.9 * len(splits) + 3, 4))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        g = df[df.model == model].set_index("split")
        xs, ys, los, his = [], [], [], []
        for j, s in enumerate(splits):
            if s not in g.index or pd.isna(g.loc[s, "pearson_delta"]):
                continue
            xs.append(j + i * width - 0.4)
            ys.append(g.loc[s, "pearson_delta"])
            los.append(max(ys[-1] - g.loc[s, "pearson_delta_lo"], 0))
            his.append(max(g.loc[s, "pearson_delta_hi"] - ys[-1], 0))
        if not xs:
            continue
        ax.bar(xs, ys, width=width, yerr=[los, his], capsize=2, label=model,
               color=PALETTE[model], edgecolor="white", linewidth=0.5)
    for j, s in enumerate(splits):
        sh = df[(df.split == s) & (df.model == "shuffled_labels")]
        if not sh.empty and pd.notna(sh.pearson_delta.iloc[0]):
            ax.hlines(sh.pearson_delta.iloc[0], j - 0.45, j + 0.45,
                      color="#616161", linestyle=":", linewidth=1.4, zorder=5)
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([SPLIT_LABEL[s] for s in splits], fontsize=8)
    ax.set_ylabel("Pearson-delta (top-DEG deltas)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Prediction accuracy by split\n(dotted line: shuffled-label control)", fontsize=10)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    _save(fig, out / "fig_model_comparison.png")


def fig_ranking_shift(res: Path, out: Path) -> None:
    """The H1b figure: does reversal-aware evaluation reorder the models?"""
    point = _read(res / "h1_point_metrics.csv")
    rank = _read(res / "h1_reversal_ranking.csv")
    if point is None or rank is None:
        return
    rank = rank[rank.get("ranking_usable", True)]
    if rank.empty:
        print("  skip: no split had enough test perturbations for a ranking")
        return
    readouts = sorted(set(rank.readout))
    fig, axes = plt.subplots(1, len(readouts), figsize=(4.6 * len(readouts), 4.2), squeeze=False)
    for ax, readout in zip(axes[0], readouts):
        sub = rank[rank.readout == readout]
        merged = point.merge(sub, on=["model", "split"], suffixes=("", "_r"))
        merged = merged.dropna(subset=["pearson_delta", "reversal_rank_rho"])
        for _, row in merged.iterrows():
            ax.scatter(row.pearson_delta, row.reversal_rank_rho, s=55,
                       color=PALETTE.get(row.model, "#333333"),
                       marker="o" if row.model != "oracle_latent" else "*",
                       edgecolor="white", linewidth=0.6, zorder=3)
            ax.annotate(row.model, (row.pearson_delta, row.reversal_rank_rho),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
        lo = min(merged.pearson_delta.min(), merged.reversal_rank_rho.min(), 0)
        ax.plot([lo, 1], [lo, 1], color="#bbbbbb", linestyle="--", linewidth=1, zorder=1)
        ax.set_xlabel("Pearson-delta (conventional)")
        ax.set_ylabel("reversal rank correlation")
        ax.set_title(f"readout: {readout}", fontsize=9)
    fig.suptitle("Models off the diagonal are ranked differently by the two evaluations",
                 fontsize=10)
    _save(fig, out / "fig_ranking_shift.png")


def fig_coverage(res: Path, out: Path) -> None:
    """Conformal coverage against nominal, separating the shifted split."""
    df = _read(res / "h2_calibration.csv")
    if df is None:
        return
    nominal = float(df.nominal_coverage.iloc[0])
    cols = [c for c in ("shift_coverage", "reversal_coverage") if c in df.columns]
    if not cols:
        print("  skip: no coverage columns in h2_calibration.csv")
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(df) + 3.5, 4))
    width = 0.8 / len(cols)
    for i, col in enumerate(cols):
        xs = [j + i * width - 0.4 for j in range(len(df))]
        lo = (df[col] - df.get(f"{col}_lo", df[col])).clip(lower=0)
        hi = (df.get(f"{col}_hi", df[col]) - df[col]).clip(lower=0)
        ax.bar(xs, df[col], width=width, yerr=[lo, hi], capsize=3,
               label=col.replace("_", " "), edgecolor="white")
    ax.axhline(nominal, color="#D55E00", linestyle="--",
               label=f"nominal {nominal:.2f}")
    labels = [
        SPLIT_LABEL.get(r.split, r.split) + ("\n(assumption\nviolated)" if r.exchangeability_violated else "")
        for r in df.itertuples()
    ]
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Conformal coverage by split", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _save(fig, out / "fig_coverage.png")


def fig_risk_coverage(res: Path, out: Path) -> None:
    df = _read(res / "h2_risk_coverage.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for split_name, g in df.groupby("split"):
        m = g.groupby("coverage").risk.mean()
        ax.plot(m.index, m.values, marker="o", markersize=3,
                label=SPLIT_LABEL.get(split_name, split_name).replace("\n", " "))
    ax.set_xlabel("coverage (fraction of perturbations retained)")
    ax.set_ylabel("mean error on retained perturbations")
    ax.set_title("Risk-coverage: abstaining on the\nmost uncertain perturbations", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _save(fig, out / "fig_risk_coverage.png")


def fig_search(res: Path, out: Path) -> None:
    """The H3 figure: verified reversal against the uncertainty penalty."""
    df = _read(res / "h3_search.csv")
    if df is None:
        return
    g = df[(~df.abstention) & (df.strategy == "greedy")].sort_values("lambda")
    if g.empty:
        print("  skip: no uncertainty-blind greedy runs in h3_search.csv")
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
    ax.errorbar(g["lambda"], g.reversal_verified,
                yerr=[(g.reversal_verified - g.reversal_verified_lo).clip(lower=0),
                      (g.reversal_verified_hi - g.reversal_verified).clip(lower=0)],
                marker="o", capsize=3, color="#0072B2", label="verified (oracle)")
    ax.errorbar(g["lambda"], g.reversal_pred, marker="s", linestyle="--",
                color="#D55E00", label="predicted by model")
    ax.set_xlabel("uncertainty penalty $\\lambda$")
    ax.set_ylabel("reversal of recommended set")
    ax.set_title("Recommendation quality vs penalty", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax2.errorbar(g["lambda"], g.optimism_gap,
                 yerr=[(g.optimism_gap - g.optimism_gap_lo).clip(lower=0),
                       (g.optimism_gap_hi - g.optimism_gap).clip(lower=0)],
                 marker="o", capsize=3, color="#CC79A7")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("uncertainty penalty $\\lambda$")
    ax2.set_ylabel("predicted - verified")
    ax2.set_title("Optimism gap (the optimiser's curse)", fontsize=10)
    _save(fig, out / "fig_search.png")


FIGURE_FNS = (fig_model_comparison, fig_ranking_shift, fig_coverage, fig_risk_coverage, fig_search)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    args = ap.parse_args()
    res = RESULTS / args.profile
    out = FIGURES / args.profile
    if not res.exists():
        raise SystemExit(f"no results directory at {res}; run the experiments first")
    print(f"building figures from {res}")
    for fn in FIGURE_FNS:
        fn(res, out)
    print("done")


if __name__ == "__main__":
    main()
