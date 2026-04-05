"""
Bucket variants by SIFT-based conservation and report model AUROC per bucket.

SIFT scores capture position-level evolutionary conservation:
  - SIFT ≈ 0  →  highly conserved (substitution is intolerable)
  - SIFT ≈ 1  →  weakly conserved (substitution is tolerated)

Buckets (by default):
  high    SIFT < 0.05   — strongly conserved
  medium  0.05 ≤ SIFT < 0.20
  low     SIFT ≥ 0.20  — weakly conserved

For each model score file supplied via --scores, the script merges scores with
the SIFT file on (protein, mutant), buckets by SIFT score, computes AUROC in
each bucket, and generates a grouped bar chart.

Usage
-----
# Compare ESM vs SIFT self-consistency across conservation buckets:
python -m vep_eval.analyze_conservation_buckets \\
    --sift-scores results/sift_run/scores.csv \\
    --scores results/esm_run/scores.csv:ESM \\
    --scores results/sift_run/scores.csv:SIFT \\
    --output-dir figures/

# Add AlphaMissense and PrimateAI-3D once scored:
python -m vep_eval.analyze_conservation_buckets \\
    --sift-scores results/sift_run/scores.csv \\
    --scores results/esm_run/scores.csv:ESM \\
    --scores results/am_run/scores.csv:AlphaMissense \\
    --scores results/pai_run/scores.csv:PrimateAI3D \\
    --output-dir figures/
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from vep_eval.run_name import add_run_name_args, build_run_name, resolve_output_dir

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATHOGENIC_LABEL = "Pathogenic"
BENIGN_LABEL = "Benign"
VALID_LABELS = {PATHOGENIC_LABEL, BENIGN_LABEL}

# Default conservation bucket thresholds (SIFT score)
DEFAULT_THRESHOLDS = (0.05, 0.20)

# Score columns that should be negated before AUROC (lower = more pathogenic)
NEGATE_COLS = {"sift_score", "pai_score"}


# ---------------------------------------------------------------------------
# Conservation bucketing
# ---------------------------------------------------------------------------


def assign_conservation_bucket(
    sift: pd.Series,
    low_thresh: float,
    high_thresh: float,
) -> pd.Series:
    """
    Assign each variant to a conservation bucket based on its SIFT score.

    Buckets (strings):
      "high"    SIFT < low_thresh
      "medium"  low_thresh ≤ SIFT < high_thresh
      "low"     SIFT ≥ high_thresh
    """
    buckets = pd.Series(index=sift.index, dtype=str)
    buckets[sift < low_thresh] = "high"
    buckets[(sift >= low_thresh) & (sift < high_thresh)] = "medium"
    buckets[sift >= high_thresh] = "low"
    return buckets


# ---------------------------------------------------------------------------
# AUROC helpers
# ---------------------------------------------------------------------------


def compute_auroc(df: pd.DataFrame, score_col: str, negate: bool) -> float | None:
    """Compute AUROC for a subset DataFrame. Returns None if < 2 classes present."""
    sub = df.dropna(subset=[score_col])
    sub = sub[sub["DMS_bin_score"].isin(VALID_LABELS)]
    if sub["DMS_bin_score"].nunique() < 2 or len(sub) < 2:
        return None

    y_true = (sub["DMS_bin_score"] == PATHOGENIC_LABEL).astype(int)
    y_score = -sub[score_col] if negate else sub[score_col]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc(fpr, tpr))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

BUCKET_ORDER = ["high", "medium", "low"]
BUCKET_LABELS = {
    "high": "High conservation\n(SIFT < {lo})",
    "medium": "Medium conservation\n({lo} ≤ SIFT < {hi})",
    "low": "Low conservation\n(SIFT ≥ {hi})",
}


def plot_conservation_auroc(
    auroc_table: pd.DataFrame,
    thresholds: tuple[float, float],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """
    Draw a grouped bar chart of AUROC by conservation bucket and model.

    *auroc_table* has index = bucket names, columns = model names, values = AUROC.
    """
    lo, hi = thresholds
    bucket_labels = [
        BUCKET_LABELS["high"].format(lo=lo),
        BUCKET_LABELS["medium"].format(lo=lo, hi=hi),
        BUCKET_LABELS["low"].format(hi=hi),
    ]

    models = auroc_table.columns.tolist()
    n_buckets = len(BUCKET_ORDER)
    n_models = len(models)

    x = np.arange(n_buckets)
    bar_width = 0.7 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

    fig, ax = plt.subplots(figsize=(max(7, 2 * n_buckets + n_models), 5))

    for i, model in enumerate(models):
        vals = [auroc_table.loc[b, model] if b in auroc_table.index else float("nan")
                for b in BUCKET_ORDER]
        bars = ax.bar(x + offsets[i], vals, width=bar_width * 0.9, label=model, zorder=3)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Random (0.5)", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title("Model AUROC by Conservation Bucket (SIFT-based)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure → %s", out_path)


def print_table(auroc_table: pd.DataFrame, thresholds: tuple[float, float]) -> None:
    """Pretty-print the AUROC table to stdout."""
    lo, hi = thresholds
    bucket_display = {
        "high": f"High conservation   (SIFT < {lo})",
        "medium": f"Medium conservation ({lo} ≤ SIFT < {hi})",
        "low": f"Low conservation    (SIFT ≥ {hi})",
    }
    lines = ["\nAUROC by Conservation Bucket", "=" * 60]
    header = f"{'Bucket':<38}" + "  ".join(f"{m:>12}" for m in auroc_table.columns)
    lines.append(header)
    lines.append("-" * len(header))
    for bucket in BUCKET_ORDER:
        if bucket not in auroc_table.index:
            continue
        row_str = f"{bucket_display[bucket]:<38}"
        for model in auroc_table.columns:
            val = auroc_table.loc[bucket, model]
            row_str += f"  {val:>12.4f}" if not np.isnan(val) else f"  {'N/A':>12}"
        lines.append(row_str)
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_score_arg(s: str) -> tuple[Path, str]:
    """Parse a 'path:label' or bare 'path' score argument."""
    if ":" in s:
        path_str, label = s.rsplit(":", 1)
    else:
        path_str = s
        label = Path(s).parent.name or Path(s).stem
    return Path(path_str), label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bucket variants by SIFT-based conservation and compute AUROC per bucket."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sift-scores", required=True, metavar="CSV",
        help="CSV with sift_score column (output of score_proteingym_sift.py).",
    )
    parser.add_argument(
        "--scores", action="append", default=[], metavar="CSV[:LABEL]",
        help=(
            "Score CSV to evaluate, optionally with a display label after ':'. "
            "Can be specified multiple times. "
            "Each CSV must have exactly one *_score column (other than DMS_bin_score)."
        ),
    )
    parser.add_argument(
        "--thresholds", nargs=2, type=float, default=list(DEFAULT_THRESHOLDS),
        metavar=("LOW", "HIGH"),
        help="SIFT thresholds for high/medium/low conservation buckets.",
    )
    parser.add_argument(
        "--output-dir", "-o", default=".", metavar="DIR",
        help="Base directory for figures.",
    )
    add_run_name_args(parser)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.getLogger().setLevel(args.log_level)

    thresholds = tuple(sorted(args.thresholds))
    lo, hi = thresholds

    # --- Load SIFT scores (used only for bucketing) ---
    sift_path = Path(args.sift_scores)
    log.info("Loading SIFT scores from %s …", sift_path)
    sift_df = pd.read_csv(sift_path, usecols=["protein", "mutant", "sift_score"])
    sift_df = sift_df.dropna(subset=["sift_score"])
    log.info("  %d variants with valid SIFT scores", len(sift_df))

    if not args.scores:
        log.error("No --scores files provided. Pass at least one with --scores path:Label.")
        sys.exit(1)

    # --- Process each model score file ---
    auroc_records: dict[str, dict[str, float]] = {b: {} for b in BUCKET_ORDER}

    for score_arg in args.scores:
        score_path, label = parse_score_arg(score_arg)
        log.info("Processing model '%s' from %s …", label, score_path)

        score_df = pd.read_csv(score_path)

        # Detect score column
        candidates = [c for c in score_df.columns if c.endswith("_score") and c != "DMS_bin_score"]
        if len(candidates) != 1:
            log.error(
                "  %s: expected exactly one *_score column, found %s — skipping",
                score_path.name, candidates,
            )
            continue
        score_col = candidates[0]
        negate = score_col in NEGATE_COLS
        log.info("  Score column: %s  (negate=%s)", score_col, negate)

        # Merge with SIFT for bucketing
        merged = score_df.merge(sift_df[["protein", "mutant", "sift_score"]],
                                on=["protein", "mutant"], how="inner")
        merged = merged[merged["DMS_bin_score"].isin(VALID_LABELS)]

        # Assign conservation buckets
        merged["_bucket"] = assign_conservation_bucket(merged["sift_score"], lo, hi)

        for bucket in BUCKET_ORDER:
            subset = merged[merged["_bucket"] == bucket]
            auroc = compute_auroc(subset, score_col, negate)
            n = len(subset.dropna(subset=[score_col]))
            if auroc is None:
                log.warning("    bucket=%s: insufficient data (n=%d) — skipping", bucket, n)
                auroc = float("nan")
            else:
                log.info("    bucket=%-6s  n=%5d  AUROC=%.4f", bucket, n, auroc)
            auroc_records[bucket][label] = auroc

    # --- Build table ---
    all_labels = [parse_score_arg(s)[1] for s in args.scores]
    auroc_table = pd.DataFrame(auroc_records, index=all_labels).T
    auroc_table = auroc_table.reindex(BUCKET_ORDER)

    print_table(auroc_table, thresholds)

    # --- Save CSV ---
    run_name = build_run_name(args.run_name, args.no_timestamp)
    out_dir = resolve_output_dir(args.output_dir, run_name)
    log.info("Run name: %s", run_name)

    table_path = out_dir / "auroc_by_conservation.csv"
    auroc_table.to_csv(table_path)
    log.info("Saved AUROC table → %s", table_path)

    # --- Plot ---
    fig_path = out_dir / "auroc_by_conservation.png"
    plot_conservation_auroc(auroc_table, thresholds, fig_path, dpi=args.dpi)


if __name__ == "__main__":
    main()
