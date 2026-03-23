"""
Visualize ESM1b log-likelihood ratio scores from a scored ProteinGym CSV.

Produces two plots:
  1. ROC curve with AUC — how well esm_score separates Pathogenic from Benign.
  2. Score histogram — distribution of esm_score, colour-coded by label.

Both plots are saved to the output directory (or displayed interactively if
--no-save is passed).

Usage
-----
python -m vep_eval.visualize_esm_scores \\
    --input all_esm_scores.csv \\
    --output-dir figures/

python -m vep_eval.visualize_esm_scores \\
    --input NP_000007.1_esm.csv \\
    --output-dir figures/ \\
    --title "NP_000007.1"
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

from vep_eval.run_name import add_run_name_args, build_run_name, resolve_output_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PATHOGENIC_LABEL = "Pathogenic"
BENIGN_LABEL = "Benign"
COLORS = {PATHOGENIC_LABEL: "#d62728", BENIGN_LABEL: "#1f77b4"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_scores(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = {"esm_score", "DMS_bin_score"} - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    n_before = len(df)
    df = df.dropna(subset=["esm_score"])
    df = df[df["DMS_bin_score"].isin([PATHOGENIC_LABEL, BENIGN_LABEL])]
    n_dropped = n_before - len(df)
    if n_dropped:
        log.warning("Dropped %d rows with missing scores or unknown labels.", n_dropped)

    log.info(
        "Loaded %d variants  (Pathogenic=%d  Benign=%d)",
        len(df),
        (df["DMS_bin_score"] == PATHOGENIC_LABEL).sum(),
        (df["DMS_bin_score"] == BENIGN_LABEL).sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_roc(df: pd.DataFrame, title: str, ax: plt.Axes) -> float:
    """Draw ROC curve on *ax*. Returns AUROC."""
    # Pathogenic = positive class; higher ESM score = more benign, so negate
    y_true = (df["DMS_bin_score"] == PATHOGENIC_LABEL).astype(int)
    y_score = -df["esm_score"]  # lower LLR → more deleterious

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color="#2ca02c", lw=2, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {title}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return auroc


def plot_histogram(df: pd.DataFrame, title: str, ax: plt.Axes, bins: int = 60):
    """Draw overlapping score histograms colour-coded by label on *ax*."""
    for label in [PATHOGENIC_LABEL, BENIGN_LABEL]:
        scores = df.loc[df["DMS_bin_score"] == label, "esm_score"]
        ax.hist(
            scores,
            bins=bins,
            alpha=0.55,
            color=COLORS[label],
            label=f"{label} (n={len(scores):,})",
            density=True,
            edgecolor="none",
        )

    ax.set_xlabel("ESM1b log-likelihood ratio")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — {title}")
    ax.legend()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize ESM1b scores from a scored ProteinGym CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="PATH",
        help="Scored CSV produced by score_proteingym_esm.py.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        metavar="DIR",
        help="Base directory for figures. Saved under <output-dir>/<run-name>/.",
    )
    add_run_name_args(parser)
    parser.add_argument(
        "--title",
        default=None,
        metavar="TEXT",
        help="Plot title prefix. Defaults to the input filename stem.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        metavar="N",
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure resolution.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Display plots interactively instead of saving to disk.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.getLogger().setLevel(args.log_level)

    input_path = Path(args.input)
    title = args.title or input_path.stem

    df = load_scores(input_path)
    if df.empty:
        log.error("No usable data in %s", input_path)
        sys.exit(1)

    # --- ROC plot -----------------------------------------------------------
    fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
    auroc = plot_roc(df, title, ax_roc)
    log.info("AUROC: %.4f", auroc)
    fig_roc.tight_layout()

    # --- Histogram plot -----------------------------------------------------
    fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
    plot_histogram(df, title, ax_hist, bins=args.bins)
    fig_hist.tight_layout()

    if args.no_save:
        plt.show()
    else:
        run_name = build_run_name(args.run_name, args.no_timestamp)
        out_dir = resolve_output_dir(args.output_dir, run_name)
        log.info("Run name: %s", run_name)

        fig_roc.savefig(out_dir / "roc.png", dpi=args.dpi, bbox_inches="tight")
        fig_hist.savefig(out_dir / "histogram.png", dpi=args.dpi, bbox_inches="tight")

        log.info("Saved ROC curve  → %s", out_dir / "roc.png")
        log.info("Saved histogram  → %s", out_dir / "histogram.png")

    plt.close("all")


if __name__ == "__main__":
    main()
