"""Shared I/O utilities for ProteinGym scoring scripts."""

import logging
import re
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_COLS = {"protein", "protein_sequence", "mutant"}


def parse_mutant(mutant: str) -> tuple[str, int, str]:
    """
    Parse a single-substitution string such as "R29L".

    Returns (wt_aa, position, mut_aa). Raises ValueError for unrecognised formats.
    """
    m = re.fullmatch(r"([A-Z\*])(\d+)([A-Z\*])", mutant.strip().upper())
    if m is None:
        raise ValueError(f"Cannot parse mutant: {mutant!r}")
    return m.group(1), int(m.group(2)), m.group(3)


def load_gene_df(csv_path: Path) -> tuple[pd.DataFrame, str, str]:
    """
    Load a ProteinGym gene CSV and return (df, protein_id, wt_seq).

    Raises ValueError if required columns are missing.
    """
    df = pd.read_csv(csv_path, index_col=0)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {missing}")
    protein_id = df["protein"].iloc[0]
    wt_seq = df["protein_sequence"].iloc[0]
    return df, protein_id, wt_seq


def collect_csv_paths(input_path: Path, max_inputs: int | None = None) -> list[Path]:
    """Return sorted list of CSV paths from a file or directory. Exits if none found."""
    paths = sorted(input_path.glob("*.csv")) if input_path.is_dir() else [input_path]
    if not paths:
        log.error("No CSV files found at %s", input_path)
        sys.exit(1)
    if max_inputs is not None:
        paths = paths[:max_inputs]
        log.info("Limiting to first %d CSV file(s) (--max-inputs)", max_inputs)
    log.info("%d gene CSV(s) to process", len(paths))
    return paths


def build_score_output(df: pd.DataFrame, score_col: str, scores: list) -> pd.DataFrame:
    """Assemble the standard output DataFrame: protein, mutant, <score_col>, DMS_bin_score."""
    out = df[["protein", "mutant"]].copy()
    out[score_col] = scores
    if "DMS_bin_score" in df.columns:
        out["DMS_bin_score"] = df["DMS_bin_score"].values
    return out
