"""
Score ProteinGym substitution CSV files with PrimateAI-3D pre-computed scores.

PrimateAI-3D provides per-variant scores indexed by ENST transcript ID,
amino-acid position, reference AA, and alternate AA. This script:
  1. Collects all unique NP_ protein accessions from the input CSVs.
  2. Maps them to ENST transcript IDs in bulk via MyGene.
  3. Reads PrimateAI-3D.hg38.txt and builds an in-memory lookup table.
  4. For each input CSV, looks up score_PAI3D by (enst_id, position, ref_aa, alt_aa).

The PrimateAI-3D file is tab-separated with one header row and 10 columns:
  #CHROM  POS  REF  ALT  gene_name  change_position_1based  ref_aa  alt_aa
  score_PAI3D  percentile_PAI3D

gene_name contains the ENST transcript ID used for lookup.
score_PAI3D ranges roughly 0–1; higher = more pathogenic.

Usage
-----
# Score all CSVs in a directory:
python -m vep_eval.score_proteingym_primateai3d \\
    --input data/clinical_ProteinGym_substitutions/ \\
    --pai-scores /path/to/PrimateAI-3D.hg38.txt \\
    --output-dir results/

# Quick test on first 20 proteins:
python -m vep_eval.score_proteingym_primateai3d \\
    --input data/clinical_ProteinGym_substitutions/ \\
    --pai-scores /path/to/PrimateAI-3D.hg38.txt \\
    --output-dir results/ \\
    --max-inputs 20
"""

import argparse
import logging
import sys
from pathlib import Path

import mygene
import pandas as pd
from tqdm import tqdm

from vep_eval.proteingym_io import (
    build_score_output,
    collect_csv_paths,
    load_gene_df,
    parse_mutant,
)
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

# Column names for the PrimateAI-3D file (header=0 overrides the file header)
PAI_COLS = [
    "#CHROM", "POS", "REF", "ALT",
    "gene_name", "change_position_1based", "ref_aa", "alt_aa",
    "score_PAI3D", "percentile_PAI3D",
]


# ---------------------------------------------------------------------------
# MyGene mapping
# ---------------------------------------------------------------------------


def map_np_to_enst(np_ids: list[str]) -> dict[str, str | None]:
    """
    Map RefSeq NP_ accessions to ENST transcript IDs via MyGene.

    When multiple transcripts are returned, takes the first one. Returns a dict
    mapping each NP_ accession to an ENST ID (or None if not found).
    """
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        np_ids, scopes="refseq", fields="ensembl.transcript", species="human", verbose=False
    )

    mapping: dict[str, str | None] = {}
    for res in results:
        refseq = res.get("query")
        enst = None
        ensembl_field = res.get("ensembl")
        if isinstance(ensembl_field, dict):
            enst = ensembl_field.get("transcript")
        elif isinstance(ensembl_field, list):
            enst = ensembl_field[0].get("transcript")
        # enst may itself be a list — take first element
        if isinstance(enst, list):
            enst = enst[0]
        mapping[refseq] = enst

    found = sum(v is not None for v in mapping.values())
    log.info("MyGene: mapped %d / %d NP_ IDs to ENST", found, len(np_ids))
    return mapping


# ---------------------------------------------------------------------------
# PrimateAI-3D loading
# ---------------------------------------------------------------------------


def load_pai_lookup(
    pai_path: Path, needed_ensts: set[str]
) -> dict[tuple, float]:
    """
    Read PrimateAI-3D.hg38.txt and return a lookup dict.

    Key: (enst_id, position, ref_aa, alt_aa)  e.g. ("ENST00000370321", 29, "R", "L")
    Value: score_PAI3D (float)

    Only retains rows for ENST IDs in *needed_ensts*.
    """
    log.info("Reading PrimateAI-3D scores from %s …", pai_path)

    df = pd.read_csv(
        pai_path,
        sep="\t",
        names=PAI_COLS,
        header=0,
        usecols=["gene_name", "change_position_1based", "ref_aa", "alt_aa", "score_PAI3D"],
    )

    df = df[df["gene_name"].isin(needed_ensts)]
    df["change_position_1based"] = df["change_position_1based"].astype(int)

    lookup: dict[tuple, float] = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing PAI3D", unit="rows"):
        key = (row["gene_name"], row["change_position_1based"], row["ref_aa"], row["alt_aa"])
        if key not in lookup:
            lookup[key] = float(row["score_PAI3D"])

    log.info(
        "PrimateAI-3D lookup table: %d entries for %d transcripts",
        len(lookup), len(needed_ensts),
    )
    return lookup


# ---------------------------------------------------------------------------
# Per-gene scoring
# ---------------------------------------------------------------------------


def score_gene_csv(
    csv_path: Path,
    np_to_enst: dict[str, str | None],
    pai_lookup: dict[tuple, float],
) -> pd.DataFrame:
    """
    Score all variants in a single ProteinGym gene CSV with PrimateAI-3D.

    Returns a DataFrame with columns: protein, mutant, pai_score, DMS_bin_score.
    """
    df, protein_id, wt_seq = load_gene_df(csv_path)
    enst_id = np_to_enst.get(protein_id)

    if enst_id is None:
        log.warning("  %s: no ENST ID found — all scores will be NaN", protein_id)

    log.info(
        "  %s  →  %s  seq_len=%d  n_variants=%d",
        protein_id, enst_id or "?", len(wt_seq), len(df),
    )

    scores: list[float] = []
    for mutant in df["mutant"]:
        if enst_id is None:
            scores.append(float("nan"))
            continue
        try:
            wt_aa, pos, mut_aa = parse_mutant(mutant)
        except ValueError:
            scores.append(float("nan"))
            continue
        scores.append(pai_lookup.get((enst_id, pos, wt_aa, mut_aa), float("nan")))

    return build_score_output(df, "pai_score", scores)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score ProteinGym substitution CSVs with PrimateAI-3D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True, metavar="PATH",
        help="Single ProteinGym CSV file or directory of CSV files.",
    )
    parser.add_argument(
        "--pai-scores", required=True, metavar="TXT",
        help=(
            "Path to PrimateAI-3D.hg38.txt "
            "(downloaded from https://primateai3d.basespace.illumina.com/)."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o", required=True, metavar="DIR",
        help="Base directory for results. Outputs are saved under <output-dir>/<run-name>/.",
    )
    add_run_name_args(parser)
    parser.add_argument(
        "--max-inputs", type=int, default=None, metavar="N",
        help="Limit processing to the first N CSV files (useful for quick tests).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.getLogger().setLevel(args.log_level)

    pai_path = Path(args.pai_scores)
    if not pai_path.exists():
        log.error(
            "PrimateAI-3D score file not found: %s\n"
            "Download it from https://primateai3d.basespace.illumina.com/",
            pai_path,
        )
        sys.exit(1)

    csv_paths = collect_csv_paths(Path(args.input), args.max_inputs)

    # --- Step 1: collect all protein IDs and map to ENST ---
    log.info("Collecting NP_ accessions from input CSVs …")
    np_ids: list[str] = []
    for p in csv_paths:
        _, protein_id, _ = load_gene_df(p)
        np_ids.append(protein_id)
    np_ids = list(dict.fromkeys(np_ids))  # deduplicate, preserve order

    np_to_enst = map_np_to_enst(np_ids)
    needed_ensts = {v for v in np_to_enst.values() if v is not None}

    # --- Step 2: load PrimateAI-3D lookup for needed transcripts ---
    pai_lookup = load_pai_lookup(pai_path, needed_ensts)

    # --- Step 3: score each gene CSV ---
    results = []
    for i, csv_path in enumerate(csv_paths, 1):
        log.info("[%d/%d] %s", i, len(csv_paths), csv_path.name)
        try:
            results.append(score_gene_csv(csv_path, np_to_enst, pai_lookup))
        except Exception as exc:
            log.error("  Failed on %s: %s", csv_path.name, exc)

    if not results:
        log.error("No results produced.")
        sys.exit(1)

    run_name = build_run_name(args.run_name, args.no_timestamp)
    out_dir = resolve_output_dir(args.output_dir, run_name)
    log.info("Run name: %s", run_name)

    output_path = out_dir / "scores.csv"
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(output_path, index=False)
    log.info("Saved %d rows → %s", len(combined), output_path)


if __name__ == "__main__":
    main()
