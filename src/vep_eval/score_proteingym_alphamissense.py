"""
Score ProteinGym substitution CSV files with AlphaMissense pre-computed scores.

AlphaMissense provides per-variant pathogenicity scores indexed by
(uniprot_id, protein_variant). This script:
  1. Collects all unique NP_ protein accessions from the input CSVs.
  2. Maps them to UniProt IDs in bulk via MyGene.
  3. Reads AlphaMissense_hg38.tsv in chunks, retaining only proteins of interest.
  4. For each input CSV, looks up am_pathogenicity by (uniprot_id, protein_variant).

The AlphaMissense TSV has 3 header/comment rows followed by tab-separated columns:
  #CHROM  POS  REF  ALT  genome_context  uniprot_id  transcript_id
  protein_variant  am_pathogenicity  am_class

am_pathogenicity ranges 0–1; higher = more likely pathogenic.

Usage
-----
# Score all CSVs in a directory:
python -m vep_eval.score_proteingym_alphamissense \\
    --input data/clinical_ProteinGym_substitutions/ \\
    --am-scores /path/to/AlphaMissense_hg38.tsv \\
    --output-dir results/

# Quick test on first 20 proteins:
python -m vep_eval.score_proteingym_alphamissense \\
    --input data/clinical_ProteinGym_substitutions/ \\
    --am-scores /path/to/AlphaMissense_hg38.tsv \\
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

AM_CHUNKSIZE = 500_000  # rows per chunk when reading the large AlphaMissense TSV
AM_SKIPROWS = 3  # comment/header rows before the column header in the TSV


# ---------------------------------------------------------------------------
# MyGene mapping
# ---------------------------------------------------------------------------


def map_np_to_uniprot(np_ids: list[str]) -> dict[str, str | None]:
    """
    Map RefSeq NP_ accessions to UniProt IDs via MyGene.

    Prefers Swiss-Prot entries; falls back to TrEMBL. Returns a dict
    mapping each NP_ accession to a UniProt ID (or None if not found).
    """
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        np_ids, scopes="refseq", fields="uniprot", species="human", verbose=False
    )

    mapping: dict[str, str | None] = {}
    for res in results:
        refseq = res.get("query")
        uniprot_field = res.get("uniprot", {})
        uniprot_id = None

        if isinstance(uniprot_field, dict):
            ids = uniprot_field.get("Swiss-Prot") or uniprot_field.get("TrEMBL")
            uniprot_id = ids[0] if isinstance(ids, list) else ids
        elif isinstance(uniprot_field, list):
            uniprot_id = uniprot_field[0]

        mapping[refseq] = uniprot_id

    found = sum(v is not None for v in mapping.values())
    log.info("MyGene: mapped %d / %d NP_ IDs to UniProt", found, len(np_ids))
    return mapping


# ---------------------------------------------------------------------------
# AlphaMissense loading
# ---------------------------------------------------------------------------


def load_am_lookup(am_path: Path, needed_uniprots: set[str]) -> dict[tuple, float]:
    """
    Read AlphaMissense_hg38.tsv and return a lookup dict.

    Key: (uniprot_id, protein_variant)  e.g. ("Q9Y6W5", "R29L")
    Value: am_pathogenicity score (float)

    Only retains rows for UniProt IDs in *needed_uniprots* to keep memory
    usage manageable when processing a subset of the full dataset.
    """
    log.info("Reading AlphaMissense scores from %s …", am_path)
    lookup: dict[tuple, float] = {}

    for chunk in tqdm(
        pd.read_csv(
            am_path,
            sep="\t",
            skiprows=AM_SKIPROWS,
            chunksize=AM_CHUNKSIZE,
            usecols=["uniprot_id", "protein_variant", "am_pathogenicity"],
        ),
        desc="Loading AlphaMissense",
        unit="chunk",
    ):
        chunk = chunk[chunk["uniprot_id"].isin(needed_uniprots)]
        for _, row in chunk.iterrows():
            key = (row["uniprot_id"], row["protein_variant"])
            # Keep first occurrence; AM should have exactly one entry per key
            if key not in lookup:
                lookup[key] = float(row["am_pathogenicity"])

    log.info("AlphaMissense lookup table: %d entries for %d proteins", len(lookup), len(needed_uniprots))
    return lookup


# ---------------------------------------------------------------------------
# Per-gene scoring
# ---------------------------------------------------------------------------


def score_gene_csv(
    csv_path: Path,
    np_to_uniprot: dict[str, str | None],
    am_lookup: dict[tuple, float],
) -> pd.DataFrame:
    """
    Score all variants in a single ProteinGym gene CSV with AlphaMissense.

    Returns a DataFrame with columns: protein, mutant, am_score, DMS_bin_score.
    """
    df, protein_id, wt_seq = load_gene_df(csv_path)
    uniprot_id = np_to_uniprot.get(protein_id)

    if uniprot_id is None:
        log.warning("  %s: no UniProt ID found — all scores will be NaN", protein_id)

    log.info(
        "  %s  →  %s  seq_len=%d  n_variants=%d",
        protein_id, uniprot_id or "?", len(wt_seq), len(df),
    )

    scores: list[float] = []
    for mutant in df["mutant"]:
        if uniprot_id is None:
            scores.append(float("nan"))
            continue
        try:
            parse_mutant(mutant)  # validate format; AM uses same notation
        except ValueError:
            scores.append(float("nan"))
            continue
        scores.append(am_lookup.get((uniprot_id, mutant), float("nan")))

    return build_score_output(df, "am_score", scores)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score ProteinGym substitution CSVs with AlphaMissense.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True, metavar="PATH",
        help="Single ProteinGym CSV file or directory of CSV files.",
    )
    parser.add_argument(
        "--am-scores", required=True, metavar="TSV",
        help="Path to AlphaMissense_hg38.tsv (downloaded from alphamissense.hegelab.org).",
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

    am_path = Path(args.am_scores)
    if not am_path.exists():
        log.error(
            "AlphaMissense TSV not found: %s\n"
            "Download it from https://alphamissense.hegelab.org/",
            am_path,
        )
        sys.exit(1)

    csv_paths = collect_csv_paths(Path(args.input), args.max_inputs)

    # --- Step 1: collect all protein IDs and map to UniProt ---
    log.info("Collecting NP_ accessions from input CSVs …")
    np_ids: list[str] = []
    for p in csv_paths:
        _, protein_id, _ = load_gene_df(p)
        np_ids.append(protein_id)
    np_ids = list(dict.fromkeys(np_ids))  # deduplicate, preserve order

    np_to_uniprot = map_np_to_uniprot(np_ids)
    needed_uniprots = {v for v in np_to_uniprot.values() if v is not None}

    # --- Step 2: load AlphaMissense lookup for needed proteins ---
    am_lookup = load_am_lookup(am_path, needed_uniprots)

    # --- Step 3: score each gene CSV ---
    results = []
    for i, csv_path in enumerate(csv_paths, 1):
        log.info("[%d/%d] %s", i, len(csv_paths), csv_path.name)
        try:
            results.append(score_gene_csv(csv_path, np_to_uniprot, am_lookup))
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
