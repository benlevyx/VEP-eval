"""
Score ProteinGym substitution CSV files with SIFT scores via the Ensembl VEP REST API.

Each input CSV (one per gene) contains columns:
  protein, protein_sequence, mutant, mutated_sequence, DMS_bin_score

For every variant (mutant column, e.g. "R29L"), SIFT scores are retrieved using
HGVS protein notation (e.g. "NP_000007.1:p.Arg29Leu") submitted in batches to the
Ensembl VEP REST API. The protein ID is taken from the CSV's ``protein`` column,
which must be a valid RefSeq NP_ accession recognised by Ensembl.

Note: SIFT scores range from 0 (deleterious) to 1 (tolerated). Lower = more damaging,
which is the opposite convention from ESM LLR.

Usage
-----
# Score a single gene CSV:
python -m vep_eval.score_proteingym_sift \\
    --input data/clinical_ProteinGym_substitutions/NP_000007.1.csv \\
    --output-dir results/

# Score all CSVs in a directory:
python -m vep_eval.score_proteingym_sift \\
    --input data/clinical_ProteinGym_substitutions/ \\
    --output-dir results/
"""

import argparse
import logging
import sys
from pathlib import Path

import backoff
import pandas as pd
import requests

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

ENSEMBL_VEP_URL = "https://rest.ensembl.org/vep/human/hgvs"
ENSEMBL_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

# Ensembl allows up to 300 HGVS notations per POST request
BATCH_SIZE = 200

# Retry settings for transient API errors
MAX_RETRIES = 5

# HGVS requires three-letter amino acid codes
AA_THREE = {
    "A": "Ala",
    "R": "Arg",
    "N": "Asn",
    "D": "Asp",
    "C": "Cys",
    "Q": "Gln",
    "E": "Glu",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "K": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val",
    "*": "Ter",
}


# ---------------------------------------------------------------------------
# HGVS helpers
# ---------------------------------------------------------------------------


def to_hgvs_protein(protein_id: str, wt_aa: str, pos: int, mut_aa: str) -> str | None:
    """
    Build an HGVS protein notation string, e.g. "NP_000007.1:p.Leu26Arg".

    Returns None if either amino acid is not in the three-letter lookup table.
    """
    wt3 = AA_THREE.get(wt_aa)
    mut3 = AA_THREE.get(mut_aa)
    if wt3 is None or mut3 is None:
        return None
    return f"{protein_id}:p.{wt3}{pos}{mut3}"


# ---------------------------------------------------------------------------
# Ensembl VEP API
# ---------------------------------------------------------------------------


class _RetryableHTTPError(Exception):
    """Raised for HTTP 429 / 5xx so backoff can intercept it."""


def _on_backoff(details):
    log.warning(
        "Retrying VEP request (attempt %d, waited %.1fs so far)",
        details["tries"], details["elapsed"],
    )


@backoff.on_exception(
    backoff.expo,
    (requests.RequestException, _RetryableHTTPError),
    max_tries=MAX_RETRIES,
    on_backoff=_on_backoff,
)
def _post_vep_batch(hgvs_list: list[str], timeout: int = 60) -> list[dict]:
    """
    POST a batch of HGVS notations to the Ensembl VEP REST API.

    Retries on network errors, HTTP 429, and 5xx with exponential back-off.
    Returns a list of VEP result objects (may be shorter than input if some
    notations are unrecognised).
    """
    resp = requests.post(
        ENSEMBL_VEP_URL,
        headers=ENSEMBL_HEADERS,
        json={"hgvs_notations": hgvs_list},
        timeout=timeout,
    )
    if resp.status_code == 429 or resp.status_code >= 500:
        raise _RetryableHTTPError(f"HTTP {resp.status_code}")
    resp.raise_for_status()
    return resp.json()


def _extract_sift_score(vep_result: dict) -> float:
    """
    Extract the most deleterious (lowest) SIFT score from a VEP result object.

    Returns NaN if no SIFT score is present.
    """
    consequences = vep_result.get("transcript_consequences", [])
    scores = [c["sift_score"] for c in consequences if "sift_score" in c]
    if not scores:
        return float("nan")
    return min(scores)  # lower = more deleterious


def fetch_sift_scores(hgvs_notations: list[str | None]) -> list[float]:
    """
    Fetch SIFT scores for a list of HGVS notations (None entries map to NaN).

    Sends requests in batches of BATCH_SIZE. Returns one score per input notation.
    """
    # Index valid notations, leaving None slots as NaN placeholders
    results = [float("nan")] * len(hgvs_notations)
    valid_indices = [(i, h) for i, h in enumerate(hgvs_notations) if h is not None]

    for batch_start in range(0, len(valid_indices), BATCH_SIZE):
        batch = valid_indices[batch_start : batch_start + BATCH_SIZE]
        indices, notations = zip(*batch)

        log.debug(
            "  VEP API batch %d–%d (%d notations)",
            batch_start + 1,
            batch_start + len(batch),
            len(batch),
        )

        try:
            vep_results = _post_vep_batch(list(notations))
        except Exception as exc:
            log.error("  VEP batch failed: %s — marking batch as NaN", exc)
            continue

        # VEP results are keyed by the input HGVS notation in the "input" field
        score_map = {r["input"]: _extract_sift_score(r) for r in vep_results}

        for i, notation in zip(indices, notations):
            results[i] = score_map.get(notation, float("nan"))

    return results


# ---------------------------------------------------------------------------
# Per-gene scoring
# ---------------------------------------------------------------------------


def score_gene_csv(csv_path: Path) -> pd.DataFrame:
    """
    Score all variants in a single ProteinGym gene CSV with SIFT.

    Returns a DataFrame with columns: protein, mutant, sift_score, DMS_bin_score.
    """
    df, protein_id, wt_seq = load_gene_df(csv_path)
    log.info("  %s  seq_len=%d  n_variants=%d", protein_id, len(wt_seq), len(df))

    hgvs_notations: list[str | None] = []
    n_skipped = 0
    for mutant in df["mutant"]:
        try:
            wt_aa, pos, mut_aa = parse_mutant(mutant)
        except ValueError as exc:
            log.debug("  Skipping %r: %s", mutant, exc)
            n_skipped += 1
            hgvs_notations.append(None)
            continue
        hgvs_notations.append(to_hgvs_protein(protein_id, wt_aa, pos, mut_aa))

    if n_skipped:
        log.warning("  %d unparseable mutant(s) in %s", n_skipped, csv_path.name)

    scores = fetch_sift_scores(hgvs_notations)
    return build_score_output(df, "sift_score", scores)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score ProteinGym substitution CSVs with SIFT via Ensembl VEP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="PATH",
        help="Single ProteinGym CSV file or directory of CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        metavar="DIR",
        help="Base directory for results. Outputs are saved under <output-dir>/<run-name>/.",
    )
    add_run_name_args(parser)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        metavar="N",
        help="Number of HGVS notations per Ensembl VEP API request.",
    )
    parser.add_argument(
        "--max-inputs",
        type=int,
        default=None,
        metavar="N",
        help="Limit processing to the first N CSV files (useful for quick tests).",
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

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    csv_paths = collect_csv_paths(Path(args.input), args.max_inputs)

    results = []
    for i, csv_path in enumerate(csv_paths, 1):
        log.info("[%d/%d] %s", i, len(csv_paths), csv_path.name)
        try:
            results.append(score_gene_csv(csv_path))
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
    log.info("Saved %d rows to %s", len(combined), output_path)


if __name__ == "__main__":
    main()
