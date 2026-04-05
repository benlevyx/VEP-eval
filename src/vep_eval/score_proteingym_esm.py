"""
Score ProteinGym substitution CSV files with ESM1b log-likelihood ratios.

Each input CSV (one per gene) contains columns:
  protein, protein_sequence, mutant, mutated_sequence, DMS_bin_score

For every variant (mutant column, e.g. "R29L") the LLR is looked up from the
full LLR matrix computed on the wild-type sequence:
  score = log P(mut_aa | context) - log P(wt_aa | context)

Long sequences (>1022 AA) are handled via sigmoid-weighted tiling.

Usage
-----
# Score a single gene CSV:
python -m vep_eval.score_proteingym_esm \\
    --input data/clinical_ProteinGym_substitutions/NP_000007.1.csv \\
    --output NP_000007.1_esm_scores.csv

# Score all CSVs in a directory (outputs one merged CSV):
python -m vep_eval.score_proteingym_esm \\
    --input data/clinical_ProteinGym_substitutions/ \\
    --output all_esm_scores.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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

AA_ORDER = list("KRHEDNQTSCGAVLIMPYFW")  # 20 canonical AAs in ESM's preferred order

# ESM1b context window
ESM_MAX_LEN = 1022


# ---------------------------------------------------------------------------
# ESM model
# ---------------------------------------------------------------------------


def load_esm_model(model_name: str, device: str):
    """Load an ESM model from torch hub. Returns (model, alphabet, batch_converter)."""
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    batch_converter = alphabet.get_batch_converter()
    return model.eval().to(device), alphabet, batch_converter


# ---------------------------------------------------------------------------
# LLR computation
# ---------------------------------------------------------------------------


def _logits_for_sequence(seq: str, model, alphabet, batch_converter, device: str) -> np.ndarray:
    """
    Run one ESM forward pass on *seq* and return log-softmax logits.

    Returns an array of shape (len(seq), vocab_size), with BOS/EOS tokens
    already stripped.
    """
    batch_labels, batch_strs, batch_tokens = batch_converter([("_", seq)])
    with torch.no_grad():
        logits = torch.log_softmax(
            model(batch_tokens.to(device), repr_layers=[], return_contacts=False)["logits"],
            dim=-1,
        )
    # Strip BOS (index 0) and EOS (index -1)
    return logits[0, 1:-1, :].cpu().numpy()


def _tiling_weights(seq_len: int, min_overlap: int = 512, s: int = 20):
    """
    Compute tiled interval indices and per-position sigmoid blend weights for
    sequences longer than ESM_MAX_LEN.

    Returns (intervals, weight_matrix_normalised) where weight_matrix_normalised
    has shape (n_tiles, seq_len) and columns sum to 1.
    """
    def _chop(idx):
        return idx[ESM_MAX_LEN - min_overlap: -ESM_MAX_LEN + min_overlap]

    def _build_intervals(idx, parts=None):
        parts = parts or []
        if len(idx) <= ESM_MAX_LEN:
            # Add a centred tile if the overlap with the previous tile is small
            if parts and parts[-2][-1] - parts[-1][0] < min_overlap:
                mid = idx[len(idx) // 2]
                parts.append(np.arange(mid - ESM_MAX_LEN // 2, mid + ESM_MAX_LEN // 2))
            return parts
        parts += [idx[:ESM_MAX_LEN], idx[-ESM_MAX_LEN:]]
        return _build_intervals(_chop(idx), parts)

    ints = _build_intervals(np.arange(seq_len))
    ints = [ints[i] for i in np.argsort([t[0] for t in ints])]

    a = min_overlap // 2
    t = np.arange(ESM_MAX_LEN)
    sigmoid_left  = 1 / (1 + np.exp(-(t[:a] - a / 2) / s))
    sigmoid_right = 1 / (1 + np.exp( (t[:a] - a / 2) / s))

    def _tile_filter(i, n):
        f = np.ones(ESM_MAX_LEN)
        if i > 0:        f[:a]            = sigmoid_left
        if i < n - 1:    f[ESM_MAX_LEN - a:] = sigmoid_right
        return f

    n = len(ints)
    M = np.zeros((n, seq_len))
    for k, idx in enumerate(ints):
        M[k, idx] = _tile_filter(k, n)
    M_norm = M / M.sum(axis=0)
    return ints, M_norm


def compute_llr_matrix(
    seq: str,
    protein_id: str,
    model,
    alphabet,
    batch_converter,
    device: str,
) -> pd.DataFrame:
    """
    Compute the full ESM log-likelihood ratio (LLR) matrix for *seq*.

    LLR[mut_aa, "<wt_aa> <pos>"] = log P(mut_aa | context) - log P(wt_aa | context)

    Handles sequences longer than ESM_MAX_LEN via sigmoid-weighted tiling.
    Returns a DataFrame with shape (20, len(seq)).
    """
    if len(seq) <= ESM_MAX_LEN:
        logits = _logits_for_sequence(seq, model, alphabet, batch_converter, device)
    else:
        log.debug("  %s: sequence length %d > %d, using tiling", protein_id, len(seq), ESM_MAX_LEN)
        ints, M_norm = _tiling_weights(len(seq))

        logits_full = np.zeros((len(seq), len(alphabet.all_toks)))
        for k, idx in enumerate(ints):
            tile_seq = "".join(np.array(list(seq))[idx])
            tile_logits = _logits_for_sequence(tile_seq, model, alphabet, batch_converter, device)
            weighted = tile_logits * M_norm[k, idx, np.newaxis]
            logits_full[idx] += weighted

        logits = logits_full

    # Select the 20 canonical AA columns and build a (20 x L) DataFrame
    aa_indices = [alphabet.tok_to_idx[aa] for aa in AA_ORDER]
    logit_df = pd.DataFrame(
        logits[:, aa_indices].T,           # shape (20, L)
        index=AA_ORDER,
        columns=[f"{aa} {i + 1}" for i, aa in enumerate(seq)],
    )

    # Subtract wild-type log-prob at each position to get LLR
    wt_log_probs = np.array([logit_df.loc[aa, f"{aa} {i + 1}"] for i, aa in enumerate(seq)])
    llr_df = logit_df - wt_log_probs

    return llr_df


# ---------------------------------------------------------------------------
# Variant lookup helpers
# ---------------------------------------------------------------------------


def lookup_llr(llr_df: pd.DataFrame, wt_aa: str, pos: int, mut_aa: str) -> float:
    """Return LLR for a substitution, or NaN if position/AA is out of range."""
    col = f"{wt_aa} {pos}"
    if col not in llr_df.columns or mut_aa not in llr_df.index:
        return float("nan")
    return float(llr_df.loc[mut_aa, col])


# ---------------------------------------------------------------------------
# Per-gene scoring
# ---------------------------------------------------------------------------


def score_gene_csv(
    csv_path: Path,
    model,
    alphabet,
    batch_converter,
    device: str,
) -> pd.DataFrame:
    """
    Score all variants in a single ProteinGym gene CSV.

    Returns a DataFrame with columns: protein, mutant, esm_score, DMS_bin_score.
    """
    df, protein_id, wt_seq = load_gene_df(csv_path)
    log.info("  %s  seq_len=%d  n_variants=%d", protein_id, len(wt_seq), len(df))

    llr_df = compute_llr_matrix(wt_seq, protein_id, model, alphabet, batch_converter, device)

    scores = []
    n_skipped = 0
    for mutant in df["mutant"]:
        try:
            wt_aa, pos, mut_aa = parse_mutant(mutant)
        except ValueError as exc:
            log.debug("  Skipping %r: %s", mutant, exc)
            n_skipped += 1
            scores.append(float("nan"))
            continue
        scores.append(lookup_llr(llr_df, wt_aa, pos, mut_aa))

    if n_skipped:
        log.warning("  %d unparseable mutant(s) in %s", n_skipped, csv_path.name)

    return build_score_output(df, "esm_score", scores)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score ProteinGym substitution CSVs with ESM1b log-likelihood ratios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True, metavar="PATH",
        help="Single ProteinGym CSV file or directory of CSV files.",
    )
    parser.add_argument(
        "--output-dir", "-o", required=True, metavar="DIR",
        help="Base directory for results. Outputs are saved under <output-dir>/<run-name>/.",
    )
    add_run_name_args(parser)
    parser.add_argument(
        "--model-name", default="esm1b_t33_650M_UR50S", metavar="NAME",
        help="ESM model name (see https://github.com/facebookresearch/esm#available).",
    )
    parser.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="Torch device (e.g. 'cuda', 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--max-inputs", type=int, default=None, metavar="N",
        help="Limit processing to the first N CSV files (useful for quick tests).",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.getLogger().setLevel(args.log_level)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading model: %s", args.model_name)
    model, alphabet, batch_converter = load_esm_model(args.model_name, device)

    csv_paths = collect_csv_paths(Path(args.input), args.max_inputs)

    results = []
    for i, csv_path in enumerate(csv_paths, 1):
        log.info("[%d/%d] %s", i, len(csv_paths), csv_path.name)
        try:
            results.append(score_gene_csv(csv_path, model, alphabet, batch_converter, device))
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
