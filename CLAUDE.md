# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for benchmarking variant pathogenicity prediction models. Evaluates DNA-based and protein-based AI models against a genome-wide ClinVar benchmark of >250,000 variants stratified by variant type (missense, frameshift, splice, stop-gain, 5'UTR, 3'UTR, etc.).

Paper: *"Genomic heterogeneity inflates the performance of variant pathogenicity predictions"* (bioRxiv).

## Running the Code

This is a Jupyter notebook-based project with no build system. Run notebooks sequentially cell-by-cell in Jupyter/JupyterLab. Each notebook uses inline `!pip install` commands for dependencies.

**Execution order:**
1. `VEP_ClinVar_Benchmarking_RefSeq.ipynb` — creates the benchmark dataset from ClinVar
2. `DNA-based Models/*.ipynb` or `protein_models/*.ipynb` — score variants with each model
3. `VEP_AUROC_figure.ipynb` — compute AUROC metrics and generate figures

## Architecture

### Data Flow

```
ClinVar + MANE annotations
        ↓
VEP_ClinVar_Benchmarking_RefSeq.ipynb
        ↓
ClinVarBenchmark_PB_*.csv  (benchmark dataset with variant coordinates + labels)
        ↓
[Model notebooks] — each loads the benchmark CSV, runs scoring, outputs all_scores_*.csv
        ↓
VEP_AUROC_figure.ipynb — merges all scores, computes AUROC + 95% CIs, generates figures
```

### Model Notebooks (shared pattern)

Each model notebook follows the same structure:
1. Install dependencies
2. Load benchmark CSV, extract variant info (chrom, pos, ref, alt)
3. Score variants through the model
4. Merge scores back into benchmark dataframe
5. Save `all_scores_<model>.csv`

**DNA models** (`DNA-based Models/`): AlphaGenome, DNABERT2, Evo2, GPNMSA, NT, PhyloGPN, PhyloP — extract hg38 genome sequences around variant positions.

**Protein models** (`protein_models/`): AlphaMissense, ESM_models, PrimateAI3D — map RefSeq IDs to UniProt IDs (via MyGene) and process protein sequences.

### Analysis Notebook

`VEP_AUROC_figure.ipynb` implements stratified bootstrap resampling (1,000 iterations) for 95% confidence intervals on AUROC scores per model per variant type. Outputs Figure 1 and Figure 2.

## Key External Data (not in repo)

- ClinVar variant data — downloaded within the benchmarking notebook
- hg38 genome FASTA — downloaded within DNA model notebooks
- Pre-trained model weights — downloaded individually within each model notebook
- AlphaMissense precomputed scores (`AlphaMissense_hg38.tsv`) — external download required

## Dependencies

No `requirements.txt` — dependencies are installed inline per notebook. Core packages: `pandas`, `numpy`, `scipy`, `sklearn`, `torch`, `transformers`, `pyfaidx`, `mygene`, `biopython`, `matplotlib`, `seaborn`.
