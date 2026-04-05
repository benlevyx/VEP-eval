# Genomic heterogeneity inflates the performance of variant pathogenicity predictions

[![bioRxiv](https://img.shields.io/badge/bioRxiv-Available-red)](https://www.biorxiv.org/content/10.1101/2025.09.05.674459v2)
This is the official repository for our paper [Genomic heterogeneity inflates the performance of variant pathogenicity predictions](https://www.biorxiv.org/content/10.1101/2025.09.05.674459v2).

It provides a **genome-wide, variant-type-stratified benchmark dataset** (>250,000 ClinVar variants) and the code to evaluate state-of-the-art DNA-based and protein-based models for variant pathogenicity prediction.

## Contents

- [Notebooks](#notebooks)
- [CLI Scoring Scripts](#cli-scoring-scripts)
- [Results](#results)
- [Citation](#citation)

## Notebooks

We provide one-click Jupyter notebook examples for each evaluated model, benchmark creation, and results visualization.  

- **DNA-based models**:  
  AlphaGenome, DNABERT2, Evo2, GPN-MSA, Nucleotide Transformer (NT), PhyloGPN, PhyloP  
  → Notebooks are available in the [`DNA-based Models/`](DNA-based%20Models/) directory.  

- **Protein-based models**:  
  ESM family models, AlphaMissense, PrimateAI-3D  
  → Notebooks are available in the [`protein_models/`](protein_models/) directory.  

- **Benchmark creation**:  
  → See [`VEP_ClinVar_Benchmarking_RefSeq.ipynb`](VEP_ClinVar_Benchmarking_RefSeq.ipynb).  

- **Visualization**:  
  → See [`VEP_AUROC_figure.ipynb`](VEP_AUROC_figure.ipynb).


## CLI Scoring Scripts

The `src/vep_eval/` package provides command-line scripts for scoring ProteinGym substitution CSVs and analyzing results. Install dependencies with `uv sync` before running.

### ESM1b

```bash
python -m vep_eval.score_proteingym_esm \
    --input data/clinical_ProteinGym_substitutions/ \
    --output-dir results/ \
    --run-name esm \
    --model-name esm1b_t33_650M_UR50S \
    --device cpu
```

### SIFT (via Ensembl VEP REST API)

```bash
python -m vep_eval.score_proteingym_sift \
    --input data/clinical_ProteinGym_substitutions/ \
    --output-dir results/ \
    --run-name sift
```

### AlphaMissense

Requires pre-computed scores downloaded from [alphamissense.hegelab.org](https://alphamissense.hegelab.org/).

```bash
python -m vep_eval.score_proteingym_alphamissense \
    --input data/clinical_ProteinGym_substitutions/ \
    --am-scores /path/to/AlphaMissense_hg38.tsv \
    --output-dir results/ \
    --run-name alphamissense
```

### PrimateAI-3D

Requires pre-computed scores downloaded from [primateai3d.basespace.illumina.com](https://primateai3d.basespace.illumina.com/).

```bash
python -m vep_eval.score_proteingym_primateai3d \
    --input data/clinical_ProteinGym_substitutions/ \
    --pai-scores /path/to/PrimateAI-3D.hg38.txt \
    --output-dir results/ \
    --run-name primateai3d
```

### Visualization

```bash
python -m vep_eval.visualize_esm_scores \
    --input results/<run-name>/scores.csv \
    --output-dir figures/ \
    --no-timestamp --run-name <model-name>
```

Pass `--negate` for scores where lower = more pathogenic (SIFT, ESM LLR).

### Conservation Bucket Analysis

Buckets variants by SIFT-based conservation (high: SIFT < 0.05, medium: 0.05–0.20, low: ≥ 0.20) and reports AUROC per bucket per model. Pass one `--scores path:Label` flag per model.

```bash
python -m vep_eval.analyze_conservation_buckets \
    --sift-scores results/<sift-run>/scores.csv \
    --scores results/<esm-run>/scores.csv:ESM1b \
    --scores results/<am-run>/scores.csv:AlphaMissense \
    --scores results/<pai-run>/scores.csv:PrimateAI3D \
    --output-dir figures/ \
    --no-timestamp --run-name conservation_analysis
```

## Results

<p align="center">
  <img src="/Figure1.svg" alt="AUROC Results by Variant Type" width="900">
</p>

**Figure 1. Pathogenicity prediction performance of frontier sequence-based models across variant types.**  
Evaluation and comparison of DNA and protein sequence AI models for their capacity to distinguish between pathogenic and benign variants across variant types, measured by the area under the receiver operating characteristic curve (AUROC). Error bars denote 95% confidence intervals estimated by stratified bootstrap resampling (1,000 iterations) within each variant group.
- **%P** indicates the proportion of pathogenic variants in each group.  
- Some groups are defined by multiple annotated effects (e.g., both missense and 3′ UTR, with respect to different transcripts).  
- **DNA models** are shown as solid bars, **protein models** as dashed bars.  

*Note: The evaluation of PrimateAI-3D on stop-gain variants includes only 19,795 variants.*

## Citation

If you find this benchmark useful for your research, please cite our paper:

```bibtex
@article{genomic2025biorxiv,
  author    = {Baiyu Lu and Xueshen Liu and Po-Yu Lin and Nadav Brandes},
  title     = {Genomic heterogeneity inflates the performance of variant pathogenicity predictions},
  journal   = {bioRxiv},
  year      = {2025},
  doi       = {10.1101/2025.09.05.674459},
  url       = {https://www.biorxiv.org/content/10.1101/2025.09.05.674459v2},
  eprint    = {https://www.biorxiv.org/content/10.1101/2025.09.05.674459v2.full.pdf}
}
