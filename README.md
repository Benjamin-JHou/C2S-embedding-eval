# C2S-Embedding-Eval: Systematic Benchmarking and Domain Evaluation of Cell2Sentence Single-Cell Embeddings

**C2S-Embedding-Eval** is a computational benchmarking and evaluation suite designed to systematically quantify the representation quality, domain dependency, subpopulation resolution, and perturbation-screening utility of **Cell2Sentence (`C2S-Pythia-410M`)** single-cell foundation model embeddings against classical linear dimensionality reduction (**PCA**).

By evaluating frozen `1,024`-dimensional C2S transformer representations alongside `50`-component PCA representations on identical single-cell transcriptomes ($100,064$ cells across $9$ major lineages in `GSE176078`, plus external validation on `pbmc3k` and `paul15`), this repository isolates how pre-training corpus composition, preprocessing alignment, and biological granularity govern foundation-model utility.

---

## Core Analytical Framework: The Three-Factor Model ($T \times P \times G$)

Our evaluation formalizes three quantitative determinants that predict when single-cell foundation-model embeddings outperform traditional linear representations:

1. **Training Corpus Representation ($T$):** The representation frequency of a target lineage's gene expression programs in the pre-training corpus (quantified via BPE tokenizer ID distributions and lineage-stratified clustering compactness). Because large-scale single-cell repositories (CellxGene, Human Cell Atlas) are heavily enriched for hematopoietic and lymphoid tissues, C2S excels in immune and stromal compartments while underperforming in tissue-specific epithelial compartments.
2. **Preprocessing Alignment ($P$):** Consistency between input data normalization/filtering and the foundation model's pre-training pipeline (log-normalization to $10,000$ counts per cell, filtering to the C2S gene vocabulary, and matched scaling).
3. **Biological Granularity ($G$):** The resolution of the downstream task—ranging from coarse global cell-type separation and microenvironment-invariant cell identity to fine-grained within-lineage subpopulation discovery.

---

## Key Benchmark Findings

- **Global vs. Lineage-Stratified Clustering (`Code/run_benchmark.py`, `Code/deep_analysis.py`):**
  - At global resolution across $100,064$ cells (`GSE176078`), C2S achieves a $60\%$ higher Calinski-Harabasz (CH) Index than PCA ($1,422$ vs. $884$, non-overlapping $95\%$ bootstrap CIs: $[1,338, 1,494]$ vs. $[791, 938]$).
  - Stratified by lineage ($n = 300$ bootstrap resamples across $9\text{ lineages} \times 5\text{ metrics} = 45\text{ comparisons}$), **6 statistically significant C2S advantages** (non-overlapping $95\%$ CIs) emerge exclusively in **immune** (B-cell CH & Davies-Bouldin, Myeloid CH) and **stromal** (Endothelial CH, CAF CH, PVL CH) lineages.
  - Conversely, **epithelial lineages** exhibit complete C2S underperformance ($0/10$ metrics won across Cancer and Normal Epithelial subsets; binomial $p = 1.0 \times 10^{-3}$), with PCA holding significant ARI advantages in both Cancer Epithelial ($0.343$ vs. $0.148$) and Normal Epithelial ($0.879$ vs. $0.441$).
- **Token Frequency & Microenvironment Robustness (`Code/I3_token_domain_analysis.py`):**
  - Across $22$ fine-grained sub-lineages, marker gene BPE token IDs correlate positively with C2S/PCA compactness ratios ($\rho = 0.30$, $p = 0.17$, $95\%$ bootstrap CI $[-0.09, 0.67]$).
  - Comparing T-cells across distinct tumor microenvironments ($\text{ER}^+$ vs. $\text{TNBC}$) demonstrates **$148\text{-fold}$ greater condition invariance** in C2S space (centroid cosine distance $0.012$) compared to PCA ($1.772$).
- **T-Cell Subpopulation Discovery & Doublet QC (`Code/I1_tcell_subclustering.py`):**
  - High-resolution clustering ($k = 15$) of $35,214$ T-cells identifies $5$ C2S-specific clusters with PCA purity $< 0.50$ (overall C2S–PCA cluster agreement $\text{ARI} = 0.350$).
  - Quantitative immunoglobulin (IG) and `CD3D` doublet quality control flags Cluster 8 ($n = 1,858$, IG mean $2.5\times$ cohort average, $\text{CD3D}^+$ $32\%$) as B-cell/T-cell doublets.
  - On the remaining **$33,356$ clean T-cells**, C2S wins $4/5$ clustering metrics ($\text{ARI } 0.592\text{ vs. }0.309$, **$+91\%$ improvement**; $\text{NMI } 0.719\text{ vs. }0.488$) and resolves $4$ validated T-cell states: `CDKN1A+` stress-responsive, `FCGR3A+/FGFBP2+` NK-like cytotoxic effector, `NR4A1+/HSPA6+` immediate-early activated, and `IGKV+` subsets.
- **Preprocessing Sensitivity on External Data (`Code/run_benchmark.py`):**
  - On default-preprocessed `pbmc3k` ($2,638$ cells), C2S loses $5/5$ metrics to PCA ($\text{ARI } 0.030\text{ vs. }0.653$). Reprocessing `pbmc3k` with the C2S-compatible pipeline improves C2S ARI **$9.7\text{-fold}$** ($0.030 \rightarrow 0.290$) and shifts C2S to winning $3/5$ metrics (Silhouette, CH, and DB).
- **Multi-Dimensional LINCS L1000 Drug Screening Benchmark (`Code/J5_full_scale_screening.py`, `Code/J6_p0_fixes.py`):**
  - Evaluates $10,755$ LINCS L1000 perturbation signatures against breast cancer epithelial DE signatures across four scoring dimensions: GSEA composite, standard CMap connectivity, C2S disease-reversal embedding alignment, and a 4-dimensional enhanced composite score.
  - Benchmarks discrimination of $15$ standard-of-care (SOC) breast cancer drugs against negative controls and background compounds, demonstrating an honest negative result across all four dimensions (Mann-Whitney U $p = 0.60, 0.11, 0.84,\text{ and }0.87$, respectively) due to the resolution limits of the 978-gene L1000 landmark panel.

---

## Repository Structure

```text
C2S-embedding-eval/
├── README.md                          # Project documentation and usage guide
├── Source_Data_Guide.md               # Schema and description for all 24 processed CSVs
├── requirements.txt                   # Python package dependencies
├── LICENSE                            # MIT License
├── Code/                              # Core computational analysis and validation scripts
│   ├── run_benchmark.py               # Cross-dataset C2S vs. PCA clustering benchmark
│   ├── deep_analysis.py               # Token frequency audit, lineage compactness & subcluster discovery
│   ├── I1_tcell_subclustering.py      # T-cell subpopulation clustering and doublet QC
│   ├── I3_token_domain_analysis.py    # BPE token ID distribution and gene-gene correlation analysis
│   ├── J5_full_scale_screening.py     # Full-scale LINCS L1000 GSEA + pathway + toxicity screening
│   ├── J6_p0_fixes.py                 # 4-dimensional drug screening benchmark (GSEA, CMap, C2S alignment)
│   └── validate_pipeline.py           # Automated integrity validator for Source_Data/ outputs
└── Source_Data/                       # Processed benchmark tables and statistical results (24 CSVs)
    ├── benchmark_results.csv
    ├── lineage_performance.csv
    ├── token_frequency_by_lineage.csv
    ├── 1.1_doublet_cluster_summary.csv
    ├── 1.2_tcell_subcluster_metrics_clean.csv
    ├── 1.2_novel_cluster_markers.csv
    ├── 1.3_token_frequency_expanded.csv
    ├── 1.3_correlation_statistics.csv
    ├── 1.4_bootstrap_ci_global.csv
    ├── 1.4_bootstrap_ci_lineage.csv
    ├── 1.4_bootstrap_ci_tcell.csv
    ├── 1.5_pbmc3k_reprocessed_metrics.csv
    ├── tumor_up_genes.csv
    ├── tumor_down_genes.csv
    ├── deg_full.csv
    ├── 2.1_pathway_gsea_results.csv
    ├── 2.1_pathway_enrichment.csv
    ├── 2.1_screening_summary.csv
    ├── 2.1_top_candidates.csv
    ├── 2.2_soc_drug_ranking.csv
    ├── p0_fixes_full_results.csv
    ├── method_comparison_summary.csv
    ├── screening_summary_stats.csv
    └── bootstrap_ci.csv
```

---

## Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:Benjamin-JHou/C2S-embedding-eval.git
   cd C2S-embedding-eval
   ```

2. **Install Python dependencies (Python 3.9+ recommended):**
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Preparation & Path Configuration

Because raw single-cell AnnData matrices (`.h5ad`) and full LINCS perturbation archives exceed GitHub file-size limits, each script in `Code/` includes a dedicated **`USER CONFIGURATION`** block at the top of the file. Before running the analysis scripts on raw data, update these placeholder paths to point to your local files:

```python
# ============================================================
# USER CONFIGURATION: Update paths below to match your local setup
# ============================================================
# Path to preprocessed GSE176078 AnnData (.h5ad) containing C2S embeddings in .obsm["X_c2s"]
GSE176078_H5AD_PATH = "/path/to/your/GSE176078_c2s_embedded.h5ad"

# Path to local C2S-Pythia-410M model directory or HuggingFace model ID
MODEL_PATH = "/path/to/your/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks"

# Path to LINCS L1000 ranked gene perturbation sentences CSV
LINCS_SENTENCES_CSV = "/path/to/your/lincs_perturbation_sentences.csv"
```

### External Public Resources
- **C2S-Pythia-410M Model & Tokenizer:** Available on HuggingFace (`vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks`).
- **GSE176078 Breast Cancer scRNA-seq Atlas:** [NCBI GEO GSE176078](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078).
- **LINCS L1000 Perturbation Signatures:** [NCBI GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742).
- **External Immune Validation Datasets:** `pbmc3k` ([10x Genomics](https://www.10xgenomics.com/datasets/)) and `paul15` (`scanpy.datasets.paul15()`).

---

## Running the Pipeline

### 1. Verify Pre-Computed Benchmark Data (No External `.h5ad` Required)
You can immediately validate all 24 processed tables in `Source_Data/` using the integrity validator:
```bash
python Code/validate_pipeline.py
```

### 2. Run Core Analytical Modules (Requires Local `.h5ad` / Model Paths Configured)
```bash
# 1. Cross-dataset C2S vs. PCA clustering benchmark
python Code/run_benchmark.py

# 2. Training-data token audit, per-lineage compactness & subcluster discovery
python Code/deep_analysis.py

# 3. High-resolution T-cell subpopulation clustering & diagnostic plots
python Code/I1_tcell_subclustering.py

# 4. BPE token frequency domain analysis & gene-gene correlation preservation
python Code/I3_token_domain_analysis.py

# 5. Full-scale LINCS L1000 compound screening
python Code/J5_full_scale_screening.py

# 6. Four-dimensional drug screening evaluation (GSEA, CMap, C2S alignment, Enhanced)
python Code/J6_p0_fixes.py
```

See [Source_Data_Guide.md](Source_Data_Guide.md) for a complete dictionary of all output files and column definitions.

---

## License

This project is released under the [MIT License](LICENSE).
