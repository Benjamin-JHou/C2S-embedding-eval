# Source Data Guide

This document describes all 24 processed data tables included in the `Source_Data/` directory, including their generating scripts, column schemas, and analytical scope.

---

## 1. Clustering and Cross-Dataset Benchmark Data

### `benchmark_results.csv`
- **Source Script:** `Code/run_benchmark.py`
- **Description:** Cross-dataset clustering evaluation comparing C2S embeddings (`1,024`-dimensional) against PCA (`50` principal components) across 11 dataset/domain partitions and 5 clustering metrics (`ARI`, `NMI`, `Silhouette`, `Calinski-Harabasz`, `Davies-Bouldin`).
- **Columns:** `dataset`, `domain`, `metric`, `c2s`, `pca`, `c2s_ratio`, `c2s_wins`, `n_cells`, `n_types`, `c2s_proxy`

### `lineage_performance.csv`
- **Source Script:** `Code/deep_analysis.py`
- **Description:** Per-lineage clustering compactness summary across the 9 major cell lineages in `GSE176078`, paired with mean BPE token IDs of lineage marker genes.
- **Columns:** `cell_type`, `domain`, `n_cells`, `c2s_compactness`, `pca_compactness`, `c2s_advantage`, `token_mean_id`

### `token_frequency_by_lineage.csv`
- **Source Script:** `Code/I3_token_domain_analysis.py` / `Code/deep_analysis.py`
- **Description:** Summary of BPE token IDs for curated lineage-specific marker gene sets across major cell lineages and housekeeping/ribosomal controls.
- **Columns:** `lineage`, `n_genes`, `mean_token_id`, `median_token_id`, `min_token_id`

---

## 2. Bootstrap Confidence Intervals

### `1.4_bootstrap_ci_global.csv`
- **Source Script:** `Code/run_benchmark.py`
- **Description:** Global bootstrap $95\%$ confidence intervals ($n = 300$ resamples, all 9 lineages pooled in `GSE176078`) across all 5 clustering metrics for C2S and PCA.
- **Columns:** `metric`, `method`, `mean`, `ci_low`, `ci_high`

### `1.4_bootstrap_ci_lineage.csv`
- **Source Script:** `Code/deep_analysis.py`
- **Description:** Per-lineage bootstrap $95\%$ confidence intervals ($9\text{ lineages} \times 5\text{ metrics} = 45\text{ rows}$, $n = 300$ resamples) evaluating non-overlapping CI significance between C2S and PCA.
- **Columns:** `lineage`, `k`, `n_cells`, `metric`, `method`, `mean`, `ci_low`, `ci_high`

### `1.4_bootstrap_ci_tcell.csv`
- **Source Script:** `Code/I1_tcell_subclustering.py`
- **Description:** Bootstrap $95\%$ confidence intervals for clean T-cell subpopulation clustering following doublet removal.
- **Columns:** `metric`, `method`, `mean`, `ci_low`, `ci_high`

---

## 3. T-Cell Subpopulation Discovery and Doublet Quality Control

### `1.1_doublet_cluster_summary.csv`
- **Source Script:** `Code/I1_tcell_subclustering.py`
- **Description:** Quantitative doublet quality-control metrics across all 15 high-resolution T-cell clusters (`k = 15`), profiling immunoglobulin (IG), B-cell, and T-cell (`CD3D`) expression indicators.
- **Columns:** `cluster_id`, `n_cells`, `ig_mean`, `ig_median`, `ig_frac_positive`, `ig_p90`, `bcell_mean`, `bcell_frac_positive`, `tcell_mean`, `n_ig_above_median`, `pct_ig_above_median`

### `1.2_tcell_subcluster_metrics_clean.csv`
- **Source Script:** `Code/I1_tcell_subclustering.py`
- **Description:** Clustering evaluation metrics on $33,356$ clean T-cells after excluding the B-cell/T-cell doublet cluster (Cluster 8, $n = 1,858$).
- **Columns:** `Metric`, `C2S`, `PCA`, `Better`, `n_cells_clean`, `n_excluded_cluster8`, `k`

### `1.2_novel_cluster_markers.csv`
- **Source Script:** `Code/I1_tcell_subclustering.py`
- **Description:** Wilcoxon rank-sum differential expression results identifying marker genes for C2S-discovered T-cell subpopulations (clusters with PCA purity $< 0.50$).
- **Columns:** `gene`, `scores`, `log2fc`, `pvalue`, `pval_adj`, `novel_cluster`

---

## 4. Expanded Token Frequency and Domain Correlation

### `1.3_token_frequency_expanded.csv`
- **Source Script:** `Code/I3_token_domain_analysis.py`
- **Description:** Expanded BPE token ID statistics and C2S/PCA centroid compactness ratios across 22 fine-grained cellular sub-lineages.
- **Columns:** `lineage`, `n_markers`, `n_found`, `found_genes`, `mean_token_id`, `median_token_id`, `min_token_id`, `max_token_id`, `n_cells`, `mean_c2s_dist`, `mean_pca_dist`, `compactness_ratio`

### `1.3_correlation_statistics.csv`
- **Source Script:** `Code/I3_token_domain_analysis.py`
- **Description:** Spearman rank correlation statistics and $1,000$-resample bootstrap confidence intervals between mean lineage token ID and C2S/PCA compactness ratio.
- **Columns:** `n_lineages`, `spearman_rho`, `p_value`, `significant_at_005`, `bootstrap_ci_lower`, `bootstrap_ci_upper`, `bootstrap_mean_rho`, `n_bootstrap`

---

## 5. Preprocessing Reproducibility Validation

### `1.5_pbmc3k_reprocessed_metrics.csv`
- **Source Script:** `Code/run_benchmark.py`
- **Description:** Clustering benchmark on the external `pbmc3k` dataset ($2,638$ cells, $8$ cell types) before and after applying C2S-compatible preprocessing (count normalization, C2S vocabulary filtering, and pipeline-matched scaling).
- **Columns:** `dataset`, `domain`, `metric`, `c2s`, `pca`, `c2s_ratio`, `c2s_wins`, `n_cells`, `n_types`

---

## 6. Differential Expression, Pathway Enrichment, and Drug Screening Benchmark

### `tumor_up_genes.csv`
- **Source Script:** `Code/deep_analysis.py`
- **Description:** Top 200 significantly upregulated genes in `Cancer Epithelial` vs. `Normal Epithelial` cells from `GSE176078` (Wilcoxon rank-sum test, BH-adjusted $\text{FDR} < 0.05$).
- **Columns:** `gene`, `mean_cancer`, `mean_normal`, `log2FC`, `pct_cancer`, `pct_normal`, `pct_diff`, `pval`, `pval_adj`, `significant`

### `tumor_down_genes.csv`
- **Source Script:** `Code/deep_analysis.py`
- **Description:** Top 200 significantly downregulated genes in `Cancer Epithelial` vs. `Normal Epithelial` cells from `GSE176078`.
- **Columns:** `gene`, `mean_cancer`, `mean_normal`, `log2FC`, `pct_cancer`, `pct_normal`, `pct_diff`, `pval`, `pval_adj`, `significant`

### `deg_full.csv`
- **Source Script:** `Code/deep_analysis.py`
- **Description:** Full transcriptome differential expression table comparing tumor versus normal epithelial compartments in `GSE176078`.
- **Columns:** `gene`, `mean_cancer`, `mean_normal`, `log2FC`, `pct_cancer`, `pct_normal`, `pct_diff`, `pval`, `pval_adj`, `significant`

### `2.1_pathway_gsea_results.csv`
- **Source Script:** `Code/deep_analysis.py`
- **Description:** Preranked Gene Set Enrichment Analysis (GSEA) results across the 50 MSigDB Hallmark pathways.
- **Columns:** `Name`, `Term`, `ES`, `NES`, `NOM p-val`, `FDR q-val`, `FWER p-val`, `Tag %`, `Gene %`, `Lead_genes`

### `2.1_pathway_enrichment.csv`
- **Source Script:** `Code/deep_analysis.py`
- **Description:** Filtered summary of significantly enriched Hallmark pathways ($\text{FDR} < 0.05$).

### `2.1_screening_summary.csv`
- **Source Script:** `Code/J5_full_scale_screening.py`
- **Description:** High-level summary of the LINCS L1000 compound screening run across $10,755$ perturbation signatures.

### `2.1_top_candidates.csv`
- **Source Script:** `Code/J5_full_scale_screening.py`
- **Description:** Top-ranked compound candidates stratified across individual scoring dimensions.

### `2.2_soc_drug_ranking.csv`
- **Source Script:** `Code/J5_full_scale_screening.py` / `Code/J6_p0_fixes.py`
- **Description:** Rankings and multi-dimensional scores of 15 standard-of-care (SOC) breast cancer drugs identified within the LINCS L1000 corpus.
- **Columns:** `drug`, `mechanism`, `enhanced_composite`, `rank`, `c2s_alignment`, `gsea_composite`, `cmap_composite`

### `p0_fixes_full_results.csv`
- **Source Script:** `Code/J6_p0_fixes.py`
- **Description:** Complete screening table across all $10,755$ LINCS L1000 compounds evaluated with all four scoring dimensions (`gsea_composite`, `cmap_composite`, `c2s_alignment`, and `enhanced_composite`) and control classifications.
- **Columns:** `compound`, `n_genes`, `gsea_up`, `gsea_down`, `gsea_combined`, `cmap_cs`, `overlap`, `toxicity`, `gsea_composite`, `cmap_composite`, `class`, `mechanism`, `gsea_rank`, `cmap_rank`, `pathway_mean`, `c2s_alignment`, `enhanced_composite`, `gsea_composite_rank`, `cmap_composite_rank`, `enhanced_composite_rank`

### `method_comparison_summary.csv`
- **Source Script:** `Code/J6_p0_fixes.py`
- **Description:** Statistical comparison of drug-screening scoring methods (Mann-Whitney U test $p$-values, Cohen's $d$ effect sizes, and inter-method Spearman rank correlations).
- **Columns:** `method`, `mwu_p`, `cohens_d`, `n_positives`, `n_negatives`, `spearman_gsea_cmap`, `c2s_alignment_corr`

### `screening_summary_stats.csv`
- **Source Script:** `Code/J5_full_scale_screening.py`
- **Description:** Aggregate screening statistics (library size, positive/negative control counts, mean ranks, and effect sizes).

### `bootstrap_ci.csv`
- **Source Script:** `Code/J5_full_scale_screening.py`
- **Description:** Bootstrap confidence intervals for compound-screening evaluation metrics.

---

## 7. Public Data Sources

All processed tables are derived from publicly accessible biological repositories:

- **GSE176078**: Human breast cancer single-cell RNA-seq atlas ($100,064$ cells across 9 major lineages) — https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078
- **pbmc3k**: 10x Genomics peripheral blood mononuclear cell dataset ($2,638$ cells) — https://www.10xgenomics.com/datasets/
- **paul15**: Myeloid progenitor scRNA-seq dataset (`GSE72857`, $2,730$ cells, accessible via `scanpy.datasets.paul15()`)
- **GSE92742 (LINCS L1000)**: Connectivity Map perturbation gene expression signatures — https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742

Large raw binary matrices (`>100 MB`, including `.h5ad` files and full LINCS `.gctx` archives) are excluded from version control and can be locally configured in the `Code/` scripts.
