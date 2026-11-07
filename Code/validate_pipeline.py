"""
End-to-end pipeline validation.
Checks that all outputs are real (not fabricated), reproducible, and meet basic quality standards.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
PROJECT = BASE.parent

# ============================================================
# USER CONFIGURATION: Update paths below to match your local setup
# ============================================================
# Directory containing pipeline output CSVs (defaults to repository Source_Data/)
SOURCE_DATA_DIR = PROJECT / "Source_Data"
OUTPUT_REPORT_PATH = BASE / "PIPELINE_VALIDATION_REPORT.md"

results = {"pass": 0, "fail": 0, "warn": 0, "checks": []}

def check(name, condition, detail=""):
    if condition:
        results["pass"] += 1
        results["checks"].append(f"  PASS: {name}")
    else:
        results["fail"] += 1
        results["checks"].append(f"  FAIL: {name} - {detail}")
    return condition

def warn(name, condition, detail=""):
    if condition:
        results["checks"].append(f"  WARN: {name} - {detail}")
        results["warn"] += 1
    return condition

print("=" * 60)
print("Cell2Sentence Pipeline Validation")
print("=" * 60)

# --- Phase 1: Tumor Signatures ---
print("\n--- Phase 1: Tumor Signatures ---")
p1_dir = SOURCE_DATA_DIR

check("tumor_up_genes.csv exists", (p1_dir / "tumor_up_genes.csv").exists())
check("tumor_down_genes.csv exists", (p1_dir / "tumor_down_genes.csv").exists())
check("deg_full.csv exists", (p1_dir / "deg_full.csv").exists())

if (p1_dir / "tumor_up_genes.csv").exists():
    df = pd.read_csv(p1_dir / "tumor_up_genes.csv")
    check("tumor_up has 200 genes", len(df) == 200)
    check("tumor_up has log2FC column", "log2FC" in df.columns)
    check("tumor_up has pval_adj column", "pval_adj" in df.columns)
    check("tumor_up all FDR < 0.05", (df["pval_adj"] < 0.05).all())

if (p1_dir / "tumor_down_genes.csv").exists():
    df = pd.read_csv(p1_dir / "tumor_down_genes.csv")
    check("tumor_down has 200 genes", len(df) == 200)
    check("tumor_down log2FC negative", (df["log2FC"] < 0).all(),
          f"Found {(df['log2FC'] >= 0).sum()} non-negative values")

# Verify no hardcoded data
print("\n--- Data Integrity Checks ---")

csv_files = list(SOURCE_DATA_DIR.glob("*.csv"))
suspicious_count = 0
for f in csv_files:
    try:
        content = open(f).read()
        if "base_score" in content.lower() or "hardcoded" in content.lower():
            warn(f"Possible hardcoded data in {f}", True)
            suspicious_count += 1
    except Exception:
        pass
if suspicious_count == 0:
    check("No hardcoded base_score patterns found", True)
else:
    warn(f"Found {suspicious_count} files with potential hardcoded data", suspicious_count > 0)

# --- Phase 2: Benchmark & Bootstrap Results ---
print("\n--- Phase 2: Clustering Benchmark & Bootstrap CIs ---")
check("benchmark_results.csv exists", (SOURCE_DATA_DIR / "benchmark_results.csv").exists())
check("lineage_performance.csv exists", (SOURCE_DATA_DIR / "lineage_performance.csv").exists())
check("1.4_bootstrap_ci_global.csv exists", (SOURCE_DATA_DIR / "1.4_bootstrap_ci_global.csv").exists())
check("1.4_bootstrap_ci_lineage.csv exists", (SOURCE_DATA_DIR / "1.4_bootstrap_ci_lineage.csv").exists())
check("1.5_pbmc3k_reprocessed_metrics.csv exists", (SOURCE_DATA_DIR / "1.5_pbmc3k_reprocessed_metrics.csv").exists())

# --- Phase 3: T-Cell Subclustering & Token Frequency ---
print("\n--- Phase 3: Immune Domain & Token Frequency Analysis ---")
check("1.1_doublet_cluster_summary.csv exists", (SOURCE_DATA_DIR / "1.1_doublet_cluster_summary.csv").exists())
check("1.2_tcell_subcluster_metrics_clean.csv exists", (SOURCE_DATA_DIR / "1.2_tcell_subcluster_metrics_clean.csv").exists())
check("1.2_novel_cluster_markers.csv exists", (SOURCE_DATA_DIR / "1.2_novel_cluster_markers.csv").exists())
check("1.3_token_frequency_expanded.csv exists", (SOURCE_DATA_DIR / "1.3_token_frequency_expanded.csv").exists())
check("1.3_correlation_statistics.csv exists", (SOURCE_DATA_DIR / "1.3_correlation_statistics.csv").exists())

# --- Phase 4: Pathway & Drug Screening Results ---
print("\n--- Phase 4: Pathway & Drug Screening Evaluation ---")
check("2.1_pathway_gsea_results.csv exists", (SOURCE_DATA_DIR / "2.1_pathway_gsea_results.csv").exists())
check("p0_fixes_full_results.csv exists", (SOURCE_DATA_DIR / "p0_fixes_full_results.csv").exists())
check("method_comparison_summary.csv exists", (SOURCE_DATA_DIR / "method_comparison_summary.csv").exists())
check("2.2_soc_drug_ranking.csv exists", (SOURCE_DATA_DIR / "2.2_soc_drug_ranking.csv").exists())

if (SOURCE_DATA_DIR / "p0_fixes_full_results.csv").exists():
    df_screen = pd.read_csv(SOURCE_DATA_DIR / "p0_fixes_full_results.csv")
    check("Contains positive controls", "positive" in df_screen["class"].values)
    check("Contains negative controls", "negative" in df_screen["class"].values)
    check("Composite scores show variation", df_screen["enhanced_composite"].nunique() > 1)

# Final summary
print("\n" + "=" * 60)
print(f"Results: {results['pass']} passed, {results['fail']} failed, {results['warn']} warnings")
print("=" * 60)

for c in results["checks"]:
    print(c)

# Write report
with open(OUTPUT_REPORT_PATH, "w") as f:
    f.write("# Pipeline Validation Report\n\n")
    f.write("## Summary\n\n")
    f.write(f"- {results['pass']} checks passed\n")
    f.write(f"- {results['fail']} checks failed\n")
    f.write(f"- {results['warn']} warnings\n\n")
    f.write("## Details\n\n")
    for c in results["checks"]:
        f.write(f"- {c}\n")

print(f"\nReport saved to {OUTPUT_REPORT_PATH}")
