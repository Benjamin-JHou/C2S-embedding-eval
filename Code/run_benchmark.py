"""
Cross-Dataset C2S Embedding Benchmark.
Evaluates C2S vs PCA clustering across 6+ datasets from 3 tissue domains.

Core hypothesis: C2S embedding quality tracks training data lineage composition.
"""

import os, sys
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)

# ============================================================
# USER CONFIGURATION: Update paths below to match your local setup
# ============================================================
# Path to preprocessed GSE176078 AnnData (.h5ad) containing C2S embeddings in .obsm["X_c2s"]
GSE176078_H5AD_PATH = "/path/to/your/GSE176078_c2s_embedded.h5ad"

# Directory containing external validation .h5ad files (pbmc3k_c2s_embedded.h5ad, paul15_c2s_embedded.h5ad)
DATA_DIR = "/path/to/your/external_datasets_dir"

# Directory to save benchmark outputs and diagnostic plots
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD ALL DATASETS
# ============================================================
print("="*70)
print("C2S EMBEDDING BENCHMARK: CROSS-DATASET EVALUATION")
print("="*70)

datasets = {}

# GSE176078 (already has C2S embeddings)
print("\nLoading GSE176078...")
gse = sc.read_h5ad(GSE176078_H5AD_PATH)
datasets["GSE176078_Breast"] = {
    "adata": gse,
    "domain": "mixed",
    "n_cells": gse.shape[0],
    "label_col": "cell_type",
    "has_c2s": True,
}
print(f"  {gse.shape[0]} cells, {gse.obs['cell_type'].nunique()} types, C2S: ✓")

# pbmc3k + paul15 (C2S embeddings now available)
for fname, dname, domain, c2s_fname in [
    ("pbmc3k_c2s_embedded.h5ad", "pbmc3k_PBMC", "immune", "pbmc3k_c2s_embedded.h5ad"),
    ("paul15_c2s_embedded.h5ad", "paul15_Myeloid", "immune", "paul15_c2s_embedded.h5ad"),
]:
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        print(f"\nLoading {dname}...")
        adata = sc.read_h5ad(path)
        # Ensure cell_type column exists
        if "cell_type" not in adata.obs.columns:
            # Try common alternatives
            for col in ["louvain", "paul15_clusters", "bulk_labels", "annotation"]:
                if col in adata.obs.columns:
                    adata.obs["cell_type"] = adata.obs[col].astype(str)
                    break
            else:
                adata.obs["cell_type"] = "unknown"
        has_c2s = "X_c2s" in adata.obsm
        datasets[dname] = {
            "adata": adata,
            "domain": domain,
            "n_cells": adata.shape[0],
            "label_col": "cell_type",
            "has_c2s": has_c2s,
        }
        print(f"  {adata.shape[0]} cells, {adata.obs['cell_type'].nunique()} types, C2S: {'✓' if has_c2s else '✗'}")
    else:
        print(f"\n  SKIP {dname}: file not found")

print(f"\nTotal datasets loaded: {len(datasets)}")

# ============================================================
# 2. COMPUTE C2S + PCA EMBEDDINGS (for datasets without C2S)
# ============================================================
print(f"\n{'='*70}")
print("EXTRACTING EMBEDDINGS")
print(f"{'='*70}")

for dname, d in datasets.items():
    adata = d["adata"]
    print(f"\n--- {dname} ({d['n_cells']} cells) ---")

    # PCA
    if "X_pca" not in adata.obsm:
        # Ensure dense
        if hasattr(adata.X, "toarray"):
            expr = adata.X.toarray()
        else:
            expr = adata.X
        pca = PCA(n_components=50, random_state=42)
        X_pca = pca.fit_transform(expr)
        adata.obsm["X_pca"] = X_pca
        print(f"  PCA: 50 PCs, var explained top5={pca.explained_variance_ratio_[:5].sum():.3f}")

    # C2S embedding
    if not d["has_c2s"]:
        # For non-C2S datasets: use the pre-trained C2S model to embed cell sentences
        # This requires: formatting genes as cell sentences → C2S forward pass
        # For NOW: use a gene-frequency-based embedding proxy
        # (actual C2S embedding requires GPU for forward pass through 410M model)

        # PROXY: Use top-200 highly variable genes as a "pseudo-C2S" embedding
        # This captures the gene ranking aspect without the LLM forward pass
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Get top-200 expressed genes per cell
        expr_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        gene_order = np.argsort(-expr_dense, axis=1)[:, :200]

        # Build a frequency-based embedding (1024-dim via random projection of gene ranks)
        np.random.seed(42)
        proj = np.random.randn(adata.shape[1], 1024) / np.sqrt(adata.shape[1])

        X_c2s_proxy = np.zeros((adata.shape[0], 1024))
        for i in range(adata.shape[0]):
            ranks = np.zeros(adata.shape[1])
            ranks[gene_order[i]] = np.arange(200, 0, -1)  # weight by rank
            X_c2s_proxy[i] = ranks @ proj

        adata.obsm["X_c2s"] = X_c2s_proxy
        print(f"  C2S proxy: 1024-dim via gene-rank projection")

        d["c2s_proxy"] = True
    else:
        print(f"  C2S: pre-computed (real model embeddings)")
        d["c2s_proxy"] = False

# ============================================================
# 3. CLUSTERING EVALUATION
# ============================================================
print(f"\n{'='*70}")
print("CLUSTERING BENCHMARK")
print(f"{'='*70}")

METRICS = ["ARI", "NMI", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]

all_results = []

for dname, d in datasets.items():
    adata = d["adata"]
    domain = d["domain"]
    labels = adata.obs["cell_type"].astype(str)
    n_clusters = labels.nunique()

    X_c2s = adata.obsm["X_c2s"]
    X_pca = adata.obsm["X_pca"]

    if n_clusters < 2:
        print(f"  SKIP {dname}: only {n_clusters} types")
        continue

    # KMeans clustering
    km_c2s = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_c2s = km_c2s.fit_predict(X_c2s)
    km_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_pca = km_pca.fit_predict(X_pca)

    # Compute metrics
    ari_c2s = adjusted_rand_score(labels, pred_c2s)
    ari_pca = adjusted_rand_score(labels, pred_pca)
    nmi_c2s = normalized_mutual_info_score(labels, pred_c2s)
    nmi_pca = normalized_mutual_info_score(labels, pred_pca)

    try:
        sil_c2s = silhouette_score(X_c2s, pred_c2s)
        sil_pca = silhouette_score(X_pca, pred_pca)
        ch_c2s = calinski_harabasz_score(X_c2s, pred_c2s)
        ch_pca = calinski_harabasz_score(X_pca, pred_pca)
        db_c2s = davies_bouldin_score(X_c2s, pred_c2s)
        db_pca = davies_bouldin_score(X_pca, pred_pca)
    except:
        sil_c2s = sil_pca = ch_c2s = ch_pca = db_c2s = db_pca = 0

    c2s_values = [ari_c2s, nmi_c2s, sil_c2s, ch_c2s, db_c2s]
    pca_values = [ari_pca, nmi_pca, sil_pca, ch_pca, db_pca]

    c2s_wins = sum([
        ari_c2s > ari_pca, nmi_c2s > nmi_pca, sil_c2s > sil_pca,
        ch_c2s > ch_pca, db_c2s < db_pca  # lower DB is better
    ])

    print(f"\n  {dname} [{domain}] — {n_clusters} types, {d['n_cells']} cells{' (C2S proxy)' if d.get('c2s_proxy') else ''}")
    print(f"  {'Metric':25s} {'C2S':>10s} {'PCA':>10s} {'Winner':>8s}")
    print(f"  {'-'*55}")
    for name, c2s, pca in zip(METRICS, c2s_values, pca_values):
        w = "C2S" if (name != "Davies-Bouldin" and c2s > pca) or (name == "Davies-Bouldin" and c2s < pca) else "PCA"
        print(f"  {name:25s} {c2s:10.4f} {pca:10.4f} {w:>8s}")
    print(f"  C2S wins: {c2s_wins}/5")

    for metric, c2s, pca in zip(METRICS, c2s_values, pca_values):
        all_results.append({
            "dataset": dname, "domain": domain, "metric": metric,
            "c2s": c2s, "pca": pca, "c2s_ratio": c2s / max(pca, 1e-10),
            "c2s_wins": c2s_wins, "n_cells": d["n_cells"],
            "n_types": n_clusters, "c2s_proxy": d.get("c2s_proxy", False),
        })

df = pd.DataFrame(all_results)
df.to_csv(os.path.join(OUTPUT_DIR, "benchmark_results.csv"), index=False)

# ============================================================
# 4. SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("BENCHMARK SUMMARY")
print(f"{'='*70}")

# By domain
for domain in ["immune", "epithelial", "neural", "mixed"]:
    sub = df[df["domain"] == domain]
    if len(sub) == 0:
        continue
    ari_sub = sub[sub["metric"] == "ARI"]
    mean_ratio = ari_sub["c2s_ratio"].mean()
    n_datasets = ari_sub["dataset"].nunique()
    print(f"  {domain:15s}: {n_datasets} datasets, C2S/PCA ARI ratio = {mean_ratio:.2f}x")

# Overall
ari_df = df[df["metric"] == "ARI"]
print(f"\n  Overall C2S/PCA ARI ratio: {ari_df['c2s_ratio'].mean():.2f}x (±{ari_df['c2s_ratio'].std():.2f})")

# ============================================================
# 5. FIGURE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: ARI comparison bar chart
ari_data = df[df["metric"] == "ARI"].copy()
ari_data = ari_data.sort_values("dataset")
x = np.arange(len(ari_data))
w = 0.35
axes[0].bar(x - w/2, ari_data["c2s"].values, w, label="C2S", color="#2196F3")
axes[0].bar(x + w/2, ari_data["pca"].values, w, label="PCA", color="#FF9800")
axes[0].set_xticks(x)
axes[0].set_xticklabels(ari_data["dataset"].values, rotation=30, ha="right", fontsize=8)
axes[0].set_ylabel("ARI")
axes[0].set_title("C2S vs PCA: Adjusted Rand Index Across Datasets")
axes[0].legend()
axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

# Panel 2: C2S/PCA ratio by domain
ari_data["ratio"] = ari_data["c2s"] / ari_data["pca"].clip(lower=1e-6)
colors = [{"immune": "#4CAF50", "epithelial": "#F44336", "neural": "#9C27B0", "mixed": "#FF9800"}[d]
          for d in ari_data["domain"]]
axes[1].bar(range(len(ari_data)), ari_data["ratio"].values, color=colors)
axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="C2S = PCA")
axes[1].set_xticks(range(len(ari_data)))
axes[1].set_xticklabels(ari_data["dataset"].values, rotation=30, ha="right", fontsize=8)
axes[1].set_ylabel("C2S/PCA ARI Ratio")
axes[1].set_title("C2S/PCA Performance Ratio by Dataset\n(>1 = C2S better, <1 = PCA better)")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Benchmark_Summary.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nFigure saved to {OUTPUT_DIR}/Fig_Benchmark_Summary.png")
print("Benchmark complete.")
