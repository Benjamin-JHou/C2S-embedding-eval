"""
I1: T Cell Subpopulation Clustering — C2S vs PCA.
Demonstrates C2S embedding superiority in the immune domain (in-distribution for C2S).

Input: Preprocessed GSE176078 h5ad with C2S + PCA embeddings
Output: Clustering metrics, UMAP visualizations, marker gene heatmaps
"""

import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)

# ============================================================
# USER CONFIGURATION: Update paths below to match your local setup
# ============================================================
# Path to preprocessed GSE176078 AnnData (.h5ad) containing C2S embeddings in .obsm["X_c2s"]
H5AD = "/path/to/your/GSE176078_c2s_embedded.h5ad"

# Directory to save T-cell subclustering outputs and diagnostic plots
OUTPUT_DIR = os.path.join(BASE, "output", "I1_tcell")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load data ---
print("Loading data...")
adata = sc.read_h5ad(H5AD)

# Extract T cells
t_adata = adata[adata.obs["cell_type"] == "T-cells"].copy()
print(f"T cells: {t_adata.shape[0]} cells, {t_adata.shape[1]} genes")

# Ground truth labels
t_adata.obs["subcluster"] = t_adata.obs["celltype_subset"].astype(str)
n_clusters = t_adata.obs["subcluster"].nunique()
print(f"T cell subclusters: {n_clusters}")

# --- C2S Embedding Clustering ---
print("\n=== C2S Embedding Clustering ===")
X_c2s = t_adata.obsm["X_c2s"]

# KMeans with ground-truth k
kmeans_c2s = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
t_adata.obs["cluster_c2s_sub"] = kmeans_c2s.fit_predict(X_c2s)
t_adata.obs["cluster_c2s_sub"] = t_adata.obs["cluster_c2s_sub"].astype(str)

ari_c2s = adjusted_rand_score(t_adata.obs["subcluster"], t_adata.obs["cluster_c2s_sub"])
nmi_c2s = normalized_mutual_info_score(t_adata.obs["subcluster"], t_adata.obs["cluster_c2s_sub"])
sil_c2s = silhouette_score(X_c2s, kmeans_c2s.labels_)
ch_c2s = calinski_harabasz_score(X_c2s, kmeans_c2s.labels_)
db_c2s = davies_bouldin_score(X_c2s, kmeans_c2s.labels_)

print(f"  ARI: {ari_c2s:.4f}")
print(f"  NMI: {nmi_c2s:.4f}")
print(f"  Silhouette: {sil_c2s:.4f}")
print(f"  Calinski-Harabasz: {ch_c2s:.1f}")
print(f"  Davies-Bouldin: {db_c2s:.4f}")

# --- PCA Embedding Clustering ---
print("\n=== PCA Embedding Clustering ===")
from sklearn.decomposition import PCA
# Recompute PCA on T cells only (fair comparison)
expr_dense = t_adata.X.toarray() if hasattr(t_adata.X, "toarray") else t_adata.X
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(expr_dense)
print(f"  PCA explained variance (top 5): {pca.explained_variance_ratio_[:5].round(4)}")

kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_pca = kmeans_pca.fit_predict(X_pca)
t_adata.obs["cluster_pca_sub"] = labels_pca.astype(str)
t_adata.obsm["X_pca"] = X_pca

ari_pca = adjusted_rand_score(t_adata.obs["subcluster"], t_adata.obs["cluster_pca_sub"])
nmi_pca = normalized_mutual_info_score(t_adata.obs["subcluster"], t_adata.obs["cluster_pca_sub"])
sil_pca = silhouette_score(X_pca, labels_pca)
ch_pca = calinski_harabasz_score(X_pca, labels_pca)
db_pca = davies_bouldin_score(X_pca, labels_pca)

print(f"  ARI: {ari_pca:.4f}")
print(f"  NMI: {nmi_pca:.4f}")
print(f"  Silhouette: {sil_pca:.4f}")
print(f"  Calinski-Harabasz: {ch_pca:.1f}")
print(f"  Davies-Bouldin: {db_pca:.4f}")

# --- Comparison Table ---
print("\n=== COMPARISON ===")
metrics = {
    "Metric": ["ARI", "NMI", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"],
    "C2S": [ari_c2s, nmi_c2s, sil_c2s, ch_c2s, db_c2s],
    "PCA": [ari_pca, nmi_pca, sil_pca, ch_pca, db_pca],
    "Better": ["C2S" if ari_c2s > ari_pca else "PCA",
               "C2S" if nmi_c2s > nmi_pca else "PCA",
               "C2S" if sil_c2s > sil_pca else "PCA",
               "C2S" if ch_c2s > ch_pca else "PCA",
               "PCA" if db_c2s > db_pca else "C2S"],  # lower DB is better
}
df = pd.DataFrame(metrics)
print(df.to_string(index=False))
df.to_csv(f"{OUTPUT_DIR}/tcell_subcluster_metrics.csv", index=False)

# --- Compute UMAP for both embeddings ---
print("\nComputing UMAP...")
sc.pp.neighbors(t_adata, use_rep="X_c2s", n_neighbors=30, key_added="c2s_neighbors")
sc.tl.umap(t_adata, neighbors_key="c2s_neighbors")
t_adata.obsm["X_umap_c2s"] = t_adata.obsm["X_umap"].copy()

sc.pp.neighbors(t_adata, use_rep="X_pca", n_neighbors=30, key_added="pca_neighbors")
sc.tl.umap(t_adata, neighbors_key="pca_neighbors")
t_adata.obsm["X_umap_pca"] = t_adata.obsm["X_umap"].copy()

# --- Figure 1: UMAP comparison ---
fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# C2S - colored by subcluster
sc.pl.umap(t_adata, color="subcluster", ax=axes[0, 0], show=False,
           title=f"C2S Embedding — T Cell Subclusters (ARI={ari_c2s:.3f})", frameon=False)
axes[0, 0].legend(fontsize=6, loc="lower right", ncol=2)

# PCA - colored by subcluster
t_adata.obsm["X_umap"] = t_adata.obsm["X_umap_pca"]
sc.pl.umap(t_adata, color="subcluster", ax=axes[0, 1], show=False,
           title=f"PCA Embedding — T Cell Subclusters (ARI={ari_pca:.3f})", frameon=False)
axes[0, 1].legend(fontsize=6, loc="lower right", ncol=2)

# C2S - key markers (separate subplots)
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
t_adata.obsm["X_umap"] = t_adata.obsm["X_umap_c2s"]
for j, (gene, title) in enumerate([("CD4", "CD4 (Helper)"), ("CD8A", "CD8A (Cytotoxic)"),
                                     ("FOXP3", "FOXP3 (Treg)"), ("PDCD1", "PDCD1 (Exhaustion)")]):
    sc.pl.umap(t_adata, color=gene, ax=axes2[j], show=False, title=title, frameon=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Fig1_Tcell_UMAP_Comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved Fig1")

# --- Figure 2: Per-subcluster compactness ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
top_subclusters = t_adata.obs["subcluster"].value_counts().head(6).index

for i, sub in enumerate(top_subclusters):
    ax = axes[i // 3, i % 3]
    mask = (t_adata.obs["subcluster"] == sub).values
    n = mask.sum()
    # Compactness: mean pairwise distance within subcluster (lower = tighter)
    c2s_dist = np.mean(np.linalg.norm(X_c2s[mask] - X_c2s[mask].mean(0), axis=1))
    pca_dist = np.mean(np.linalg.norm(X_pca[mask] - X_pca[mask].mean(0), axis=1))

    ax.bar(["C2S", "PCA"], [c2s_dist, pca_dist], color=["#2196F3", "#FF9800"])
    ax.set_title(f"{sub[:40]}\n(n={n})")
    ax.set_ylabel("Mean dist to centroid (↓better)")

plt.suptitle("Per-Subcluster Compactness: C2S vs PCA (lower = tighter)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Fig2_Subcluster_Compactness.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Figure 3: Marker gene expression in C2S embedding space ---
t_adata.obsm["X_umap"] = t_adata.obsm["X_umap_c2s"]
fig3, axes3 = plt.subplots(1, 4, figsize=(20, 5))
for j, (gene, title) in enumerate([("CD4", "CD4"), ("CD8A", "CD8A"),
                                     ("FOXP3", "FOXP3 (Treg)"), ("GZMB", "GZMB (Cyto)")]):
    sc.pl.umap(t_adata, color=gene, ax=axes3[j], show=False, title=title, frameon=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Fig3_C2S_Marker_Expression.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Save ---
t_adata.write(f"{OUTPUT_DIR}/tcell_subcluster_analysis.h5ad")
print(f"\nResults saved to {OUTPUT_DIR}/")

# --- Final summary ---
print(f"\n{'='*60}")
print(f"I1 COMPLETE: T Cell Subpopulation Clustering")
print(f"  C2S ARI: {ari_c2s:.4f} | PCA ARI: {ari_pca:.4f} | Δ = {ari_c2s - ari_pca:+.4f}")
print(f"  C2S NMI: {nmi_c2s:.4f} | PCA NMI: {nmi_pca:.4f} | Δ = {nmi_c2s - nmi_pca:+.4f}")
c2s_wins = sum([ari_c2s > ari_pca, nmi_c2s > nmi_pca, sil_c2s > sil_pca, ch_c2s > ch_pca, db_c2s < db_pca])
print(f"  C2S wins {c2s_wins}/5 metrics vs PCA")
print(f"{'='*60}")
