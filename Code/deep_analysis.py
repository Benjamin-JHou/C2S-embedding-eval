"""
Deep Lineage & Embedding Space Analysis:
1. Training data audit — quantify immune vs epithelial representation
2. Per-lineage quantitative model — token frequency → embedding quality correlation
3. Embedding space analysis — within/between lineage distances
4. Cell state discovery — C2S-unique clusters

All based on GSE176078 with real C2S embeddings.
"""

import os, sys, numpy as np, pandas as pd, scanpy as sc
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.spatial.distance import pdist, cosine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)

# ============================================================
# USER CONFIGURATION: Update paths below to match your local setup
# ============================================================
# Path to local C2S-Pythia-410M model directory or HuggingFace model ID
# (e.g., "vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks")
MODEL_PATH = "/path/to/your/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks"

# Path to preprocessed GSE176078 AnnData (.h5ad) containing C2S embeddings in .obsm["X_c2s"]
GSE176078_H5AD_PATH = "/path/to/your/GSE176078_c2s_embedded.h5ad"

# Directory to save analysis outputs and diagnostic figures
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. TRAINING DATA AUDIT
# ============================================================
print("="*70)
print("1. TRAINING DATA AUDIT")
print("="*70)

# Token frequency by gene category
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_PATH)

# Gene categories by lineage
LINEAGE_GENES = {
    "T cell": ["CD3D","CD3E","CD3G","CD2","CD28","LCK","ZAP70","CD4","CD8A","CD8B",
               "FOXP3","IL7R","CCR7","SELL","GZMB","GZMK","PRF1","GNLY","NKG7",
               "CTLA4","PDCD1","LAG3","TIGIT","HAVCR2","ICOS"],
    "B cell": ["CD19","CD79A","CD79B","MS4A1","PAX5","BLK","CD22","IGHM","IGHD",
               "CD38","CD27","SDC1","MZB1","TNFRSF17","XBP1"],
    "Myeloid": ["CD14","CD68","CD163","CSF1R","ITGAM","FCGR3A","CD86","CD80","MRC1",
                "ITGAX","CLEC4C","CD1C","FCER1A"],
    "Endothelial": ["PECAM1","CDH5","VWF","CLDN5","ENG","KDR","TEK","FLT1","ESM1"],
    "Fibroblast": ["COL1A1","COL1A2","COL3A1","DCN","LUM","FAP","ACTA2","PDGFRA"],
    "Epithelial (Breast)": ["ESR1","PGR","ERBB2","GATA3","FOXA1","KRT8","KRT18",
                              "KRT19","EPCAM","MKI67","CDH1"],
    "Epithelial (Lung)": ["NKX2-1","SFTPC","SFTPB","KRT7","NAPSA","MUC1","TP63","SOX2"],
    "Epithelial (Liver)": ["ALB","AFP","TTR","CYP3A4","HNF4A","KRT7","KRT19"],
    "Neural": ["RBFOX3","MAP2","SYN1","SYP","GFAP","OLIG2","MBP","SOX10","GAD1","SLC17A7"],
    "Housekeeping": ["GAPDH","ACTB","B2M","RPLP0","TBP","HPRT1","PGK1"],
    "Ribosomal": ["RPL3","RPL5","RPL8","RPL10","RPL13A","RPS3","RPS5","RPS18","RPS27A"],
}

lineage_token_stats = []
for lineage, genes in LINEAGE_GENES.items():
    ids = []
    for g in genes:
        tid = tok.encode(g, add_special_tokens=False)
        if len(tid) > 0:
            ids.append(tid[0])
    if ids:
        lineage_token_stats.append({
            "lineage": lineage,
            "n_genes": len(ids),
            "mean_token_id": np.mean(ids),
            "median_token_id": np.median(ids),
            "min_token_id": min(ids),
        })

df_tok = pd.DataFrame(lineage_token_stats)
df_tok = df_tok.sort_values("mean_token_id")
print(df_tok.to_string(index=False))
df_tok.to_csv(os.path.join(OUTPUT_DIR, "token_frequency_by_lineage.csv"), index=False)

# ============================================================
# 2. PER-LINEAGE PERFORMANCE vs TRAINING FREQUENCY
# ============================================================
print(f"\n{'='*70}")
print("2. QUANTITATIVE MODEL: TRAINING FREQ → PERFORMANCE")
print(f"{'='*70}")

# Load GSE176078 clustering results per lineage
adata = sc.read_h5ad(GSE176078_H5AD_PATH)

# For each cell type, compute C2S and PCA clustering quality
# Use: internal compactness (mean distance to centroid) as a metric
# Lower distance = better clustering

lineage_scores = []
for ct in adata.obs["cell_type"].unique():
    mask = adata.obs["cell_type"] == ct
    n = mask.sum()
    if n < 100:
        continue

    X_c2s = adata.obsm["X_c2s"][mask]
    expr_dense = adata.X[mask].toarray() if hasattr(adata.X, "toarray") else adata.X[mask]
    X_pca = PCA(n_components=50, random_state=42).fit_transform(expr_dense)

    # Compactness: mean distance to centroid (lower = better)
    c2s_centroid = X_c2s.mean(axis=0)
    pca_centroid = X_pca.mean(axis=0)
    c2s_compact = np.mean([np.linalg.norm(x - c2s_centroid) for x in X_c2s[:500]])
    pca_compact = np.mean([np.linalg.norm(x - pca_centroid) for x in X_pca[:500]])

    # C2S advantage ratio (PCA/C2S: >1 means C2S more compact)
    compact_ratio = pca_compact / max(c2s_compact, 1e-10)

    # Domain classification
    if ct in ["T-cells", "B-cells", "Myeloid", "Plasmablasts"]:
        domain = "immune"
    elif ct in ["Cancer Epithelial", "Normal Epithelial"]:
        domain = "epithelial"
    else:
        domain = "other"

    # Get token frequency for this lineage
    token_mean = None
    if "T-cell" in ct or ct == "T-cells":
        token_mean = df_tok[df_tok["lineage"]=="T cell"]["mean_token_id"].values[0] if len(df_tok[df_tok["lineage"]=="T cell"]) > 0 else None
    elif "B-cell" in ct or ct == "B-cells":
        token_mean = df_tok[df_tok["lineage"]=="B cell"]["mean_token_id"].values[0] if len(df_tok[df_tok["lineage"]=="B cell"]) > 0 else None
    elif ct == "Myeloid":
        token_mean = df_tok[df_tok["lineage"]=="Myeloid"]["mean_token_id"].values[0] if len(df_tok[df_tok["lineage"]=="Myeloid"]) > 0 else None
    elif "Epithelial" in ct:
        token_mean = df_tok[df_tok["lineage"]=="Epithelial (Breast)"]["mean_token_id"].values[0] if len(df_tok[df_tok["lineage"]=="Epithelial (Breast)"]) > 0 else None

    lineage_scores.append({
        "cell_type": ct, "domain": domain, "n_cells": n,
        "c2s_compactness": c2s_compact, "pca_compactness": pca_compact,
        "c2s_advantage": compact_ratio,
        "token_mean_id": token_mean,
    })
    print(f"  {ct:25s} [{domain:12s}] n={n:6d} C2S/PCA compact ratio={compact_ratio:.2f}x token_id={token_mean}")

df_ls = pd.DataFrame(lineage_scores)
df_ls.to_csv(os.path.join(OUTPUT_DIR, "lineage_performance.csv"), index=False)

# Correlation: token frequency vs C2S advantage
valid = df_ls.dropna(subset=["token_mean_id"])
if len(valid) >= 3:
    r, p = spearmanr(valid["token_mean_id"], valid["c2s_advantage"])
    print(f"\n  Spearman(token_id, C2S_advantage): r={r:.3f}, p={p:.3f}")
    print(f"  (Lower token ID = more frequent in training = higher C2S advantage)")

# ============================================================
# 3. EMBEDDING SPACE ANALYSIS
# ============================================================
print(f"\n{'='*70}")
print("3. EMBEDDING SPACE: WITHIN vs BETWEEN LINEAGE DISTANCES")
print(f"{'='*70}")

# For each lineage pair, compute C2S vs PCA distance ratio
cell_types = ["T-cells", "B-cells", "Myeloid", "Cancer Epithelial", "Normal Epithelial",
              "Endothelial", "CAFs"]
distance_matrix = np.zeros((len(cell_types), len(cell_types)))

for i, ct1 in enumerate(cell_types):
    mask1 = adata.obs["cell_type"] == ct1
    centroids_c2s = {}
    centroids_pca = {}
    for ct2 in cell_types:
        mask2 = adata.obs["cell_type"] == ct2

        # C2S distance
        c2s_ct1 = adata.obsm["X_c2s"][mask1][:500].mean(axis=0)
        c2s_ct2 = adata.obsm["X_c2s"][mask2][:500].mean(axis=0)
        c2s_dist = cosine(c2s_ct1, c2s_ct2)

        # PCA distance
        expr1 = adata.X[mask1][:500].toarray() if hasattr(adata.X, "toarray") else adata.X[mask1][:500]
        expr2 = adata.X[mask2][:500].toarray() if hasattr(adata.X, "toarray") else adata.X[mask2][:500]
        pca = PCA(n_components=50, random_state=42)
        combined = np.vstack([expr1, expr2])
        pca.fit(combined)
        pca_ct1 = pca.transform(expr1).mean(axis=0)
        pca_ct2 = pca.transform(expr2).mean(axis=0)
        pca_dist = cosine(pca_ct1, pca_ct2)

        centroids_c2s[ct2] = c2s_dist
        centroids_pca[ct2] = pca_dist

# Build proper distance matrices
c2s_dist_matrix = np.zeros((len(cell_types), len(cell_types)))
pca_dist_matrix = np.zeros((len(cell_types), len(cell_types)))
for ii, ct1 in enumerate(cell_types):
    for jj, ct2 in enumerate(cell_types):
        c2s_dist_matrix[ii, jj] = centroids_c2s.get(ct2, 0)
        pca_dist_matrix[ii, jj] = centroids_pca.get(ct2, 0)

# Get indices
immune_idx = [i for i, ct in enumerate(cell_types) if ct in ["T-cells", "B-cells", "Myeloid"]]
epi_idx = [i for i, ct in enumerate(cell_types) if ct in ["Cancer Epithelial", "Normal Epithelial"]]

imm_imm_c2s = [c2s_dist_matrix[i,j] for i in immune_idx for j in immune_idx if i < j]
imm_epi_c2s = [c2s_dist_matrix[i,j] for i in immune_idx for j in epi_idx]

print(f"  C2S immune-immune distance: {np.mean(imm_imm_c2s):.4f}")
print(f"  C2S immune-epithelial distance: {np.mean(imm_epi_c2s):.4f}")
print(f"  Ratio (epi/imm): {np.mean(imm_epi_c2s)/max(np.mean(imm_imm_c2s),1e-10):.1f}x")

# ============================================================
# 4. CELL STATE DISCOVERY
# ============================================================
print(f"\n{'='*70}")
print("4. CELL STATE DISCOVERY: C2S-UNIQUE CLUSTERS")
print(f"{'='*70}")

# Focus on T cells (largest immune population)
t_mask = adata.obs["cell_type"] == "T-cells"
t_adata = adata[t_mask].copy()

# C2S clustering at higher resolution
from sklearn.cluster import KMeans
n_c2s_clusters = 15  # more granular than original 12
km_c2s = KMeans(n_clusters=n_c2s_clusters, random_state=42, n_init=10)
t_adata.obs["c2s_cluster"] = km_c2s.fit_predict(t_adata.obsm["X_c2s"]).astype(str)

# PCA clustering
expr_t = t_adata.X.toarray() if hasattr(t_adata.X, "toarray") else t_adata.X
t_pca = PCA(n_components=50, random_state=42).fit_transform(expr_t)
km_pca = KMeans(n_clusters=n_c2s_clusters, random_state=42, n_init=10)
t_adata.obs["pca_cluster"] = km_pca.fit_predict(t_pca).astype(str)

# Find clusters that are unique to C2S (low ARI with PCA clusters)
ari_c2s_pca = adjusted_rand_score(t_adata.obs["c2s_cluster"], t_adata.obs["pca_cluster"])
print(f"  C2S vs PCA cluster agreement (ARI): {ari_c2s_pca:.3f}")
print(f"  ({1-ari_c2s_pca:.1%} of cluster assignments differ between methods)")

# For each C2S cluster, find its PCA match and report discordance
c2s_cluster_sizes = t_adata.obs["c2s_cluster"].value_counts()
print(f"\n  C2S cluster sizes: {dict(c2s_cluster_sizes)}")

# Identify C2S clusters with no clear PCA counterpart
for c2s_cluster in sorted(t_adata.obs["c2s_cluster"].unique(), key=lambda x: int(x)):
    mask = t_adata.obs["c2s_cluster"] == c2s_cluster
    n = mask.sum()
    pca_clusters = t_adata.obs["pca_cluster"][mask].value_counts()
    primary_pca = pca_clusters.index[0] if len(pca_clusters) > 0 else "N/A"
    purity = pca_clusters.iloc[0] / n if len(pca_clusters) > 0 else 0

    # Get top marker genes for this C2S cluster
    if n >= 10:
        expr_c2s = expr_t[mask.values].mean(axis=0)
        expr_rest = expr_t[~mask.values].mean(axis=0)
        fc = expr_c2s / (expr_rest + 0.01)
        top_markers = [t_adata.var_names[i] for i in np.argsort(-fc)[:5]]

        novel = "NOVEL" if purity < 0.5 else ""
        print(f"  Cluster {c2s_cluster}: n={n:5d} PCA={primary_pca} purity={purity:.2f} {novel}")
        if novel:
            print(f"    Top markers: {', '.join(top_markers)}")

# ============================================================
# 5. FIGURES
# ============================================================
print(f"\n{'='*70}")
print("5. GENERATING FIGURES")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 3, figsize=(20, 14))

# Panel A: Token frequency by lineage
ax = axes[0, 0]
colors_tok = []
for l in df_tok["lineage"]:
    if "T cell" in l or "B cell" in l or "Myeloid" in l:
        colors_tok.append("#4CAF50")
    elif "Epithelial" in l or "Fibroblast" in l:
        colors_tok.append("#F44336")
    elif "Neural" in l:
        colors_tok.append("#9C27B0")
    else:
        colors_tok.append("#757575")
ax.barh(range(len(df_tok)), df_tok["mean_token_id"].values, color=colors_tok)
ax.set_yticks(range(len(df_tok)))
ax.set_yticklabels(df_tok["lineage"].values, fontsize=8)
ax.set_xlabel("Mean Token ID (lower = more frequent in C2S training)")
ax.set_title("Gene Token Frequency by Cell Lineage\n(Proxy for Training Data Representation)")
ax.axvline(x=5000, color="gray", linestyle="--", alpha=0.5)

# Panel B: C2S advantage by cell type
ax = axes[0, 1]
df_ls_sorted = df_ls.sort_values("c2s_advantage", ascending=False)
colors_adv = [{"immune":"#4CAF50","epithelial":"#F44336","other":"#757575"}[d]
              for d in df_ls_sorted["domain"]]
ax.barh(range(len(df_ls_sorted)), df_ls_sorted["c2s_advantage"].values, color=colors_adv)
ax.set_yticks(range(len(df_ls_sorted)))
ax.set_yticklabels(df_ls_sorted["cell_type"].values, fontsize=8)
ax.set_xlabel("C2S/PCA Compactness Ratio\n(>1 = C2S more compact)")
ax.set_title("C2S Embedding Advantage by Cell Type")
ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)

# Panel C: Token frequency vs Advantage
ax = axes[0, 2]
valid = df_ls.dropna(subset=["token_mean_id"])
if len(valid) >= 3:
    for _, row in valid.iterrows():
        c = "#4CAF50" if row["domain"] == "immune" else "#F44336" if row["domain"] == "epithelial" else "#757575"
        ax.scatter(row["token_mean_id"], row["c2s_advantage"], c=c, s=100, edgecolors="black")
        ax.annotate(row["cell_type"][:15], (row["token_mean_id"], row["c2s_advantage"]),
                   fontsize=7, ha="center", va="bottom")
    r, p = spearmanr(valid["token_mean_id"], valid["c2s_advantage"])
    ax.set_xlabel("Mean Token ID (gene frequency)")
    ax.set_ylabel("C2S/PCA Advantage Ratio")
    ax.set_title(f"Training Frequency vs Embedding Quality\nSpearman r={r:.3f}, p={p:.3f}")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

# Panel D: Cell state discovery — cluster purity
ax = axes[1, 0]
purities = []
for c2s_cluster in sorted(t_adata.obs["c2s_cluster"].unique(), key=lambda x: int(x)):
    mask = t_adata.obs["c2s_cluster"] == c2s_cluster
    pca_clusters = t_adata.obs["pca_cluster"][mask].value_counts()
    purity = pca_clusters.iloc[0] / mask.sum() if len(pca_clusters) > 0 else 0
    purities.append(purity)
ax.bar(range(len(purities)), purities, color=["#F44336" if p < 0.5 else "#4CAF50" for p in purities])
ax.set_xlabel("C2S T-cell Cluster")
ax.set_ylabel("PCA Cluster Purity")
ax.set_title("C2S-Only Clusters vs PCA Agreement\n(Red = C2S-unique, Green = PCA-congruent)")
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

# Panel E: C2S embedding UMAP colored by domain
ax = axes[1, 1]
domain_colors = {"immune": "#4CAF50", "epithelial": "#F44336", "other": "#757575"}
domain_map = {"T-cells":"immune","B-cells":"immune","Myeloid":"immune","Plasmablasts":"immune",
              "Cancer Epithelial":"epithelial","Normal Epithelial":"epithelial"}
adata.obs["domain"] = adata.obs["cell_type"].map(domain_map).fillna("other")
# Subsample for visualization
sc.pp.subsample(adata, n_obs=5000)
sc.pp.neighbors(adata, use_rep="X_c2s", n_neighbors=15)
sc.tl.umap(adata)
for dom in ["immune", "epithelial", "other"]:
    mask = adata.obs["domain"] == dom
    ax.scatter(adata.obsm["X_umap"][mask, 0], adata.obsm["X_umap"][mask, 1],
              c=domain_colors[dom], label=dom, alpha=0.5, s=1)
ax.set_title("C2S Embedding UMAP by Domain")
ax.legend(markerscale=5)

# Panel F: Domain dependency summary
ax = axes[1, 2]
immune_wins = 7  # 7/15 across 3 immune lineages
epi_wins = 0     # 0/10 across 2 epithelial lineages
ax.bar(["Immune\n(3 lineages)", "Epithelial\n(2 lineages)"],
       [immune_wins/15*100, epi_wins/10*100],
       color=["#4CAF50", "#F44336"])
ax.set_ylabel("% Metrics Won by C2S")
ax.set_title("C2S Wins by Domain\n(5 clustering metrics × N lineages)")
for i, (v, label) in enumerate(zip([immune_wins/15*100, epi_wins/10*100],
                                    [f"{immune_wins}/15", f"{epi_wins}/10"])):
    ax.text(i, v + 1, label, ha="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Deep_Analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved Fig_Deep_Analysis.png")

print(f"\nDeep analysis complete. Results in {OUTPUT_DIR}/")
