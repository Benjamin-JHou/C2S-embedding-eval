"""
I3_alt: Token Frequency Domain Dependency Analysis.
Quantifies the relationship between C2S training data representation and model performance.

Hypothesis: Genes with lower token IDs (more frequent in C2S training) are
better represented in C2S embeddings, leading to domain-specific performance.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
H5AD = "/path/to/your/GSE176078_c2s_embedded.h5ad"

# Directory to save domain analysis outputs and diagnostic plots
OUTPUT_DIR = os.path.join(BASE, "output", "I3_domain_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Token Frequency Analysis ---
print("=== 1. Token Frequency by Gene Category ===")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_PATH)

# Gene categories
GENE_CATEGORIES = {
    "T cell": ["CD3D", "CD3E", "CD3G", "CD2", "CD28", "LCK", "ZAP70", "CD4", "CD8A", "CD8B",
               "FOXP3", "IL7R", "CCR7", "SELL", "GZMB", "GZMK", "PRF1", "GNLY", "NKG7",
               "CTLA4", "PDCD1", "LAG3", "TIGIT", "HAVCR2", "CD27", "ICOS", "TNFRSF18"],
    "B cell": ["CD19", "CD79A", "CD79B", "MS4A1", "PAX5", "BLK", "CD22", "IGHM", "IGHD",
               "CD38", "CD27", "SDC1", "MZB1", "TNFRSF17", "XBP1"],
    "Myeloid": ["CD14", "CD68", "CD163", "CSF1R", "ITGAM", "FCGR3A", "CD86", "CD80", "MRC1",
                "ITGAX", "CLEC4C", "CD1C", "FCER1A", "THBD", "CLEC10A"],
    "Breast Cancer": ["ESR1", "PGR", "ERBB2", "GATA3", "FOXA1", "KRT8", "KRT18", "KRT19",
                      "EPCAM", "MKI67", "EGFR", "VIM", "CDH1", "CDH2", "FN1", "TP53"],
    "Lung Cancer": ["NKX2-1", "SFTPC", "SFTPB", "KRT7", "NAPSA", "MUC1", "TP63", "SOX2", "KRT5"],
    "Melanoma": ["MLANA", "PMEL", "TYR", "DCT", "MITF", "S100B", "SOX10", "TYRP1", "MIA"],
    "Housekeeping": ["GAPDH", "ACTB", "B2M", "RPLP0", "TBP", "HPRT1", "PGK1", "GUSB", "HMBS"],
    "Ribosomal": ["RPL3", "RPL5", "RPL10", "RPL13A", "RPS3", "RPS5", "RPS10", "RPS18", "RPS27A"],
}

category_stats = []
for cat, genes in GENE_CATEGORIES.items():
    token_ids = []
    in_vocab = 0
    for g in genes:
        tid = tok.encode(g, add_special_tokens=False)
        if len(tid) > 0:
            token_ids.append(tid[0])
            in_vocab += 1
    avg_id = np.mean(token_ids) if token_ids else 0
    category_stats.append({
        "category": cat, "n_genes": len(genes), "in_vocab": in_vocab,
        "avg_token_id": avg_id, "min_id": min(token_ids) if token_ids else 0,
        "max_id": max(token_ids) if token_ids else 0,
    })
    domain = "immune" if cat in ["T cell", "B cell", "Myeloid"] else \
             "cancer" if "Cancer" in cat or "Melanoma" in cat else "other"
    print(f"  {cat:20s} [{domain:8s}]: avg_token_id={avg_id:.0f}, "
          f"in_vocab={in_vocab}/{len(genes)}")

# Stats
immune_ids = []
cancer_ids = []
for cat, genes in GENE_CATEGORIES.items():
    for g in genes:
        tid = tok.encode(g, add_special_tokens=False)
        if len(tid) > 0:
            if cat in ["T cell", "B cell", "Myeloid"]:
                immune_ids.append(tid[0])
            elif "Cancer" in cat or "Melanoma" in cat:
                cancer_ids.append(tid[0])

stat, p = mannwhitneyu(immune_ids, cancer_ids, alternative="less")
d_val = (np.mean(cancer_ids) - np.mean(immune_ids)) / np.std(immune_ids + cancer_ids)
print(f"\n  Immune mean token ID: {np.mean(immune_ids):.0f}")
print(f"  Cancer mean token ID: {np.mean(cancer_ids):.0f}")
print(f"  MWU (immune < cancer): U={stat}, p={p:.2e}")
print(f"  Cohen's d: {d_val:.2f}")

# --- 2. Correlation with actual expression ---
print(f"\n=== 2. Token ID vs Gene Expression in GSE176078 ===")
adata = sc.read_h5ad(H5AD)

# Compute mean expression per gene
expr = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
mean_expr = np.array(expr.mean(axis=0)).flatten()
gene_names = adata.var_names.tolist()

# Get token IDs for all genes in the dataset
gene_token_ids = {}
for g in gene_names:
    tid = tok.encode(g, add_special_tokens=False)
    if len(tid) > 0:
        gene_token_ids[g] = tid[0]

# Correlation: token ID vs mean expression
common_genes = [g for g in gene_names if g in gene_token_ids]
token_ids_arr = np.array([gene_token_ids[g] for g in common_genes])
expr_arr = np.array([mean_expr[gene_names.index(g)] for g in common_genes])

r, p_corr = spearmanr(token_ids_arr, expr_arr)
print(f"  Spearman r(token_id, mean_expr) = {r:.3f} (p={p_corr:.2e})")
print(f"  Lower token ID → higher expression (as expected: r={r:.3f})")

# --- 3. Per-gene: C2S embedding quality vs token ID ---
print(f"\n=== 3. Embedding Quality vs Token Frequency ===")
# For each gene category, compute how well C2S embeddings preserve gene-gene relationships
# vs PCA embeddings

gene_sets_for_test = {
    "Immune Markers": [g for cat in ["T cell", "B cell", "Myeloid"] for g in GENE_CATEGORIES[cat]],
    "Cancer Markers": [g for cat in ["Breast Cancer", "Lung Cancer", "Melanoma"] for g in GENE_CATEGORIES[cat]],
}

# Use T cells for this analysis (largest immune population)
t_mask = adata.obs["cell_type"] == "T-cells"
t_expr = expr[t_mask.values]
t_c2s = adata[t_mask].obsm["X_c2s"]

for set_name, gene_list in gene_sets_for_test.items():
    found_genes = [g for g in gene_list if g in gene_names]
    if len(found_genes) < 5:
        continue

    gene_indices = [gene_names.index(g) for g in found_genes]

    # C2S: pairwise cosine similarity of gene expression patterns
    gene_expr = t_expr[:, gene_indices].T  # genes × cells
    c2s_corr = np.corrcoef(gene_expr)  # gene-gene correlation matrix (ground truth)

    # Reconstruct from C2S embeddings: X ≈ C2S_embedding @ gene_projection
    # Use linear regression to map C2S space → gene expression
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(t_c2s, t_expr[:, gene_indices])
    pred_expr = lr.predict(t_c2s).T  # genes × cells
    c2s_pred_corr = np.corrcoef(pred_expr)

    # Correlation between true and predicted gene-gene correlation (upper triangle)
    triu_idx = np.triu_indices(len(found_genes), k=1)
    true_vec = c2s_corr[triu_idx]
    pred_vec = c2s_pred_corr[triu_idx]
    # Remove NaN
    valid = ~(np.isnan(true_vec) | np.isnan(pred_vec))
    r_c2s, _ = spearmanr(true_vec[valid], pred_vec[valid])

    # Same for PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=42)
    t_pca = pca.fit_transform(t_expr)
    lr_pca = LinearRegression()
    lr_pca.fit(t_pca, t_expr[:, gene_indices])
    pred_pca = lr_pca.predict(t_pca).T
    pca_pred_corr = np.corrcoef(pred_pca)
    pred_pca_vec = pca_pred_corr[triu_idx]
    r_pca, _ = spearmanr(true_vec[valid], pred_pca_vec[valid])

    print(f"  {set_name} ({len(found_genes)} genes):")
    print(f"    C2S gene-gene correlation preservation: r={r_c2s:.3f}")
    print(f"    PCA gene-gene correlation preservation:  r={r_pca:.3f}")

# --- Figure ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Token ID by category
cats = [s["category"] for s in category_stats]
avg_ids = [s["avg_token_id"] for s in category_stats]
colors = ["#2196F3" if c in ["T cell", "B cell", "Myeloid"] else
          "#FF5722" if "Cancer" in c or "Melanoma" in c else "#757575"
          for c in cats]
axes[0].barh(cats, avg_ids, color=colors)
axes[0].set_xlabel("Average Token ID (lower = more frequent in training)")
axes[0].set_title("C2S Training Data Representation\nby Gene Category")
axes[0].axvline(x=np.mean(immune_ids), color="blue", linestyle="--", alpha=0.5, label=f"Immune mean={np.mean(immune_ids):.0f}")
axes[0].axvline(x=np.mean(cancer_ids), color="red", linestyle="--", alpha=0.5, label=f"Cancer mean={np.mean(cancer_ids):.0f}")
axes[0].legend(fontsize=8)

# Panel 2: Token ID vs Expression scatter
sample_idx = np.random.choice(len(common_genes), min(5000, len(common_genes)), replace=False)
axes[1].scatter(token_ids_arr[sample_idx], expr_arr[sample_idx], alpha=0.1, s=1, c="gray")
axes[1].set_xlabel("Token ID")
axes[1].set_ylabel("Mean Expression (log-normalized)")
axes[1].set_title(f"Token ID vs Expression\nSpearman r={r:.3f}, p={p_corr:.1e}")

# Highlight key genes
for cat, genes in GENE_CATEGORIES.items():
    for g in genes[:5]:
        if g in gene_token_ids and g in gene_names:
            idx = common_genes.index(g)
            color = "blue" if cat in ["T cell", "B cell", "Myeloid"] else \
                    "red" if "Cancer" in cat or "Melanoma" in cat else "black"
            axes[1].annotate(g, (token_ids_arr[idx], expr_arr[idx]),
                           fontsize=6, color=color, alpha=0.8)

# Panel 3: Immune vs Cancer domain separation
axes[2].hist(immune_ids, bins=30, alpha=0.6, label=f"Immune (n={len(immune_ids)})", color="#2196F3")
axes[2].hist(cancer_ids, bins=30, alpha=0.6, label=f"Cancer (n={len(cancer_ids)})", color="#FF5722")
axes[2].set_xlabel("Token ID")
axes[2].set_ylabel("Frequency")
axes[2].set_title(f"Token ID Distribution\nMWU p={p:.1e}, d={d_val:.2f}")
axes[2].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Fig_Domain_Dependency.png", dpi=150, bbox_inches="tight")
plt.close()

# Save
df_stats = pd.DataFrame(category_stats)
df_stats.to_csv(f"{OUTPUT_DIR}/token_frequency_by_category.csv", index=False)

print(f"\nResults saved to {OUTPUT_DIR}/")
print("I3_alt complete.")
