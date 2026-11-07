"""
J5_FULL: Full-Scale Compound Screening.
Screens the LINCS L1000 library against breast cancer tumor signatures.
Identifies novel candidates beyond known natural compounds.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, hypergeom
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)

# ============================================================
# USER CONFIGURATION: Update paths below to match your local setup
# ============================================================
# Paths to tumor signature CSVs (defaults to repository Source_Data/ directory;
# replace with your own file paths if using custom signatures)
TUMOR_UP_CSV = os.path.join(PROJECT, "Source_Data", "tumor_up_genes.csv")
TUMOR_DOWN_CSV = os.path.join(PROJECT, "Source_Data", "tumor_down_genes.csv")

# Path to LINCS L1000 perturbation sentences CSV (columns: compound_name, sentence)
LINCS_SENTENCES_CSV = "/path/to/your/lincs_perturbation_sentences.csv"

# Directory to save screening outputs
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=== Loading Data ===")
tumor_up = pd.read_csv(TUMOR_UP_CSV)
tumor_down = pd.read_csv(TUMOR_DOWN_CSV)
TUMOR_UP = set(tumor_up["gene"].tolist())
TUMOR_DOWN = set(tumor_down["gene"].tolist())
TUMOR_ALL = TUMOR_UP | TUMOR_DOWN

lincs = pd.read_csv(LINCS_SENTENCES_CSV)
print(f"  LINCS entries: {len(lincs)}")
print(f"  Unique compounds: {lincs['compound_name'].nunique()}")

# ============================================================
# 2. DEFINE KNOWN COMPOUNDS (for validation, not discovery)
# ============================================================
KNOWN_BREAST_CANCER_DRUGS = {
    "tamoxifen", "doxorubicin", "paclitaxel", "fulvestrant",
    "anastrozole", "letrozole", "lapatinib",
}

KNOWN_NEGATIVE_CONTROLS = {
    "aspirin", "ibuprofen", "atorvastatin", "metformin", "lisinopril",
    "metoprolol", "omeprazole", "esomeprazole", "cetirizine",
    "levocetirizine", "loratadine", "desloratadine", "diphenhydramine",
}

KNOWN_NATURAL = {
    "genistein", "aminogenistein", "curcumin", "resveratrol",
    "quercetin", "epigallocatechin",
}

# ============================================================
# 3. GSEA SCORING (same as J1)
# ============================================================
def gsea_enrichment(pred_genes, gene_set, direction="down"):
    N = len(pred_genes)
    gene_set_found = gene_set & set(pred_genes)
    n_hits = len(gene_set_found)
    if n_hits == 0:
        return 0.0, 0.0

    hit_vec = np.zeros(N)
    for i, g in enumerate(pred_genes):
        if g in gene_set_found:
            hit_vec[i] = i / N if direction == "down" else (N - i) / N

    hit_weight = hit_vec.sum()
    if hit_weight == 0:
        return 0.0, 0.0

    miss_penalty = 1.0 / max(N - n_hits, 1)
    running_sum = 0
    max_es = 0
    for i in range(N):
        if pred_genes[i] in gene_set_found:
            running_sum += hit_vec[i] / hit_weight
        else:
            running_sum -= miss_penalty
        max_es = max(max_es, running_sum)

    es_norm = (max_es + 1) / 2
    return max_es, es_norm

def score_compound(genes_str):
    genes = genes_str.split()
    if len(genes) < 10:
        return None

    # GSEA scores
    es_up_raw, es_up = gsea_enrichment(genes, TUMOR_UP, direction="down")
    es_down_raw, es_down = gsea_enrichment(genes, TUMOR_DOWN, direction="up")
    gsea = (es_up + es_down) / 2

    # Overlap
    pred_set = set(genes[:200])
    overlap = len(pred_set & TUMOR_ALL) / 200

    gene_score = 0.7 * gsea + 0.3 * overlap

    # Pathway reversal (simplified for speed - key pathways only)
    pathway_genes = {
        "proliferation": {"genes": {"MKI67", "TOP2A", "PCNA", "CCNB1", "CCNB2",
                          "CDC20", "CDK1", "BIRC5", "AURKA"}, "dir": "up"},
        "er_signaling": {"genes": {"ESR1", "PGR", "GATA3", "TFF1", "GREB1",
                         "XBP1"}, "dir": "down"},
        "emt": {"genes": {"VIM", "CDH2", "FN1", "ZEB1", "SNAI1",
                "SNAI2", "TWIST1", "MMP2"}, "dir": "up"},
        "pi3k_mtor": {"genes": {"AKT1", "MTOR", "PIK3CA", "PTEN", "RHEB",
                      "RPTOR"}, "dir": "up"},
    }

    pathway_scores = []
    for pname, pinfo in pathway_genes.items():
        overlap_p = pred_set & pinfo["genes"]
        if len(overlap_p) > 0:
            cov = len(overlap_p) / len(pinfo["genes"])
            pos_score = 0
            for g in overlap_p:
                r = genes.index(g)
                pos_score += (r / len(genes)) if pinfo["dir"] == "up" else ((len(genes) - r) / len(genes))
            pos_score /= len(overlap_p)
            pathway_scores.append(0.5 * cov + 0.5 * pos_score)

    p_mean = np.mean(pathway_scores) if pathway_scores else 0.0

    # Toxicity
    toxicity_genes = {"RPL3", "RPL5", "RPL8", "RPL10", "RPS3", "RPS5", "RPS10",
                      "RPS18", "MT-ND1", "MT-ND4", "MT-CO1", "MT-CO2", "MT-ATP6"}
    tox = len(pred_set & toxicity_genes) / 200

    # Composite
    composite = 0.25 * gene_score + 0.45 * (p_mean - 0.5 * tox) + 0.30 * 0

    return {
        "gsea_es": gsea, "gene_score": gene_score,
        "pathway_mean": p_mean, "toxicity": tox,
        "composite_score": composite,
        "overlap": len(pred_set & TUMOR_ALL),
    }

# ============================================================
# 4. SCREEN ALL COMPOUNDS
# ============================================================
print("\n=== Screening ALL LINCS Compounds ===")
print("(This may take 2-3 minutes for 9,119 compounds)")

all_results = []
for _, row in tqdm(lincs.iterrows(), total=len(lincs)):
    compound_name = row["compound_name"]
    genes_str = str(row["sentence"])
    scores = score_compound(genes_str)
    if scores is None:
        continue

    # Classify
    name_lower = compound_name.lower()
    is_known_pos = any(d in name_lower for d in KNOWN_BREAST_CANCER_DRUGS)
    is_known_neg = any(d in name_lower for d in KNOWN_NEGATIVE_CONTROLS)
    is_known_nat = any(d in name_lower for d in KNOWN_NATURAL)

    if is_known_pos:
        cls = "known_positive"
    elif is_known_neg:
        cls = "known_negative"
    elif is_known_nat:
        cls = "known_natural"
    else:
        cls = "candidate"

    all_results.append({
        "compound": compound_name,
        "class": cls,
        **scores,
    })

df_all = pd.DataFrame(all_results)
print(f"\n  Scored {len(df_all)} compounds")
print(f"  Known positives: {(df_all['class']=='known_positive').sum()}")
print(f"  Known negatives: {(df_all['class']=='known_negative').sum()}")
print(f"  Known naturals:  {(df_all['class']=='known_natural').sum()}")
print(f"  Novel candidates: {(df_all['class']=='candidate').sum()}")

# ============================================================
# 5. ANALYSIS
# ============================================================

# --- 5a. Global ranking ---
df_all = df_all.sort_values("composite_score", ascending=False).reset_index(drop=True)
df_all["rank"] = range(1, len(df_all) + 1)

# --- 5b. Validation: known positives vs negatives ---
pos = df_all[df_all["class"] == "known_positive"]
neg = df_all[df_all["class"] == "known_negative"]
nat = df_all[df_all["class"] == "known_natural"]

pos_scores = pos["composite_score"].values
neg_scores = neg["composite_score"].values

stat, p_mwu = mannwhitneyu(pos_scores, neg_scores, alternative="greater")
cohens_d = (pos_scores.mean() - neg_scores.mean()) / max(np.std(np.concatenate([pos_scores, neg_scores])), 1e-10)

print(f"\n=== Internal Validation (within 9,119-compound screen) ===")
print(f"  Positive mean rank: {pos['rank'].mean():.0f} / {len(df_all)}")
print(f"  Negative mean rank: {neg['rank'].mean():.0f} / {len(df_all)}")
print(f"  Natural mean rank:  {nat['rank'].mean():.0f} / {len(df_all)}")
print(f"  MWU p={p_mwu:.4f}, Cohen's d={cohens_d:.2f}")

# Enrichment of known positives in top N
for top_n in [50, 100, 200, 500, 1000]:
    top = df_all.head(top_n)
    n_pos = (top["class"] == "known_positive").sum()
    print(f"  Top-{top_n}: {n_pos} known positives (expected: {7*top_n/len(df_all):.1f})")

# --- 5c. Top novel candidates ---
print(f"\n=== TOP 30 NOVEL CANDIDATES (excluding known drugs) ===")
novel = df_all[df_all["class"] == "candidate"].head(30)
print(f"{'Rank':<8} {'Compound':35s} {'Score':<10} {'GSEA':<10}")
print("-" * 65)
for _, row in novel.iterrows():
    print(f"{row['rank']:<8} {row['compound']:35s} {row['composite_score']:<10.4f} {row['gsea_es']:<10.4f}")

# --- 5d. Distribution statistics ---
print(f"\n=== Score Distribution ===")
for cls_name, cls_label in [("known_positive", "Positive"), ("known_negative", "Negative"),
                              ("known_natural", "Natural"), ("candidate", "Candidate")]:
    subset = df_all[df_all["class"] == cls_name]
    print(f"  {cls_label:15s}: n={len(subset):5d}, mean={subset['composite_score'].mean():.4f}, "
          f"median={subset['composite_score'].median():.4f}, "
          f"max={subset['composite_score'].max():.4f}")

# --- 5e. Percentage of candidates scoring above negative mean ---
neg_mean = neg_scores.mean()
n_above_neg = (df_all[df_all["class"] == "candidate"]["composite_score"] > neg_mean).sum()
n_candidates = len(df_all[df_all["class"] == "candidate"])
print(f"\n  Candidate compounds above negative mean: {n_above_neg}/{n_candidates} ({100*n_above_neg/max(n_candidates,1):.1f}%)")

# ============================================================
# 6. SAVE
# ============================================================
df_all.to_csv(os.path.join(OUTPUT_DIR, "full_screening_9119_compounds.csv"), index=False)

# Highlight top 100 novel candidates for wet lab consideration
top100 = df_all[df_all["class"] == "candidate"].head(100)
top100.to_csv(os.path.join(OUTPUT_DIR, "top100_novel_candidates.csv"), index=False)

# Summary stats
summary_stats = {
    "total_screened": len(df_all),
    "known_positives": len(pos),
    "known_negatives": len(neg),
    "known_naturals": len(nat),
    "novel_candidates": n_candidates,
    "positive_mean_score": pos_scores.mean(),
    "negative_mean_score": neg_scores.mean(),
    "positive_mean_rank": pos["rank"].mean(),
    "negative_mean_rank": neg["rank"].mean(),
    "mwu_p_value": p_mwu,
    "cohens_d": cohens_d,
    "candidates_above_neg_mean": n_above_neg,
    "top10_novel": ", ".join(novel.head(10)["compound"].tolist()),
}
pd.DataFrame([summary_stats]).to_csv(os.path.join(OUTPUT_DIR, "screening_summary_stats.csv"), index=False)

# ============================================================
# 7. REPORT
# ============================================================
print(f"\n{'='*70}")
print(f"FULL-SCALE SCREENING COMPLETE")
print(f"{'='*70}")
print(f"""
Screening Summary:
  Library size:        {len(df_all):,} compounds (LINCS L1000)
  Known positives:     {len(pos)} (internal validation)
  Known negatives:     {len(neg)} (internal validation)
  Novel candidates:    {n_candidates:,}

Validation:
  Positive mean rank:  {pos['rank'].mean():.0f} / {len(df_all):,}
  Negative mean rank:  {neg['rank'].mean():.0f} / {len(df_all):,}
  MWU p-value:         {p_mwu:.4f}
  Cohen's d:           {cohens_d:.2f}

Top Novel Candidates (for wet lab follow-up):
""")
for i, (_, row) in enumerate(novel.head(10).iterrows()):
    print(f"  {i+1}. {row['compound']:35s} score={row['composite_score']:.4f}")

print(f"\nFull results: {OUTPUT_DIR}/full_screening_9119_compounds.csv")
print(f"Top 100 novel: {OUTPUT_DIR}/top100_novel_candidates.csv")
print("J5_FULL complete.")
