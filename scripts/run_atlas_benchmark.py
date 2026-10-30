"""
Real-World Single-Cell Atlas Case Study: 10x Genomics PBMC Single-Cell Transcriptomics.
Demonstrates the empirical collapse of standard feature screening on single-cell dropouts
and its resolution via AutoNullConsensus and Two-Part HurdleGate.
Generates Figure 6 publication vector graphic (fig6_atlas_case_study.pdf).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
)

import nullconsensus as nc

# Set publication style
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.edgecolor"] = "#333333"


def run_atlas_case_study():
    print("Loading 10x Genomics PBMC single-cell atlas...")
    adata = sc.datasets.pbmc3k_processed()
    # Use the true raw single-cell count matrix containing pervasive technical dropouts (93.7% zeros)
    X_raw = adata.raw.X.toarray()
    ground_truth = adata.obs["louvain"].astype(str).values
    unique_types, true_labels = np.unique(ground_truth, return_inverse=True)
    K = len(unique_types)
    N, D = X_raw.shape
    zero_frac = float(np.mean(X_raw == 0.0))

    print(f"Atlas loaded: N={N} cells, D={D} genes, K={K} immune types.")
    print(f"Pervasive zero-inflation dropout fraction: {zero_frac:.1%}")
    print("Cell type distribution:")
    for ct, count in adata.obs["louvain"].value_counts().items():
        print(f"  - {ct:20s}: {count:4d} ({count/N*100:4.1f}%)")

    # 1. Run AutoNullConsensus (Auto-routes to HurdleGate with k-DPP anchors)
    print("\nRunning AutoNullConsensus (Regime Auto-Router)...")
    model_auto = nc.AutoNullConsensus(n_clusters=K, mode="auto", anchor_method="dpp", random_state=42)
    anc_labels = model_auto.fit_predict(X_raw)

    # 2. Run Standard NullGate (Without Hurdle Correction) to demonstrate dropout collapse
    print("Running Standard NullGate Consensus (Without Hurdle Decomposition)...")
    model_null = nc.NullConsensus(n_clusters=K, mode="nullgate", random_state=42)
    null_labels = model_null.fit_predict(X_raw)

    # 3. Run Standard KMeans on raw counts
    print("Running KMeans on raw counts...")
    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    km_labels = km.fit_predict(X_raw)

    # Evaluate metrics
    metrics = {
        "AutoNullConsensus": {
            "ARI": float(adjusted_rand_score(true_labels, anc_labels)),
            "AMI": float(adjusted_mutual_info_score(true_labels, anc_labels)),
            "Homogeneity": float(homogeneity_score(true_labels, anc_labels)),
            "Completeness": float(completeness_score(true_labels, anc_labels)),
            "Routed_Mode": model_auto.mode_,
            "Anchor_Method": model_auto.anchor_method_,
        },
        "Standard_NullGate": {
            "ARI": float(adjusted_rand_score(true_labels, null_labels)),
            "AMI": float(adjusted_mutual_info_score(true_labels, null_labels)),
            "Homogeneity": float(homogeneity_score(true_labels, null_labels)),
            "Completeness": float(completeness_score(true_labels, null_labels)),
        },
        "KMeans": {
            "ARI": float(adjusted_rand_score(true_labels, km_labels)),
            "AMI": float(adjusted_mutual_info_score(true_labels, km_labels)),
            "Homogeneity": float(homogeneity_score(true_labels, km_labels)),
            "Completeness": float(completeness_score(true_labels, km_labels)),
        },
    }

    print("\nEvaluation Metrics:")
    for algo, res in metrics.items():
        print(f"  {algo:20s}: ARI={res['ARI']:.4f}, AMI={res['AMI']:.4f}, Homogeneity={res['Homogeneity']:.4f}")

    # Check rare cell type purity (Megakaryocytes: 15 cells = 0.57%)
    megakaryocyte_idx = np.where(ground_truth == "Megakaryocytes")[0]
    anc_megak_cluster = pd.Series(anc_labels[megakaryocyte_idx]).mode()[0]
    anc_megak_purity = float(np.mean(anc_labels[megakaryocyte_idx] == anc_megak_cluster))

    metrics["Rare_Megakaryocytes_Purity"] = {
        "True_Count": len(megakaryocyte_idx),
        "AutoNullConsensus_Purity": anc_megak_purity,
    }
    print(f"\nMegakaryocyte Sub-Type Resolution: AutoNullConsensus Purity = {anc_megak_purity:.1%}")

    # Save metrics to JSON
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "atlas_case_study.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 4. Generate Publication Vector Figure 6
    print("Generating Figure 6 vector visualization...")
    fig_dir = Path("paper/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.0])

    umap_coords = adata.obsm["X_umap"]

    # (a) Ground Truth Cell Types
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=true_labels,
        cmap="tab10",
        s=12,
        alpha=0.85,
        edgecolors="none",
    )
    ax1.set_title("(a) Ground Truth Immune Subtypes ($K=8$)", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("UMAP 1", fontsize=9.5)
    ax1.set_ylabel("UMAP 2", fontsize=9.5)
    ax1.tick_params(labelsize=8)

    # Highlight rare megakaryocytes
    ax1.scatter(
        umap_coords[megakaryocyte_idx, 0],
        umap_coords[megakaryocyte_idx, 1],
        s=65,
        facecolors="none",
        edgecolors="black",
        linewidths=1.3,
        label="Megakaryocytes (0.57%)",
    )
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # (b) Standard NullGate (Dropout Collapse)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=null_labels,
        cmap="tab10",
        s=12,
        alpha=0.85,
        edgecolors="none",
    )
    ax2.set_title(
        f"(b) Standard Gating Collapse (ARI = {metrics['Standard_NullGate']['ARI']:.3f})",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax2.set_xlabel("UMAP 1", fontsize=9.5)
    ax2.set_ylabel("UMAP 2", fontsize=9.5)
    ax2.tick_params(labelsize=8)

    # (c) AutoNullConsensus HurdleGate Recovery Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    methods = ["Standard Gating\n(NullGate)", "K-Means", "AutoNullConsensus\n(HurdleGate + DPP)"]
    aris = [metrics["Standard_NullGate"]["ARI"], metrics["KMeans"]["ARI"], metrics["AutoNullConsensus"]["ARI"]]
    amis = [metrics["Standard_NullGate"]["AMI"], metrics["KMeans"]["AMI"], metrics["AutoNullConsensus"]["AMI"]]

    x = np.arange(len(methods))
    width = 0.35

    ax3.bar(x - width / 2, aris, width, label="ARI", color="#d62728", edgecolor="#333333", linewidth=0.8)
    ax3.bar(x + width / 2, amis, width, label="AMI", color="#1f77b4", edgecolor="#333333", linewidth=0.8)

    ax3.set_ylabel("Metric Score", fontsize=10)
    ax3.set_title("(c) 10x Single-Cell Dropout Recovery", fontsize=11, fontweight="bold", pad=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=15, ha="right", fontsize=8.5)
    ax3.set_ylim(0, 0.65)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", fontsize=8.5)

    # Save vector PDF and PNG
    pdf_path = fig_dir / "fig6_atlas_case_study.pdf"
    png_path = fig_dir / "fig6_atlas_case_study.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved vector figure: {pdf_path}")
    print(f"Saved raster figure: {png_path}")


if __name__ == "__main__":
    run_atlas_case_study()
