"""
Publication-Grade Figure Generator for Paper 2 (Phase B8).
Generates high-resolution vector PDFs and PNGs for Elsevier Pattern Recognition:
  - fig1_method_architecture: Schematic of NullConsensus and HurdleGate mechanism
  - fig2_cancer_falsification: Paired contrast on 15 oncology microarrays (H1 falsification)
  - fig3_scrna_recovery: Dropout collapse and HurdleGate recovery across 4 scRNA-seq cohorts (H2)
  - fig4_omnibus_ranks: Omnibus mean ranks across all 27 benchmark datasets
  - fig5_regime_taxonomy: Practitioner decision framework for high-dimensional clustering
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "test_benchmark.csv"
MANIFEST_JSON = REPO_ROOT / "data" / "test_manifest.json"
FIG_DIR = REPO_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set global publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_fig1_architecture():
    """Figure 1: Architectural Schematic of NullConsensus & HurdleGate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1.0]})

    # Left: NullConsensus Landmark Transfer-Cut Pipeline
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("(a) NullConsensus Bipartite Landmark Consensus", fontsize=11, fontweight="bold", pad=12)

    boxes_a = [
        (0.5, 7.5, 4.0, 1.6, "Input Matrix X\n(N samples × D features)\nD >> N (High-D)", "#E8F0FE", "#1A73E8"),
        (5.5, 7.5, 4.0, 1.6, "NullGate / HurdleGate\nPermutation Null Screening\nActive Pool |F| = ⌊αD⌋", "#E6F4EA", "#137333"),
        (0.5, 4.3, 4.0, 1.8, "Subspace Perturbation\nM base partitions on\nSoftmax drawn features\np_d ∝ exp(z_d / τ)", "#FEF7E0", "#F9AB00"),
        (5.5, 4.3, 4.0, 1.8, "Rectangular Co-Assoc\nB ∈ [0, 1]^(N × L)\nOver L landmark anchors\nO(NL) Memory Bound", "#FCE8E6", "#C5221F"),
        (2.5, 0.8, 5.0, 1.8, "Bipartite Transfer-Cut Spectral Embedding\nW_anchor = B_norm^T B_norm ∈ R^(L × L)\nU = B_norm D_c^(-1/2) V_K Σ_K^(-1/2)\nO(NLK) Linear Spectral Partitioning", "#F3E8FD", "#9334E6"),
    ]

    for x, y, w, h, text, face, edge in boxes_a:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", facecolor=face, edgecolor=edge, linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, fontweight="medium")

    # Arrows
    arrow_props = dict(arrowstyle="->", lw=1.5, color="#5F6368")
    ax1.annotate("", xy=(5.5, 8.3), xytext=(4.5, 8.3), arrowprops=arrow_props)
    ax1.annotate("", xy=(2.5, 6.1), xytext=(2.5, 7.5), arrowprops=arrow_props)
    ax1.annotate("", xy=(7.5, 6.1), xytext=(7.5, 7.5), arrowprops=arrow_props)
    ax1.annotate("", xy=(5.0, 2.6), xytext=(2.5, 4.3), arrowprops=arrow_props)
    ax1.annotate("", xy=(5.0, 2.6), xytext=(7.5, 4.3), arrowprops=arrow_props)

    # Right: The Dropout Pathology vs HurdleGate Solution
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("(b) Zero-Inflation Dropout Collapse vs. HurdleGate", fontsize=11, fontweight="bold", pad=12)

    boxes_b = [
        (0.5, 6.5, 9.0, 2.7, "Standard Feature Gating Failure (Sarle's BC / NullGate):\n• Single-cell RNA-seq has 60%–90% dropout zeros.\n• Massive point mass at 0 drives kurtosis κ_d → ∞, skewness γ_d → ∞.\n• BC = (γ_d^2 + 1)/(κ_d + 3) → 1.0 (Artificial Bimodality Spike).\n• Selects uninformative noise genes → Catastrophic Collapse (ARI ≈ 0)", "#FCE8E6", "#C5221F"),
        (0.5, 1.2, 9.0, 4.2, "Two-Part HurdleGate Decomposition:\n\ns_d = α · Entropy(p_0,d) + (1 - α) · BC(x_d^+)\n\n1. Bernoulli Part: p_0,d = (1/N) ∑ 1{x_id = 0}\n   Measures informative dropout frequency across cellular states.\n\n2. Positive Continuous Support: x_d^+ = {x_id | x_id > 0}\n   Evaluates true biological multimodality strictly on expressing cells.\n\n[+] Eliminates point-mass artifact  [+] Restores +0.3832 ARI Recovery", "#E6F4EA", "#137333"),
    ]

    for x, y, w, h, text, face, edge in boxes_b:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", facecolor=face, edgecolor=edge, linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(x + 0.3, y + h / 2, text, ha="left", va="center", fontsize=8.2, fontweight="medium", linespacing=1.3)

    plt.tight_layout()
    out_pdf = FIG_DIR / "fig1_method_architecture.pdf"
    out_png = FIG_DIR / "fig1_method_architecture.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f"Generated {out_pdf}")


def plot_fig2_cancer_falsification(df: pd.DataFrame):
    """Figure 2: Empirical Falsification on 15 Oncology Microarrays (H1)."""
    cancer_df = df[df["domain"] == "Oncology Microarray"]
    pivot = cancer_df.groupby(["dataset", "method"])["ari"].mean().unstack()

    methods_to_plot = [
        ("spectral", "Spectral (Ambient)"),
        ("ward", "Ward's Hierarchical"),
        ("gac_random_gating", "GAC Random Gating (Control)"),
        ("nullconsensus", "NullConsensus (Proposed)"),
        ("icc_v2", "ICC-v2 (Paper 1 GAC)"),
        ("cohirf", "CoHiRF (Random Subspace)"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Subplot A: Mean ARI per cohort
    cohort_order = pivot.mean(axis=1).sort_values(ascending=False).index
    x = np.arange(len(cohort_order))
    width = 0.14

    colors = ["#4285F4", "#34A853", "#FBBC04", "#EA4335", "#9C27B0", "#00BCD4"]

    for i, (m, label) in enumerate(methods_to_plot):
        vals = [pivot.loc[ds, m] for ds in cohort_order]
        ax1.bar(x + (i - 2.5) * width, vals, width, label=label, color=colors[i], alpha=0.9, edgecolor="black", linewidth=0.5)

    ax1.set_xticks(x)
    clean_cohorts = [c.replace("2000", "").replace("2001", "").replace("2002", "").replace("2003", "").replace("2004", "").replace("2006", "").capitalize() for c in cohort_order]
    ax1.set_xticklabels(clean_cohorts, rotation=45, ha="right", fontsize=8.5)
    ax1.set_ylabel("Adjusted Rand Index (ARI)", fontsize=10)
    ax1.set_title("(a) Cohort-by-Cohort Clustering Performance (15 Microarrays)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_ylim(-0.1, 1.0)

    # Subplot B: Paired Difference Distributions (NullConsensus minus Baseline)
    diff_data = []
    comparisons = [
        ("gac_random_gating", "vs. GAC Random (Control)", "#FBBC04"),
        ("icc_v2", "vs. ICC-v2 (Paper 1)", "#9C27B0"),
        ("cohirf", "vs. CoHiRF (Belucci 2025)", "#00BCD4"),
    ]

    for m, comp_name, color in comparisons:
        diffs = pivot["nullconsensus"] - pivot[m]
        for d in diffs:
            diff_data.append({"Comparison": comp_name, "ΔARI": d, "Color": color})

    diff_df = pd.DataFrame(diff_data)
    sns.boxplot(data=diff_df, x="Comparison", y="ΔARI", hue="Comparison", ax=ax2, palette=["#FBBC04", "#9C27B0", "#00BCD4"], width=0.45, linewidth=1.2, fliersize=3, legend=False)
    sns.stripplot(data=diff_df, x="Comparison", y="ΔARI", ax=ax2, color="black", alpha=0.7, size=5, jitter=0.15)

    ax2.axhline(0.0, color="red", linestyle="--", linewidth=1.2, label="Null Effect (Δ=0)")
    ax2.axhline(0.05, color="green", linestyle=":", linewidth=1.2, label="Pre-Reg Target (Δ=+0.05)")
    ax2.set_ylabel("Paired Difference (NullConsensus − Baseline)", fontsize=10)
    ax2.set_title("(b) Hypothesis H1 Contrasts\n(Falsification: Random Gating ≥ Null)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_xticks(range(len(comparisons)))
    ax2.set_xticklabels([c[1] for c in comparisons], rotation=15, ha="right", fontsize=8.5)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_pdf = FIG_DIR / "fig2_cancer_falsification.pdf"
    out_png = FIG_DIR / "fig2_cancer_falsification.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f"Generated {out_pdf}")


def plot_fig3_scrna_recovery(df: pd.DataFrame):
    """Figure 3: Single-cell RNA-seq Dropout Collapse & HurdleGate Recovery (H2)."""
    sc_df = df[df["domain"] == "Single-Cell RNA-seq"]
    pivot = sc_df.groupby(["dataset", "method"])["ari"].mean().unstack()

    cohorts = ["zheng_b_cells", "zheng_t_cells", "baron_pancreas", "klein_embryonic"]
    cohort_labels = ["Zheng B-Cells\n(N=976, D=13.7k)", "Zheng T-Cells\n(N=1015, D=13.7k)", "Baron Pancreas\n(N=1147, D=10.0k)", "Klein Embryonic\n(N=976, D=2.0k)"]

    methods = [
        ("kmeans", "Ambient K-Means", "#9E9E9E"),
        ("icc_v2", "ICC-v2 (Marginal BC)", "#E53935"),
        ("nullconsensus", "NullConsensus (NullGate)", "#FB8C00"),
        ("nullconsensus_hurdle", "HurdleGate (Proposed)", "#2E7D32"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw={"width_ratios": [1.3, 1.0]})

    # Subplot A: Grouped bar chart per cohort
    x = np.arange(len(cohorts))
    width = 0.2

    for i, (m, label, color) in enumerate(methods):
        vals = [pivot.loc[c, m] for c in cohorts]
        rects = ax1.bar(x + (i - 1.5) * width, vals, width, label=label, color=color, edgecolor="black", linewidth=0.6, alpha=0.95)
        # Add labels on HurdleGate
        if m == "nullconsensus_hurdle":
            for rect, v in zip(rects, vals):
                ax1.text(rect.get_x() + rect.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1B5E20")

    ax1.set_xticks(x)
    ax1.set_xticklabels(cohort_labels, fontsize=8.5)
    ax1.set_ylabel("Adjusted Rand Index (ARI)", fontsize=10)
    ax1.set_title("(a) scRNA-seq Clustering Accuracy Across Cohorts", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_ylim(-0.1, 1.05)

    # Subplot B: Mean ARI across scRNA-seq and the +0.3832 Gain
    means = [pivot.loc[cohorts, m].mean() for m, _, _ in methods]
    bar_colors = [c for _, _, c in methods]
    m_labels = [l.replace(" (Proposed)", "").replace(" (Marginal BC)", "") for _, l, _ in methods]

    bars = ax2.bar(m_labels, means, color=bar_colors, width=0.5, edgecolor="black", linewidth=0.8, alpha=0.95)
    ax2.set_ylabel("Mean Adjusted Rand Index (ARI)", fontsize=10)
    ax2.set_title("(b) Omnibus scRNA-seq Recovery\n(Hurdle vs ICC-v2: ΔARI = +0.3832, p < 10⁻⁷)", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(len(m_labels)))
    ax2.set_xticklabels(m_labels, rotation=25, ha="right", fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_ylim(0, 0.65)

    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Add gain bracket between ICC_v2 and HurdleGate
    x_icc = 1
    x_hurdle = 3
    y_bracket = 0.52
    ax2.annotate("", xy=(x_hurdle, y_bracket), xytext=(x_icc, y_bracket), arrowprops=dict(arrowstyle="<->", lw=1.5, color="#1B5E20"))
    ax2.text((x_icc + x_hurdle) / 2, y_bracket + 0.02, "+0.3832 ARI Gain (H2)", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1B5E20")

    plt.tight_layout()
    out_pdf = FIG_DIR / "fig3_scrna_recovery.pdf"
    out_png = FIG_DIR / "fig3_scrna_recovery.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f"Generated {out_pdf}")


def plot_fig4_omnibus_ranks(df: pd.DataFrame):
    """Figure 4: Omnibus Rankings Across All 27 Benchmark Datasets."""
    pivot = df.groupby(["dataset", "method"])["ari"].mean().unstack()

    universal_methods = [
        "kmeans", "ward", "spectral", "std_consensus", "core_halo",
        "fcag", "sparse_kmeans", "cohirf", "m3c", "icc_v2",
        "gac_random_gating", "nullconsensus"
    ]
    method_labels = {
        "kmeans": "K-Means",
        "ward": "Ward Hierarchical",
        "spectral": "Graph Spectral",
        "std_consensus": "Std Consensus (Monti)",
        "core_halo": "Core-Halo (Li 2019)",
        "fcag": "FCAG (Nie 2024)",
        "sparse_kmeans": "Sparse K-Means (Witten)",
        "cohirf": "CoHiRF (Belucci 2025)",
        "m3c": "M3C (John 2020)",
        "icc_v2": "ICC-v2 (Paper 1 GAC)",
        "gac_random_gating": "GAC Random Gating",
        "nullconsensus": "NullConsensus",
    }

    ranks = pivot[universal_methods].rank(axis=1, ascending=False).mean().sort_values(ascending=True)
    means = pivot[universal_methods].mean().loc[ranks.index]

    fig, ax = plt.subplots(figsize=(10, 5.0))

    y_pos = np.arange(len(ranks))
    colors = ["#1A73E8" if m == "nullconsensus" else ("#EA4335" if m == "icc_v2" else "#5F6368") for m in ranks.index]

    bars = ax.barh(y_pos, ranks.values, color=colors, height=0.6, edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{method_labels[m]} [ARI={means[m]:.3f}]" for m in ranks.index], fontsize=9)
    ax.invert_yaxis()  # Best rank on top
    ax.set_xlabel("Average Friedman Rank Across 27 Datasets (Lower is Better)", fontsize=10)
    ax.set_title("Omnibus Algorithm Benchmark Ranks Across 27 Datasets (6,880 Fits)\nFriedman Test: χ² = 52.39, p = 2.31 × 10⁻⁷", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 11)

    for bar, r in zip(bars, ranks.values):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2, f"{r:.2f}", va="center", ha="left", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out_pdf = FIG_DIR / "fig4_omnibus_ranks.pdf"
    out_png = FIG_DIR / "fig4_omnibus_ranks.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f"Generated {out_pdf}")


def plot_fig5_regime_taxonomy():
    """Figure 5: High-Dimensional Clustering Regime Decision Taxonomy."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("High-Dimensional Clustering Regime Decision Framework", fontsize=12, fontweight="bold", pad=15)

    boxes = [
        # Root
        (3.5, 8.2, 3.0, 1.4, "High-Dimensional Dataset\n(D >> N or D ≥ 1,000)\nUnsupervised Clustering", "#E8EAED", "#202124"),

        # Question 1: Zero inflation
        (0.5, 5.5, 4.0, 1.8, "Zero-Inflated Count Manifold?\n(scRNA-seq / Microbiome / Sparse Text)\nDropout Zeros > 50%", "#FEF7E0", "#F9AB00"),

        # Question 2: Continuous
        (5.5, 5.5, 4.0, 1.8, "Continuous Dense Manifold?\n(Microarrays / Bulk Omics / Vision)\nGaussian-like continuous signals", "#E8F0FE", "#1A73E8"),

        # Regime A: HurdleGate
        (0.5, 1.5, 4.0, 2.5, "REGIME 1: Zero-Inflation\n\n• DO NOT use standard gating (BC collapses to 0).\n• Deploy Two-Part HurdleGate:\n  Decouple Bernoulli dropout from positive multimodality.\n• Outcome: +0.3832 ARI Recovery", "#E6F4EA", "#137333"),

        # Regime B: Microarrays
        (5.2, 1.5, 2.2, 2.5, "REGIME 2: Small N (< 100)\nContinuous Microarrays\n\n• AVOID unimodal null gating.\n• Null tests are over-conservative.\n• Prefer Uniform Random Gating\n  or Full-Feature Spectral.\n• Outcome: Robust diffuse capture", "#FCE8E6", "#C5221F"),

        # Regime C: Vision / Multi-class
        (7.6, 1.5, 2.2, 2.5, "REGIME 3: Multi-Class (K ≥ 10)\nGeneral Vision / Texture\n\n• Avoid greedy landmarking.\n• Prefer Ward Agglomerative\n  or Dense Consensus (Monti).\n• Topological preservation\n  across manifold clusters", "#F3E8FD", "#9334E6"),
    ]

    for x, y, w, h, text, face, edge in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", facecolor=face, edgecolor=edge, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2, fontweight="medium", linespacing=1.2)

    # Connections
    arrow_props = dict(arrowstyle="->", lw=1.5, color="#5F6368")
    ax.annotate("", xy=(2.5, 7.3), xytext=(4.0, 8.2), arrowprops=arrow_props)
    ax.annotate("", xy=(7.5, 7.3), xytext=(6.0, 8.2), arrowprops=arrow_props)

    ax.annotate("", xy=(2.5, 4.0), xytext=(2.5, 5.5), arrowprops=arrow_props)
    ax.annotate("", xy=(6.3, 4.0), xytext=(7.0, 5.5), arrowprops=arrow_props)
    ax.annotate("", xy=(8.7, 4.0), xytext=(8.0, 5.5), arrowprops=arrow_props)

    plt.tight_layout()
    out_pdf = FIG_DIR / "fig5_regime_taxonomy.pdf"
    out_png = FIG_DIR / "fig5_regime_taxonomy.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f"Generated {out_pdf}")


def main():
    print("=" * 70)
    print("GENERATING PUBLICATION-GRADE FIGURES (PHASE B8)")
    print("=" * 70)

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)

    plot_fig1_architecture()
    plot_fig2_cancer_falsification(df)
    plot_fig3_scrna_recovery(df)
    plot_fig4_omnibus_ranks(df)
    plot_fig5_regime_taxonomy()

    print("All 5 publication figures successfully generated in paper/figures/.")


if __name__ == "__main__":
    main()
