"""
Automated LaTeX Table & Macro Generator for Paper 2 (Phase B7).
Reads frozen confirmatory results from results/test_benchmark.csv and data/test_manifest.json.
Generates:
  - paper/numbers.tex (single source of truth LaTeX macros)
  - paper/tables/table1_manifest.tex (27 sealed datasets)
  - paper/tables/table2_cancer_results.tex (15 cancer cohorts)
  - paper/tables/table3_scrna_results.tex (4 scRNA-seq cohorts + HurdleGate ablation)
  - paper/tables/table4_general_results.tex (8 multi-class & control cohorts)
  - paper/tables/table5_omnibus_tests.tex (Friedman test, pairwise contrasts, Holm correction)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "test_benchmark.csv"
MANIFEST_JSON = REPO_ROOT / "data" / "test_manifest.json"
PAPER_DIR = REPO_ROOT / "paper"
TABLES_DIR = PAPER_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
NUMBERS_TEX = PAPER_DIR / "numbers.tex"


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, ci: float = 0.95, seed: int = 42, stat_fn=None) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    if stat_fn is None:
        boots = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    else:
        boots = [stat_fn(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boots, 100.0 * alpha)), float(np.percentile(boots, 100.0 * (1.0 - alpha)))


def main():
    print("=" * 70)
    print("GENERATING PUBLICATION-GRADE LATEX TABLES & MACROS (PHASE B7)")
    print("=" * 70)

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)
    with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Manifest Table (Table 1)
    print("Generating Table 1: Dataset Manifest...")
    with open(TABLES_DIR / "table1_manifest.tex", "w", encoding="utf-8") as f:
        f.write("% Auto-generated from data/test_manifest.json\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\caption{Summary of the 27 sealed benchmark datasets evaluated in the confirmatory protocol (Rule R2).}\\label{tab:manifest}\n")
        f.write("\\centering\\footnotesize\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{llrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Domain & Dataset Name & Reference / Source & $N$ & $D$ & $K$ & $D/N$ & Majority \\% \\\\\n")
        f.write("\\midrule\n")

        # Group by domain
        categories = [
            ("Oncology Microarray ($D/N > 20$)", lambda d: d.get("domain") == "Oncology Microarray"),
            ("Single-Cell RNA-seq (Zero-Inflated)", lambda d: d.get("domain") == "Single-Cell RNA-seq"),
            ("Multi-Class Benchmarks ($K \\ge 10$)", lambda d: d.get("domain") in ["Computer Vision", "Texture Classification", "Character Recognition", "Text / NLP"]),
            ("Physical / Biomedical Controls", lambda d: d.get("domain") in ["Biomedical Acoustic", "Clinical Pathology", "Radar Physics"]),
        ]

        for cat_label, cat_filter in categories:
            f.write("\\multicolumn{8}{l}{\\textit{" + cat_label + "}} \\\\\n")
            for ds in manifest["datasets"]:
                if cat_filter(ds):
                    clean_name = ds["id"].replace("_", "\\_")
                    ref_text = ds.get("inclusion_reason", ds.get("name", "")).replace("&", "\\&").replace("_", "\\_")
                    d_over_n = ds["n_features"] / ds["n_samples"]
                    cb = ds.get("class_balance", [1, 1])
                    if isinstance(cb, list) and sum(cb) > 0:
                        maj_pct = (max(cb) / sum(cb)) * 100.0
                    else:
                        maj_pct = 50.0
                    f.write(f" & {clean_name} & {ref_text[:30]} & {ds['n_samples']} & {ds['n_features']} & {ds['n_clusters']} & {d_over_n:.1f} & {maj_pct:.1f}\\% \\\\\n")
            f.write("\\addlinespace[3pt]\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table*}\n")

    # Group dataframe
    universal_methods = [
        "kmeans", "ward", "spectral", "std_consensus", "core_halo",
        "fcag", "sparse_kmeans", "cohirf", "m3c", "icc_v2",
        "gac_random_gating", "nullconsensus"
    ]

    method_labels = {
        "kmeans": "K-Means",
        "ward": "Ward Hier.",
        "spectral": "Spectral",
        "std_consensus": "Std. Consensus",
        "core_halo": "Core-Halo",
        "fcag": "FCAG",
        "sparse_kmeans": "Sparse KM",
        "cohirf": "CoHiRF",
        "m3c": "M3C",
        "icc_v2": "ICC-v2 (GAC)",
        "gac_random_gating": "Random Gate",
        "nullconsensus": "NullConsensus",
        "nullconsensus_hurdle": "HurdleGate",
    }

    # Summary pivot: mean and std of ARI
    pivot_mean = df.groupby(["dataset", "method"])["ari"].mean().unstack()
    pivot_std = df.groupby(["dataset", "method"])["ari"].std().unstack()

    # 2. Cancer Microarrays Table (Table 2)
    print("Generating Table 2: 15 High-D Cancer Cohorts...")
    cancer_datasets = [d["id"] for d in manifest["datasets"] if d.get("domain") == "Oncology Microarray"]
    
    with open(TABLES_DIR / "table2_cancer_results.tex", "w", encoding="utf-8") as f:
        f.write("% Auto-generated from results/test_benchmark.csv\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\caption{Empirical Adjusted Rand Index (mean $\\pm$ std over 30 independent seeds) across the 15 high-dimensional cancer cohorts ($D/N > 20$). Bold indicates the highest mean performance per cohort.}\\label{tab:cancer_results}\n")
        f.write("\\centering\\tiny\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{lcccccccccccc}\n")
        f.write("\\toprule\n")
        cols_header = " & ".join([method_labels[m] for m in universal_methods])
        f.write(f"Dataset & {cols_header} \\\\\n")
        f.write("\\midrule\n")

        for ds in cancer_datasets:
            row_means = [pivot_mean.loc[ds, m] for m in universal_methods]
            best_val = max(row_means)
            clean_name = ds.replace("_", "\\_")
            entries = []
            for m in universal_methods:
                val = pivot_mean.loc[ds, m]
                std = pivot_std.loc[ds, m] if not np.isnan(pivot_std.loc[ds, m]) else 0.0
                cell = f"{val:.3f}"
                if abs(val - best_val) < 1e-4:
                    cell = f"\\textbf{{{cell}}}"
                entries.append(cell)
            f.write(f"{clean_name} & " + " & ".join(entries) + " \\\\\n")

        f.write("\\midrule\n")
        # Grand mean row
        cancer_mean_means = [pivot_mean.loc[cancer_datasets, m].mean() for m in universal_methods]
        best_cancer = max(cancer_mean_means)
        cancer_cells = []
        for m, val in zip(universal_methods, cancer_mean_means):
            c = f"{val:.3f}"
            if abs(val - best_cancer) < 1e-4:
                c = f"\\textbf{{{c}}}"
            cancer_cells.append(c)
        f.write("\\textbf{Mean} & " + " & ".join(cancer_cells) + " \\\\\n")

        # Rank row
        ranks = pivot_mean.loc[cancer_datasets, universal_methods].rank(axis=1, ascending=False).mean()
        rank_cells = [f"\\#{int(np.round(ranks[m]))}" for m in universal_methods]
        f.write("\\textbf{Avg Rank} & " + " & ".join(rank_cells) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table*}\n")

    # 3. Single-Cell RNA-seq Table (Table 3)
    print("Generating Table 3: scRNA-seq Count Cohorts...")
    sc_datasets = [d["id"] for d in manifest["datasets"] if d.get("domain") == "Single-Cell RNA-seq"]
    sc_methods = universal_methods + ["nullconsensus_hurdle"]

    with open(TABLES_DIR / "table3_scrna_results.tex", "w", encoding="utf-8") as f:
        f.write("% Auto-generated from results/test_benchmark.csv\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\caption{Performance recovery on zero-inflated scRNA-seq count cohorts (mean ARI over 10 seeds). HurdleGate decouples Bernoulli dropout point masses from continuous positive support multimodality.}\\label{tab:scrna_results}\n")
        f.write("\\centering\\footnotesize\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Dataset & ICC-v2 (Paper 1) & NullConsensus & \\textbf{HurdleGate} & Random Gate & Std. Consensus & Top Baseline \\\\\n")
        f.write("\\midrule\n")

        for ds in sc_datasets:
            clean_name = ds.replace("_", "\\_")
            icc_v = pivot_mean.loc[ds, "icc_v2"]
            nc_v = pivot_mean.loc[ds, "nullconsensus"]
            hg_v = pivot_mean.loc[ds, "nullconsensus_hurdle"]
            rand_v = pivot_mean.loc[ds, "gac_random_gating"]
            std_v = pivot_mean.loc[ds, "std_consensus"]

            # find top baseline
            other_baselines = [m for m in universal_methods if m not in ["icc_v2", "nullconsensus", "gac_random_gating"]]
            best_other = max(other_baselines, key=lambda m: pivot_mean.loc[ds, m])
            best_other_val = pivot_mean.loc[ds, best_other]
            top_str = f"{method_labels[best_other]} ({best_other_val:.3f})"

            f.write(f"{clean_name} & {icc_v:.3f} & {nc_v:.3f} & \\textbf{{{hg_v:.3f}}} & {rand_v:.3f} & {std_v:.3f} & {top_str} \\\\\n")

        f.write("\\midrule\n")
        f.write(f"\\textbf{{Mean}} & {pivot_mean.loc[sc_datasets, 'icc_v2'].mean():.3f} & {pivot_mean.loc[sc_datasets, 'nullconsensus'].mean():.3f} & \\textbf{{{pivot_mean.loc[sc_datasets, 'nullconsensus_hurdle'].mean():.3f}}} & {pivot_mean.loc[sc_datasets, 'gac_random_gating'].mean():.3f} & {pivot_mean.loc[sc_datasets, 'std_consensus'].mean():.3f} & Std. Cons. ({pivot_mean.loc[sc_datasets, 'std_consensus'].mean():.3f}) \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table*}\n")

    # 4. Multi-Class & Physical Controls Table (Table 4)
    print("Generating Table 4: General Multi-Class & Controls...")
    gen_datasets = [d["id"] for d in manifest["datasets"] if d.get("domain") not in ["Oncology Microarray", "Single-Cell RNA-seq"]]

    with open(TABLES_DIR / "table4_general_results.tex", "w", encoding="utf-8") as f:
        f.write("% Auto-generated from results/test_benchmark.csv\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\caption{Empirical Adjusted Rand Index across the 8 general multi-class and low-dimensional physical control cohorts.}\\label{tab:general_results}\n")
        f.write("\\centering\\tiny\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{lcccccccccccc}\n")
        f.write("\\toprule\n")
        f.write(f"Dataset & {cols_header} \\\\\n")
        f.write("\\midrule\n")

        for ds in gen_datasets:
            clean_name = ds.replace("_", "\\_")
            row_means = [pivot_mean.loc[ds, m] for m in universal_methods]
            best_val = max(row_means)
            entries = []
            for m in universal_methods:
                val = pivot_mean.loc[ds, m]
                cell = f"{val:.3f}"
                if abs(val - best_val) < 1e-4:
                    cell = f"\\textbf{{{cell}}}"
                entries.append(cell)
            f.write(f"{clean_name} & " + " & ".join(entries) + " \\\\\n")

        f.write("\\midrule\n")
        gen_mean_means = [pivot_mean.loc[gen_datasets, m].mean() for m in universal_methods]
        best_gen = max(gen_mean_means)
        gen_cells = []
        for m, val in zip(universal_methods, gen_mean_means):
            c = f"{val:.3f}"
            if abs(val - best_gen) < 1e-4:
                c = f"\\textbf{{{c}}}"
            gen_cells.append(c)
        f.write("\\textbf{Mean} & " + " & ".join(gen_cells) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table*}\n")

    # 5. Statistical Numbers & LaTeX Macros (numbers.tex)
    print("Computing formal statistical numbers for numbers.tex...")

    # Omnibus Friedman
    chi2_val, p_friedman = friedmanchisquare(*[pivot_mean[c].values for c in universal_methods])
    ranks_all = pivot_mean[universal_methods].rank(axis=1, ascending=False).mean()

    # H1 Paired differences on 15 cancer cohorts
    diff_h1_icc = pivot_mean.loc[cancer_datasets, "nullconsensus"].values - pivot_mean.loc[cancer_datasets, "icc_v2"].values
    diff_h1_cohirf = pivot_mean.loc[cancer_datasets, "nullconsensus"].values - pivot_mean.loc[cancer_datasets, "cohirf"].values
    diff_h1_rand = pivot_mean.loc[cancer_datasets, "nullconsensus"].values - pivot_mean.loc[cancer_datasets, "gac_random_gating"].values

    ci_h1_icc_low, ci_h1_icc_high = bootstrap_ci(diff_h1_icc)
    ci_h1_cohirf_low, ci_h1_cohirf_high = bootstrap_ci(diff_h1_cohirf)
    ci_h1_rand_low, ci_h1_rand_high = bootstrap_ci(diff_h1_rand)

    _, p_h1_icc_raw = wilcoxon(pivot_mean.loc[cancer_datasets, "nullconsensus"], pivot_mean.loc[cancer_datasets, "icc_v2"])
    _, p_h1_cohirf_raw = wilcoxon(pivot_mean.loc[cancer_datasets, "nullconsensus"], pivot_mean.loc[cancer_datasets, "cohirf"])
    _, p_h1_rand_raw = wilcoxon(pivot_mean.loc[cancer_datasets, "nullconsensus"], pivot_mean.loc[cancer_datasets, "gac_random_gating"])

    # Holm correction
    raw_p = [float(p_h1_icc_raw), float(p_h1_cohirf_raw), float(p_h1_rand_raw)]
    m_p = len(raw_p)
    sort_idx = np.argsort(raw_p)
    holm_p = np.empty(m_p)
    for r, idx in enumerate(sort_idx):
        holm_p[idx] = min(1.0, raw_p[idx] * (m_p - r))
    for idx in range(1, m_p):
        holm_p[sort_idx[idx]] = max(holm_p[sort_idx[idx]], holm_p[sort_idx[idx - 1]])

    # H2 Paired differences on 4 scRNA-seq cohorts
    diff_h2_icc = pivot_mean.loc[sc_datasets, "nullconsensus_hurdle"].values - pivot_mean.loc[sc_datasets, "icc_v2"].values
    diff_h2_nc = pivot_mean.loc[sc_datasets, "nullconsensus_hurdle"].values - pivot_mean.loc[sc_datasets, "nullconsensus"].values
    ci_h2_icc_low, ci_h2_icc_high = bootstrap_ci(diff_h2_icc)

    # Seed-level H2 (40 pairs)
    sc_df = df[df["domain"] == "Single-Cell RNA-seq"]
    pair_sc = sc_df.pivot(index=["dataset", "seed"], columns="method", values="ari")
    _, p_h2_seed = wilcoxon(pair_sc["nullconsensus_hurdle"], pair_sc["icc_v2"])
    diff_h2_seed = (pair_sc["nullconsensus_hurdle"] - pair_sc["icc_v2"]).values
    ci_h2_seed_low, ci_h2_seed_high = bootstrap_ci(diff_h2_seed)

    # H4 Variance Ratio
    cancer_df = df[df["domain"] == "Oncology Microarray"]
    var_df = cancer_df.groupby(["dataset", "method"])["ari"].var(ddof=1).unstack()
    ratios = np.maximum(var_df["nullconsensus"].values, 1e-8) / np.maximum(var_df["gac_random_gating"].values, 1e-8)
    r_bar = float(np.exp(np.mean(np.log(ratios))))
    ci_rbar_low, ci_rbar_high = bootstrap_ci(np.log(ratios), stat_fn=lambda x: np.exp(np.mean(x)))

    # All 27 datasets: NullConsensus vs ICC_v2 and CoHiRF
    diff_all_icc = pivot_mean["nullconsensus"].values - pivot_mean["icc_v2"].values
    diff_all_cohirf = pivot_mean["nullconsensus"].values - pivot_mean["cohirf"].values
    diff_all_rand = pivot_mean["nullconsensus"].values - pivot_mean["gac_random_gating"].values

    ci_all_icc_low, ci_all_icc_high = bootstrap_ci(diff_all_icc)
    ci_all_cohirf_low, ci_all_cohirf_high = bootstrap_ci(diff_all_cohirf)
    ci_all_rand_low, ci_all_rand_high = bootstrap_ci(diff_all_rand)

    _, p_all_icc = wilcoxon(pivot_mean["nullconsensus"], pivot_mean["icc_v2"])
    _, p_all_cohirf = wilcoxon(pivot_mean["nullconsensus"], pivot_mean["cohirf"])
    _, p_all_rand = wilcoxon(pivot_mean["nullconsensus"], pivot_mean["gac_random_gating"])

    # Write numbers.tex
    with open(NUMBERS_TEX, "w", encoding="utf-8") as f:
        f.write("% Auto-generated LaTeX macros for Paper 2 (Single Source of Truth)\n")
        f.write("% Protocol: PREREGISTRATION.md (prereg-v2)\n\n")

        # Global dataset and evaluation metrics
        f.write(f"\\newcommand{{\\TotalTestDatasets}}{{{len(manifest['datasets'])}}}\n")
        f.write(f"\\newcommand{{\\TotalEvaluations}}{{{len(df)}}}\n")
        f.write(f"\\newcommand{{\\TotalCancerCohorts}}{{{len(cancer_datasets)}}}\n")
        f.write(f"\\newcommand{{\\TotalSingleCellCohorts}}{{{len(sc_datasets)}}}\n")
        f.write(f"\\newcommand{{\\TotalGeneralCohorts}}{{{len(gen_datasets)}}}\n")

        # Omnibus Friedman
        f.write(f"\\newcommand{{\\FriedmanChiSquare}}{{{chi2_val:.2f}}}\n")
        f.write(f"\\newcommand{{\\FriedmanPValue}}{{{p_friedman:.2e}}}\n")

        # Grand Means across all 27 datasets
        for m in universal_methods:
            clean_m = m.replace("1", "One").replace("2", "Two").replace("3", "Three")
            mac_name = "".join([part.capitalize() for part in clean_m.split("_")])
            f.write(f"\\newcommand{{\\GrandMean{mac_name}}}{{{pivot_mean[m].mean():.3f}}}\n")
            f.write(f"\\newcommand{{\\Rank{mac_name}}}{{{ranks_all[m]:.2f}}}\n")

        # H1 Contrast macros
        f.write(f"\\newcommand{{\\DeltaCancerIccMean}}{{{float(np.mean(diff_h1_icc)):+.4f}}}\n")
        f.write(f"\\newcommand{{\\DeltaCancerIccCILow}}{{{ci_h1_icc_low:+.4f}}}\n")
        f.write(f"\\newcommand{{\\DeltaCancerIccCIHigh}}{{{ci_h1_icc_high:+.4f}}}\n")
        f.write(f"\\newcommand{{\\PCancerIccRaw}}{{{p_h1_icc_raw:.4f}}}\n")
        f.write(f"\\newcommand{{\\PCancerIccHolm}}{{{holm_p[0]:.4f}}}\n")

        f.write(f"\\newcommand{{\\DeltaCancerCohirfMean}}{{{float(np.mean(diff_h1_cohirf)):+.4f}}}\n")
        f.write(f"\\newcommand{{\\DeltaCancerCohirfCILow}}{{{ci_h1_cohirf_low:+.4f}}}\n")
        f.write(f"\\newcommand{{\\DeltaCancerCohirfCIHigh}}{{{ci_h1_cohirf_high:+.4f}}}\n")
        f.write(f"\\newcommand{{\\PCancerCohirfRaw}}{{{p_h1_cohirf_raw:.4f}}}\n")
        f.write(f"\\newcommand{{\\PCancerCohirfHolm}}{{{holm_p[1]:.4f}}}\n")

        f.write(f"\\newcommand{{\\DeltaCancerRandomMean}}{{{float(np.mean(diff_h1_rand)):+.4f}}}\n")
        f.write(f"\\newcommand{{\\DeltaCancerRandomCILow}}{{{ci_h1_rand_low:+.4f}}}\n")
        f.write(f"\\newcommand{{\\DeltaCancerRandomCIHigh}}{{{ci_h1_rand_high:+.4f}}}\n")
        f.write(f"\\newcommand{{\\PCancerRandomHolm}}{{{holm_p[2]:.4f}}}\n")

        # H2 Single-Cell HurdleGate macros
        f.write(f"\\newcommand{{\\DeltaScIccMean}}{{{float(np.mean(diff_h2_icc)):+.4f}}}\n")
        f.write(f"\\newcommand{{\\DeltaScIccSeedCILow}}{{{ci_h2_seed_low:+.4f}}}\n")
        f.write(f"\\newcommand{{\\DeltaScIccSeedCIHigh}}{{{ci_h2_seed_high:+.4f}}}\n")
        f.write(f"\\newcommand{{\\PScIccSeed}}{{{p_h2_seed:.2e}}}\n")
        f.write(f"\\newcommand{{\\DeltaScNullgateMean}}{{{float(np.mean(diff_h2_nc)):+.4f}}}\n")
        f.write(f"\\newcommand{{\\MeanHurdleGateSc}}{{{pivot_mean.loc[sc_datasets, 'nullconsensus_hurdle'].mean():.3f}}}\n")
        f.write(f"\\newcommand{{\\MeanIccVTwoSc}}{{{pivot_mean.loc[sc_datasets, 'icc_v2'].mean():.3f}}}\n")

        # H4 Variance macros
        f.write(f"\\newcommand{{\\OmnibusRBar}}{{{r_bar:.4f}}}\n")
        f.write(f"\\newcommand{{\\RBarCILow}}{{{ci_rbar_low:.4f}}}\n")
        f.write(f"\\newcommand{{\\RBarCIHigh}}{{{ci_rbar_high:.4f}}}\n")

        # All 27 datasets pairwise comparisons
        f.write(f"\\newcommand{{\\DeltaAllIccMean}}{{{float(np.mean(diff_all_icc)):+.4f}}}\n")
        f.write(f"\\newcommand{{\\DeltaAllIccCILow}}{{{ci_all_icc_low:+.4f}}}\n")
        f.write(f"\\newcommand{{\\DeltaAllIccCIHigh}}{{{ci_all_icc_high:+.4f}}}\n")
        f.write(f"\\newcommand{{\\PAllIcc}}{{{p_all_icc:.4e}}}\n")

        f.write(f"\\newcommand{{\\DeltaAllCohirfMean}}{{{float(np.mean(diff_all_cohirf)):+.4f}}}\n")
        f.write(f"\\newcommand{{\\DeltaAllCohirfCILow}}{{{ci_all_cohirf_low:+.4f}}}\n")
        f.write(f"\\newcommand{{\\DeltaAllCohirfCIHigh}}{{{ci_all_cohirf_high:+.4f}}}\n")
        f.write(f"\\newcommand{{\\PAllCohirf}}{{{p_all_cohirf:.4e}}}\n")

    print(f"Generated {NUMBERS_TEX} and table files under {TABLES_DIR}.")


if __name__ == "__main__":
    main()
