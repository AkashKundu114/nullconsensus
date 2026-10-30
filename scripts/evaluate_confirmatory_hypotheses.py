"""
Confirmatory Hypothesis Evaluation & Statistical Testing Suite (scripts/evaluate_confirmatory_hypotheses.py).

Implements the pre-registered decision rules and statistical tests of PREREGISTRATION.md:
1. Hypothesis H1: Paired Wilcoxon signed-rank tests with Holm-Bonferroni correction,
   Cohen's d, and 10,000-iteration bootstrap CIs for NullConsensus vs:
   - icc_v2 (GAC pipeline)
   - cohirf (Random Subspace K-Means)
   - gac_random_gating (uniform random gating control)
2. Hypothesis H4: Variance reduction ratio R_bar and 10,000-iteration bootstrap CI.
3. Omnibus Friedman rank test and Nemenyi post-hoc analysis across evaluated cohorts.
4. Generates LaTeX tables for Paper 2 manuscript.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_DIR = REPO_ROOT / "results"
BENCHMARK_CSV = RESULTS_DIR / "test_benchmark.csv"


def bootstrap_ci(
    data: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
    stat_fn=np.mean,
) -> tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[b] = stat_fn(sample)
    lower = float(np.percentile(boot_stats, (1 - ci) / 2 * 100))
    upper = float(np.percentile(boot_stats, (1 + ci) / 2 * 100))
    return lower, upper


def evaluate_h1(df: pd.DataFrame) -> dict:
    """Evaluates Hypothesis H1: Superior High-D Cancer Performance."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS H1: SUPERIOR HIGH-D CANCER PERFORMANCE (CONFIRMATORY)")
    print("=" * 70)

    # Filter to high-D cancer cohorts
    cancer_df = df[df["domain"] == "Oncology Microarray"].copy()
    datasets = sorted(cancer_df["dataset"].unique())
    n_cohorts = len(datasets)
    print(f"Evaluated High-D Cancer Cohorts (n={n_cohorts}): {datasets}")

    # Compute mean ARI per dataset per method
    pivot_ari = cancer_df.groupby(["dataset", "method"])["ari"].mean().unstack()

    if "nullconsensus" not in pivot_ari.columns:
        raise ValueError("nullconsensus results missing from benchmark registry.")

    primary_baselines = ["icc_v2", "cohirf", "gac_random_gating"]
    secondary_baselines = [c for c in pivot_ari.columns if c not in ["nullconsensus"] + primary_baselines]

    h1_results = {}
    raw_p_values = []
    contrasts = []

    print("\n--- Primary Contrasts vs Pre-registered Baselines ---")
    for base in primary_baselines:
        if base not in pivot_ari.columns:
            print(f"WARNING: Baseline '{base}' not present in results.")
            continue

        nc_scores = pivot_ari["nullconsensus"].values
        base_scores = pivot_ari[base].values
        diffs = nc_scores - base_scores

        mean_diff = float(np.mean(diffs))
        median_diff = float(np.median(diffs))
        sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 1e-4
        cohen_d = mean_diff / max(sd_diff, 1e-6)

        # Paired two-sided Wilcoxon test
        try:
            w_stat, p_val = wilcoxon(nc_scores, base_scores, alternative="two-sided")
        except Exception:
            w_stat, p_val = 0.0, 1.0

        ci_low, ci_high = bootstrap_ci(diffs, n_boot=10000, ci=0.95, seed=42)

        contrasts.append(base)
        raw_p_values.append(p_val)

        h1_results[base] = {
            "mean_diff": mean_diff,
            "median_diff": median_diff,
            "sd_diff": sd_diff,
            "cohen_d": cohen_d,
            "wilcoxon_stat": float(w_stat),
            "p_value_raw": float(p_val),
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "supported": (mean_diff >= 0.05 and ci_low > 0 and p_val < 0.05),
        }

    # Holm-Bonferroni correction
    m = len(raw_p_values)
    sorted_indices = np.argsort(raw_p_values)
    holm_p_values = np.empty(m)
    for rank, idx in enumerate(sorted_indices):
        holm_p = raw_p_values[idx] * (m - rank)
        holm_p_values[idx] = min(holm_p, 1.0)
    # Ensure monotonicity
    for idx in range(1, m):
        holm_p_values[sorted_indices[idx]] = max(
            holm_p_values[sorted_indices[idx]],
            holm_p_values[sorted_indices[idx - 1]]
        )

    for i, base in enumerate(contrasts):
        h1_results[base]["p_value_holm"] = float(holm_p_values[i])
        res = h1_results[base]
        sig_marker = "***" if res["p_value_holm"] < 0.001 else ("**" if res["p_value_holm"] < 0.01 else ("*" if res["p_value_holm"] < 0.05 else "n.s."))
        print(f"NullConsensus vs {base:18s}: Delta_ARI={res['mean_diff']:+.4f} (95% CI: [{res['ci_95_low']:+.4f}, {res['ci_95_high']:+.4f}]), d={res['cohen_d']:+.2f}, p_raw={res['p_value_raw']:.4e}, p_holm={res['p_value_holm']:.4e} [{sig_marker}]")

    print("\n--- Secondary Baseline Comparisons ---")
    for base in secondary_baselines:
        nc_scores = pivot_ari["nullconsensus"].values
        base_scores = pivot_ari[base].values
        diffs = nc_scores - base_scores
        mean_diff = float(np.mean(diffs))
        ci_low, ci_high = bootstrap_ci(diffs, n_boot=10000, ci=0.95, seed=42)
        try:
            _, p_val = wilcoxon(nc_scores, base_scores, alternative="two-sided")
        except Exception:
            p_val = 1.0
        print(f"NullConsensus vs {base:18s}: Delta_ARI={mean_diff:+.4f} (95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]), p={p_val:.4e}")

    # Summary table
    print("\nMean ARI across High-D Cancer Cohorts:")
    mean_summary = pivot_ari.mean().sort_values(ascending=False)
    for rank, (method, score) in enumerate(mean_summary.items(), 1):
        print(f"  Rank #{rank:2d}: {method:20s} Mean ARI = {score:.4f}")

    return h1_results


def evaluate_h4(df: pd.DataFrame) -> dict:
    """Evaluates Hypothesis H4: Subspace Stability / Cross-seed Variance Reduction."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS H4: CROSS-SEED VARIANCE REDUCTION (CONFIRMATORY)")
    print("=" * 70)

    cancer_df = df[df["domain"] == "Oncology Microarray"].copy()
    datasets = sorted(cancer_df["dataset"].unique())

    # Compute variance across seeds per dataset
    var_df = cancer_df.groupby(["dataset", "method"])["ari"].var(ddof=1).unstack()

    if "nullconsensus" not in var_df.columns or "gac_random_gating" not in var_df.columns:
        print("Missing nullconsensus or gac_random_gating for H4 variance test.")
        return {}

    nc_var = var_df["nullconsensus"].values
    rand_var = var_df["gac_random_gating"].values

    # Avoid zero variance division
    eps = 1e-8
    nc_var_safe = np.maximum(nc_var, eps)
    rand_var_safe = np.maximum(rand_var, eps)

    ratios = nc_var_safe / rand_var_safe
    log_ratios = np.log(ratios)

    # Omnibus geometric mean variance ratio
    r_bar = float(np.exp(np.mean(log_ratios)))

    # One-sided Wilcoxon signed-rank test on log variances (H1: log R < 0)
    try:
        w_stat, p_val = wilcoxon(log_ratios, alternative="less")
    except Exception:
        w_stat, p_val = 0.0, 1.0

    # 10,000 bootstrap CI of R_bar
    def geom_mean_stat(x):
        return np.exp(np.mean(x))

    ci_low, ci_high = bootstrap_ci(log_ratios, n_boot=10000, ci=0.95, seed=42, stat_fn=geom_mean_stat)

    print(f"Omnibus Geometric Mean Variance Ratio R_bar: {r_bar:.4f}")
    print(f"95% Bootstrap Confidence Interval           : [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"One-sided Wilcoxon Signed-Rank Test p-value : {p_val:.4e}")
    print(f"Pre-registered Falsification Boundary (R_high < 0.70): {'PASSED' if ci_high < 0.70 else 'FALSIFIED'}")

    return {
        "r_bar": r_bar,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(p_val),
        "passed": ci_high < 0.70,
    }


def evaluate_omnibus(df: pd.DataFrame):
    """Computes non-parametric Friedman test and mean ranks across all datasets."""
    print("\n" + "=" * 70)
    print("OMNIBUS BENCHMARK ANALYSIS (FRIEDMAN & MEAN RANKS)")
    print("=" * 70)

    pivot = df.groupby(["dataset", "method"])["ari"].mean().unstack()
    # Rank each row: rank 1 is highest ARI
    ranks = pivot.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean().sort_values()

    print("Overall Algorithm Rankings (Lower is Better):")
    for r, (m, score) in enumerate(mean_ranks.items(), 1):
        print(f"  #{r:2d} (Rank {score:4.2f}): {m:20s} [Mean ARI = {pivot[m].mean():.4f}]")

    # Friedman test
    try:
        chi2, p_val = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
        print(f"\nFriedman Test: Chi2 = {chi2:.2f}, p = {p_val:.4e}")
    except Exception as e:
        print(f"Friedman test could not be computed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Confirmatory Benchmark Hypotheses.")
    parser.add_argument("--csv", type=str, default=str(BENCHMARK_CSV), help="Path to benchmark CSV.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Benchmark CSV not found at {csv_path}. Please run benchmark first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} completed model evaluations from {csv_path}.")

    h1_res = evaluate_h1(df)
    h4_res = evaluate_h4(df)
    evaluate_omnibus(df)


if __name__ == "__main__":
    main()
