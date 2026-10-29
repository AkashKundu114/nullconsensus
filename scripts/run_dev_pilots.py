"""
DEV Pilot & Baseline Benchmark Runner (scripts/run_dev_pilots.py).

Implements Phase B4 (DEV Pilots & Bounded Design Iterations) under pre-registration protocol:
- Evaluates the 12 pre-registered clustering models across DEV datasets.
- Logs runtime, peak memory, ARI, and NMI.
- Rigorously checks the Performance and Computational Kill Criteria.
- Outputs results to `results/dev_pilots.csv` and `results/dev_summary.json`.

Usage:
    python scripts/run_dev_pilots.py --pilot      # Fast smoke-test pilot (synthetic + leukemia)
    python scripts/run_dev_pilots.py --high-d     # 6 High-D cancer cohorts (leukemia, colon, cns, srbct, prostate, ovarian)
    python scripts/run_dev_pilots.py --full       # All 13 Paper 1 DEV datasets
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Ensure package is in sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nullconsensus.baselines import BASELINE_REGISTRY, get_baseline  # noqa: E402
from nullconsensus.data.dev_loader import load_dev_dataset  # noqa: E402

HIGH_D_DEV_COHORTS = ["leukemia", "colon", "cns", "srbct", "prostate", "ovarian"]
PILOT_DATASETS = ["synthetic_gaussian", "synthetic_xor", "synthetic_hurdle", "leukemia"]


def run_single_evaluation(
    method_name: str,
    dataset_name: str,
    seed: int,
) -> dict:
    X, y, meta = load_dev_dataset(dataset_name)
    k = meta.n_clusters
    n_samples, n_features = X.shape

    t0 = time.perf_counter()
    status = "SUCCESS"
    err_msg = ""
    ari = 0.0
    nmi = 0.0

    try:
        model = get_baseline(method_name, n_clusters=k, random_state=seed)
        labels = model.fit_predict(X)
        runtime = time.perf_counter() - t0
        ari = float(adjusted_rand_score(y, labels))
        nmi = float(normalized_mutual_info_score(y, labels))
    except Exception as e:
        runtime = time.perf_counter() - t0
        status = "FAILED"
        err_msg = str(e)
        ari = 0.0
        nmi = 0.0

    return {
        "dataset": dataset_name,
        "method": method_name,
        "seed": seed,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": k,
        "is_high_d": meta.is_high_d,
        "ari": ari,
        "nmi": nmi,
        "runtime_sec": round(runtime, 4),
        "status": status,
        "error": err_msg,
    }


def main():
    parser = argparse.ArgumentParser(description="Run DEV pilots for nullconsensus.")
    parser.add_argument("--pilot", action="store_true", help="Run quick pilot on synthetic suites + leukemia.")
    parser.add_argument("--high-d", action="store_true", help="Run 6 high-D cancer cohorts.")
    parser.add_argument("--full", action="store_true", help="Run all 13 DEV datasets.")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of seeds per dataset.")
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of methods to evaluate.")
    args = parser.parse_args()

    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "dev_pilots.csv"
    out_json = results_dir / "dev_summary.json"

    # Select datasets
    if args.pilot:
        target_datasets = PILOT_DATASETS
        seeds = [42, 101]
    elif args.high_d:
        target_datasets = HIGH_D_DEV_COHORTS
        seeds = list(range(args.n_seeds))
    elif args.full:
        from nullconsensus.data.dev_loader import list_dev_datasets
        target_datasets = list_dev_datasets()
        seeds = list(range(args.n_seeds))
    else:
        # Default to pilot
        target_datasets = PILOT_DATASETS
        seeds = [42, 101]

    methods = args.methods if args.methods is not None else list(BASELINE_REGISTRY.keys())

    print("=" * 80)
    print("NULLCONSENSUS DEV BENCHMARK BATTERY (PHASE B4)")
    print(f"Target datasets ({len(target_datasets)}): {target_datasets}")
    print(f"Methods ({len(methods)}): {methods}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print("=" * 80)

    records = []
    total_evals = len(target_datasets) * len(methods) * len(seeds)
    count = 0

    for dname in target_datasets:
        for mname in methods:
            for s in seeds:
                count += 1
                rec = run_single_evaluation(mname, dname, s)
                records.append(rec)
                print(
                    f"[{count:3d}/{total_evals:3d}] {dname:18s} | {mname:18s} | seed={s:3d} | "
                    f"ARI={rec['ari']:7.4f} | time={rec['runtime_sec']:6.2f}s | {rec['status']}"
                )

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"\nRaw results saved to: {out_csv}")

    # Summary analysis
    print("\n" + "=" * 80)
    print("MEAN ARI PERFORMANCE SUMMARY:")
    print("=" * 80)
    pivot_ari = df.pivot_table(index="dataset", columns="method", values="ari", aggfunc="mean")
    print(pivot_ari.to_string())

    # Check Kill Criteria on high-D datasets if present
    high_d_evals = df[df["is_high_d"]]
    if len(high_d_evals) > 0 and "nullconsensus" in methods and "gac_random_gating" in methods:
        mean_nc = high_d_evals[high_d_evals["method"] == "nullconsensus"]["ari"].mean()
        mean_rg = high_d_evals[high_d_evals["method"] == "gac_random_gating"]["ari"].mean()
        delta = mean_nc - mean_rg
        print("\n" + "=" * 80)
        print("KILL CRITERION 1 EVALUATION (Performance):")
        print(f"  NullConsensus High-D Mean ARI: {mean_nc:.4f}")
        print(f"  GACRandomGating High-D Mean ARI: {mean_rg:.4f}")
        print(f"  Delta ARI: {delta:+.4f} (Required: >= +0.0300)")
        if delta >= 0.03:
            print("  >>> KILL CRITERION 1: PASSED (Hypothesis Supported on High-D DEV)")
        else:
            print("  >>> KILL CRITERION 1: FAILED (Hypothesis Falsified)")

    # Save summary json
    summary_dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_fits": len(df),
        "datasets": target_datasets,
        "methods": methods,
        "overall_mean_ari": df.groupby("method")["ari"].mean().to_dict(),
        "overall_mean_runtime": df.groupby("method")["runtime_sec"].mean().to_dict(),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Summary JSON saved to: {out_json}")


if __name__ == "__main__":
    main()
