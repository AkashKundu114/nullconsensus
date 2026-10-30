"""
Confirmatory Test Benchmark Battery Runner (scripts/run_test_benchmark.py).

Implements Phase B6 (Confirmatory Test Set Evaluation) under strict Pre-Registration Protocol:
- Evaluates the 12 pre-registered models across all 27 Confirmatory TEST datasets.
- 15 High-D Cancer Cohorts: 30 independent seeds (1000 to 1029) per method = 5,400 fits.
- 12 Remaining Cohorts: 10 independent seeds (1000 to 1009) per method = 1,440 fits.
- Total planned evaluations: 6,840 model fits.
- Logs ARI, NMI, runtime, peak RAM, and error status to results/test_benchmark.csv.
- Fully resumable and checkpointed after every single model fit.

Strictly enforced by Rule R2:
Requires FREEZE_TAG=freeze-v1 and clean working tree via guard.py.
"""

from __future__ import annotations

import os

# Limit OpenBLAS/OMP thread pools per process to avoid memory exhaustion on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nullconsensus.baselines import BASELINE_REGISTRY, get_baseline  # noqa: E402
from nullconsensus.data.guard import verify_test_access_allowed  # noqa: E402
from nullconsensus.data.test_loader import list_test_datasets, load_test_dataset  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_CSV = RESULTS_DIR / "test_benchmark.csv"

CANCER_DATASETS = [
    "armstrong2002",
    "bhattacharjee2001",
    "chowdary2006",
    "dyrskjot2003",
    "gordon2002",
    "pomeroy2002",
    "ramaswamy2001",
    "shipp2002",
    "singh2002",
    "su2001",
    "west2001",
    "yeoh2002",
    "alizadeh2000",
    "nutt2003",
    "lapointe2004",
]

CSV_HEADER = [
    "dataset",
    "domain",
    "is_high_d",
    "n_samples",
    "n_features",
    "n_clusters",
    "method",
    "seed",
    "ari",
    "nmi",
    "runtime_sec",
    "status",
    "error",
]


def load_existing_results(csv_path: Path) -> set[tuple[str, str, int]]:
    """Loads already completed (dataset, method, seed) triplets from CSV."""
    completed = set()
    if not csv_path.exists():
        return completed
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row["dataset"], row["method"], int(row["seed"])))
    return completed


def init_benchmark_csv(csv_path: Path):
    """Initializes CSV with header if it does not exist."""
    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def evaluate_task(task_args: tuple) -> dict:
    """Worker task evaluating a single (dataset, method, seed) combination."""
    dataset_name, method_name, seed = task_args

    t0 = time.perf_counter()
    status = "SUCCESS"
    err_msg = ""
    ari = 0.0
    nmi = 0.0
    n_samples, n_features, k = 0, 0, 2
    domain = "Unknown"
    is_high_d = False

    try:
        # Rule R2 check in subprocess
        verify_test_access_allowed(dataset_name)

        X, y, meta = load_test_dataset(dataset_name)
        k = meta.n_clusters
        n_samples, n_features = X.shape
        domain = meta.domain
        is_high_d = meta.is_high_d

        # Special case: Hurdle mode evaluation on count cohorts
        if method_name == "nullconsensus_hurdle":
            from nullconsensus.api import NullConsensus
            model = NullConsensus(n_clusters=k, mode="hurdle", random_state=seed)
        else:
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
        "domain": domain,
        "is_high_d": is_high_d,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": k,
        "method": method_name,
        "seed": seed,
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
        "runtime_sec": round(runtime, 4),
        "status": status,
        "error": err_msg,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Confirmatory TEST Benchmark Battery.")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (2 datasets, 2 seeds).")
    parser.add_argument("--cancer", action="store_true", help="Run only the 15 cancer cohorts (H1 test).")
    parser.add_argument("--sc", action="store_true", help="Run only the 4 scRNA-seq count cohorts (H2 test).")
    parser.add_argument("--all", action="store_true", help="Run the full 27-dataset battery (6,840 fits).")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes.")
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of methods to evaluate.")
    args = parser.parse_args()

    # Pre-flight guard check
    verify_test_access_allowed("armstrong2002")

    all_methods = list(BASELINE_REGISTRY.keys())
    # Add hurdle mode for scRNA-seq count evaluation (Hypothesis H2)
    all_methods.append("nullconsensus_hurdle")

    methods = args.methods if args.methods is not None else all_methods

    all_datasets = list_test_datasets()

    tasks = []
    if args.smoke:
        target_datasets = ["armstrong2002", "parkinsons"]
        methods_subset = ["kmeans", "icc_v2", "nullconsensus"]
        for ds in target_datasets:
            for m in methods_subset:
                for s in [1000, 1001]:
                    tasks.append((ds, m, s))
    elif args.cancer:
        for ds in CANCER_DATASETS:
            for m in methods:
                if m == "nullconsensus_hurdle":
                    continue  # Hurdle mode is reserved for scRNA-seq count data
                for s in range(1000, 1030):  # 30 seeds: 1000 to 1029
                    tasks.append((ds, m, s))
    elif args.sc:
        sc_datasets = ["zheng_b_cells", "zheng_t_cells", "baron_pancreas", "klein_embryonic"]
        for ds in sc_datasets:
            for m in methods:
                for s in range(1000, 1010):  # 10 seeds: 1000 to 1009
                    tasks.append((ds, m, s))
    elif args.all:
        for ds in all_datasets:
            is_cancer = ds in CANCER_DATASETS
            n_seeds = 30 if is_cancer else 10
            for m in methods:
                # Only run hurdle on count cohorts
                if m == "nullconsensus_hurdle" and ds not in ["zheng_b_cells", "zheng_t_cells", "baron_pancreas", "klein_embryonic"]:
                    continue
                for s in range(1000, 1000 + n_seeds):
                    tasks.append((ds, m, s))
    else:
        print("Please specify --smoke, --cancer, --sc, or --all.")
        sys.exit(1)

    init_benchmark_csv(BENCHMARK_CSV)
    completed_keys = load_existing_results(BENCHMARK_CSV)
    pending_tasks = [t for t in tasks if (t[0], t[1], t[2]) not in completed_keys]

    print("=" * 70)
    print("CONFIRMATORY TEST BENCHMARK BATTERY (PHASE B6)")
    print(f"Total Tasks Planned   : {len(tasks)}")
    print(f"Already Completed     : {len(tasks) - len(pending_tasks)}")
    print(f"Pending Execution     : {len(pending_tasks)}")
    print(f"Parallel Workers      : {args.workers}")
    print(f"Output CSV Registry   : {BENCHMARK_CSV}")
    print("=" * 70, flush=True)

    if not pending_tasks:
        print("All requested tasks are already completed in CSV. Exiting.")
        return

    n_done = 0
    t_start = time.perf_counter()

    with open(BENCHMARK_CSV, "a", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=CSV_HEADER)

        if args.workers <= 1:
            for task in pending_tasks:
                res = evaluate_task(task)
                writer.writerow(res)
                f_out.flush()
                n_done += 1
                if n_done % 10 == 0 or n_done == len(pending_tasks):
                    rate = n_done / max(time.perf_counter() - t_start, 1e-4)
                    print(f"[{n_done}/{len(pending_tasks)}] {res['dataset']} | {res['method']} (seed {res['seed']}) -> ARI={res['ari']:.4f} ({rate:.1f} fits/s)", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_task = {executor.submit(evaluate_task, t): t for t in pending_tasks}
                for future in as_completed(future_to_task):
                    res = future.result()
                    writer.writerow(res)
                    f_out.flush()
                    n_done += 1
                    if n_done % 20 == 0 or n_done == len(pending_tasks):
                        elapsed = time.perf_counter() - t_start
                        rate = n_done / max(elapsed, 1e-4)
                        remain = (len(pending_tasks) - n_done) / max(rate, 1e-4)
                        print(f"[{n_done:4d}/{len(pending_tasks)}] {res['dataset']:18s} | {res['method']:16s} (s={res['seed']}) -> ARI={res['ari']:.4f} | {rate:.1f} fits/s, ETA: {remain/60:.1f}m", flush=True)

    total_time = time.perf_counter() - t_start
    print(f"\nConfirmatory battery execution finished: {n_done} fits evaluated in {total_time/60:.2f} minutes.")


if __name__ == "__main__":
    main()
