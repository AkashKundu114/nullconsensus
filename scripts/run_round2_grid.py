"""
Round 2: Hyperparameter Robustness Grid (scripts/run_round2_grid.py).

Evaluates sensitivity of NullConsensus to:
- Temperature tau in {0.5, 1.0, 2.0}
- Anchor count L in {20, 40, 60}
- Subspace fraction alpha in {0.10, 0.15, 0.20}

Tested across synthetic manifolds and high-D cancer cohorts.
Outputs:
- results/round2_grid.csv
- results/round2_summary.json
"""

import json
import sys
import time
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nullconsensus.api import NullConsensus  # noqa: E402
from nullconsensus.data.dev_loader import load_dev_dataset  # noqa: E402

DATASETS = ["synthetic_gaussian", "leukemia", "srbct", "colon"]
TEMPERATURES = [0.5, 1.0, 2.0]
ANCHORS = [20, 40, 60]
SUBSPACES = [0.10, 0.15, 0.20]
SEEDS = [42, 101, 202]


def main():
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "round2_grid.csv"
    out_json = results_dir / "round2_summary.json"

    grid = list(product(DATASETS, TEMPERATURES, ANCHORS, SUBSPACES, SEEDS))
    total_runs = len(grid)

    print("=" * 80)
    print("ROUND 2: HYPERPARAMETER ROBUSTNESS GRID")
    print(f"Total configurations to evaluate: {total_runs}")
    print(f"Datasets ({len(DATASETS)}): {DATASETS}")
    print(f"Temperatures ({len(TEMPERATURES)}): {TEMPERATURES}")
    print(f"Anchors ({len(ANCHORS)}): {ANCHORS}")
    print(f"Subspace fractions ({len(SUBSPACES)}): {SUBSPACES}")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}")
    print("=" * 80)

    records = []
    # Cache loaded datasets
    loaded_data = {}
    for dname in DATASETS:
        X, y, meta = load_dev_dataset(dname)
        loaded_data[dname] = (X, y, meta.n_clusters)

    count = 0
    t_start = time.perf_counter()

    for dname, tau, L, alpha, s in grid:
        count += 1
        X, y, k = loaded_data[dname]
        t0 = time.perf_counter()

        try:
            model = NullConsensus(
                n_clusters=k,
                n_base_models=25,
                n_anchors=L,
                subspace_fraction=alpha,
                temperature=tau,
                random_state=s,
            )
            labels = model.fit_predict(X)
            runtime = time.perf_counter() - t0
            ari = float(adjusted_rand_score(y, labels))
            nmi = float(normalized_mutual_info_score(y, labels))
            status = "SUCCESS"
        except Exception as e:
            runtime = time.perf_counter() - t0
            ari = 0.0
            nmi = 0.0
            status = f"FAILED: {e}"

        rec = {
            "dataset": dname,
            "temperature": tau,
            "n_anchors": L,
            "subspace_fraction": alpha,
            "seed": s,
            "ari": ari,
            "nmi": nmi,
            "runtime_sec": round(runtime, 4),
            "status": status,
        }
        records.append(rec)
        if count % 20 == 0 or count == total_runs:
            print(
                f"[{count:3d}/{total_runs:3d}] {dname:18s} | tau={tau:.1f} | L={L:2d} | alpha={alpha:.2f} | "
                f"ARI={ari:7.4f} | time={runtime:5.2f}s"
            )

    elapsed = time.perf_counter() - t_start
    print(f"\nGrid completed in {elapsed:.1f}s ({elapsed/60:.2f} min).")

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Raw results saved to: {out_csv}")

    # Summary by parameter
    print("\n" + "=" * 80)
    print("MEAN ARI BY TEMPERATURE:")
    print("=" * 80)
    p_tau = df.pivot_table(index="dataset", columns="temperature", values="ari", aggfunc="mean")
    print(p_tau.to_string())

    print("\n" + "=" * 80)
    print("MEAN ARI BY ANCHORS L:")
    print("=" * 80)
    p_L = df.pivot_table(index="dataset", columns="n_anchors", values="ari", aggfunc="mean")
    print(p_L.to_string())

    print("\n" + "=" * 80)
    print("MEAN ARI BY SUBSPACE FRACTION ALPHA:")
    print("=" * 80)
    p_alpha = df.pivot_table(index="dataset", columns="subspace_fraction", values="ari", aggfunc="mean")
    print(p_alpha.to_string())

    # Overall optimal default validation
    default_runs = df[
        (df["temperature"] == 1.0) & (df["n_anchors"] == 40) & (df["subspace_fraction"] == 0.15)
    ]
    mean_default_ari = default_runs.groupby("dataset")["ari"].mean().to_dict()
    print("\n" + "=" * 80)
    print("DEFAULT CONFIGURATION PERFORMANCE (tau=1.0, L=40, alpha=0.15):")
    for d, val in mean_default_ari.items():
        print(f"  {d:20s}: ARI = {val:.4f}")

    summary_dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_evaluations": total_runs,
        "elapsed_seconds": round(elapsed, 2),
        "mean_ari_by_temperature": df.groupby(["dataset", "temperature"])["ari"].mean().unstack().to_dict(),
        "mean_ari_by_anchors": df.groupby(["dataset", "n_anchors"])["ari"].mean().unstack().to_dict(),
        "mean_ari_by_subspace": df.groupby(["dataset", "subspace_fraction"])["ari"].mean().unstack().to_dict(),
        "default_configuration_ari": mean_default_ari,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Summary JSON saved to: {out_json}")


if __name__ == "__main__":
    main()
