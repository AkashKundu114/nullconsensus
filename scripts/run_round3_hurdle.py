"""
Round 3: Count-Aware Hurdle Calibration (scripts/run_round3_hurdle.py).

Evaluates Hypothesis H2 (Exploratory):
- Benchmarks HurdleGate (mode='hurdle') vs NullGate (mode='nullgate') vs ICC-v2 vs KMeans
- Evaluated on single-cell PBMC counts (zheng68k) and synthetic zero-inflated Poisson manifolds
  with varying dropout rates (50%, 70%, 85%).
Outputs:
- results/round3_hurdle.csv
- results/round3_summary.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nullconsensus.api import NullConsensus  # noqa: E402
from nullconsensus.baselines.classical import KMeansBaseline  # noqa: E402
from nullconsensus.baselines.gac import GACBaseline  # noqa: E402
from nullconsensus.data.dev_loader import load_dev_dataset  # noqa: E402
from nullconsensus.data.synthetic import make_zero_inflated_counts  # noqa: E402

SEEDS = [42, 101, 202, 303, 404]
DROPOUT_RATES = [0.50, 0.70, 0.85]


def main():
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "round3_hurdle.csv"
    out_json = results_dir / "round3_summary.json"

    print("=" * 80)
    print("ROUND 3: COUNT-AWARE HURDLEGATE CALIBRATION (HYPOTHESIS H2)")
    print("Evaluating positive-support multimodality screening on zero-inflated counts")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}")
    print(f"Synthetic dropout rates: {DROPOUT_RATES}")
    print("=" * 80)

    records = []
    t_start = time.perf_counter()

    # 1. Real single-cell PBMC cohort (zheng68k)
    X_zheng, y_zheng, meta_zheng = load_dev_dataset("zheng68k")
    k_zheng = meta_zheng.n_clusters
    print(f"\nEvaluating on real scRNA-seq: ZHENG68K (N={len(X_zheng)}, D={X_zheng.shape[1]}, K={k_zheng})...")

    models = {
        "nullconsensus_hurdle": lambda s: NullConsensus(n_clusters=k_zheng, mode="hurdle", random_state=s),
        "nullconsensus_nullgate": lambda s: NullConsensus(n_clusters=k_zheng, mode="nullgate", random_state=s),
        "icc_v2": lambda s: GACBaseline(n_clusters=k_zheng, random_state=s),
        "kmeans": lambda s: KMeansBaseline(n_clusters=k_zheng, random_state=s),
    }

    for mname, mfn in models.items():
        for s in SEEDS:
            t0 = time.perf_counter()
            try:
                m = mfn(s)
                labels = m.fit_predict(X_zheng)
                runtime = time.perf_counter() - t0
                ari = float(adjusted_rand_score(y_zheng, labels))
                nmi = float(normalized_mutual_info_score(y_zheng, labels))
                status = "SUCCESS"
            except Exception as e:
                runtime = time.perf_counter() - t0
                ari = 0.0
                nmi = 0.0
                status = f"FAILED: {e}"

            records.append({
                "dataset": "zheng68k",
                "dropout_rate": float(np.mean(X_zheng == 0.0)),
                "method": mname,
                "seed": s,
                "ari": ari,
                "nmi": nmi,
                "runtime_sec": round(runtime, 4),
                "status": status,
            })
            print(f"  zheng68k | {mname:22s} | seed={s:3d} | ARI={ari:7.4f} | time={runtime:5.2f}s")

    # 2. Controlled zero-inflated Poisson count benchmarks
    for dr in DROPOUT_RATES:
        dname = f"synthetic_hurdle_drop_{int(dr*100)}"
        print(f"\nEvaluating on {dname} (dropout={dr:.0%})...")

        for s in SEEDS:
            X_syn, y_syn = make_zero_inflated_counts(
                n_samples=300,
                n_features=500,
                n_informative=25,
                n_clusters=3,
                dropout_rate=dr,
                random_state=s,
            )
            k_syn = 3

            syn_models = {
                "nullconsensus_hurdle": NullConsensus(n_clusters=k_syn, mode="hurdle", random_state=s),
                "nullconsensus_nullgate": NullConsensus(n_clusters=k_syn, mode="nullgate", random_state=s),
                "icc_v2": GACBaseline(n_clusters=k_syn, random_state=s),
                "kmeans": KMeansBaseline(n_clusters=k_syn, random_state=s),
            }

            for mname, m in syn_models.items():
                t0 = time.perf_counter()
                try:
                    labels = m.fit_predict(X_syn)
                    runtime = time.perf_counter() - t0
                    ari = float(adjusted_rand_score(y_syn, labels))
                    nmi = float(normalized_mutual_info_score(y_syn, labels))
                    status = "SUCCESS"
                except Exception as e:
                    runtime = time.perf_counter() - t0
                    ari = 0.0
                    nmi = 0.0
                    status = f"FAILED: {e}"

                records.append({
                    "dataset": dname,
                    "dropout_rate": dr,
                    "method": mname,
                    "seed": s,
                    "ari": ari,
                    "nmi": nmi,
                    "runtime_sec": round(runtime, 4),
                    "status": status,
                })

    elapsed = time.perf_counter() - t_start
    print(f"\nRound 3 evaluation completed in {elapsed:.1f}s.")

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Raw results saved to: {out_csv}")

    # Summary table
    print("\n" + "=" * 80)
    print("MEAN ARI SUMMARY ACROSS ZERO-INFLATED BENCHMARKS:")
    print("=" * 80)
    p_summary = df.pivot_table(index="dataset", columns="method", values="ari", aggfunc="mean")
    print(p_summary.to_string())

    # Check Hypothesis H2 contrast
    mean_hurdle = df[df["method"] == "nullconsensus_hurdle"]["ari"].mean()
    mean_nullgate = df[df["method"] == "nullconsensus_nullgate"]["ari"].mean()
    mean_iccv2 = df[df["method"] == "icc_v2"]["ari"].mean()
    delta_vs_iccv2 = mean_hurdle - mean_iccv2
    delta_vs_nullgate = mean_hurdle - mean_nullgate

    print("\n" + "=" * 80)
    print("HYPOTHESIS H2 EVALUATION (Count Recovery via HurdleGate):")
    print(f"  NullConsensus (Hurdle Mode) Overall Mean ARI: {mean_hurdle:.4f}")
    print(f"  NullConsensus (NullGate Mode) Overall Mean ARI: {mean_nullgate:.4f}")
    print(f"  ICC-v2 (Marginal Screening) Overall Mean ARI: {mean_iccv2:.4f}")
    print(f"  Delta ARI (Hurdle vs ICC-v2): {delta_vs_iccv2:+.4f} (Pre-reg target: >= +0.0500)")
    print(f"  Delta ARI (Hurdle vs NullGate): {delta_vs_nullgate:+.4f}")

    if delta_vs_iccv2 >= 0.05:
        print("  >>> HYPOTHESIS H2: STRONGLY SUPPORTED on count data")
    elif delta_vs_iccv2 > 0:
        print("  >>> HYPOTHESIS H2: MODERATELY SUPPORTED (positive advantage)")
    else:
        print("  >>> HYPOTHESIS H2: NOT SUPPORTED")

    summary_dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_evaluations": len(df),
        "mean_hurdle_ari": round(mean_hurdle, 4),
        "mean_nullgate_ari": round(mean_nullgate, 4),
        "mean_iccv2_ari": round(mean_iccv2, 4),
        "delta_vs_iccv2": round(delta_vs_iccv2, 4),
        "mean_ari_matrix": p_summary.to_dict(),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Summary JSON saved to: {out_json}")


if __name__ == "__main__":
    main()
