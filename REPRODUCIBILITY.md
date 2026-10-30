# Reproducibility Guide: Permutation-Null Calibrated Consensus Clustering (`nullconsensus`)

This document provides complete, push-button instructions to verify every empirical finding, reproduce all 6,880 confirmatory model fits, generate all LaTeX tables and vector figures, and compile the manuscript submitted to *Pattern Recognition* (Elsevier).

---

## 1. Environment & Hardware Specifications

The benchmarks were executed and validated on the following hardware platform:
- **Operating System:** Windows 11 Enterprise (Build 10.0.26300) / Linux (Ubuntu 22.04 LTS / 24.04 LTS)
- **CPU:** AMD Ryzen 9 8940HX (16 physical cores, 32 hardware threads)
- **RAM:** 16.0 GB DDR5 host memory
- **Python Version:** 3.10, 3.11, 3.12, or 3.14 (Verified on Python 3.14.6)
- **Key Dependencies:** `scikit-learn>=1.4.0`, `scipy>=1.12.0`, `numpy>=1.26.0`, `pandas>=2.2.0`, `matplotlib>=3.8.0`, `seaborn>=0.13.0`

### Developer Installation
```bash
git clone https://github.com/nullconsensus/nullconsensus.git
cd nullconsensus

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install editable package with dev dependencies
pip install -e ".[dev]"
```

---

## 2. Pre-Registration Protocol & Integrity Verification

To protect against post-hoc p-hacking and publication bias, this study adheres strictly to **Rule R2** (Zero Test-Set Peeking):
- Pre-registration protocol version 2.0.0 is codified in [`PREREGISTRATION.md`](file:///d:/nullconsensus/PREREGISTRATION.md) (commit `ac0fcd3`, tag `prereg-v2`).
- The development phase (Rounds 1–4) was permanently concluded and tagged at commit `568d7ed` (`freeze-v1`).
- All 27 confirmatory test datasets are programmatically guarded by [`src/nullconsensus/data/guard.py`](file:///d:/nullconsensus/src/nullconsensus/data/guard.py). Attempting to load test data without a clean git tree at tag `freeze-v1` raises a fatal `DataPeekingViolation`.

### Run Unit Test Battery
Verify the 32 unit tests across baseline estimators, permutation gating, hurdle decomposition, and data access guards:
```bash
pytest tests/
```
*Expected Result:* `32 passed in ~8s` (100% pass rate).

---

## 3. End-to-End Workflow Execution

### Step 1: Acquire Confirmatory Benchmark Datasets
Downloads and caches the 27 sealed public benchmark datasets (15 oncology microarrays, 4 zero-inflated scRNA-seq cohorts, 5 multi-class vision/text datasets, and 3 physical controls) into `data/cache/` while asserting exact cryptographic SHA256 checksums from `data/test_manifest.json`:
```bash
python scripts/fetch_test_data.py
```

### Step 2: Run Confirmatory Test Benchmark (6,880 Evaluations)
Runs all 12 universal algorithms (`kmeans`, `ward`, `spectral`, `std_consensus`, `core_halo`, `fcag`, `sparse_kmeans`, `cohirf`, `m3c`, `icc_v2`, `gac_random_gating`, `nullconsensus`) across 30 independent seeds for high-D cohorts and 10 seeds for general cohorts:
```bash
python scripts/run_test_benchmark.py
```
*Output File:* `results/test_benchmark.csv` (6,880 rows, SHA256: `4C930C8720206E6EAA563B90B9A72219E2734AA1322983F73C1A79B267BC4D1B`).

### Step 3: Evaluate Confirmatory Hypotheses & Statistical Tests
Computes paired Wilcoxon signed-rank tests, Holm-Bonferroni corrections, 10,000-iteration clustered bootstrap confidence intervals, and the non-parametric omnibus Friedman rank test:
```bash
python scripts/evaluate_confirmatory_hypotheses.py
```
*Key Verified Empirical Findings:*
1. **Hypothesis H1 (Cancer Microarrays):** FALSIFIED ($\Delta\text{ARI} = -0.0433, p_{\text{holm}} = 0.2293$). Uniform random subspace gating matches or outperforms statistical null screening when $N < 100$.
2. **Hypothesis H2 (scRNA-seq Zero-Inflation):** STRONGLY SUPPORTED ($\Delta\text{ARI} = +0.3832, 95\%\text{ CI: } [+0.3157, +0.4596], p = 3.57 \times 10^{-8}$). Branch B (Regime Paper) triggered.
3. **Hypothesis H4 (Cross-Seed Variance Ratio):** FALSIFIED ($\overline{R} = 0.6640, 95\%\text{ CI: } [0.2419, 2.0056]$).
4. **Omnibus Friedman Test:** $\chi^2 = 52.39, p = 2.31 \times 10^{-7}$ across all 27 cohorts.

### Step 4: Regenerate Publication Tables & Single-Source-of-Truth Macros
Pulls metrics from `results/test_benchmark.csv` to generate all LaTeX tables and macros:
```bash
python scripts/generate_paper_artifacts.py
```
*Generated Files:*
- `paper/numbers.tex`: All empirical macro constants.
- `paper/tables/table1_manifest.tex`: Full 27-dataset manifest.
- `paper/tables/table2_cancer_results.tex`: 15 oncology microarrays results table.
- `paper/tables/table3_scrna_results.tex`: 4 scRNA-seq zero-inflation results table.
- `paper/tables/table4_general_results.tex`: 8 multi-class & control cohorts table.
- `paper/tables/table5_omnibus_tests.tex`: Omnibus statistical contrasts table.

### Step 5: Regenerate Vector Publication Figures
Generates 300 DPI publication figures in both vector PDF and PNG formats:
```bash
python scripts/generate_paper_figures.py
```
*Generated Figures in `paper/figures/`:*
- `fig1_method_architecture.pdf`: Architecture schematic & HurdleGate mechanism.
- `fig2_cancer_falsification.pdf`: Microarray cohort performance & H1 contrast distributions.
- `fig3_scrna_recovery.pdf`: Single-cell dropout collapse & HurdleGate recovery.
- `fig4_omnibus_ranks.pdf`: Omnibus Friedman average rankings across 27 cohorts.
- `fig5_regime_taxonomy.pdf`: Practitioner decision framework & regime taxonomy.

### Step 6: Compile Elsevier Pattern Recognition Manuscript
Compiles the manuscript using the bundled Tectonic compiler:
```bash
cd paper
tectonic main.tex
```
*Generated Manuscript:* `paper/main.pdf` (322 KB).

---

## 4. Cryptographic Provenance & Checksums

| Artifact File | Role in Study | SHA256 Checksum |
| :--- | :--- | :--- |
| `results/test_benchmark.csv` | Raw 6,880 Confirmatory Model Evaluations | `4C930C8720206E6EAA563B90B9A72219E2734AA1322983F73C1A79B267BC4D1B` |
| `paper/numbers.tex` | Single Source of Truth LaTeX Macros | `A4E6EBC79E3B31697A76B732FDADF2580C2D29C7D46A03F452FDC5851CFE458B` |
| `paper/main.pdf` | Final Pattern Recognition Manuscript | `49957FCB105CB410ED8F6378C75444C76726C42179430746538E3C22E082038C` |
| `data/test_manifest.json` | 27 Pre-Registered Benchmark Manifest | `44FDE584E85DE52E63920F99401C0AE665BF4CE1156641884DC530D911BA8D3D` |

---

## 5. Contact & Support

For queries or assistance reproducing any phase of the study, contact:  
**Akash Kundu** (`akashkundu1152@gmail.com`)  
Department of Computer Science and Engineering, Techno India University, Kolkata, India.
