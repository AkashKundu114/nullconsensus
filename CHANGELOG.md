# Changelog

All notable changes to the `nullconsensus` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-28

### Added
- **Core Estimator (`NullConsensus`):** Scikit-learn compatible estimator featuring `fit`, `predict`, and `fit_predict`.
- **Permutation-Null Feature Gating (`NullGate`):** Non-parametric column-permutation testing against empirical null benchmarks.
- **Count-Aware Hurdle Gating (`HurdleGate`):** Two-part hurdle decomposition for zero-inflated scRNA-seq and sparse text.
- **Saliency-Weighted Subspace Generation:** Soft Multinomial subspace sampling with temperature scaling.
- **Evaluation Battery:** Standardized baselines (K-Means, Ward, Spectral, Dense Consensus, Anchor Graph, CoHiRF, Sparse K-Means), paired Wilcoxon tests, and bootstrap confidence intervals.
- **Test-Set Guard (`data.guard`):** Pre-registration access barrier preventing test-set peeking prior to freeze tag.


---

## [0.4.0-dev-complete] - 2026-10-30

### Added
- **Round 2 Hyperparameter Grid Runner (`scripts/run_round2_grid.py`):** Evaluated 324 configurations across $\tau \in \{0.5, 1.0, 2.0\}$, $L \in \{20, 40, 60\}$, and $\alpha \in \{0.10, 0.15, 0.20\}$; empirically validated lack of performance cliffs and confirmed canonical defaults ($\tau=1.0, L=40, \alpha=0.15$).
- **Round 3 HurdleGate Calibration Runner (`scripts/run_round3_hurdle.py`):** Verified Hypothesis H2 on single-cell PBMC counts (`zheng68k`) and synthetic zero-inflated Poisson manifolds across varying dropout rates (50%, 70%, 85%); Hurdle mode achieved $+0.1497$ ARI advantage over `icc_v2` (beating pre-registered target $\ge +0.05$).
- **Round 4 Scalability & Memory Stress-Test Runner (`scripts/run_round4_stress.py`):** Validated Pre-Registered Kill Criterion 2 up to $N=2{,}000, D=15{,}000$ (30 million elements); achieved single-fit execution time of 4.19s ($\ll 600$s cap) and peak memory of 1.01 GB ($\ll 12.0$ GB cap).
- **Results Registries:** Logged all raw experimental runs in `results/round2_grid.csv`, `results/round3_hurdle.csv`, and `results/round4_stress.csv`.

---

## [0.3.0-dev-round1] - 2026-10-29

### Added
- **Unified 12-Model Baseline Registry (`src/nullconsensus/baselines/`):** Full implementations of `KMeansBaseline`, `WardBaseline`, `SpectralBaseline`, `SparseKMeansBaseline`, `StandardConsensusBaseline`, `CoreHaloBaseline`, `FCAGBaseline`, `CoHiRFBaseline`, `M3CBaseline`, `GACBaseline`, `GACRandomGatingBaseline`, and `NullConsensus` via `get_baseline()` factory.
- **DEV Benchmark Suite Loader (`src/nullconsensus/data/dev_loader.py`):** Partition-guarded loader for the 13 Paper 1 DEV datasets and synthetic suites with `DataPeekingViolation` enforcement.
- **DEV Pilot Runner (`scripts/run_dev_pilots.py`):** Automated runner supporting `--pilot`, `--high-d`, and `--full` execution modes with automated Kill Criteria auditing.
- **Unit Test Battery Extension:** Added `tests/test_all_baselines.py` and `tests/test_dev_loader.py`, bringing test suite to 29 clean passing tests.

### Changed
- **Transfer-Cut Spectral Embedding Refinement (`src/nullconsensus/api.py`):** Upgraded bipartite spectral consensus to select non-trivial eigenvectors ($V_{\text{core}} = \text{evecs}[:, 1:K+1]$), eliminating corruption by the stationary trivial left singular vector and boosting Leukemia ARI from negative baseline values to **0.8690**.
- **Two-Stage Active Pool Softmax Subspace Sampling:** Restricted softmax saliency subspace drawing to active candidate pool $|\mathcal{F}_{\text{active}}| = \lfloor \alpha D \rfloor$, doubling synthetic Gaussian clustering ARI to **0.5225** and achieving mean High-D ARI of **0.2541** (vs. 0.0567 for `gac_random_gating`, clearing Kill Criterion 1 with $+0.1974$ margin).

---

## [0.2.0-prereg-v2] - 2026-10-29

### Added
- **M3C Baseline Estimator (`src/nullconsensus/baselines/m3c.py`):** Standalone scikit-learn compatible implementation of Monte Carlo Reference-Based Consensus Clustering.
- **Random Gating Baseline (`src/nullconsensus/baselines/random_gating.py`):** Controlled ablation baseline matching active feature count with uniform random gating.
- **Unit Test Battery (`tests/test_baselines.py`):** 3 new unit tests validating baseline contracts, convergence, and parameter sensitivity (13/13 passing).
- **Novelty & Prior-Art Audit (`docs/NOVELTY.md`):** Complete structural comparison against M3C, hurdle/dropout methods, and related consensus clustering algorithms.
- **Null Calibration Design Specification (`docs/DESIGN_LOG.md`):** Formal justification for $B_{\text{null}} \ge 100$ and pooled null benchmarks ($B_{\text{pool}} \ge 10^5$).
- **Disambiguated Test Manifest (`data/test_manifest.json`):** 27 pre-registered benchmark datasets with exact URLs, licenses, and expected sizes.
