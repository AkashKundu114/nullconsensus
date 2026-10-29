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

## [0.2.0-prereg-v2] - 2026-10-29

### Added
- **M3C Baseline Estimator (`src/nullconsensus/baselines/m3c.py`):** Standalone scikit-learn compatible implementation of Monte Carlo Reference-Based Consensus Clustering.
- **Random Gating Baseline (`src/nullconsensus/baselines/random_gating.py`):** Controlled ablation baseline matching active feature count with uniform random gating.
- **Unit Test Battery (`tests/test_baselines.py`):** 3 new unit tests validating baseline contracts, convergence, and parameter sensitivity (13/13 passing).
- **Novelty & Prior-Art Audit (`docs/NOVELTY.md`):** Complete structural comparison against M3C, hurdle/dropout methods, and related consensus clustering algorithms.
- **Null Calibration Design Specification (`docs/DESIGN_LOG.md`):** Formal justification for $B_{\text{null}} \ge 100$ and pooled null benchmarks ($B_{\text{pool}} \ge 10^5$).
- **Disambiguated Test Manifest (`data/test_manifest.json`):** 27 pre-registered benchmark datasets with exact URLs, licenses, and expected sizes.
