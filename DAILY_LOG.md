# NullConsensus Daily Engineering Log (DAILY_LOG.md)

This log documents every task executed, evidence gathered, and verified outcomes in chronological order for the `nullconsensus` repository.

---

## 2026-10-28

### Iteration 1: Repository Initialization & Method Implementation
- **Task:** Establish standalone repository at `d:\nullconsensus`, implement Candidate 12 (NullGate) as primary method and Candidate 2 (HurdleGate) as count-aware module.
- **Evidence:**
  - Implemented `NullGateFeatureSelector` with finite-sample Gaussian and uniform null calibration benchmarks providing exact Type-I error control under high-dimensional isotropic distractor noise.
  - Implemented `HurdleGateFeatureSelector` with positive-support bimodality and Bernoulli dropout entropy/excess for zero-inflated count data (scRNA-seq / text).
  - Implemented `NullConsensus` scikit-learn estimator integrating soft saliency-weighted subspace sampling, bipartite anchor graph consensus via Transfer-Cut spectral embedding, and closed-form inductive centroid projection.
  - Initialized git repository at `d:\nullconsensus`.
  - Added unit test suite in `tests/` (`test_api.py`, `test_guard.py`, `test_hurdlegate.py`, `test_nullgate.py`). All 10 tests passing (100% pass rate in 3.87s).
- **Next Task:** Commit initial codebase, tag `prereg-v1`, and proceed to Phase B4: DEV Pilots on synthetic benchmarks and Paper 1 DEV datasets.


---

## 2026-10-29

### Iteration 2: Pre-Registration Protocol v2 (prereg-v2) & GitHub Readiness
- **Task:** Establish pre-registration protocol v2 (`prereg-v2`), complete M3C prior-art audit, formalize H1 power analysis, calibrate null permutation parameters, add baseline implementations, disambiguate 27-dataset test manifest, and make repository GitHub-ready.
- **Evidence:**
  - **M3C Novelty Audit (`docs/NOVELTY.md`):** Audited John et al. (*Sci Rep* 2020) and recorded exact differences (non-parametric permutation vs Gaussian null, $O(NL)$ anchors vs $O(N^2)$ dense consensus, dual-stage feature/subspace gating vs post-hoc $K$ diagnostic). Reclassified H3 as exploratory.
  - **H1 Contrast & Power Analysis (`PREREGISTRATION.md`):** Established primary contrast of NullConsensus vs GAC pipeline vs CoHiRF across 15 oncology cohorts. Conducted statistical power analysis ($1-\beta = 0.824, n=15, d=0.833, \alpha=0.05$).
  - **Null Calibration Protocol (`docs/DESIGN_LOG.md`):** Established $B_{\text{null}} \ge 100$ and pooled null benchmark ($B_{\text{pool}} = B_{\text{null}} \times D \ge 10^5$), reducing Monte Carlo estimation error by $>68\%$.
  - **Baselines Implemented:** Added `M3CBaseline` (`src/nullconsensus/baselines/m3c.py`) and `GACRandomGatingBaseline` (`src/nullconsensus/baselines/random_gating.py`).
  - **Unit Test Suite Expanded:** Added `tests/test_baselines.py`. All 13 unit tests passing (100% pass rate in 4.97s).
  - **Test Manifest Disambiguated:** Added `size` field across all 27 datasets in `data/test_manifest.json`; verified zero test datasets downloaded prior to freeze tag.
  - **Tags & Releases:** Committed and tagged `prereg-v2`. Configured GitHub remote `https://github.com/AkashKundu114/nullconsensus.git`.
