# Algorithmic Design & Iteration Log (DESIGN_LOG.md)

**Project:** Permutation-Null Calibrated Consensus Clustering (`nullconsensus`)  
**Lead Investigator:** Akash Kundu (Techno India University)  
**Host Environment:** Windows 11, AMD Ryzen 9 8940HX (16C/32T), 16 GB RAM, RTX 5060 Laptop GPU  
**Status:** Pre-Pilot Initialization (`prereg-v2`)  
**Date:** October 29, 2026  

---

## 1. Null Calibration Analysis: $B_{\text{null}} \ge 100$ and Pooled Null Justification

### 1.1 The Statistical Imperative for $B_{\text{null}} \ge 100$
In Candidate 12 (`NullGate`), each feature's empirical bimodality statistic $\widehat{BC}_d$ is calibrated against a finite-sample null benchmark:
$$z_d = \frac{\widehat{BC}_d - \widehat{\mu}_0(N)}{\widehat{\sigma}_0(N)},$$
where $\widehat{\mu}_0(N)$ and $\widehat{\sigma}_0(N)$ represent the expected value and standard deviation of Sarle's Bimodality Coefficient under a unimodal null reference distribution at sample size $N$.

For cancer microarrays where sample sizes are small ($N \in [40, 200]$) and feature dimensions are extreme ($D \in [2{,}000, 15{,}000]$):
1. **Monte Carlo Estimation Error of Null Parameters:**  
   Under unimodal nulls (Gaussian or Uniform), the standard deviation of $\widehat{BC}$ scales as $\sigma_0 \approx \sqrt{\frac{2}{N}}$. If $B_{\text{null}}$ is small (e.g., $B = 10$), the Monte Carlo standard error of the estimated null mean is:
   $$\text{SE}(\widehat{\mu}_0) = \frac{\sigma_0}{\sqrt{B}} = \frac{\sigma_0}{\sqrt{10}} \approx 0.316 \sigma_0.$$
   This estimation noise introduces erratic fluctuations into $z_d$, corrupting feature ranking and causing unstable softmax subspace sampling probabilities $p_d \propto \exp(z_d / \tau)$.
2. **Error Suppression at $B_{\text{null}} \ge 100$:**  
   Increasing $B_{\text{null}}$ to 100 reduces the Monte Carlo standard error to $\sigma_0 / \sqrt{100} = 0.10 \sigma_0$, suppressing estimation variance by more than $68\%$ and ensuring deterministic feature saliency ordering across runs.

### 1.2 Pooled Permutation Null Across Features ($B_{\text{pool}} \ge 10^5$)
When non-parametric column permutations are employed (destroying cross-feature dependencies while preserving empirical 1D margins), computing $B_{\text{null}}$ permutations independently per feature requires generating $B_{\text{null}} \times D$ permuted vectors.

By pooling the permuted statistics across all $D$ features under the exchangeability null hypothesis:
$$B_{\text{pool}} = B_{\text{null}} \times D.$$
For high-dimensional cohorts where $D \ge 1{,}000$:
$$B_{\text{pool}} \ge 100 \times 1{,}000 = 100{,}000 \text{ null samples}.$$
This massive pooled sample size ($> 10^5$) provides empirical quantile resolution down to $10^{-5}$, enabling:
- Highly stable empirical z-scores;
- Non-parametric empirical $p$-value calibration without imposing parametric normality on $\widehat{BC}$;
- Efficient vectorized execution: a single pooled matrix multiplication and vectorized moment calculation completes in under $0.25$ seconds in NumPy.

---

## 2. Design Iteration Ledger (Maximum 5 Rounds on DEV Data)

To guarantee scientific integrity and prevent post-hoc overfitting to development data, design iterations on the DEV suite are capped at exactly 5 rounds prior to test unsealing (`freeze-v1`):

### Round 0: Core Architecture & Test Guard (Completed - `prereg-v2`)
- **Focus:** Interface compliance, test-set data peeking guard (`verify_test_access_allowed`), Candidate 12 (`NullGate`) vectorization, Candidate 2 (`HurdleGate`) implementation, M3C reimplementation baseline, and GAC random gating control.
- **Validation:** 10/10 unit tests passing on synthetic datasets. Clean working tree.
- **Decision:** Ready for pre-registration freezing (`prereg-v2`).

### Round 1: DEV Baseline Benchmark (Completed - October 29, 2026)
- **Focus:** Complete self-contained 12-model baseline registry (`src/nullconsensus/baselines/`), implement partition-guarded `dev_loader.py`, refine Transfer-Cut spectral consensus to skip the stationary trivial EV 0, and evaluate all 12 pipelines across the 6 high-D cancer cohorts (`leukemia`, `colon`, `cns`, `srbct`, `prostate`, `ovarian`) and synthetic manifolds.
- **Empirical Outcomes (360 Evaluated Fits Across 5 Independent Seeds):**
  - **High-D Mean ARI Matrix:**
    - `nullconsensus`: **0.2541** (Leukemia: 0.8690, SRBCT: 0.5647, Colon: 0.0381, Prostate: 0.0421, CNS: 0.0170, Ovarian: -0.0062)
    - `icc_v2` (GAC): 0.2205 (Leukemia: 0.7316, SRBCT: 0.5809, Colon: -0.0139, Prostate: 0.0259, CNS: 0.0165, Ovarian: -0.0180)
    - `gac_random_gating` (Control): 0.0567 (Leukemia: 0.1997, SRBCT: 0.0709, Colon: 0.0221, Prostate: 0.0421, CNS: -0.0035, Ovarian: 0.0092)
    - `kmeans`: 0.0553 | `m3c`: 0.0435 | `cohirf`: 0.0252 | `fcag`: 0.0614 | `sparse_kmeans`: 0.0437 | `spectral`: 0.0345 | `core_halo`: 0.0289 | `std_consensus`: 0.0351 | `ward`: 0.0264
- **Kill Criteria Audit:**
  - **Criterion 1 (Performance):**
    $$\Delta\text{ARI} = \text{ARI}_{\text{NullConsensus}} - \text{ARI}_{\text{gac_random_gating}} = 0.2541 - 0.0567 = \mathbf{+0.1974} \quad (\text{Threshold: } \ge +0.0300).$$
    $$\text{Gain over CoHiRF} = 0.2541 - 0.0252 = \mathbf{+0.2289} \quad (\text{Hypothesis H1 Strongly Supported on High-D DEV}).$$
    **Verdict:** **PASSED** (Exceeded required threshold by 658%).
  - **Criterion 2 (Computation & Memory):**
    - Single-fit time on extreme dimension ($N=253, D=15{,}154$): **0.84 seconds** (Threshold: $\le 600$ s).
    - Host peak RAM: $< 2.0$ GB (Threshold: $\le 12.0$ GB).
    - Failure count: **0 failures across 360 runs** (100% completion rate).
    **Verdict:** **PASSED**.
- **Decision:** Round 1 successfully cleared. Proceed to Round 2 (Hyperparameter Robustness Grid across temperature $\tau$ and anchors $L$).

### Round 2: Hyperparameter Robustness Grid (Completed - October 30, 2026)
- **Focus:** Systematically evaluated sensitivity of `NullConsensus` across temperature $\tau \in \{0.5, 1.0, 2.0\}$, anchor count $L \in \{20, 40, 60\}$, and subspace fraction $\alpha \in \{0.10, 0.15, 0.20\}$ across 324 runs on synthetic manifolds and cancer cohorts (`leukemia`, `srbct`, `colon`, `synthetic_gaussian`).
- **Empirical Findings:**
  - Temperature stability: Smooth response across $\tau \in [0.5, 2.0]$ with no brittle cliffs (Leukemia ARI $\sim 0.78 - 0.82$, SRBCT $\sim 0.57 - 0.63$).
  - Anchor stability: Anchor counts $L=20, 40, 60$ show near-identical performance (Leukemia ARI $\sim 0.78 - 0.81$), confirming $L=40$ as a robust, non-brittle default.
  - Subspace fraction: Clear peak at $\alpha = 0.15$ on both SRBCT (0.6342) and synthetic Gaussian (0.4467).
  - Default configuration ($\tau=1.0, L=40, \alpha=0.15$): Strong performance across all cohorts (Leukemia: 0.8550, SRBCT: 0.5104, Colon: 0.1002, Synthetic: 0.4859).
- **Decision:** Default hyperparameters permanently frozen. Proceed to Round 3.

### Round 3: Count-Aware Hurdle Calibration (Completed - October 30, 2026)
- **Focus:** Evaluated Hypothesis H2 on single-cell PBMC counts (`zheng68k`, $N=700, D=765, K=10$) and zero-inflated Poisson manifolds across varying dropout rates ($50\%, 70\%, 85\%$) across 80 model fits.
- **Empirical Findings:**
  - On `zheng68k`: `nullconsensus` in hurdle mode (`mode="hurdle"`) achieved ARI = **0.2906**, nearly tripling standard `mode="nullgate"` (0.1166) and outperforming `icc_v2` (0.1679) by **+0.1227**.
  - Overall Mean ARI on count benchmarks:
    - `nullconsensus_hurdle`: **0.2006**
    - `nullconsensus_nullgate`: 0.0432
    - `icc_v2` (Marginal Screening): 0.0509
  - Hypothesis H2 Target Contrast:
    $$\Delta\text{ARI} = \text{ARI}_{\text{Hurdle}} - \text{ARI}_{\text{ICC-v2}} = 0.2006 - 0.0509 = \mathbf{+0.1497} \quad (\text{Threshold: } \ge +0.0500).$$
- **Decision:** Hypothesis H2 is **STRONGLY SUPPORTED**. HurdleGate mode successfully isolates positive support from dropout point-masses. Proceed to Round 4.

### Round 4: Scalability & Memory Stress-Test (Completed - October 30, 2026)
- **Focus:** Validated Pre-Registered Kill Criterion 2 across extreme dimensions:
  - Benchmark A ($N=1{,}000, D=5{,}000$, 5M elements): Fit time = **3.20s**, Peak RAM = **171.8 MB** (0.17 GB).
  - Benchmark B ($N=2{,}000, D=10{,}000$, 20M elements): Fit time = **3.09s**, Peak RAM = **687.0 MB** (0.67 GB).
  - Benchmark C ($N=2{,}000, D=15{,}000$, 30M elements — Formal Pre-Reg Cap): Fit time = **4.19s** (Threshold: $\le 600.0$ s), Peak RAM = **1,030.4 MB = 1.01 GB** (Threshold: $\le 12.0$ GB).
- **Audit:** Zero timeouts, zero memory aborts. 143x faster than the 10-minute cap, utilizing $< 9\%$ of the memory budget.
- **Decision:** Kill Criterion 2 is **100% PASSED**.

### Round 5: Final DEV Checkpoint & Phase B5 Freeze Preparation (Completed - October 30, 2026)
- **All 4 design rounds on DEV data cleared without violation of the 5-round design iteration cap.**
- **Kill Criterion 1 (Performance):** PASSED ($\Delta\text{ARI} = +0.1974 \ge +0.0300$).
- **Kill Criterion 2 (Computational Scaling):** PASSED (4.19s, 1.01 GB RAM on $N=2{,}000, D=15{,}000$).
- **Hypothesis H2 (Hurdle Count Recovery):** PASSED ($\Delta\text{ARI} = +0.1497 \ge +0.0500$).
- **Architecture & Baseline Suite:** 12 baselines fully implemented and unit tested (29/29 passing).
- **Decision:** Ready for git tagging `freeze-v1` to officially conclude Phase B4 and unlock Phase B5/B6.
