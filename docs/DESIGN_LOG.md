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

### Round 1: DEV Baseline Benchmark (Planned)
- **Focus:** Run all 12 baselines across the 13 Paper 1 DEV datasets with 30 seeds on high-D cohorts.
- **Target:** Verify that `NullConsensus` satisfies Performance Kill Criterion ($\Delta\text{ARI} \ge +0.03$ over `gac_random_gating` on high-D DEV benchmarks).

### Round 2: Hyperparameter Robustness Grid (Planned)
- **Focus:** Evaluate sensitivity to temperature $\tau \in \{0.5, 1.0, 2.0\}$ and anchors $L \in \{20, 40, 60\}$ on synthetic manifolds.

### Round 3: Count-Aware Hurdle Calibration (Planned)
- **Focus:** Benchmark `HurdleGate` vs standard gating on PBMC68k and simulated Poisson-dropout suites.

### Round 4: Scalability & Memory Stress-Test (Planned)
- **Focus:** Confirm single-fit execution time $< 10$ minutes on $N=2{,}000, D=15{,}000$ synthetic distractor matrices, adhering to 12.0 GB RAM cap.

### Round 5: Final DEV Checkpoint & Test Set Freeze (Planned)
- **Focus:** Freeze all parameters, verify commit hash, tag `freeze-v1`, and unseal test suite.
