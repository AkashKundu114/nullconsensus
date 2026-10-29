# Confirmatory Pre-Registration Protocol (PREREGISTRATION.md)

**Title:** Permutation-Null Calibrated Consensus Clustering for High-Dimensional Manifolds (`nullconsensus`)  
**Lead Investigator & Software Architect:** Akash Kundu  
**Host Institution:** Techno India University, Kolkata, India  
**Pre-Registration Tag:** `prereg-v2` (Supersedes `prereg-v1`)  
**Date:** October 29, 2026  
**Protocol Version:** 2.0.0  

---

## 1. Study Overview & Mission
This pre-registration protocol establishes a strict, falsification-driven experimental contract for evaluating **Permutation-Null Calibrated Consensus Clustering (`NullConsensus`)** against classical and contemporary clustering algorithms.

To prevent p-hacking, publication bias, and post-hoc rationalization, all hypotheses, baselines, hyperparameters, datasets, seeds, and statistical tests are pre-registered and permanently frozen prior to unsealing the independent test suite.

Following the prior art audit in `docs/NOVELTY.md`, this revision (`prereg-v2`):
1. Establishes the **primary contrast** of `NullConsensus` against `icc_v2` (GAC pipeline) and `cohirf`, adding the essential `gac_random_gating` control.
2. Incorporates a formal statistical power analysis for the $n=15$ high-D cancer cohorts.
3. Fixes finite-sample null calibration to $B_{\text{null}} \ge 100$ (with pooled null $B_{\text{pool}} \ge 10^5$).
4. Formulates a rigorous variance ratio metric and bootstrap CI for H4 while reclassifying H2 and H3 as exploratory.
5. Codifies explicit kill criteria and a maximum 5-round design iteration cap on DEV data.

---

## 2. Experimental Data Partitioning (Rule R2)

### A. Development (DEV) Suite (Seen Data)
Used exclusively for initial parameter calibration, ablation testing, and bounded design iterations ($\le 5$ rounds):
- The 13 datasets of Paper 1: CNS, Colon, Leukemia, Ovarian, Prostate, SRBCT, Sonar, Madelon, Isolet, PBMC68k, 20 Newsgroups, Indian Pines, Extended Yale B.
- Controlled synthetic generators (`nullconsensus.data.synthetic`): Gaussian shift distractor mixtures, 2D XOR parity interactions, and zero-inflated Poisson count dropouts.

### B. Confirmatory Test (TEST) Suite (Sealed Data)
The test suite consists of **27 fresh public benchmark datasets** documented in `data/test_manifest.json` that have never been used in Paper 1 or during method design:
- 15 high-dimensional cancer microarrays from the curated Souto et al. collection (Armstrong, Bhattacharjee, Chowdary, Dyrskjot, Gordon, Pomeroy, Ramaswamy, Shipp, Singh, Su, West, Yeoh, Alizadeh, Nutt, Lapointe).
- 4 zero-inflated scRNA-seq cohorts (Zheng B-Cells, Zheng T-Cells, Klein Stem Cells, Baron Pancreas).
- 5 large-$K$ benchmarks with $K \ge 15$ (COIL-20, UMIST-Faces, Texture-16, Letter-Recognition, News-20-Fresh-Subset).
- 3 low-dimensional biomedical/physical controls (Parkinsons, WDBC, Ionosphere).

**Technical Enforcement:** Access to any TEST dataset is programmatically gated by `nullconsensus.data.guard.verify_test_access_allowed()`. Any attempt to load TEST data prior to git tag `freeze-v1` and a clean working tree raises a fatal `DataPeekingViolation`.

---

## 3. Pre-Registered Hypotheses

### Primary Hypothesis (H1: Superior High-D Cancer Performance)
> Across the 15 high-dimensional cancer cohorts ($D/N > 20$) in the TEST suite, `NullConsensus` achieves a statistically superior mean Adjusted Rand Index (ARI) compared to:
> 1. The `icc_v2` (GAC) pipeline (`icc.consensus_v2`);
> 2. `cohirf` (Random Subspace K-Means + Ward consensus);
> 3. `gac_random_gating` (the GAC anchor pipeline with uniform random gating at equal $|\mathcal{F}_{\text{active}}| = \lfloor \alpha D \rfloor$).
>
> **Primary Contrast & Decision Rule:**  
> The primary contrast is evaluated via pairwise two-sided Wilcoxon signed-rank tests across the $n=15$ cohorts with Holm-Bonferroni correction at family-wise error rate $\alpha = 0.05$. The minimum effect of interest is pre-registered as:
> $$\Delta\text{ARI} = \text{ARI}_{\text{NullConsensus}} - \text{ARI}_{\text{Baseline}} \ge +0.05.$$
> Furthermore, the 95% dataset-clustered bootstrap confidence interval of the paired difference must strictly exclude zero. Standard consensus (`std_consensus`) and ambient `kmeans` are evaluated as secondary baselines.
>
> **Statistical Power Analysis ($n=15$):**  
> Based on empirical variance observed across DEV benchmarks, the standard deviation of paired differences across high-D cancer cohorts is bounded by $\sigma_{\Delta} \approx 0.060$. For a minimum detectable effect size of $\Delta = 0.050$, the standardized effect size is:
> $$d = \frac{\Delta}{\sigma_{\Delta}} = \frac{0.050}{0.060} \approx 0.833 \quad (\text{Cohen's } d \text{ large}).$$
> For a two-tailed paired test at $\alpha = 0.05$ with $n = 15$ cohorts, the achieved statistical power is:
> $$1 - \beta \approx 0.824 \quad (82.4\% \text{ power}),$$
> satisfying the standard $>80\%$ power requirement for confirmatory discovery without sample over-extension.

---

### Exploratory Hypothesis (H2: Count-Data Recovery via HurdleGate)
> **[EXPLORATORY]** On the 4 zero-inflated scRNA-seq count benchmarks, `NullConsensus` operating in hurdle mode (`mode="hurdle"`) outperforms marginal bimodality screening (`icc_v2`) by at least $+0.05$ mean ARI, demonstrating that isolating the positive support $\mathbf{x}^+$ prevents dropout-driven false bimodality.

---

### Exploratory Hypothesis (H3: Resilience to Stability-Accuracy Trap — Adaptation of M3C)
> **[EXPLORATORY ADAPTATION]** Under pure isotropic distractor noise ($D_{\text{noise}} \ge 500$ with zero true cluster separation), `NullConsensus` detects absence of cluster structure via non-parametric permutation-referenced stability ($\Delta\text{PAC} \le 0$), successfully flagging null partitions where classical consensus asserts falsely confident agreement ($\text{Stability} > 0.80$). This hypothesis represents an exploratory adaptation of M3C (John et al., 2020) to column permutations and bipartite anchor graphs.

---

### Confirmatory Hypothesis (H4: Subspace Stability without Seed Bimodality)
> Saliency-weighted soft subspace sampling ($p_d \propto \exp(z_d / \tau)$) significantly reduces cross-seed performance variance compared to uniform random subspace sampling on small-sample high-D cohorts.
>
> **Variance Metric & Formal Contrast:**  
> For each dataset $j \in \{1, \dots, n\}$, let $s_j^2(\text{NullConsensus})$ and $s_j^2(\text{UniformSubspace})$ denote the sample variance of ARI across the 30 independent seeds. Define the variance ratio:
> $$R_j = \frac{s_j^2(\text{NullConsensus})}{s_j^2(\text{UniformSubspace})}.$$
> The omnibus metric is the geometric mean variance ratio across cohorts:
> $$\overline{R} = \exp\left(\frac{1}{n}\sum_{j=1}^n \ln R_j\right).$$
> - **Hypothesis Test:** A one-sided Wilcoxon signed-rank test on paired log-variances $\ln s_j^2(\text{NullConsensus}) - \ln s_j^2(\text{UniformSubspace})$ tests $H_0: \text{median}(\ln R) \ge 0$ against $H_1: \text{median}(\ln R) < 0$ at $\alpha = 0.05$.
> - **Effect Size & CI:** A 10,000-iteration clustered bootstrap confidence interval is computed for $\overline{R}$. Falsification occurs unless the upper bound of the 95% bootstrap CI satisfies $\overline{R}_{\text{upper}} < 0.70$ (confirming at least a 30% reduction in seed-to-seed variance).

---

## 4. Evaluated Baselines & Frozen Defaults

All baselines are evaluated using standard scikit-learn compatible interfaces with published canonical hyperparameters:

| Method Code | Algorithmic Class | Reference / Implementation Source | Frozen Hyperparameters |
| :--- | :--- | :--- | :--- |
| `kmeans` | Classical Partitioning | Lloyd (1982), `sklearn.cluster.KMeans` | `n_init=10`, `random_state=seed` |
| `ward` | Hierarchical Agglomerative | Ward (1963), `sklearn.cluster.AgglomerativeClustering` | `linkage='ward'` |
| `spectral` | Graph Spectral | Ng, Jordan, Weiss (2001), `sklearn.cluster.SpectralClustering` | `affinity='nearest_neighbors'`, `n_neighbors=10` |
| `std_consensus` | Dense Cluster Ensemble | Monti et al. (2003), Fred & Jain (2005) | $M=25$ base K-Means, dense $N \times N$ average co-association |
| `core_halo` | Stability-Weighted Consensus | Li et al. (Artificial Intelligence 273, 2019) | $M=25, \tau_{\text{core}}=0.50$ |
| `fcag` | Anchor Graph Spectral | Nie et al. (IEEE TPAMI 46(4), 2024) | $L=40$ balanced anchors, $d=10$ nearest anchors |
| `sparse_kmeans` | Sparse Feature Weighting | Witten & Tibshirani (JRSS-B, 2010) | Heuristic $L_1$ bound tuning over 10 grid points |
| `cohirf` | Random Subspace K-Means | Belucci, Lounici, Meziani (arXiv:2502.00380, 2025) | $M=25, d_{\text{sub}}=\lfloor 0.15 D \rfloor$ random features |
| `m3c` | Monte Carlo Reference Consensus | John et al. (Sci Rep 10:1816, 2020), `nullconsensus.baselines.m3c` | $M=25, B=100$, Gaussian PAC null reference |
| `icc_v2` | GAC Inductive Pipeline (Paper 1) | Kundu et al. (2026), `icc.consensus_v2` | $M=25, L=40, \alpha=0.15, \tau_{\text{core}}=0.50$ (Frozen) |
| `gac_random_gating` | GAC with Uniform Random Gating | Control baseline, `nullconsensus.baselines.gac_random_gating` | $M=25, L=40, |\mathcal{F}_{\text{active}}|=\lfloor 0.15 D \rfloor$ (Random) |
| `nullconsensus` | **Proposed Method** | `nullconsensus.NullConsensus` | $M=25, L=40, \alpha=0.15, B_{\text{null}}\ge 100, \tau=1.0$ |

---

## 5. Seed Allocation & Experimental Battery

- **High-D Cohorts ($D/N > 20$):** 30 independent seeds per method ($\text{seed} \in [1000, 1029]$).
- **General / Multi-Class Cohorts ($D/N \le 20$):** 10 independent seeds per method ($\text{seed} \in [1000, 1009]$).
- **Independence:** All TEST seeds ($[1000, 1029]$) are strictly disjoint from Paper 1 / DEV seeds ($[0, 29]$).
- **Total Planned Model Fits on TEST:**  
  - 15 cancer cohorts ($D/N > 20$): $15 \times 12 \text{ methods} \times 30 \text{ seeds} = 5{,}400$ fits.
  - 12 remaining cohorts: $12 \times 12 \text{ methods} \times 10 \text{ seeds} = 1{,}440$ fits.
  - Total: **6,840 independent model evaluations**.

---

## 6. Execution Caps, Resource Boundaries & Kill Criteria

### Execution Caps
- **Time Cap:** Maximum 30 minutes per single model fit; maximum 6.0 hours per experiment battery.
- **Memory Cap:** Maximum 13.0 GB RAM per process (on 16 GB host).
- **Failure Accounting:** Any fit that crashes, exceeds memory limits, or times out is logged as a **failed run** (assigned $\text{ARI} = 0.0$ for ranking purposes). Failures are never discarded, re-run with relaxed constraints, or omitted from reporting.

### Explicit Kill Criteria & Bounded Iteration Limit
To prevent open-ended model chasing on DEV data:
1. **Performance Kill Criterion:** If on the DEV benchmark suite (Paper 1 datasets), `NullConsensus` fails to beat `gac_random_gating` by at least $\Delta\text{ARI} \ge +0.03$ or fails to beat `cohirf` on high-D cohorts, the hypothesis is falsified.
2. **Computational Kill Criterion:** If single-fit execution time on $N=2{,}000, D=15{,}000$ exceeds 10 minutes or host memory exceeds 12.0 GB RAM, the implementation is disqualified.
3. **Design Iteration Limit:** A strict limit of **maximum 5 design rounds** is permitted on the DEV suite. Each round must be logged in `DESIGN_LOG.md`. If Round 5 completes without satisfying Criteria 1 and 2, the pipeline is terminated immediately and the project transitions unconditionally to **Branch C (Benchmark & Diagnostic Paper)**.

---

## 7. Statistical Testing & Decision Framework

1. **Primary Omnibus Test:** Non-parametric Friedman rank test across all 27 TEST datasets with post-hoc two-tailed Nemenyi Critical Difference at $\alpha = 0.05$.
2. **Pairwise Comparisons:** Two-sided Wilcoxon signed-rank tests with Holm-Bonferroni multiple-comparison correction across all 11 competitor baselines.
3. **Effect Sizes & Uncertainty:** Vargha-Delaney $A_{12}$ non-parametric effect sizes and 10,000-iteration clustered bootstrap confidence intervals.
4. **Falsification & Fallback Decision Tree:**
   - **Branch A (Method Paper):** H1 is supported (paired Wilcoxon Holm $p < 0.05$, bootstrap CI strictly $> 0$, and Rank #1 on $D/N > 20$).
   - **Branch B (Regime Paper):** H1 fails globally, but H2 or H3 is supported with statistical significance in a clearly delineated regime (e.g., scRNA-seq count dropouts or pure noise screening).
   - **Branch C (Benchmark & Diagnostic Paper):** If neither H1 nor H2 yields a statistically significant advantage over the strongest baseline (`icc_v2` or `cohirf`), the paper transitions unconditionally to a benchmark and diagnostic paper reporting the null-stability diagnostic and failure taxonomy with equal scientific rigor.

---

## 8. Protocol Sign-Off & Hash Verification

- **Protocol Version:** 2.0.0
- **Pre-Registration Commit Tag:** `prereg-v2`
- **Integrity Guarantee:** Zero test-set evaluation runs will occur prior to this pre-registration sign-off and git tag `freeze-v1`.
