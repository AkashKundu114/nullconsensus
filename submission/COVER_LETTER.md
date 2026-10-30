# Cover Letter for Manuscript Submission to *Pattern Recognition* (Elsevier)

**Date:** October 30, 2026  
**To:** The Editor-in-Chief, *Pattern Recognition* (Elsevier)  
**From:** Akash Kundu  
Department of Computer Science and Engineering  
Techno India University, Kolkata, West Bengal 700091, India  
Email: `akashkundu1152@gmail.com` / `akash.kundu@technoindiaeducation.com`  

---

**Dear Editor-in-Chief,**

I am pleased to submit our original research manuscript entitled:

> **"Empirical Boundaries of Feature-Gated Consensus Clustering: Permutation-Null Calibration and Hurdle Recovery on High-Dimensional Manifolds"**

for consideration as a Regular Paper in ***Pattern Recognition***.

### Background & Research Scope
In unsupervised pattern recognition on high-dimensional biological and physical manifolds where features far outnumber observations ($D \gg N$), clustering algorithms are severely compromised by the curse of dimensionality, uninformative distractor dimensions, and technical artifacts such as dropout zeros. While heuristic feature gating and consensus ensembles are widely employed, their rigorous statistical necessity, empirical boundaries, and failure modes have remained poorly understood.

### Confirmatory Protocol & Core Contributions
To eliminate publication bias, selective reporting, and post-hoc rationalization, this study was executed under an unalterable, cryptographically enforced pre-registration protocol (Protocol v2.0.0, commit tag `freeze-v1`, Rule R2). We evaluated **Permutation-Null Calibrated Consensus Clustering (`NullConsensus`)** and **Two-Part Hurdle Gating (`HurdleGate`)** against 11 established and contemporary baselines across 27 sealed benchmark cohorts spanning **6,880 independent model fits**:

1. **Empirical Falsification on Continuous Microarrays (Hypothesis H1):** On 15 high-dimensional oncology microarrays ($D/N > 20$), our pre-registered hypothesis that statistical null gating universally surpasses matched uniform random subspaces was **falsified** ($\Delta\text{ARI} = -0.0433, p_{\text{holm}} = 0.2293$). Matched uniform random gating achieved Rank #3 (Mean ARI = 0.255) while null screening achieved Rank #8 (Mean ARI = 0.211), demonstrating that unimodal permutation nulls behave over-conservatively on small sample sizes ($N < 100$).
2. **Discovery & Resolution of Technical Dropout Collapse (Hypothesis H2):** On zero-inflated single-cell RNA-sequencing manifolds, conventional multimodality screening collapses catastrophically ($\text{ARI} \approx 0.04$) because technical dropout point masses mimic extreme bimodality. By decoupling Bernoulli dropout entropy from continuous positive support multimodality, `HurdleGate` achieves an extraordinary recovery of **$\Delta\text{ARI} = +0.3832$ ($p = 3.57 \times 10^{-8}$)** over Paper 1's GAC pipeline, validating Hypothesis H2 and triggering pre-registered Branch B.
3. **Omnibus Superiority & Regime Taxonomy:** Across the full 27-cohort benchmark, `NullConsensus` demonstrates statistically significant superiority over Paper 1's GAC pipeline ($\Delta\text{ARI} = +0.0542, p = 0.0112$). We synthesize these empirical boundaries into a formal decision taxonomy guiding practitioners on when to deploy random subspaces versus hurdle-corrected gating.

### Declarations & Compliance
- **Originality:** This manuscript represents original work that has not been published previously and is not under consideration for publication elsewhere.
- **Reproducibility & Open Science:** All raw evaluation CSV logs, 27 standardized benchmark datasets with SHA256 checksums, automated table/figure generation pipelines, and source code are provided under the MIT License at `https://github.com/nullconsensus/nullconsensus`.
- **Formatting:** The manuscript is formatted according to Elsevier's official `elsarticle` guidelines, including structured Highlights, CRediT Authorship Statement, and standard declarations.

Thank you for your time, consideration, and management of the peer-review process.

Sincerely,

**Akash Kundu**  
Department of Computer Science and Engineering  
Techno India University, Kolkata, India  
