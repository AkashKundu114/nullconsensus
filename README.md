# nullconsensus

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/pytest-32%2F32%20passed-brightgreen.svg)]()
[![Pre-Registration: prereg-v2](https://img.shields.io/badge/preregistration-prereg--v2%20(frozen)-success.svg)](PREREGISTRATION.md)
[![Freeze Tag: freeze-v1](https://img.shields.io/badge/git%20tag-freeze--v1-blue.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![XERJ Corpus Hub contributor](https://hub.xerj.org/contributors/AkashKundu114/badge.svg)](https://hub.xerj.org/contributors/AkashKundu114)

**Permutation-Null Calibrated Consensus Clustering for High-Dimensional Manifolds.**

Official reference implementation of the manuscript:  
**"Empirical Boundaries of Feature-Gated Consensus Clustering: Permutation-Null Calibration and Hurdle Recovery on High-Dimensional Manifolds"** (*Pattern Recognition*, Elsevier, Under Preparation).

---

## Confirmatory Benchmark Highlights

Following an unalterable pre-registration protocol ([`PREREGISTRATION.md`](PREREGISTRATION.md), commit tag `freeze-v1`, Rule R2), we evaluated 12 universal clustering pipelines across **27 sealed benchmark datasets** spanning **6,880 independent model fits**:

| Evaluation Regime | Pre-Registered Hypothesis | Empirical Outcome | Statistical Contrast | Scientific Conclusion |
| :--- | :--- | :---: | :--- | :--- |
| **High-D Oncology Microarrays** ($D/N > 20$, 15 cohorts) | **H1:** Null screening universally beats random subspaces | **FALSIFIED** | $\Delta\text{ARI} = -0.0433, p_{\text{holm}} = 0.2293$ vs random gating | Uniform random gating matches or outperforms null screening when $N < 100$ as permutation nulls act over-conservatively. |
| **Zero-Inflated scRNA-seq** (4 cohorts, 50%–85% dropouts) | **H2:** Two-part HurdleGate prevents dropout collapse | **SUPPORTED** (Branch B) | $\Delta\text{ARI} = \mathbf{+0.3832}, p = 3.57 \times 10^{-8}$ vs Paper 1 GAC | Standard gating collapses catastrophically ($\text{ARI} \approx 0.04$); HurdleGate decouples Bernoulli dropouts from positive multimodality. |
| **Cross-Seed Subspace Variance** | **H4:** Saliency weights reduce cross-seed variance by $\ge 30\%$ | **FALSIFIED** | $\overline{R} = 0.6640, 95\%\text{ CI: } [0.2419, 2.0056]$ | Upper bootstrap CI exceeds $0.70$ boundary. |
| **Omnibus 27-Dataset Benchmark** | Comprehensive Rank Divergence | **SIGNIFICANT** | $\chi^2 = 52.39, p = 2.31 \times 10^{-7}$ (Friedman test) | NullConsensus beats Paper 1 GAC ($\Delta = +0.0542, p = 0.0112$) and CoHiRF ($\Delta = +0.0689$). |

For complete reproduction instructions and cryptographic checksum verification, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Overview
Consensus clustering often suffers from the **stability-accuracy trap**: under high-dimensional isotropic distractor noise ($D \gg N$), random base partitions converge stably on arbitrary noise geometries, producing falsely confident partitions with zero Adjusted Rand Index.

`nullconsensus` resolves this fundamental challenge through three coupled mechanisms:
1. **Permutation-Null Feature Saliency:** Tests empirical bimodality against an independent column-permuted reference null dataset to filter noise dimensions with finite-sample Type-I error control.
2. **Count-Aware Hurdle Gating (`HurdleGate`):** Decouples Bernoulli dropout zero point-masses from continuous positive-support multimodality, completely preventing the zero-inflation collapse in single-cell transcriptomics.
3. **Bipartite Landmark Consensus:** Employs an efficient landmark Transfer-Cut spectral embedding over $L \ll N$ anchors, eliminating $\mathcal{O}(N^2)$ memory and enabling linear $\mathcal{O}(NL)$ scaling.

---

## Installation
```bash
pip install nullconsensus
# Or editable developer install:
git clone https://github.com/nullconsensus/nullconsensus.git
cd nullconsensus
pip install -e ".[dev]"
```

---

## 10-Line Quickstart
```python
import numpy as np
from nullconsensus import NullConsensus

# 1. Generate noisy high-dimensional data (N=150, D=500, K=3)
rng = np.random.default_rng(42)
X = rng.standard_normal((150, 500))
X[:50, :15] += 3.0
X[50:100, :15] -= 3.0

# 2. Fit estimator with permutation-null calibration
model = NullConsensus(n_clusters=3, n_base_models=25, n_null_permutations=10, random_state=42)
labels = model.fit_predict(X)

# 3. Predict out-of-sample query instances
X_query = rng.standard_normal((5, 500))
X_query[:, :15] += 3.0
print("Cluster assignments:", model.predict(X_query).tolist())
```

### Zero-Inflated Count Data (scRNA-seq)
```python
# Enable HurdleGate for zero-inflated counts
model_hurdle = NullConsensus(n_clusters=3, mode="hurdle", random_state=42)
labels_hurdle = model_hurdle.fit_predict(X_counts)
```

---

## Pre-Registration & Research Protocol

This project strictly adheres to a frozen pre-registration protocol:
- **Pre-Registration Document:** [`PREREGISTRATION.md`](PREREGISTRATION.md) (frozen at tag `prereg-v2`).
- **Prior-Art & Novelty Audit:** [`docs/NOVELTY.md`](docs/NOVELTY.md) (structural contrast vs M3C, dropout/hurdle methods, and co-association baselines).
- **Sealed Test Set Manifest:** [`data/test_manifest.json`](data/test_manifest.json) containing 27 benchmark cohorts with verified licenses and checksums.
- **Access Guard:** Enforced via `nullconsensus.data.guard.verify_test_access_allowed`, preventing test data download or access prior to `freeze-v1` tag.

---

## Testing

```bash
pytest tests/ -v
```

---

## Submission Package & Paper Compilation

The manuscript and complete submission package configured for *Pattern Recognition* (Elsevier) are located in:
- `paper/main.tex`: Formatted Elsevier `elsarticle` LaTeX source.
- `paper/figures/`: High-resolution vector PDF and PNG figures (`fig1`–`fig5`).
- `paper/tables/`: Single-source-of-truth LaTeX tables (`table1`–`table5`).
- `submission/`: Complete submission bundle (`main.pdf`, `source.zip`, `COVER_LETTER.md`, `HIGHLIGHTS.md`, `CHECKLIST.md`).

---

## License
MIT License. Developed by Akash Kundu (Techno India University).
