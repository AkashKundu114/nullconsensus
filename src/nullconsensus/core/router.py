"""
Automated Manifold Diagnostics and Regime Routing Engine for NullConsensus.
Analyzes input data geometry, sparsity patterns, and intrinsic dimensionality
to automatically select the optimal feature gating and landmark anchor strategy.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp


@dataclass
class ManifoldProfile:
    """Diagnostic profile of an input data matrix."""
    n_samples: int
    n_features: int
    d_over_n: float
    zero_fraction: float
    intrinsic_dim: float
    recommended_mode: Literal["hurdlegate", "uniform", "nullgate"]
    recommended_anchor_method: Literal["dpp", "kmeans", "random"]
    diagnostic_summary: str


def estimate_intrinsic_dim(X: np.ndarray, n_subsample: int = 500, random_state: int = 42) -> float:
    """
    Estimates intrinsic manifold dimensionality using the Two-Nearest-Neighbors (TwoNN) ratio estimator
    (Facco et al., Scientific Reports 2017). Operates in O(N_sub * D) time.
    """
    N = X.shape[0]
    if N < 10:
        return float(min(X.shape[1], 2))
    
    rng = np.random.default_rng(random_state)
    sub_size = min(N, n_subsample)
    idx = rng.choice(N, size=sub_size, replace=False)
    X_sub = X[idx]

    # Compute pairwise Euclidean distances for subsample
    diff = X_sub[:, None, :] - X_sub[None, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist_mat, np.inf)

    # Sort distances to get r1 (first nearest neighbor) and r2 (second nearest neighbor)
    sorted_dists = np.sort(dist_mat, axis=1)
    r1 = sorted_dists[:, 0]
    r2 = sorted_dists[:, 1]

    # Filter out identical duplicate instances
    valid = (r1 > 1e-10) & (r2 > r1)
    if np.sum(valid) < 5:
        return float(min(X.shape[1], 5))

    mu = r2[valid] / r1[valid]
    # TwoNN maximum likelihood estimator
    d_hat = float(np.sum(valid) / np.sum(np.log(mu)))
    return max(1.0, min(float(X.shape[1]), d_hat))


def profile_manifold(
    X: np.ndarray | sp.spmatrix,
    n_clusters: int = 3,
    random_state: int = 42,
) -> ManifoldProfile:
    """
    Performs comprehensive diagnostic inspection of input matrix geometry and routes
    to the optimal feature gating regime.

    Decision Rules:
      1. Zero-Inflated Count Regime (Dropouts > 30%):
         Standard bimodality screening collapses due to zero point-mass inflation.
         -> Route to 'hurdlegate' (Two-Part Hurdle Gate).
      2. Continuous High-Dimensional Small-N Regime (D/N > 10, N < 150, Zeros <= 30%):
         Unimodal permutation nulls behave over-conservatively on subtle multigenic shifts.
         -> Route to 'uniform' (Uniform Random Subspaces), per empirical H1 findings.
      3. Standard Continuous / Moderate-D Regime:
         -> Route to 'nullgate' (Permutation-null calibrated soft sampling).
      
      Anchor Rule:
         If K >= 8 or N >= 2,000:
         -> Deploy 'dpp' (k-Determinantal Point Process) to preserve rare sub-populations.
         Else:
         -> Deploy 'kmeans' (Balanced Landmark Seeding).
    """
    if sp.issparse(X):
        N, D = X.shape
        zero_fraction = float(1.0 - (X.nnz / (N * D)))
        # Subsample for intrinsic dimension
        sub_size = min(N, 500)
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(N, size=sub_size, replace=False)
        X_dense_sample = X[sample_idx].toarray()
    else:
        X_arr = np.asarray(X, dtype=float)
        N, D = X_arr.shape
        zero_fraction = float(np.mean(X_arr == 0.0))
        sub_size = min(N, 500)
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(N, size=sub_size, replace=False)
        X_dense_sample = X_arr[sample_idx]

    d_over_n = float(D / max(1, N))
    intrinsic_dim = estimate_intrinsic_dim(X_dense_sample, n_subsample=sub_size, random_state=random_state)

    # 1. Gating Mode Routing
    if zero_fraction >= 0.30:
        recommended_mode = "hurdlegate"
        rationale = (
            f"High zero-inflation detected (zeros={zero_fraction:.1%}). "
            "Deploying Two-Part HurdleGate to decouple dropout frequency from continuous multimodality."
        )
    elif d_over_n > 10.0 and N < 150:
        recommended_mode = "uniform"
        rationale = (
            f"Severe small-N high-D continuous regime detected (D/N={d_over_n:.1f}, N={N}). "
            "Deploying Uniform Random Subspaces to retain diffuse multigenic manifold signals without null over-pruning."
        )
    else:
        recommended_mode = "nullgate"
        rationale = (
            f"Moderate continuous geometry (zeros={zero_fraction:.1%}, D/N={d_over_n:.2f}). "
            "Deploying Permutation-Null Calibrated Gating (NullGate)."
        )

    # 2. Landmark Anchor Seeding Method
    if n_clusters >= 8 or N >= 2000:
        recommended_anchor = "dpp"
        anchor_rationale = "High complexity/scale detected; deploying k-DPP diversity anchor seeding."
    else:
        recommended_anchor = "kmeans"
        anchor_rationale = "Standard scale; deploying Balanced K-Means anchor seeding."

    summary = f"{rationale} | {anchor_rationale} [d_int={intrinsic_dim:.1f}]"

    return ManifoldProfile(
        n_samples=N,
        n_features=D,
        d_over_n=d_over_n,
        zero_fraction=zero_fraction,
        intrinsic_dim=intrinsic_dim,
        recommended_mode=recommended_mode,
        recommended_anchor_method=recommended_anchor,
        diagnostic_summary=summary,
    )
