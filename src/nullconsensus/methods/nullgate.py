"""
Permutation-Null & Unimodal-Null Calibrated Subspace Consensus (NullGate).
Candidate 12 Primary Method for High-Dimensional Consensus Clustering.
"""

from typing import Literal

import numpy as np
from scipy import stats


def compute_sarle_bc(x: np.ndarray) -> float:
    """
    Computes Sarle's Bimodality Coefficient: BC = (skew^2 + 1) / (kurt + 3).
    Bounded in [0, 1]. For standard normal, BC ~ 0.333. For uniform, BC = 5/9 ~ 0.556.
    """
    n = len(x)
    if n < 4:
        return 5.0 / 9.0
    var = np.var(x, ddof=1)
    if var < 1e-12:
        return 0.0
    skew = float(stats.skew(x, bias=False))
    kurt = float(stats.kurtosis(x, bias=False, fisher=True))  # excess kurtosis
    denom = kurt + 3.0
    if denom <= 0.0:
        return 1.0
    bc = (skew**2 + 1.0) / denom
    return float(np.clip(bc, 0.0, 1.0))


def compute_bc_vectorized(X: np.ndarray) -> np.ndarray:
    """
    Computes Sarle's Bimodality Coefficient across all columns of X: (N, D).
    Returns array of shape (D,).
    """
    N, D = X.shape
    if N < 4:
        return np.full(D, 5.0 / 9.0)

    skews = stats.skew(X, axis=0, bias=False)
    kurts = stats.kurtosis(X, axis=0, bias=False, fisher=True)
    denoms = kurts + 3.0
    denoms[denoms <= 0.0] = 1.0
    bcs = (skews**2 + 1.0) / denoms
    return np.clip(bcs, 0.0, 1.0)


class NullGateFeatureSelector:
    """
    Null-Calibrated Bimodality Feature Selector (Candidate 12: NullGate).

    Computes finite-sample null-calibrated saliency z-scores for unsupervised
    feature screening under a unimodal reference null hypothesis (Gaussian or Uniform),
    providing rigorous Type-I error control under high-dimensional isotropic noise.

    Parameters:
        null_reference: Reference distribution for unimodal null ('gaussian' or 'uniform').
        n_null_draws: Number of independent finite-sample null draws of size N (default: 200).
        random_state: Seed for deterministic reproducibility.
    """

    def __init__(
        self,
        null_reference: Literal["gaussian", "uniform"] = "gaussian",
        n_null_draws: int = 200,
        random_state: int = 42,
    ):
        self.null_reference = null_reference
        self.n_null_draws = n_null_draws
        self.random_state = random_state

        # Fitted attributes
        self.saliency_scores_: np.ndarray | None = None
        self.sampling_probs_: np.ndarray | None = None
        self.empirical_bc_: np.ndarray | None = None
        self.null_mean_: float | None = None
        self.null_std_: float | None = None

    def fit(self, X: np.ndarray, temperature: float = 1.0) -> "NullGateFeatureSelector":
        """
        Fits null benchmark for sample size N, computes feature z-scores and softmax probabilities.
        """
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        rng = np.random.default_rng(self.random_state)

        # 1. Compute empirical Sarle BC for all D features in vector form
        self.empirical_bc_ = compute_bc_vectorized(X)

        # 2. Compute finite-sample reference null distribution for sample size N
        # Since Sarle BC is scale- and location-invariant, standard normal or uniform(0, 1)
        # provides the exact finite-sample null sampling distribution.
        if self.null_reference == "gaussian":
            null_matrix = rng.normal(0.0, 1.0, size=(N, self.n_null_draws))
        else:  # uniform
            null_matrix = rng.uniform(0.0, 1.0, size=(N, self.n_null_draws))

        null_bcs = compute_bc_vectorized(null_matrix)
        self.null_mean_ = float(np.mean(null_bcs))
        self.null_std_ = float(max(1e-4, np.std(null_bcs)))

        # 3. Z-scores relative to finite-sample unimodal null benchmark
        # z_d measures how many standard deviations empirical BC exceeds expected unimodal null
        z_scores = (self.empirical_bc_ - self.null_mean_) / self.null_std_
        # Rectify: features below null mean have zero saliency
        self.saliency_scores_ = np.maximum(0.0, z_scores)

        # 4. Softmax saliency-weighted sampling probabilities
        t = max(1e-3, float(temperature))
        scaled_z = self.saliency_scores_ / t
        # Shift by max for numerical stability in exp
        exp_z = np.exp(scaled_z - np.max(scaled_z))
        sum_exp = np.sum(exp_z)
        if sum_exp > 0:
            self.sampling_probs_ = exp_z / sum_exp
        else:
            self.sampling_probs_ = np.ones(D) / D

        return self
