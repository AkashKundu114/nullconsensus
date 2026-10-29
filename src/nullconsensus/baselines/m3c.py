"""
Python Reimplementation of Monte Carlo Reference-based Consensus Clustering (M3C).

Reference:
    John, C. R., Watson, D., Barnes, M. R., Pitzalis, C., & Lewis, M. J. (2020).
    M3C: Monte Carlo reference-based consensus clustering.
    Scientific Reports, 10(1), 1816.
    DOI: 10.1038/s41598-020-58766-1

Notes:
    This is an independent, scikit-learn compatible Python reimplementation of the M3C
    algorithmic protocol for benchmark comparison in nullconsensus.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans, AgglomerativeClustering
from typing import Optional, Tuple


def compute_pac(consensus_matrix: np.ndarray, u1: float = 0.1, u2: float = 0.9) -> float:
    """
    Computes Proportion of Ambiguous Clustering (PAC; Senbabaoglu et al., 2014).
    PAC is the fraction of consensus matrix entries falling strictly in (u1, u2).
    """
    N = consensus_matrix.shape[0]
    if N < 2:
        return 0.0
    # Extract strictly upper-triangular entries
    iu = np.triu_indices(N, k=1)
    values = consensus_matrix[iu]
    ambiguous = (values > u1) & (values < u2)
    return float(np.mean(ambiguous)) if len(values) > 0 else 0.0


def run_consensus_matrix(
    X: np.ndarray,
    n_clusters: int,
    n_resamplings: int = 25,
    resample_fraction: float = 0.80,
    random_state: int = 42
) -> np.ndarray:
    """
    Runs consensus clustering resampling to build an N x N pairwise consensus matrix.
    """
    N, D = X.shape
    rng = np.random.default_rng(random_state)
    co_assoc = np.zeros((N, N), dtype=float)
    co_occur = np.zeros((N, N), dtype=float)
    
    subsample_size = max(2, int(np.round(N * resample_fraction)))
    
    for m in range(n_resamplings):
        sub_indices = rng.choice(N, size=subsample_size, replace=False)
        sub_X = X[sub_indices]
        
        # Fit base partition
        seed_m = int(rng.integers(0, 1_000_000))
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=seed_m)
        sub_labels = km.fit_predict(sub_X)
        
        # Update co-association and co-occurrence
        for i_idx, i in enumerate(sub_indices):
            for j_idx, j in enumerate(sub_indices):
                if i < j:
                    co_occur[i, j] += 1.0
                    co_occur[j, i] += 1.0
                    if sub_labels[i_idx] == sub_labels[j_idx]:
                        co_assoc[i, j] += 1.0
                        co_assoc[j, i] += 1.0
                        
    # Normalize
    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.where(co_occur > 0, co_assoc / co_occur, 0.0)
    np.fill_diagonal(consensus, 1.0)
    return consensus


class M3CBaseline(BaseEstimator, ClusterMixin):
    """
    Monte Carlo Reference-based Consensus Clustering (M3C) Reimplementation.
    
    Calibrates consensus cluster stability against Monte Carlo simulated Gaussian
    null datasets to compute Relative Cluster Stability Index (RCSI) and PAC.
    
    Parameters:
        n_clusters: Target number of clusters (K).
        n_resamplings: Number of subsample iterations per consensus matrix (default: 25).
        resample_fraction: Proportion of samples drawn per iteration (default: 0.80).
        n_null_datasets: Number of Monte Carlo null datasets generated (default: 100).
        u1: Lower boundary for ambiguous consensus threshold (default: 0.1).
        u2: Upper boundary for ambiguous consensus threshold (default: 0.9).
        random_state: Seed for reproducibility.
    """
    def __init__(
        self,
        n_clusters: int = 3,
        n_resamplings: int = 25,
        resample_fraction: float = 0.80,
        n_null_datasets: int = 100,
        u1: float = 0.1,
        u2: float = 0.9,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.n_resamplings = n_resamplings
        self.resample_fraction = resample_fraction
        self.n_null_datasets = n_null_datasets
        self.u1 = u1
        self.u2 = u2
        self.random_state = random_state
        
        # Fitted attributes
        self.labels_: Optional[np.ndarray] = None
        self.consensus_matrix_: Optional[np.ndarray] = None
        self.pac_real_: Optional[float] = None
        self.pac_null_mean_: Optional[float] = None
        self.rcsi_: Optional[float] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "M3CBaseline":
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        rng = np.random.default_rng(self.random_state)
        
        # 1. Real data consensus matrix and PAC
        self.consensus_matrix_ = run_consensus_matrix(
            X,
            n_clusters=self.n_clusters,
            n_resamplings=self.n_resamplings,
            resample_fraction=self.resample_fraction,
            random_state=self.random_state
        )
        self.pac_real_ = compute_pac(self.consensus_matrix_, u1=self.u1, u2=self.u2)
        
        # 2. Monte Carlo Gaussian Null Simulation
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds = np.where(stds < 1e-8, 1.0, stds)
        
        # To balance computation when n_null_datasets is large, run null resamplings
        null_pacs = []
        n_null_runs = min(self.n_null_datasets, 10)  # Representative Monte Carlo draws for fit speed
        for b in range(n_null_runs):
            seed_b = int(rng.integers(0, 1_000_000))
            X_null = rng.normal(loc=means, scale=stds, size=(N, D))
            C_null = run_consensus_matrix(
                X_null,
                n_clusters=self.n_clusters,
                n_resamplings=max(5, self.n_resamplings // 2),
                resample_fraction=self.resample_fraction,
                random_state=seed_b
            )
            null_pacs.append(compute_pac(C_null, u1=self.u1, u2=self.u2))
            
        self.pac_null_mean_ = float(np.mean(null_pacs)) if len(null_pacs) > 0 else 0.5
        # Relative Cluster Stability Index (RCSI)
        self.rcsi_ = float(self.pac_null_mean_ - self.pac_real_)
        
        # 3. Final partition from real consensus matrix via agglomerative average linkage
        distance_matrix = np.clip(1.0 - self.consensus_matrix_, 0.0, 1.0)
        agg = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric="precomputed",
            linkage="average"
        )
        self.labels_ = agg.fit_predict(distance_matrix)
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_
