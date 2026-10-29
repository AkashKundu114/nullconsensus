"""
Dense cluster ensemble baselines for nullconsensus.
Methods:
- std_consensus: Monti et al. (2003) / Fred & Jain (2005) Dense Co-association Consensus
- core_halo: Li et al. (Artificial Intelligence 273, 2019) Core-Halo Consensus
"""

from typing import Any

import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans


class StandardConsensusBaseline(BaseEstimator, ClusterMixin):
    """
    Fred & Jain (IEEE TPAMI 2005) / Monti et al. (2003) Dense Consensus Ensemble.
    Computes dense N x N average co-association matrix across M base K-Means models,
    followed by average linkage hierarchical clustering.
    """

    def __init__(self, n_clusters: int = 3, n_base_models: int = 25, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_base_models = n_base_models
        self.random_state = random_state
        self.labels_: np.ndarray | None = None
        self.coassoc_matrix_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "StandardConsensusBaseline":
        X = np.asarray(X, dtype=float)
        N = len(X)
        k = self.n_clusters
        rng = np.random.default_rng(self.random_state)

        # Base partitions
        partitions = []
        for _ in range(self.n_base_models):
            seed_m = int(rng.integers(0, 1000000))
            km = KMeans(n_clusters=k, n_init=1, random_state=seed_m)
            partitions.append(km.fit_predict(X))

        # Co-association calculation via indicator outer products
        H = np.zeros((N, self.n_base_models * k), dtype=np.float32)
        for idx, lab in enumerate(partitions):
            valid = lab >= 0
            H[valid, idx * k + lab[valid]] = 1.0
        S = (H @ H.T).astype(float) / float(self.n_base_models)
        self.coassoc_matrix_ = S

        dist_mat = 1.0 - S
        np.fill_diagonal(dist_mat, 0.0)

        condensed_dist = dist_mat[np.triu_indices(N, k=1)]
        condensed_dist = np.clip(condensed_dist, 0.0, 1.0)
        Z = sch.linkage(condensed_dist, method="average")
        self.labels_ = sch.fcluster(Z, t=k, criterion="maxclust") - 1
        return self

    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_


class CoreHaloBaseline(BaseEstimator, ClusterMixin):
    """
    Li et al. (Artificial Intelligence 273, 2019) Core-Halo Consensus.
    Calculates sample-level stability from co-association, identifies certified core instances
    above threshold tau_core, clusters core via hierarchical linkage, and projects halo samples.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_base_models: int = 25,
        tau_core: float = 0.50,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.n_base_models = n_base_models
        self.tau_core = tau_core
        self.random_state = random_state
        self.labels_: np.ndarray | None = None
        self.core_indices_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "CoreHaloBaseline":
        X = np.asarray(X, dtype=float)
        N = len(X)
        k = self.n_clusters
        rng = np.random.default_rng(self.random_state)

        partitions = []
        for _ in range(self.n_base_models):
            seed_m = int(rng.integers(0, 1000000))
            km = KMeans(n_clusters=k, n_init=1, random_state=seed_m)
            partitions.append(km.fit_predict(X))

        H = np.zeros((N, self.n_base_models * k), dtype=np.float32)
        for idx, lab in enumerate(partitions):
            valid = lab >= 0
            H[valid, idx * k + lab[valid]] = 1.0
        S = (H @ H.T).astype(float) / float(self.n_base_models)

        # Sample stability
        np.fill_diagonal(S, 0.0)
        sample_stability = np.sum(S**2, axis=1) / max(1, N - 1)
        np.fill_diagonal(S, 1.0)

        tau_stab = float(np.quantile(sample_stability, self.tau_core))
        is_core = sample_stability >= tau_stab
        core_idx = np.where(is_core)[0]
        halo_idx = np.where(~is_core)[0]

        if len(core_idx) < k:
            core_idx = np.argsort(sample_stability)[-k:]
            halo_idx = np.setdiff1d(np.arange(N), core_idx)

        self.core_indices_ = core_idx

        # Cluster core
        S_core = S[np.ix_(core_idx, core_idx)]
        dist_core = 1.0 - S_core
        np.fill_diagonal(dist_core, 0.0)
        condensed_dist = dist_core[np.triu_indices(len(core_idx), k=1)]
        condensed_dist = np.clip(condensed_dist, 0.0, 1.0)

        Z = sch.linkage(condensed_dist, method="average")
        core_labels = sch.fcluster(Z, t=k, criterion="maxclust") - 1

        final_labels = np.zeros(N, dtype=int)
        final_labels[core_idx] = core_labels

        # Assign halo instances by maximum average affinity to core clusters
        if len(halo_idx) > 0:
            for h in halo_idx:
                affinities = []
                for c in range(k):
                    c_pts = core_idx[core_labels == c]
                    if len(c_pts) > 0:
                        affinities.append(float(np.mean(S[h, c_pts])))
                    else:
                        affinities.append(0.0)
                final_labels[h] = int(np.argmax(affinities))

        self.labels_ = final_labels
        return self

    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_
