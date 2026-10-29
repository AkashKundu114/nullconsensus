"""
Consensus Hierarchical Random Features (CoHiRF) clustering baseline.
Reference:
    Belucci, Lounici, Meziani, Kostic.
    "CoHiRF: Consensus Hierarchical Random Forests for High-Dimensional Clustering."
    arXiv:2502.00380, 2025.
"""

from typing import Any

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans


class CoHiRFBaseline(BaseEstimator, ClusterMixin):
    """
    Consensus Hierarchical Random Features (CoHiRF) clustering model.
    Samples random feature subspaces of size d_sub = floor(alpha * D), generates
    multiple base K-Means clusterings, and hierarchically merges samples with consensus >= threshold.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        subspace_fraction: float = 0.15,
        n_base_models: int = 25,
        consensus_threshold: float = 0.80,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.subspace_fraction = subspace_fraction
        self.n_base_models = n_base_models
        self.consensus_threshold = consensus_threshold
        self.random_state = random_state
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "CoHiRFBaseline":
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        k = self.n_clusters
        d_sub = max(2, min(D, int(np.floor(self.subspace_fraction * D))))
        rng = np.random.default_rng(self.random_state)

        # 1. Base clustering in random subspaces
        partitions = []
        for _ in range(self.n_base_models):
            sub_feats = rng.choice(D, size=d_sub, replace=False)
            seed_m = int(rng.integers(0, 1000000))
            km = KMeans(n_clusters=k, n_init=1, random_state=seed_m)
            partitions.append(km.fit_predict(X[:, sub_feats]))

        # 2. Pairwise co-association matrix
        H = np.zeros((N, self.n_base_models * k), dtype=np.float32)
        for idx, lab in enumerate(partitions):
            valid = lab >= 0
            H[valid, idx * k + lab[valid]] = 1.0
        S = (H @ H.T).astype(float) / float(self.n_base_models)

        # 3. High-confidence consensus graph
        adj = (S >= self.consensus_threshold).astype(int)
        n_comps, comp_labels = connected_components(
            adj, directed=False, return_labels=True
        )

        if n_comps <= k:
            # If contracted below or equal to k, assign remaining via Ward on original X
            if n_comps == k:
                self.labels_ = comp_labels
                return self
            # Fall back to Ward on co-association distance
            dist_mat = np.clip(1.0 - S, 0.0, 1.0)
            np.fill_diagonal(dist_mat, 0.0)
            ward = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage="average"
            )
            self.labels_ = ward.fit_predict(dist_mat)
            return self

        # Contraction: compute centroid of each component in original space
        comp_centers = np.zeros((n_comps, D), dtype=float)
        for c in range(n_comps):
            pts = X[comp_labels == c]
            comp_centers[c] = pts.mean(axis=0)

        # Ward clustering on component centers
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward")
        comp_final_labels = ward.fit_predict(comp_centers)

        # Map back to samples
        self.labels_ = comp_final_labels[comp_labels]
        return self

    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_
