"""
Classical single-model clustering baselines for nullconsensus.
Methods:
- kmeans: Lloyd (1982), KMeans with n_init=10
- ward: Ward (1963), AgglomerativeClustering with Ward linkage
- spectral: Ng, Jordan, Weiss (2001), SpectralClustering with nearest_neighbors
- sparse_kmeans: Witten & Tibshirani (JRSS-B, 2010), Sparse K-Means
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering


class KMeansBaseline(BaseEstimator, ClusterMixin):
    """K-Means clustering baseline with 10 initializations."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "KMeansBaseline":
        X = np.asarray(X, dtype=float)
        km = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.random_state
        )
        self.labels_ = km.fit_predict(X)
        return self

    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_


class WardBaseline(BaseEstimator, ClusterMixin):
    """Ward hierarchical agglomerative clustering baseline."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "WardBaseline":
        X = np.asarray(X, dtype=float)
        # Add tiny deterministic jitter to break ties consistently across platforms
        rng = np.random.default_rng(self.random_state)
        jitter = rng.normal(scale=1e-7, size=X.shape)
        ward = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="ward")
        self.labels_ = ward.fit_predict(X + jitter)
        return self

    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_


class SpectralBaseline(BaseEstimator, ClusterMixin):
    """Spectral clustering baseline with k-nearest-neighbors affinity graph."""

    def __init__(
        self, n_clusters: int = 3, n_neighbors: int = 10, random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "SpectralBaseline":
        X = np.asarray(X, dtype=float)
        N = len(X)
        k_neighbors = min(self.n_neighbors, max(2, N - 1))
        sc = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=k_neighbors,
            random_state=self.random_state,
            assign_labels="kmeans",
        )
        self.labels_ = sc.fit_predict(X)
        return self

    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_


class SparseKMeansBaseline(BaseEstimator, ClusterMixin):
    """
    Sparse K-Means (Witten & Tibshirani, JRSS-B 2010).
    Optimizes L1-penalized feature weights for unsupervised sparse clustering.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        s: float | None = None,
        max_iter: int = 10,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.s = s
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "SparseKMeansBaseline":
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        k = self.n_clusters

        # Bound tuning parameter s in [1.1, sqrt(D)]
        s_val = self.s
        if s_val is None:
            s_val = 1.1 + 0.5 * (np.sqrt(D) - 1.1)
        s_val = max(1.1, min(s_val, np.sqrt(D) * 0.9))

        labels = KMeans(
            n_clusters=k, n_init=3, random_state=self.random_state
        ).fit_predict(X)

        for _ in range(self.max_iter):
            a = np.zeros(D)
            for c in range(k):
                members = X[labels == c]
                if len(members) > 0:
                    mean_c = members.mean(axis=0)
                    a += len(members) * (mean_c**2)

            if np.all(a <= 1e-12):
                break

            # Binary search for soft-threshold delta
            low, high = 0.0, float(np.max(a))
            best_delta = 0.0
            for _ in range(25):
                delta = (low + high) / 2.0
                soft = np.maximum(0.0, a - delta)
                norm2 = float(np.linalg.norm(soft))
                if norm2 < 1e-12:
                    high = delta
                    continue
                l1_over_l2 = float(np.sum(soft) / norm2)
                if l1_over_l2 > s_val:
                    low = delta
                else:
                    high = delta
                    best_delta = delta

            soft = np.maximum(0.0, a - best_delta)
            norm2 = float(np.linalg.norm(soft))
            w = soft / (norm2 + 1e-12)

            X_w = X * np.sqrt(w)
            new_labels = KMeans(
                n_clusters=k, n_init=1, random_state=self.random_state
            ).fit_predict(X_w)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

        self.labels_ = labels
        self.weights_ = w
        return self

    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_
