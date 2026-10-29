"""
Anchor-based clustering baselines for nullconsensus.
Method:
- fcag: Nie, Xue, Yu, Li (IEEE TPAMI 46(4), 2024), Fast Clustering with Anchor Guidance
"""

from typing import Any

import numpy as np
import scipy.linalg as sla
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans


class FCAGBaseline(BaseEstimator, ClusterMixin):
    """
    Fast Clustering with Anchor Guidance (FCAG, Nie et al., IEEE TPAMI 2024).
    Selects balanced anchors using KMeans++, constructs an adaptive nearest-anchor
    bipartite similarity graph, and solves bipartite spectral clustering in linear time.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_anchors: int = 40,
        n_neighbors: int = 10,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.n_anchors = n_anchors
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.labels_: np.ndarray | None = None
        self.anchors_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "FCAGBaseline":
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        k = self.n_clusters
        m = min(self.n_anchors, N)
        s = min(self.n_neighbors, m - 1)
        rng = np.random.default_rng(self.random_state)

        # 1. Anchor selection via KMeans++
        first_idx = int(rng.integers(0, N))
        anchor_indices = [first_idx]
        centers = [X[first_idx]]
        for _ in range(1, m):
            dists = np.min(
                cdist(X, np.array(centers), metric="sqeuclidean"), axis=1
            )
            d_sum = float(dists.sum())
            if d_sum < 1e-12:
                probs = np.ones(N) / N
            else:
                probs = dists / d_sum
            next_idx = int(rng.choice(N, p=probs))
            anchor_indices.append(next_idx)
            centers.append(X[next_idx])

        anchors = np.array(centers)
        self.anchors_ = anchors

        # 2. Construct adaptive sample-to-anchor bipartite similarity matrix Z: (N, m)
        dists_to_anchors = cdist(X, anchors, metric="sqeuclidean")
        Z = np.zeros((N, m), dtype=float)

        for i in range(N):
            row_dists = dists_to_anchors[i]
            sorted_idx = np.argsort(row_dists)
            knn_idx = sorted_idx[: s + 1]
            d_sp1 = row_dists[knn_idx[s]]
            denom = s * d_sp1 - np.sum(row_dists[knn_idx[:s]])
            if denom > 1e-12:
                weights = (d_sp1 - row_dists[knn_idx[:s]]) / denom
                weights = np.maximum(0.0, weights)
                w_sum = np.sum(weights)
                if w_sum > 0:
                    Z[i, knn_idx[:s]] = weights / w_sum
                else:
                    Z[i, knn_idx[0]] = 1.0
            else:
                Z[i, knn_idx[0]] = 1.0

        # 3. Spectral analysis on bipartite graph: S = D_x^(-1/2) * Z * D_y^(-1/2)
        dx = np.sum(Z, axis=1)
        dy = np.sum(Z, axis=0)
        dx[dx < 1e-12] = 1e-12
        dy[dy < 1e-12] = 1e-12

        Dx_inv_sqrt = 1.0 / np.sqrt(dx)
        Dy_inv_sqrt = 1.0 / np.sqrt(dy)

        # M_mat = Dy^(-1/2) * Z^T * Dx^(-1) * Z * Dy^(-1/2): (m, m)
        Zx = Z * Dx_inv_sqrt[:, None]
        M_mat = (Zx.T @ Zx) * (Dy_inv_sqrt[:, None] * Dy_inv_sqrt[None, :])

        try:
            evals, evecs = sla.eigh(M_mat)
            k_eff = min(k, m)
            V = evecs[:, -k_eff:]  # top eigenvectors
            # Project samples: F = Dx^(-1/2) * Z * Dy^(-1/2) * V
            F = (Zx * Dy_inv_sqrt[None, :]) @ V
            norms = np.linalg.norm(F, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1e-12
            F_norm = F / norms
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            self.labels_ = km.fit_predict(F_norm)
        except Exception:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            self.labels_ = km.fit_predict(Z)

        return self

    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_
