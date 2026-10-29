"""
Geometry-Calibrated Anchor Consensus (GAC / ICC-v2) Pipeline Baseline.
Reference:
    Kundu et al. (2026), "When Do Feature Gating and Anchor Consensus Help High-Dimensional Clustering?"
    Springer Nature (DAMI).
"""

from typing import Any

import numpy as np
import scipy.linalg as sla
from scipy import stats
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans, SpectralClustering


def _compute_sarle_bc(X: np.ndarray) -> np.ndarray:
    """Computes Sarle's Bimodality Coefficient for columns of X."""
    N, D = X.shape
    if N < 4:
        return np.full(D, 5.0 / 9.0)
    skews = stats.skew(X, axis=0, bias=False)
    kurts = stats.kurtosis(X, axis=0, bias=False, fisher=True)
    denoms = kurts + 3.0
    denoms[denoms <= 0.0] = 1.0
    bcs = (skews**2 + 1.0) / denoms
    return np.clip(bcs, 0.0, 1.0)


class GACBaseline(BaseEstimator, ClusterMixin):
    """
    GAC (Geometry-Calibrated Anchor Consensus / ICC-v2) Pipeline.
    Frozen Paper 1 pipeline featuring:
      1. Bimodality coefficient feature screening (top alpha * D features).
      2. Random subspace draws across candidate pool.
      3. Invariant core anchor selection (tau_core = 0.50).
      4. Bipartite Transfer-Cut spectral embedding.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_base_models: int = 25,
        n_anchors: int = 40,
        subspace_fraction: float = 0.15,
        tau_core: float = 0.50,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.n_base_models = n_base_models
        self.n_anchors = n_anchors
        self.subspace_fraction = subspace_fraction
        self.tau_core = tau_core
        self.random_state = random_state
        self.labels_: np.ndarray | None = None
        self.active_features_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "GACBaseline":
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        k = self.n_clusters
        rng = np.random.default_rng(self.random_state)

        # 1. Feature Screening via Sarle BC
        bcs = _compute_sarle_bc(X)
        n_pool = max(2, int(np.ceil(self.subspace_fraction * D)))
        # Floor of 100 or min(D, n_pool)
        n_active = min(D, max(100, n_pool)) if D >= 100 else D
        active_idx = np.argsort(-bcs)[:n_active]
        self.active_features_ = active_idx

        # 2. Base Partitions
        sub_size = max(2, int(len(active_idx) * 0.30))
        base_engine = "spectral" if N <= 100 else "kmeans"

        partitions = []
        for _ in range(self.n_base_models):
            seed_m = int(rng.integers(0, 1000000))
            sub_feats = rng.choice(active_idx, size=sub_size, replace=False)
            X_sub = X[:, sub_feats]
            if base_engine == "spectral":
                try:
                    sp = SpectralClustering(
                        n_clusters=k,
                        affinity="nearest_neighbors",
                        n_neighbors=min(10, max(2, N - 1)),
                        random_state=seed_m,
                    )
                    partitions.append(sp.fit_predict(X_sub))
                except Exception:
                    km = KMeans(n_clusters=k, n_init=3, random_state=seed_m)
                    partitions.append(km.fit_predict(X_sub))
            else:
                km = KMeans(n_clusters=k, n_init=3, random_state=seed_m)
                partitions.append(km.fit_predict(X_sub))

        P_stack = np.array(partitions)  # (M, N)

        # 3. Sample-level Co-association Stability & Core Anchor Certification
        H = np.zeros((N, self.n_base_models * k), dtype=np.float32)
        for idx, lab in enumerate(partitions):
            valid = lab >= 0
            H[valid, idx * k + lab[valid]] = 1.0
        S = (H @ H.T).astype(float) / float(self.n_base_models)
        np.fill_diagonal(S, 0.0)
        stability = np.sum(S**2, axis=1) / max(1, N - 1)

        tau_stab = float(np.quantile(stability, self.tau_core))
        core_cands = np.where(stability >= tau_stab)[0]
        if len(core_cands) < k:
            core_cands = np.argsort(stability)[-k:]

        L = min(self.n_anchors, len(core_cands))
        anchor_idx = rng.choice(core_cands, size=L, replace=False)

        # 4. Rectangular Bipartite Co-association Matrix B: (N, L)
        P_anchors = P_stack[:, anchor_idx]
        B = np.zeros((N, L), dtype=float)
        for m in range(self.n_base_models):
            B += (P_stack[m, :, None] == P_anchors[m, None, :]).astype(float)
        B /= float(self.n_base_models)

        # 5. Transfer-Cut Spectral Embedding
        row_sums = B.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        B_norm = B / row_sums

        W_anchor = B_norm.T @ B_norm
        deg = W_anchor.sum(axis=1)
        deg_inv_sqrt = 1.0 / np.sqrt(np.fmax(deg, 1e-12))
        L_norm = np.eye(L) - (deg_inv_sqrt[:, None] * W_anchor * deg_inv_sqrt[None, :])

        try:
            evals, evecs = sla.eigh(L_norm)
            if L > k:
                V_core = evecs[:, 1 : k + 1]
                sig = np.fmax(1.0 - evals[1 : k + 1], 1e-9)
            else:
                V_core = evecs[:, :k]
                sig = np.fmax(1.0 - evals[:k], 1e-9)

            U_emb = B_norm @ (deg_inv_sqrt[:, None] * V_core) / np.sqrt(sig)
            emb_norm = np.linalg.norm(U_emb, axis=1, keepdims=True)
            emb_norm[emb_norm < 1e-12] = 1.0
            embeddings = U_emb / emb_norm
        except Exception:
            embeddings = B_norm

        km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        self.labels_ = km.fit_predict(embeddings)
        return self

    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_
