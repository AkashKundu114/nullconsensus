"""
GAC Anchor Pipeline with Random Feature Gating Baseline (gac_random_gating).

Negative control for nullconsensus and Paper 1:
Evaluates the exact same anchor consensus pipeline but replaces null/bimodality gating
with a uniform random feature gate of identical size |F_active| = floor(alpha * D).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from typing import Optional


class GACRandomGatingBaseline(BaseEstimator, ClusterMixin):
    """
    GAC (Anchor Consensus) Pipeline with Uniform Random Gating.
    
    Serves as the primary control baseline in H1: evaluates whether feature gating
    significantly exceeds arbitrary random subspace reduction under the identical
    bipartite anchor consensus architecture.
    
    Parameters:
        n_clusters: Target number of clusters (K).
        n_base_models: Number of base models in ensemble (M). Default: 25.
        n_anchors: Number of landmark anchors (L). Default: 40.
        gating_fraction: Fraction of features retained in active pool (|F_active| / D). Default: 0.15.
        subspace_fraction: Fraction of active features drawn per base model. Default: 0.50.
        random_state: Seed for reproducibility.
    """
    def __init__(
        self,
        n_clusters: int = 3,
        n_base_models: int = 25,
        n_anchors: int = 40,
        gating_fraction: float = 0.15,
        subspace_fraction: float = 0.50,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.n_base_models = n_base_models
        self.n_anchors = n_anchors
        self.gating_fraction = gating_fraction
        self.subspace_fraction = subspace_fraction
        self.random_state = random_state
        
        # Fitted attributes
        self.labels_: Optional[np.ndarray] = None
        self.active_features_: Optional[np.ndarray] = None
        self.centroids_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "GACRandomGatingBaseline":
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        rng = np.random.default_rng(self.random_state)
        
        # 1. Random Gating: select floor(alpha * D) features uniformly without replacement
        n_active = max(2, int(np.floor(self.gating_fraction * D)))
        self.active_features_ = rng.choice(D, size=n_active, replace=False)
        X_active = X[:, self.active_features_]
        
        # 2. Select L Landmark Anchors
        L = min(self.n_anchors, N)
        anchor_idx = rng.choice(N, size=L, replace=False)
        
        # 3. Base Ensemble Partitions
        n_sub = max(2, int(np.floor(self.subspace_fraction * n_active)))
        B = np.zeros((N, L), dtype=float)
        
        for m in range(self.n_base_models):
            seed_m = int(rng.integers(0, 1_000_000))
            sub_feats = rng.choice(n_active, size=n_sub, replace=False)
            X_sub = X_active[:, sub_feats]
            
            km = KMeans(n_clusters=self.n_clusters, n_init=1, random_state=seed_m)
            base_labels = km.fit_predict(X_sub)
            
            anchor_labels = base_labels[anchor_idx]
            match_matrix = (base_labels[:, None] == anchor_labels[None, :]).astype(float)
            B += match_matrix / self.n_base_models
            
        # 4. Transfer Cut Spectral Embedding
        dx = np.sum(B, axis=1)
        dy = np.sum(B, axis=0)
        dx[dx < 1e-12] = 1e-12
        dy[dy < 1e-12] = 1e-12
        
        Dx_inv_sqrt = np.diag(1.0 / np.sqrt(dx))
        Dy_inv_sqrt = np.diag(1.0 / np.sqrt(dy))
        S = Dx_inv_sqrt @ B @ Dy_inv_sqrt
        
        try:
            U, s, Vt = np.linalg.svd(S, full_matrices=False)
            k_eff = min(self.n_clusters, U.shape[1])
            embeddings = U[:, :k_eff]
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1e-12
            embeddings = embeddings / norms
        except Exception:
            embeddings = B[:, :min(self.n_clusters, L)]
            
        final_km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        self.labels_ = final_km.fit_predict(embeddings)
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_
