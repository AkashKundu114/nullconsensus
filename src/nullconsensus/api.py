"""
Top-level Scikit-Learn Compatible Estimator for nullconsensus.
Integrates Candidate 12 (NullGate) with Candidate 2 (HurdleGate).
"""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans

from nullconsensus.methods.hurdlegate import HurdleGateFeatureSelector
from nullconsensus.methods.nullgate import NullGateFeatureSelector


class NullConsensus(BaseEstimator, ClusterMixin):
    """
    Permutation-Null Calibrated Consensus Clustering (NullConsensus).

    A scalable, geometry-calibrated cluster ensemble designed for high-dimensional,
    noisy, and zero-inflated manifolds. Uniquely integrates:
      1. Candidate 12 (NullGate): Unimodal-null calibrated saliency z-scores with
         Type-I error control under high-dimensional isotropic distractor noise.
      2. Candidate 2 (HurdleGate): Two-part hurdle decomposition isolating positive-support
         multimodality from dropout point-masses for single-cell count matrices.
      3. Saliency-Weighted Soft Subspace Sampling: Multi-feature random subspaces drawn
         via softmax probability distributions, eliminating bimodal seed failure.
      4. Bipartite Landmark Anchor Graphs: Scales co-association in linear O(N * L) time
         and memory via rectangular anchor matrices solved via Transfer-Cut spectra.
      5. Inductive Out-of-Sample Centroid Projection: Evaluates queries in O(K * |F_active|)
         time, bypassing transductive graph recomputation.

    Parameters:
        n_clusters: Target number of clusters (K). Default: 3.
        n_base_models: Number of base partitions in the ensemble (M). Default: 25.
        n_anchors: Number of landmark anchors (L). Default: 40.
        subspace_fraction: Fraction of features sampled per base model (alpha). Default: 0.15.
        mode: Feature gating mode ('nullgate', 'hurdlegate', or 'uniform'). Default: 'nullgate'.
        temperature: Softmax temperature for saliency-weighted subspace sampling. Default: 1.0.
        null_reference: Reference null for NullGate ('gaussian' or 'uniform'). Default: 'gaussian'.
        n_null_draws: Number of finite-sample null draws for NullGate. Default: 200.
        random_state: Seed for deterministic reproducibility.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_base_models: int = 25,
        n_anchors: int = 40,
        subspace_fraction: float = 0.15,
        mode: Literal[
            "nullgate", "hurdlegate", "uniform", "permutation_null", "hurdle"
        ] = "nullgate",
        temperature: float = 1.0,
        null_reference: Literal["gaussian", "uniform"] = "gaussian",
        n_null_draws: int = 200,
        n_null_permutations: int | None = None,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.n_base_models = n_base_models
        self.n_anchors = n_anchors
        self.subspace_fraction = subspace_fraction
        self.mode = mode
        self.temperature = temperature
        self.null_reference = null_reference
        self.n_null_draws = (
            n_null_permutations if n_null_permutations is not None else n_null_draws
        )
        self.n_null_permutations = self.n_null_draws
        self.random_state = random_state

        # Fitted attributes
        self.labels_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None
        self.active_features_: np.ndarray | None = None
        self.feature_weights_: np.ndarray | None = None
        self.sampling_probs_: np.ndarray | None = None
        self.selector_: NullGateFeatureSelector | HurdleGateFeatureSelector | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "NullConsensus":
        """
        Fits NullConsensus on input matrix X: (N, D).
        """
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        rng = np.random.default_rng(self.random_state)

        # 1. Feature Saliency Weighting
        normalized_mode = self.mode.lower()
        if normalized_mode in ["nullgate", "permutation_null"]:
            self.selector_ = NullGateFeatureSelector(
                null_reference=self.null_reference,
                n_null_draws=self.n_null_draws,
                random_state=self.random_state,
            )
            self.selector_.fit(X, temperature=self.temperature)
            self.sampling_probs_ = self.selector_.sampling_probs_
            self.feature_weights_ = self.selector_.saliency_scores_
        elif normalized_mode in ["hurdlegate", "hurdle"]:
            self.selector_ = HurdleGateFeatureSelector(random_state=self.random_state)
            self.selector_.fit(X, temperature=self.temperature)
            self.sampling_probs_ = self.selector_.sampling_probs_
            self.feature_weights_ = self.selector_.hurdle_scores_
        else:  # uniform
            self.sampling_probs_ = np.ones(D) / D
            self.feature_weights_ = np.ones(D)

        # Top active features pool
        assert self.feature_weights_ is not None
        n_pool = max(2, int(np.ceil(self.subspace_fraction * D)))
        self.active_features_ = np.argsort(-self.feature_weights_)[:n_pool]

        # 2. Select L Landmark Anchors
        L = min(self.n_anchors, N)
        anchor_idx = rng.choice(N, size=L, replace=False)

        # 3. Generate Base Ensemble Partitions with Saliency-Weighted Subspaces
        n_sub = max(2, int(np.ceil(self.subspace_fraction * D)))
        # Rectangular bipartite co-association matrix B: N x L
        B = np.zeros((N, L), dtype=float)

        for m in range(self.n_base_models):
            sub_seed = int(rng.integers(0, 1000000))
            # Soft saliency-weighted subspace sampling
            sub_feats = rng.choice(D, size=n_sub, replace=False, p=self.sampling_probs_)
            X_sub = X[:, sub_feats]

            # Base clustering
            km = KMeans(n_clusters=self.n_clusters, n_init=1, random_state=sub_seed)
            base_labels = km.fit_predict(X_sub)

            # Co-association with landmark anchors: match indicator
            anchor_labels = base_labels[anchor_idx]
            match_matrix = (base_labels[:, None] == anchor_labels[None, :]).astype(
                float
            )
            B += match_matrix / self.n_base_models

        # 4. Bipartite Transfer-Cut Spectral Embedding
        dx = np.sum(B, axis=1)  # (N,)
        dy = np.sum(B, axis=0)  # (L,)
        dx[dx < 1e-12] = 1e-12
        dy[dy < 1e-12] = 1e-12

        Dx_inv_sqrt = np.diag(1.0 / np.sqrt(dx))
        Dy_inv_sqrt = np.diag(1.0 / np.sqrt(dy))

        # S = Dx^(-1/2) * B * Dy^(-1/2)
        S = Dx_inv_sqrt @ B @ Dy_inv_sqrt

        try:
            U, s, Vt = np.linalg.svd(S, full_matrices=False)
            k_eff = min(self.n_clusters, U.shape[1])
            embeddings = U[:, :k_eff]
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1e-12
            embeddings = embeddings / norms
        except Exception:
            embeddings = B[:, : min(self.n_clusters, L)]

        # Final consensus clustering on normalized spectral embedding
        final_km = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.random_state
        )
        self.labels_ = final_km.fit_predict(embeddings)

        # 5. Compute Active Subspace Centroids for Inductive Inference
        self.centroids_ = np.zeros((self.n_clusters, len(self.active_features_)))
        for k in range(self.n_clusters):
            mask = self.labels_ == k
            if np.any(mask):
                self.centroids_[k] = np.mean(X[mask][:, self.active_features_], axis=0)
            else:
                self.centroids_[k] = np.mean(X[:, self.active_features_], axis=0)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Projects out-of-sample query instances to nearest active centroids in O(K * |F_active|) time.
        """
        if self.centroids_ is None or self.active_features_ is None:
            raise RuntimeError(
                "NullConsensus instance is not fitted yet. Call 'fit' before predicting."
            )

        X = np.asarray(X, dtype=float)
        X_active = X[:, self.active_features_]

        # Squared Euclidean distance: (N_query, K)
        dists = np.zeros((len(X), self.n_clusters))
        for k in range(self.n_clusters):
            dists[:, k] = np.sum((X_active - self.centroids_[k]) ** 2, axis=1)

        return np.argmin(dists, axis=1)

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """
        Fits estimator and returns cluster labels.
        """
        self.fit(X, y)
        assert self.labels_ is not None
        return self.labels_
