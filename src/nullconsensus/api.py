"""
Top-level Scikit-Learn Compatible Estimators for nullconsensus.
Provides:
  - NullConsensus: Permutation-Null and Hurdle-gated consensus clustering.
  - AutoNullConsensus: Self-tuning, geometry-adaptive universal consensus clustering engine.
"""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans, SpectralClustering

from nullconsensus.core.anchors import select_anchors
from nullconsensus.core.gpu_backend import solve_transfer_cut
from nullconsensus.core.router import ManifoldProfile, profile_manifold
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
        mode: Feature gating mode ('nullgate', 'hurdlegate', 'uniform', or 'auto'). Default: 'nullgate'.
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
            "nullgate", "hurdlegate", "uniform", "permutation_null", "hurdle", "auto"
        ] = "nullgate",
        temperature: float = 1.0,
        null_reference: Literal["gaussian", "uniform"] = "gaussian",
        n_null_draws: int = 200,
        n_null_permutations: int | None = None,
        base_clusterer: Literal["auto", "kmeans", "spectral"] = "auto",
        anchor_method: Literal["auto", "dpp", "kmeans", "random"] = "random",
        device: Literal["auto", "cpu", "cuda"] = "auto",
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
        self.base_clusterer = base_clusterer
        self.anchor_method = anchor_method
        self.device = device
        self.random_state = random_state

        # Fitted attributes
        self.labels_: np.ndarray | None = None
        self.embedding_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None
        self.active_features_: np.ndarray | None = None
        self.feature_weights_: np.ndarray | None = None
        self.sampling_probs_: np.ndarray | None = None
        self.selector_: NullGateFeatureSelector | HurdleGateFeatureSelector | None = None
        self.profile_: ManifoldProfile | None = None
        self.mode_: str = mode

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "NullConsensus":
        """
        Fits NullConsensus on input matrix X: (N, D).
        """
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        rng = np.random.default_rng(self.random_state)

        # 0. Automatic Regime Routing if requested
        if self.mode.lower() == "auto":
            self.profile_ = profile_manifold(X, n_clusters=self.n_clusters, random_state=self.random_state)
            self.mode_ = self.profile_.recommended_mode
        else:
            self.mode_ = self.mode.lower()

        # 1. Feature Saliency Weighting
        if self.mode_ in ["nullgate", "permutation_null"]:
            self.selector_ = NullGateFeatureSelector(
                null_reference=self.null_reference,
                n_null_draws=self.n_null_draws,
                random_state=self.random_state,
            )
            self.selector_.fit(X, temperature=self.temperature)
            self.sampling_probs_ = self.selector_.sampling_probs_
            self.feature_weights_ = self.selector_.saliency_scores_
        elif self.mode_ in ["hurdlegate", "hurdle"]:
            self.selector_ = HurdleGateFeatureSelector(random_state=self.random_state)
            self.selector_.fit(X, temperature=self.temperature)
            self.sampling_probs_ = self.selector_.sampling_probs_
            self.feature_weights_ = self.selector_.hurdle_scores_
        else:  # uniform
            self.sampling_probs_ = np.ones(D) / D
            self.feature_weights_ = np.ones(D)

        # Top active features pool
        assert self.feature_weights_ is not None
        assert self.sampling_probs_ is not None
        n_pool = max(2, int(np.ceil(self.subspace_fraction * D)))
        self.active_features_ = np.argsort(-self.feature_weights_)[:n_pool]

        # Normalized saliency-weighted probabilities over active candidate pool
        pool_probs = self.sampling_probs_[self.active_features_]
        prob_sum = float(np.sum(pool_probs))
        if prob_sum > 0:
            pool_probs = pool_probs / prob_sum
        else:
            pool_probs = np.ones(len(self.active_features_)) / len(self.active_features_)

        # 2. Select L Landmark Anchors
        L = min(self.n_anchors, N)
        if self.anchor_method == "random":
            anchor_idx = rng.choice(N, size=L, replace=False)
        else:
            anchor_method_to_use = self.anchor_method
            if anchor_method_to_use == "auto":
                anchor_method_to_use = self.profile_.recommended_anchor_method if self.profile_ else "dpp"
            anchor_idx = select_anchors(X, L=L, method=anchor_method_to_use, random_state=self.random_state)

        # Determine base clustering engine
        if self.base_clusterer == "auto":
            base_engine = "spectral" if N <= 100 else "kmeans"
        else:
            base_engine = self.base_clusterer

        # 3. Generate Base Ensemble Partitions with Saliency-Weighted Subspaces
        sub_size = max(
            2, min(len(self.active_features_), int(np.ceil(0.30 * len(self.active_features_))))
        )
        # Rectangular bipartite co-association matrix B: N x L
        B = np.zeros((N, L), dtype=float)

        for m in range(self.n_base_models):
            sub_seed = int(rng.integers(0, 1000000))
            # Soft saliency-weighted subspace sampling within active pool
            sub_feats = rng.choice(
                self.active_features_, size=sub_size, replace=False, p=pool_probs
            )
            X_sub = X[:, sub_feats]

            # Base clustering
            if base_engine == "spectral":
                try:
                    sp = SpectralClustering(
                        n_clusters=self.n_clusters,
                        affinity="nearest_neighbors",
                        n_neighbors=min(10, max(2, N - 1)),
                        random_state=sub_seed,
                        assign_labels="kmeans",
                    )
                    base_labels = sp.fit_predict(X_sub)
                except Exception:
                    km = KMeans(
                        n_clusters=self.n_clusters, n_init=3, random_state=sub_seed
                    )
                    base_labels = km.fit_predict(X_sub)
            else:
                km = KMeans(
                    n_clusters=self.n_clusters, n_init=3, random_state=sub_seed
                )
                base_labels = km.fit_predict(X_sub)

            # Co-association with landmark anchors: match indicator
            anchor_labels = base_labels[anchor_idx]
            match_matrix = (base_labels[:, None] == anchor_labels[None, :]).astype(
                float
            )
            B += match_matrix / self.n_base_models

        # 4. Bipartite Transfer-Cut Spectral Embedding (CPU/GPU)
        try:
            self.embedding_, _ = solve_transfer_cut(B, n_clusters=self.n_clusters, device=self.device)
        except Exception:
            self.embedding_ = B[:, : min(self.n_clusters, L)]

        # Final consensus clustering on normalized spectral embedding
        final_km = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.random_state
        )
        self.labels_ = final_km.fit_predict(self.embedding_)

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


class AutoNullConsensus(NullConsensus):
    """
    Self-Tuning, Geometry-Adaptive Universal Consensus Clustering (AutoNullConsensus).

    Autonomously audits input data geometry, sparsity patterns, and intrinsic dimensionality:
      - On zero-inflated manifolds (scRNA-seq with dropouts > 30%): automatically activates Two-Part HurdleGate.
      - On high-dimensional continuous manifolds (D/N > 10, N < 150): deploys uniform random subspaces.
      - On complex/multi-class benchmarks (K >= 8 or N >= 2000): deploys k-DPP diversity anchor seeding.
      - On CUDA GPU hardware: executes in sub-second time via PyTorch tensor Lanczos Transfer-Cuts.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        n_base_models: int = 25,
        n_anchors: int = 40,
        subspace_fraction: float = 0.15,
        mode: Literal["auto", "nullgate", "hurdlegate", "uniform"] = "auto",
        anchor_method: Literal["auto", "dpp", "kmeans", "random"] = "auto",
        temperature: float = 1.0,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            n_clusters=n_clusters,
            n_base_models=n_base_models,
            n_anchors=n_anchors,
            subspace_fraction=subspace_fraction,
            mode=mode,
            temperature=temperature,
            anchor_method=anchor_method,
            device=device,
            random_state=random_state,
            **kwargs,
        )
