"""
Unit tests for AutoNullConsensus, Manifold Profiler, DPP Anchor Seeding, and Scanpy Integration.
"""

import numpy as np
import pytest

from nullconsensus.api import AutoNullConsensus
from nullconsensus.core.anchors import select_anchors
from nullconsensus.core.gpu_backend import solve_transfer_cut
from nullconsensus.core.router import estimate_intrinsic_dim, profile_manifold
import nullconsensus as nc


def test_estimate_intrinsic_dim():
    rng = np.random.default_rng(42)
    # Generate points along a low-dimensional manifold (3D embedded in 50D)
    N = 100
    t = rng.uniform(-1, 1, size=(N, 3))
    proj = rng.standard_normal((3, 50))
    X = t @ proj
    d_est = estimate_intrinsic_dim(X, n_subsample=100, random_state=42)
    assert 1.0 <= d_est <= 20.0


def test_profile_manifold_routing():
    rng = np.random.default_rng(42)

    # 1. Zero-inflated scRNA-seq profile
    X_sc = rng.poisson(lam=1.5, size=(60, 200)).astype(float)
    X_sc[rng.uniform(0, 1, size=X_sc.shape) < 0.60] = 0.0  # 60% zeros
    prof_sc = profile_manifold(X_sc, n_clusters=4, random_state=42)
    assert prof_sc.zero_fraction > 0.30
    assert prof_sc.recommended_mode == "hurdlegate"

    # 2. Continuous high-D microarray profile (D/N > 10, N < 150)
    X_micro = rng.standard_normal((50, 600))
    prof_micro = profile_manifold(X_micro, n_clusters=3, random_state=42)
    assert prof_micro.zero_fraction < 0.10
    assert prof_micro.d_over_n > 10.0
    assert prof_micro.recommended_mode == "uniform"

    # 3. Standard moderate continuous profile
    X_std = rng.standard_normal((200, 30))
    prof_std = profile_manifold(X_std, n_clusters=3, random_state=42)
    assert prof_std.recommended_mode == "nullgate"

    # 4. Multi-class anchor recommendation
    prof_multi = profile_manifold(X_std, n_clusters=10, random_state=42)
    assert prof_multi.recommended_anchor_method == "dpp"


def test_dpp_anchor_selection():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 20))
    anchors_dpp = select_anchors(X, L=25, method="dpp", random_state=42)
    assert len(anchors_dpp) == 25
    assert len(np.unique(anchors_dpp)) == 25
    assert np.all(anchors_dpp >= 0) and np.all(anchors_dpp < 100)

    anchors_km = select_anchors(X, L=25, method="kmeans", random_state=42)
    assert len(anchors_km) == 25
    assert len(np.unique(anchors_km)) == 25


def test_auto_nullconsensus_fit_predict():
    rng = np.random.default_rng(42)
    # Synthetic 3-cluster dataset
    c1 = rng.standard_normal((30, 40)) + 3.0
    c2 = rng.standard_normal((30, 40)) - 3.0
    c3 = rng.standard_normal((30, 40))
    X = np.vstack([c1, c2, c3])

    model = AutoNullConsensus(n_clusters=3, mode="auto", anchor_method="dpp", random_state=42)
    labels = model.fit_predict(X)
    assert len(labels) == 90
    assert len(np.unique(labels)) == 3
    assert model.embedding_ is not None
    assert model.embedding_.shape == (90, 3)
    assert model.centroids_ is not None

    # Test inductive out-of-sample prediction
    X_query = rng.standard_normal((5, 40)) + 3.0
    preds = model.predict(X_query)
    assert len(preds) == 5
    assert np.all(preds >= 0) and np.all(preds < 3)


def test_solve_transfer_cut_cpu():
    rng = np.random.default_rng(42)
    B = rng.uniform(0, 1, size=(50, 20))
    U_emb, evals = solve_transfer_cut(B, n_clusters=3, device="cpu")
    assert U_emb.shape == (50, 3)
    assert len(evals) == 3
    # Check normalization
    norms = np.linalg.norm(U_emb, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_scanpy_integration():
    try:
        import anndata
    except ImportError:
        pytest.skip("anndata not installed")

    rng = np.random.default_rng(42)
    X = rng.poisson(lam=1.0, size=(50, 80)).astype(float)
    X[rng.uniform(0, 1, size=X.shape) < 0.50] = 0.0  # single-cell counts

    adata = anndata.AnnData(X=X)
    nc.tl.cluster(adata, n_clusters=3, key_added="nullconsensus_clusters")

    assert "nullconsensus_clusters" in adata.obs
    assert len(adata.obs["nullconsensus_clusters"]) == 50
    assert "X_nullconsensus" in adata.obsm
    assert adata.obsm["X_nullconsensus"].shape == (50, 3)
    assert "nullconsensus_clusters" in adata.uns
    assert adata.uns["nullconsensus_clusters"]["params"]["n_clusters"] == 3
