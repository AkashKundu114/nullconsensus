"""
Unit tests for nullconsensus baselines (M3C and GACRandomGating).
"""

import numpy as np
import pytest
from nullconsensus.baselines.m3c import M3CBaseline, compute_pac
from nullconsensus.baselines.gac_random_gating import GACRandomGatingBaseline


def test_compute_pac_bounds():
    # Empty / small matrix
    assert compute_pac(np.eye(1)) == 0.0
    
    # Perfectly crisp consensus matrix (all 0s or 1s)
    C_crisp = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ])
    assert compute_pac(C_crisp, u1=0.1, u2=0.9) == 0.0
    
    # Completely ambiguous matrix (all 0.5)
    C_ambig = np.full((4, 4), 0.5)
    np.fill_diagonal(C_ambig, 1.0)
    assert compute_pac(C_ambig, u1=0.1, u2=0.9) == 1.0


def test_m3c_baseline_execution():
    rng = np.random.default_rng(42)
    # 2 clusters in 20 dimensions
    X1 = rng.normal(loc=-2.0, scale=0.5, size=(15, 20))
    X2 = rng.normal(loc=2.0, scale=0.5, size=(15, 20))
    X = np.vstack([X1, X2])
    
    m3c = M3CBaseline(n_clusters=2, n_resamplings=5, n_null_datasets=2, random_state=42)
    labels = m3c.fit_predict(X)
    
    assert len(labels) == 30
    assert len(np.unique(labels)) == 2
    assert m3c.pac_real_ is not None
    assert m3c.rcsi_ is not None
    assert 0.0 <= m3c.pac_real_ <= 1.0


def test_gac_random_gating_execution():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(25, 30))
    
    gac_rand = GACRandomGatingBaseline(
        n_clusters=2,
        n_base_models=5,
        n_anchors=10,
        gating_fraction=0.30,
        random_state=42
    )
    labels = gac_rand.fit_predict(X)
    
    assert len(labels) == 25
    assert len(np.unique(labels)) <= 2
    assert gac_rand.active_features_ is not None
    assert len(gac_rand.active_features_) == int(np.floor(0.30 * 30))
