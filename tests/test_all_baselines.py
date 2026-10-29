"""
Unit tests for the complete 12-model baseline registry.
"""

import numpy as np
import pytest

from nullconsensus.baselines import BASELINE_REGISTRY, get_baseline


@pytest.fixture
def small_synthetic():
    rng = np.random.default_rng(42)
    X1 = rng.normal(loc=-2.0, scale=0.5, size=(12, 10))
    X2 = rng.normal(loc=2.0, scale=0.5, size=(12, 10))
    X = np.vstack([X1, X2])
    y = np.array([0] * 12 + [1] * 12)
    return X, y


@pytest.mark.parametrize("method_name", list(BASELINE_REGISTRY.keys()))
def test_all_baselines_fit_predict(method_name, small_synthetic):
    X, y = small_synthetic
    model = get_baseline(method_name, n_clusters=2, random_state=42)
    labels = model.fit_predict(X)
    assert len(labels) == len(X)
    assert len(np.unique(labels)) <= 2
    assert np.all(labels >= 0)
