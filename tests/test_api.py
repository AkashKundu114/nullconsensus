"""
Unit tests for NullConsensus API and parameters.
"""

import numpy as np
import pytest
from nullconsensus import NullConsensus, NullGateFeatureSelector, HurdleGateFeatureSelector
from nullconsensus.data.synthetic import make_gaussian_distractor, make_zero_inflated_counts


def test_nullconsensus_estimator_modes():
    X, y = make_gaussian_distractor(n_samples=50, n_features=40, n_informative=8, n_clusters=2, random_state=123)
    
    # Test nullgate mode
    nc_null = NullConsensus(n_clusters=2, n_base_models=5, n_anchors=15, mode="nullgate", random_state=123)
    labels_null = nc_null.fit_predict(X)
    assert len(labels_null) == 50
    assert isinstance(nc_null.selector_, NullGateFeatureSelector)
    
    # Test hurdlegate mode
    nc_hurdle = NullConsensus(n_clusters=2, n_base_models=5, n_anchors=15, mode="hurdlegate", random_state=123)
    labels_hurdle = nc_hurdle.fit_predict(X)
    assert len(labels_hurdle) == 50
    assert isinstance(nc_hurdle.selector_, HurdleGateFeatureSelector)
    
    # Test uniform mode
    nc_unif = NullConsensus(n_clusters=2, n_base_models=5, n_anchors=15, mode="uniform", random_state=123)
    labels_unif = nc_unif.fit_predict(X)
    assert len(labels_unif) == 50
    assert nc_unif.selector_ is None


def test_unfitted_predict_raises():
    nc = NullConsensus()
    X = np.random.randn(10, 5)
    with pytest.raises(RuntimeError, match="not fitted yet"):
        nc.predict(X)
