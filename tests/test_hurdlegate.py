"""
Unit tests for HurdleGateFeatureSelector (Candidate 2).
"""

import numpy as np
import pytest
from nullconsensus.methods.hurdlegate import HurdleGateFeatureSelector
from nullconsensus.data.synthetic import make_zero_inflated_counts


def test_hurdlegate_identifies_informative_features():
    # 80 samples, 200 features, 20 informative zero-inflated features, 180 noise
    X, y = make_zero_inflated_counts(
        n_samples=80,
        n_features=200,
        n_informative=20,
        n_clusters=2,
        dropout_rate=0.5,
        random_state=42
    )
    
    selector = HurdleGateFeatureSelector(min_positive_samples=5, random_state=42)
    selector.fit(X, temperature=1.0)
    
    assert selector.hurdle_scores_ is not None
    assert len(selector.hurdle_scores_) == 200
    assert len(selector.sampling_probs_) == 200
    assert np.isclose(np.sum(selector.sampling_probs_), 1.0)
    
    # Informative features should receive significantly higher average score than noise features
    mean_info_score = np.mean(selector.hurdle_scores_[:20])
    mean_noise_score = np.mean(selector.hurdle_scores_[20:])
    
    assert mean_info_score > mean_noise_score
    print(f"HurdleGate: Informative={mean_info_score:.3f} vs Noise={mean_noise_score:.3f}")


def test_hurdlegate_all_zero_features_handled():
    X = np.zeros((50, 10))
    # Give feature 0 some positive expression
    X[:25, 0] = 5.0
    
    selector = HurdleGateFeatureSelector(min_positive_samples=5, random_state=42)
    selector.fit(X)
    
    assert selector.hurdle_scores_[0] > 0.0
    assert np.all(selector.hurdle_scores_[1:] == 0.0)
    assert selector.sampling_probs_[0] > np.max(selector.sampling_probs_[1:])
