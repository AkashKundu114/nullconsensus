"""
Unit tests for DEV benchmark dataset loader and test guard separation.
"""

import pytest

from nullconsensus.data.dev_loader import list_dev_datasets, load_dev_dataset
from nullconsensus.data.guard import DataPeekingViolation


def test_list_dev_datasets():
    datasets = list_dev_datasets()
    assert len(datasets) == 13
    assert "leukemia" in datasets
    assert "colon" in datasets
    assert "cns" in datasets


def test_load_synthetic_dev():
    X, y, meta = load_dev_dataset("synthetic_gaussian")
    assert X.shape == (150, 500)
    assert len(y) == 150
    assert meta.n_clusters == 3

    X_xor, y_xor, meta_xor = load_dev_dataset("synthetic_xor")
    assert X_xor.shape == (300, 302)
    assert meta_xor.n_clusters == 2


def test_test_data_access_blocked():
    # An attempt to load Armstrong (which is in test_manifest.json) must trigger DataPeekingViolation
    with pytest.raises(DataPeekingViolation):
        load_dev_dataset("armstrong")

    with pytest.raises(DataPeekingViolation):
        load_dev_dataset("chowdary")


def test_load_real_dev_dataset():
    X, y, meta = load_dev_dataset("leukemia")
    assert X.shape == (72, 7129)
    assert len(y) == 72
    assert meta.n_clusters == 2
    assert meta.is_high_d is True
