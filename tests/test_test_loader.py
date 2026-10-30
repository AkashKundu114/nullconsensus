"""
Unit tests for Confirmatory Test Loader (tests/test_test_loader.py).
Verifies Rule R2 compliance, dataset listing, and data integrity.
"""

import os
import pytest
import numpy as np

from nullconsensus.data.guard import DataPeekingViolation
from nullconsensus.data.test_loader import list_test_datasets, load_test_dataset, get_test_manifest


def test_list_test_datasets():
    """Verify that all 27 pre-registered datasets are listed."""
    datasets = list_test_datasets()
    assert len(datasets) == 27
    manifest = get_test_manifest()
    assert set(datasets) == set(manifest.keys())


def test_load_test_dataset_denied_without_freeze_tag(monkeypatch):
    """Verify that Rule R2 strictly blocks test data loading without valid FREEZE_TAG."""
    monkeypatch.delenv("FREEZE_TAG", raising=False)
    with pytest.raises(DataPeekingViolation, match="ACCESS DENIED"):
        load_test_dataset("armstrong2002")


def test_load_test_dataset_allowed_with_freeze_tag(monkeypatch):
    """Verify that test data loading succeeds when FREEZE_TAG=freeze-v1 is set."""
    monkeypatch.setenv("FREEZE_TAG", "freeze-v1")
    X, y, meta = load_test_dataset("armstrong2002")

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == 72
    assert X.shape[1] == 2194
    assert meta.n_clusters == 3
    assert len(np.unique(y)) == 3
    assert meta.is_high_d is True
    assert len(meta.sha256) == 64
