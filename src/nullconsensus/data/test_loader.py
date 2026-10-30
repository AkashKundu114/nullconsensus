"""
Confirmatory Test Benchmark Suite Loader (data/test_loader.py).

Provides loading and metadata inspection for the 27 pre-registered Confirmatory TEST datasets:
- 15 High-D Cancer Microarrays (Souto collection)
- 4 Zero-inflated scRNA-seq Count Cohorts
- 5 Large-K Benchmarks with K >= 15
- 3 Low-D Physical/Biomedical Controls

Strictly enforces Rule R2:
Calls verify_test_access_allowed() prior to any data access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from nullconsensus.data.guard import verify_test_access_allowed


def get_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / ".git").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path.cwd()


TEST_CACHE_PATHS = [
    get_repo_root() / "data" / "cache" / "test",
    Path("data/cache/test"),
]

TEST_MANIFEST_PATH = get_repo_root() / "data" / "test_manifest.json"


@dataclass
class TestDatasetMeta:
    id: str
    name: str
    domain: str
    n_samples: int
    n_features: int
    n_clusters: int
    d_over_n: float
    is_high_d: bool
    class_balance: list[int]
    preprocessing: str
    sha256: str


def get_test_manifest() -> dict[str, dict]:
    """Loads the test dataset dictionary keyed by dataset ID."""
    if not TEST_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Test manifest not found at {TEST_MANIFEST_PATH}")
    with open(TEST_MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {ds["id"]: ds for ds in data.get("datasets", [])}


def list_test_datasets() -> list[str]:
    """Returns the pre-registered IDs of all 27 TEST datasets."""
    manifest = get_test_manifest()
    return list(manifest.keys())


def load_test_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, TestDatasetMeta]:
    """
    Loads a Confirmatory TEST dataset by ID under pre-registered Rule R2.

    Parameters:
        name: ID of the TEST dataset (e.g., 'armstrong2002', 'wdbc')

    Returns:
        X: Preprocessed feature matrix of shape (N, D)
        y: Ground truth cluster label vector of shape (N,) with contiguous integers in [0, K-1]
        meta: TestDatasetMeta containing dimensions, domain, and regime flags

    Raises:
        DataPeekingViolation: If access conditions (FREEZE_TAG, clean tree) are not satisfied.
        FileNotFoundError: If the dataset has not been cached yet via scripts/fetch_test_data.py.
    """
    clean_name = name.lower().strip()
    manifest = get_test_manifest()

    if clean_name not in manifest:
        raise ValueError(
            f"Dataset '{clean_name}' is not recognized in the 27 pre-registered TEST datasets. "
            f"Available IDs: {list(manifest.keys())}"
        )

    # RULE R2: Strictly verify legal unsealing conditions
    verify_test_access_allowed(clean_name)

    # Locate npz file in cache
    npz_file: Path | None = None
    for p in TEST_CACHE_PATHS:
        cand = p / f"{clean_name}.npz"
        if cand.exists():
            npz_file = cand
            break

    if npz_file is None:
        raise FileNotFoundError(
            f"TEST dataset '{clean_name}' has not been cached in {TEST_CACHE_PATHS}. "
            f"Please run scripts/fetch_test_data.py to download and preprocess the test suite."
        )

    data = np.load(npz_file)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=int)

    ds_info = manifest[clean_name]
    N, D = X.shape
    k = len(np.unique(y))
    d_over_n = D / N
    is_high_d = d_over_n > 20.0

    meta = TestDatasetMeta(
        id=clean_name,
        name=ds_info.get("name", clean_name),
        domain=ds_info.get("domain", "Unknown"),
        n_samples=N,
        n_features=D,
        n_clusters=k,
        d_over_n=d_over_n,
        is_high_d=is_high_d,
        class_balance=list(np.bincount(y)),
        preprocessing=ds_info.get("preprocessing", "StandardScaler()"),
        sha256=ds_info.get("sha256", "UNKNOWN"),
    )
    return X, y, meta
