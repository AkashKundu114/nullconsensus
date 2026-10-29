"""
Development Benchmark Suite Loader (data/dev_loader.py).

Provides loading and metadata inspection for the 13 Paper 1 DEV datasets:
- CNS, Colon, Leukemia, Ovarian, Prostate, SRBCT (6 High-D cancer cohorts)
- Indian Pines, Isolet, Madelon, Newsgroups20, Sonar, Yale B, Zheng68k (7 General/Multi-class cohorts)
- Controlled synthetic generators (Gaussian distractor, XOR parity, Zero-inflated counts)

Enforces strict partition boundary:
Any attempt to request a TEST dataset raises a fatal DataPeekingViolation via guard.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from nullconsensus.data.guard import verify_test_access_allowed
from nullconsensus.data.synthetic import (
    make_gaussian_distractor,
    make_xor_parity,
    make_zero_inflated_counts,
)

DEV_DATASETS = [
    "cns",
    "colon",
    "leukemia",
    "ovarian",
    "prostate",
    "srbct",
    "indian_pines",
    "isolet",
    "madelon",
    "newsgroups20",
    "sonar",
    "yale_b",
    "zheng68k",
]

DEV_CACHE_PATHS = [
    Path(r"d:\Final Year Project\data\cache"),
    Path(r"data\cache"),
]


@dataclass
class DevDatasetMeta:
    name: str
    n_samples: int
    n_features: int
    n_clusters: int
    d_over_n: float
    is_high_d: bool


def list_dev_datasets() -> list[str]:
    """Returns the names of the 13 pre-registered DEV datasets."""
    return list(DEV_DATASETS)


def load_dev_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, DevDatasetMeta]:
    """
    Loads a DEV dataset by name.

    Parameters:
        name: Name of the DEV dataset (e.g., 'leukemia', 'synthetic_gaussian')

    Returns:
        X: Feature matrix of shape (N, D)
        y: Ground truth cluster label vector of shape (N,)
        meta: DevDatasetMeta containing dataset dimensions and regime flags
    """
    clean_name = name.lower().strip()

    # Synthetic handling
    if clean_name == "synthetic_gaussian":
        X, y = make_gaussian_distractor(n_samples=150, n_features=500, n_informative=15, n_clusters=3, separation=3.0, random_state=42)
        meta = DevDatasetMeta(name=clean_name, n_samples=len(X), n_features=X.shape[1], n_clusters=3, d_over_n=X.shape[1] / len(X), is_high_d=False)
        return X, y, meta

    if clean_name == "synthetic_xor":
        X, y = make_xor_parity(n_samples=300, n_distractors=300, random_state=42)
        meta = DevDatasetMeta(name=clean_name, n_samples=len(X), n_features=X.shape[1], n_clusters=2, d_over_n=X.shape[1] / len(X), is_high_d=False)
        return X, y, meta

    if clean_name == "synthetic_hurdle":
        X, y = make_zero_inflated_counts(n_samples=200, n_features=300, n_informative=20, n_clusters=3, random_state=42)
        meta = DevDatasetMeta(name=clean_name, n_samples=len(X), n_features=X.shape[1], n_clusters=3, d_over_n=X.shape[1] / len(X), is_high_d=False)
        return X, y, meta

    # Check if this name is a TEST dataset
    if clean_name not in DEV_DATASETS:
        # Trigger guard verification which raises DataPeekingViolation for TEST datasets
        verify_test_access_allowed(clean_name)
        raise ValueError(f"Unknown dataset '{clean_name}'. Not in DEV manifest or recognized TEST manifest.")

    # Locate npz file
    npz_file: Path | None = None
    for p in DEV_CACHE_PATHS:
        cand = p / f"{clean_name}.npz"
        if cand.exists():
            npz_file = cand
            break

    if npz_file is None:
        raise FileNotFoundError(
            f"DEV dataset '{clean_name}' could not be located in cache paths: {DEV_CACHE_PATHS}. "
            f"Please verify Paper 1 data cache exists."
        )

    data = np.load(npz_file)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=int)
    N, D = X.shape
    k = len(np.unique(y))
    d_over_n = D / N
    is_high_d = d_over_n > 20.0

    meta = DevDatasetMeta(
        name=clean_name,
        n_samples=N,
        n_features=D,
        n_clusters=k,
        d_over_n=d_over_n,
        is_high_d=is_high_d,
    )
    return X, y, meta
