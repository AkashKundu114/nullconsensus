"""
Benchmark baselines for nullconsensus.
Provides uniform scikit-learn estimators for all 12 pre-registered models:
1. kmeans: Classical K-Means (Lloyd 1982)
2. ward: Ward Agglomerative Hierarchical (Ward 1963)
3. spectral: Graph Spectral (Ng, Jordan, Weiss 2001)
4. std_consensus: Dense Consensus (Monti 2003, Fred & Jain 2005)
5. core_halo: Stability-Weighted Consensus (Li et al. 2019)
6. fcag: Fast Clustering with Anchor Guidance (Nie et al. 2024)
7. sparse_kmeans: Sparse K-Means (Witten & Tibshirani 2010)
8. cohirf: Consensus Hierarchical Random Features (Belucci et al. 2025)
9. m3c: Monte Carlo Reference Consensus (John et al. 2020)
10. icc_v2: Geometry-Calibrated Anchor Consensus (GAC, Paper 1)
11. gac_random_gating: Matched Anchor Gating Negative Control
12. nullconsensus: Permutation-Null Calibrated Consensus Clustering (Proposed)
"""

from typing import Any

from sklearn.base import BaseEstimator

from nullconsensus.api import NullConsensus
from nullconsensus.baselines.anchor import FCAGBaseline
from nullconsensus.baselines.classical import (
    KMeansBaseline,
    SparseKMeansBaseline,
    SpectralBaseline,
    WardBaseline,
)
from nullconsensus.baselines.cohirf import CoHiRFBaseline
from nullconsensus.baselines.consensus import CoreHaloBaseline, StandardConsensusBaseline
from nullconsensus.baselines.gac import GACBaseline
from nullconsensus.baselines.gac_random_gating import GACRandomGatingBaseline
from nullconsensus.baselines.m3c import M3CBaseline

BASELINE_REGISTRY = {
    "kmeans": KMeansBaseline,
    "ward": WardBaseline,
    "spectral": SpectralBaseline,
    "sparse_kmeans": SparseKMeansBaseline,
    "std_consensus": StandardConsensusBaseline,
    "core_halo": CoreHaloBaseline,
    "fcag": FCAGBaseline,
    "cohirf": CoHiRFBaseline,
    "m3c": M3CBaseline,
    "icc_v2": GACBaseline,
    "gac_random_gating": GACRandomGatingBaseline,
    "nullconsensus": NullConsensus,
}


def get_baseline(
    name: str, n_clusters: int = 3, random_state: int = 42, **kwargs: Any
) -> BaseEstimator:
    """
    Factory function returning an initialized baseline estimator with canonical defaults.
    """
    key = name.lower()
    if key not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{name}'. Available: {list(BASELINE_REGISTRY.keys())}"
        )
    cls = BASELINE_REGISTRY[key]
    return cls(n_clusters=n_clusters, random_state=random_state, **kwargs)


__all__ = [
    "BASELINE_REGISTRY",
    "get_baseline",
    "KMeansBaseline",
    "WardBaseline",
    "SpectralBaseline",
    "SparseKMeansBaseline",
    "StandardConsensusBaseline",
    "CoreHaloBaseline",
    "FCAGBaseline",
    "CoHiRFBaseline",
    "M3CBaseline",
    "GACBaseline",
    "GACRandomGatingBaseline",
    "NullConsensus",
]
