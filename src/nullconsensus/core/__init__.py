"""
Core Algorithmic Engines for AutoNullConsensus.
"""

from nullconsensus.core.anchors import select_anchors, select_anchors_dpp, select_anchors_kmeans
from nullconsensus.core.gpu_backend import is_gpu_available, solve_transfer_cut
from nullconsensus.core.module_gating import detect_feature_modules
from nullconsensus.core.router import ManifoldProfile, estimate_intrinsic_dim, profile_manifold

__all__ = [
    "profile_manifold",
    "estimate_intrinsic_dim",
    "ManifoldProfile",
    "select_anchors",
    "select_anchors_dpp",
    "select_anchors_kmeans",
    "detect_feature_modules",
    "solve_transfer_cut",
    "is_gpu_available",
]
