"""
Permutation-Null Calibrated Consensus Clustering (nullconsensus).
"""

from nullconsensus.api import NullConsensus
from nullconsensus.methods.nullgate import NullGateFeatureSelector, compute_sarle_bc, compute_bc_vectorized
from nullconsensus.methods.hurdlegate import HurdleGateFeatureSelector

__version__ = "0.1.0"
__all__ = [
    "NullConsensus",
    "NullGateFeatureSelector",
    "HurdleGateFeatureSelector",
    "compute_sarle_bc",
    "compute_bc_vectorized"
]
