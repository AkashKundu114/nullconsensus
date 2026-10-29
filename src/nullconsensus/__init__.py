"""
Permutation-Null Calibrated Consensus Clustering (nullconsensus).
"""

from nullconsensus.api import NullConsensus
from nullconsensus.methods.hurdlegate import HurdleGateFeatureSelector
from nullconsensus.methods.nullgate import (
    NullGateFeatureSelector,
    compute_bc_vectorized,
    compute_sarle_bc,
)

__version__ = "0.1.0"
__all__ = [
    "HurdleGateFeatureSelector",
    "NullConsensus",
    "NullGateFeatureSelector",
    "compute_bc_vectorized",
    "compute_sarle_bc",
]
