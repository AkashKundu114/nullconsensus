"""
Permutation-Null Calibrated and Self-Tuning Consensus Clustering (nullconsensus).
"""

from nullconsensus.api import AutoNullConsensus, NullConsensus
import nullconsensus.integrations as tl
from nullconsensus.methods.hurdlegate import HurdleGateFeatureSelector
from nullconsensus.methods.nullgate import (
    NullGateFeatureSelector,
    compute_bc_vectorized,
    compute_sarle_bc,
)

__version__ = "0.2.0"
__all__ = [
    "AutoNullConsensus",
    "NullConsensus",
    "HurdleGateFeatureSelector",
    "NullGateFeatureSelector",
    "compute_bc_vectorized",
    "compute_sarle_bc",
    "tl",
]
