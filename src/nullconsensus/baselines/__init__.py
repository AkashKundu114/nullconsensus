"""
Benchmark baselines for nullconsensus.
"""

from nullconsensus.baselines.m3c import M3CBaseline
from nullconsensus.baselines.gac_random_gating import GACRandomGatingBaseline

__all__ = [
    "M3CBaseline",
    "GACRandomGatingBaseline",
]
