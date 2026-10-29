"""
Benchmark baselines for nullconsensus.
"""

from nullconsensus.baselines.gac_random_gating import GACRandomGatingBaseline
from nullconsensus.baselines.m3c import M3CBaseline

__all__ = [
    "GACRandomGatingBaseline",
    "M3CBaseline",
]
