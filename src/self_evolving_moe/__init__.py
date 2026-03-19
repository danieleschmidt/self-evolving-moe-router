"""
self-evolving-moe-router
========================
Neuroevolution framework for discovering sparse expert routing topologies
in Mixture-of-Experts (MoE) language models.

Public API
----------
MoERouter          – sparse router controlled by a binary connectivity mask
SlimmableExpert    – MLP expert that runs at 100 / 75 / 50 % width
EvolutionarySearch – evolutionary loop: mutate → score → select
"""

__version__ = "0.2.0"
__author__ = "Daniel Schmidt"

from .router import MoERouter
from .experts import SlimmableExpert
from .evolution import EvolutionarySearch, Individual

__all__ = [
    "MoERouter",
    "SlimmableExpert",
    "EvolutionarySearch",
    "Individual",
]
