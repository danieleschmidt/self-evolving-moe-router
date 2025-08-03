"""
Self-Evolving MoE-Router: Evolutionary Discovery of Sparse Expert Routing

A neuro-evolution framework for discovering optimal routing topologies
in Mixture of Experts models through evolutionary algorithms.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

from .evolution import EvolvingMoERouter
from .experts import ExpertPool, SlimmableMoE
from .routing import TopologyGenome
from .data import TopologyRepository, ExperimentRepository, EvolutionCache, ModelStorage

__all__ = [
    "EvolvingMoERouter",
    "ExpertPool", 
    "SlimmableMoE",
    "TopologyGenome",
    "TopologyRepository",
    "ExperimentRepository", 
    "EvolutionCache",
    "ModelStorage",
]