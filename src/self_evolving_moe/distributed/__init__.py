"""
Distributed computing support for Self-Evolving MoE-Router.

This module provides distributed evolution, parallel evaluation,
and multi-node training capabilities for large-scale deployments.
"""

from .parallel_evolution import DistributedEvolver, ParallelFitnessEvaluator
from .model_parallel import ModelParallelMoE, ExpertParallelRouter
from .communication import EvolutionCommunicator, TopologyBroadcaster
from .load_balancing import DynamicLoadBalancer, WorkerPool

__all__ = [
    "DistributedEvolver",
    "ParallelFitnessEvaluator", 
    "ModelParallelMoE",
    "ExpertParallelRouter",
    "EvolutionCommunicator",
    "TopologyBroadcaster",
    "DynamicLoadBalancer",
    "WorkerPool"
]