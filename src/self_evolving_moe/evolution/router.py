"""
Core evolutionary algorithm engine for MoE routing discovery.

This module implements the EvolvingMoERouter, the central component that
orchestrates the evolutionary process to discover optimal expert routing
topologies for Mixture of Experts models.
"""

import random
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
import copy
from pathlib import Path
import json

from ..routing.topology import TopologyGenome, RoutingParams
from ..experts.pool import ExpertPool
# from ..monitoring.metrics import EvolutionMetrics  # Temporarily disabled
# from ..utils.monitoring import EvolutionMetrics  # Temporarily disabled

# Temporary stub for EvolutionMetrics
class EvolutionMetrics:
    def __init__(self):
        self.metrics = {}
    
    def update_generation_metrics(self, stats):
        self.metrics.update(stats)
    
    def get_metrics(self):
        return self.metrics
from ..utils.exceptions import EvolutionError, TopologyError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm."""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_rate: float = 0.1
    tournament_size: int = 3
    selection_method: str = "tournament"  # tournament, roulette, rank
    
    # Objective weights
    accuracy_weight: float = 1.0
    latency_weight: float = -0.1  # Negative to minimize
    memory_weight: float = -0.05
    sparsity_weight: float = 0.1
    
    # Constraints
    max_active_experts: int = 4
    min_expert_usage: float = 0.05
    target_sparsity: float = 0.1


class FitnessFunction:
    """Base class for fitness evaluation."""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.evaluation_cache = {}
    
    def evaluate(
        self,
        topology: TopologyGenome,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Evaluate topology fitness on multiple objectives.
        
        Args:
            topology: Routing topology to evaluate
            model: MoE model to test with
            data_loader: Validation data
            device: Computation device
            
        Returns:
            Dictionary of objective scores
        """
        # Create cache key
        cache_key = hash((str(topology.routing_matrix), str(topology.routing_params)))
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        scores = {}
        
        # Apply topology to model
        model.set_routing_topology(topology)
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            total_latency = 0
            memory_usage = 0
            
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Measure latency
                start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
                end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
                
                if device == "cuda":
                    start_time.record()
                
                outputs = model(inputs)
                
                if device == "cuda":
                    end_time.record()
                    torch.cuda.synchronize()
                    batch_latency = start_time.elapsed_time(end_time)
                else:
                    batch_latency = 0  # Simplified for CPU
                
                total_latency += batch_latency
                
                # Calculate accuracy
                predictions = outputs.argmax(dim=-1)
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)
                
                # Memory measurement (simplified)
                if device == "cuda":
                    memory_usage = max(memory_usage, torch.cuda.memory_allocated(device))
                
                # Limit evaluation time for efficiency
                if batch_idx > 10:
                    break
        
        # Calculate metrics
        scores['accuracy'] = total_correct / max(total_samples, 1)
        scores['latency'] = total_latency / max(len(data_loader), 1)
        scores['memory'] = memory_usage / (1024 ** 3)  # GB
        scores['sparsity'] = self._calculate_sparsity(topology)
        scores['load_balance'] = self._calculate_load_balance(topology)
        scores['diversity'] = self._calculate_diversity(topology)
        
        # Cache result
        self.evaluation_cache[cache_key] = scores
        
        return scores
    
    def _calculate_sparsity(self, topology: TopologyGenome) -> float:
        """Calculate routing sparsity."""
        total_connections = topology.routing_matrix.sum().item()
        possible_connections = topology.routing_matrix.numel()
        return 1.0 - (total_connections / possible_connections)
    
    def _calculate_load_balance(self, topology: TopologyGenome) -> float:
        """Calculate expert load balance (higher is better)."""
        expert_loads = topology.routing_matrix.sum(dim=0)
        if expert_loads.sum() == 0:
            return 0.0
        expert_probs = expert_loads / expert_loads.sum()
        # Use entropy as load balance measure
        entropy = -(expert_probs * torch.log(expert_probs + 1e-8)).sum()
        max_entropy = np.log(topology.num_experts)
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_diversity(self, topology: TopologyGenome) -> float:
        """Calculate routing diversity."""
        # Measure how different the routing patterns are for different tokens
        routing_patterns = topology.routing_matrix.float()
        if routing_patterns.size(0) <= 1:
            return 0.0
        
        # Calculate pairwise distances between token routing patterns
        distances = []
        for i in range(routing_patterns.size(0)):
            for j in range(i + 1, routing_patterns.size(0)):
                dist = torch.norm(routing_patterns[i] - routing_patterns[j])
                distances.append(dist.item())
        
        return np.mean(distances) if distances else 0.0
    
    def compute_fitness(self, scores: Dict[str, float]) -> float:
        """Combine multiple objectives into single fitness score."""
        fitness = (
            self.config.accuracy_weight * scores.get('accuracy', 0.0) +
            self.config.latency_weight * scores.get('latency', 0.0) +
            self.config.memory_weight * scores.get('memory', 0.0) +
            self.config.sparsity_weight * scores.get('sparsity', 0.0)
        )
        return fitness


class EvolvingMoERouter:
    """
    Main evolutionary algorithm engine for discovering optimal MoE routing topologies.
    
    This class orchestrates the evolutionary process:
    1. Maintains population of routing topologies
    2. Evaluates fitness of each topology 
    3. Applies selection, crossover, and mutation
    4. Tracks evolution progress and best solutions
    """
    
    def __init__(
        self,
        num_experts: int,
        num_tokens: int = 512,
        config: Optional[EvolutionConfig] = None,
        fitness_function: Optional[FitnessFunction] = None,
        device: str = "cpu"
    ):
        """
        Initialize evolutionary MoE router.
        
        Args:
            num_experts: Number of available experts
            num_tokens: Sequence length for routing
            config: Evolution configuration
            fitness_function: Custom fitness evaluation function
            device: Computation device
        """
        self.num_experts = num_experts
        self.num_tokens = num_tokens
        self.config = config or EvolutionConfig()
        self.device = device
        
        # Fitness evaluation
        self.fitness_function = fitness_function or FitnessFunction(self.config)
        
        # Evolution state
        self.population: List[TopologyGenome] = []
        self.generation = 0
        self.fitness_history: List[List[float]] = []
        self.best_topology: Optional[TopologyGenome] = None
        self.best_fitness: float = float('-inf')
        
        # Metrics tracking
        self.metrics = EvolutionMetrics()
        
        # Initialize population
        self._initialize_population()
        
        logger.info(f"Initialized EvolvingMoERouter with {self.config.population_size} topologies")
    
    def _initialize_population(self):
        """Initialize random population of topologies."""
        self.population = []
        
        for _ in range(self.config.population_size):
            topology = TopologyGenome(
                num_tokens=self.num_tokens,
                num_experts=self.num_experts,
                sparsity=self.config.target_sparsity,
                device=self.device
            )
            self.population.append(topology)
        
        logger.info(f"Initialized population with {len(self.population)} topologies")
    
    def evolve_one_generation(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """
        Execute one generation of evolution.
        
        Args:
            model: MoE model for fitness evaluation
            data_loader: Validation data
            
        Returns:
            Generation statistics
        """
        logger.info(f"Starting generation {self.generation}")
        
        # Evaluate fitness for entire population
        fitness_scores = []
        detailed_scores = []
        
        for i, topology in enumerate(self.population):
            try:
                scores = self.fitness_function.evaluate(
                    topology, model, data_loader, self.device
                )
                fitness = self.fitness_function.compute_fitness(scores)
                fitness_scores.append(fitness)
                detailed_scores.append(scores)
                
                # Track best solution
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_topology = copy.deepcopy(topology)
                    logger.info(f"New best fitness: {fitness:.4f}")
                
            except Exception as e:
                logger.warning(f"Fitness evaluation failed for topology {i}: {e}")
                fitness_scores.append(float('-inf'))
                detailed_scores.append({})
        
        self.fitness_history.append(fitness_scores)
        
        # Create next generation
        new_population = self._create_next_generation(fitness_scores)
        self.population = new_population
        self.generation += 1
        
        # Calculate generation statistics
        stats = self._calculate_generation_stats(fitness_scores, detailed_scores)
        
        # Update metrics
        self.metrics.update_generation_metrics(stats)
        
        logger.info(f"Generation {self.generation-1} complete. Best: {max(fitness_scores):.4f}, Avg: {np.mean(fitness_scores):.4f}")
        
        return stats
    
    def _create_next_generation(self, fitness_scores: List[float]) -> List[TopologyGenome]:
        """Create next generation through selection, crossover, and mutation."""
        new_population = []
        
        # Elitism - keep best individuals
        elite_count = int(self.config.elitism_rate * self.config.population_size)
        if elite_count > 0:
            # Get indices of best individuals
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(copy.deepcopy(self.population[idx]))
        
        # Generate rest through selection, crossover, and mutation
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self._select_parent(fitness_scores)
            parent2 = self._select_parent(fitness_scores)
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = copy.deepcopy(parent1)
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child.mutate(self.config.mutation_rate)
            
            new_population.append(child)
        
        # Ensure exact population size
        return new_population[:self.config.population_size]
    
    def _select_parent(self, fitness_scores: List[float]) -> TopologyGenome:
        """Select parent using configured selection method."""
        if self.config.selection_method == "tournament":
            return self._tournament_selection(fitness_scores)
        elif self.config.selection_method == "roulette":
            return self._roulette_selection(fitness_scores)
        elif self.config.selection_method == "rank":
            return self._rank_selection(fitness_scores)
        else:
            raise ValueError(f"Unknown selection method: {self.config.selection_method}")
    
    def _tournament_selection(self, fitness_scores: List[float]) -> TopologyGenome:
        """Tournament selection."""
        tournament_indices = random.sample(
            range(len(self.population)),
            min(self.config.tournament_size, len(self.population))
        )
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return copy.deepcopy(self.population[best_idx])
    
    def _roulette_selection(self, fitness_scores: List[float]) -> TopologyGenome:
        """Roulette wheel selection."""
        # Handle negative fitness scores
        min_fitness = min(fitness_scores)
        adjusted_scores = [f - min_fitness + 1e-8 for f in fitness_scores]
        total_fitness = sum(adjusted_scores)
        
        if total_fitness <= 0:
            # Fallback to random selection
            return copy.deepcopy(random.choice(self.population))
        
        probabilities = [f / total_fitness for f in adjusted_scores]
        selected_idx = np.random.choice(len(self.population), p=probabilities)
        return copy.deepcopy(self.population[selected_idx])
    
    def _rank_selection(self, fitness_scores: List[float]) -> TopologyGenome:
        """Rank-based selection."""
        sorted_indices = np.argsort(fitness_scores)
        ranks = np.arange(1, len(sorted_indices) + 1)
        probabilities = ranks / ranks.sum()
        
        selected_rank_idx = np.random.choice(len(sorted_indices), p=probabilities)
        selected_idx = sorted_indices[selected_rank_idx]
        return copy.deepcopy(self.population[selected_idx])
    
    def _calculate_generation_stats(
        self,
        fitness_scores: List[float],
        detailed_scores: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Calculate statistics for the current generation."""
        valid_scores = [s for s in fitness_scores if s != float('-inf')]
        
        if not valid_scores:
            return {'error': 'No valid fitness scores'}
        
        stats = {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': max(valid_scores),
            'worst_fitness': min(valid_scores),
            'mean_fitness': np.mean(valid_scores),
            'std_fitness': np.std(valid_scores),
            'median_fitness': np.median(valid_scores)
        }
        
        # Add detailed objective statistics
        if detailed_scores:
            valid_detailed = [s for s in detailed_scores if s]
            if valid_detailed:
                for objective in ['accuracy', 'latency', 'memory', 'sparsity', 'load_balance', 'diversity']:
                    values = [s.get(objective, 0.0) for s in valid_detailed]
                    if values:
                        stats[f'{objective}_mean'] = np.mean(values)
                        stats[f'{objective}_std'] = np.std(values)
                        stats[f'{objective}_best'] = max(values) if objective == 'accuracy' else min(values)
        
        return stats
    
    def evolve(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        generations: Optional[int] = None
    ) -> TopologyGenome:
        """
        Run full evolution process.
        
        Args:
            model: MoE model for evaluation
            data_loader: Validation data
            generations: Number of generations (uses config default if None)
            
        Returns:
            Best discovered topology
        """
        target_generations = generations or self.config.generations
        
        logger.info(f"Starting evolution for {target_generations} generations")
        
        for gen in range(target_generations):
            stats = self.evolve_one_generation(model, data_loader)
            
            # Log progress periodically
            if gen % 10 == 0 or gen == target_generations - 1:
                logger.info(
                    f"Generation {gen}: "
                    f"Best={stats.get('best_fitness', 0):.4f}, "
                    f"Avg={stats.get('mean_fitness', 0):.4f}, "
                    f"Std={stats.get('std_fitness', 0):.4f}"
                )
        
        logger.info(f"Evolution complete. Best fitness: {self.best_fitness:.4f}")
        
        if self.best_topology is None:
            raise EvolutionError("Evolution failed to find valid topology")
        
        return self.best_topology
    
    def get_best_topology(self) -> Optional[TopologyGenome]:
        """Get the best topology discovered so far."""
        return copy.deepcopy(self.best_topology) if self.best_topology else None
    
    def get_population_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Calculate Hamming distance between routing matrices
                diff = (self.population[i].routing_matrix != self.population[j].routing_matrix).float()
                distance = diff.mean().item()
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def save_evolution_state(self, filepath: Union[str, Path]) -> None:
        """Save complete evolution state."""
        filepath = Path(filepath)
        
        state = {
            'config': self.config.__dict__,
            'generation': self.generation,
            'fitness_history': self.fitness_history,
            'best_fitness': self.best_fitness,
            'best_topology': self.best_topology.to_dict() if self.best_topology else None,
            'population': [topology.to_dict() for topology in self.population],
            'num_experts': self.num_experts,
            'num_tokens': self.num_tokens
        }
        
        # Convert numpy types to Python natives for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(item) for item in data]
            else:
                return convert_numpy(data)
        
        clean_state = clean_for_json(state)
        
        with open(filepath, 'w') as f:
            json.dump(clean_state, f, indent=2)
        
        logger.info(f"Saved evolution state to {filepath}")
    
    def load_evolution_state(self, filepath: Union[str, Path]) -> None:
        """Load evolution state from file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore config
        self.config = EvolutionConfig(**state['config'])
        self.generation = state['generation']
        self.fitness_history = state['fitness_history']
        self.best_fitness = state['best_fitness']
        
        # Restore best topology
        if state['best_topology']:
            self.best_topology = TopologyGenome.from_dict(state['best_topology'])
        
        # Restore population
        self.population = [
            TopologyGenome.from_dict(topo_dict)
            for topo_dict in state['population']
        ]
        
        self.num_experts = state['num_experts']
        self.num_tokens = state['num_tokens']
        
        logger.info(f"Loaded evolution state from {filepath}")
    
    def get_evolution_metrics(self) -> Dict[str, Any]:
        """Get comprehensive evolution metrics."""
        return self.metrics.get_metrics()
    
    def reset(self):
        """Reset evolution state."""
        self.population = []
        self.generation = 0
        self.fitness_history = []
        self.best_topology = None
        self.best_fitness = float('-inf')
        self.metrics = EvolutionMetrics()
        self.fitness_function.evaluation_cache = {}
        
        self._initialize_population()
        logger.info("Reset evolution state")
