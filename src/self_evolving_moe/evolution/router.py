"""
Evolutionary MoE router implementation.

This module implements the main EvolvingMoERouter class that coordinates
the evolutionary optimization of routing topologies for MoE models.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import random
import numpy as np
import copy
from dataclasses import dataclass
from pathlib import Path
import time
import logging

from ..routing.topology import TopologyGenome
from ..experts.pool import ExpertPool


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization."""
    population_size: int = 50
    generations: int = 500
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_ratio: float = 0.1
    tournament_size: int = 3
    selection_method: str = "tournament"  # "tournament", "roulette", "rank"
    
    # Multi-objective weights
    accuracy_weight: float = 1.0
    latency_weight: float = -0.1  # Negative to minimize
    memory_weight: float = -0.05
    sparsity_weight: float = 0.1
    diversity_weight: float = 0.05
    
    # Constraints
    max_active_experts: int = 8
    min_expert_usage: float = 0.1
    target_sparsity: float = 0.8
    
    # Convergence criteria
    patience: int = 50
    min_improvement: float = 1e-4
    
    # Hardware constraints
    memory_budget: Optional[int] = None  # bytes
    latency_budget: Optional[float] = None  # ms


class FitnessEvaluator:
    """Multi-objective fitness evaluation for routing topologies."""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.evaluation_cache: Dict[str, float] = {}
        
    def evaluate(
        self,
        topology: TopologyGenome,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader,
        device: str = "cpu"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate fitness of a routing topology.
        
        Args:
            topology: Routing topology to evaluate
            model: MoE model to evaluate with
            train_data: Training data loader
            val_data: Validation data loader
            device: Device for evaluation
            
        Returns:
            Tuple of (fitness_score, detailed_metrics)
        """
        # Create cache key
        cache_key = self._get_cache_key(topology)
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key], {}
        
        metrics = {}
        
        # Set topology in model
        if hasattr(model, 'set_routing_topology'):
            model.set_routing_topology(topology)
        
        model.eval()
        
        # Evaluate accuracy
        accuracy = self._evaluate_accuracy(model, val_data, device)
        metrics['accuracy'] = accuracy
        
        # Evaluate efficiency metrics
        latency = self._measure_latency(model, val_data, device)
        metrics['latency'] = latency
        
        memory_usage = self._estimate_memory_usage(topology, model)
        metrics['memory'] = memory_usage
        
        # Evaluate routing metrics
        sparsity = topology.compute_sparsity()
        metrics['sparsity'] = sparsity
        
        diversity = self._compute_routing_diversity(topology)
        metrics['diversity'] = diversity
        
        load_balance = self._compute_load_balance(topology, model, val_data, device)
        metrics['load_balance'] = load_balance
        
        # Constraint violations
        constraint_penalty = self._compute_constraint_penalty(topology, metrics)
        metrics['constraint_penalty'] = constraint_penalty
        
        # Compute weighted fitness
        fitness = (
            self.config.accuracy_weight * accuracy +
            self.config.latency_weight * latency +
            self.config.memory_weight * (memory_usage / 1e9) +  # Normalize to GB
            self.config.sparsity_weight * sparsity +
            self.config.diversity_weight * diversity -
            constraint_penalty
        )
        
        metrics['fitness'] = fitness
        
        # Cache result
        self.evaluation_cache[cache_key] = fitness
        
        return fitness, metrics
    
    def _get_cache_key(self, topology: TopologyGenome) -> str:
        """Generate cache key for topology."""
        # Simple hash based on routing matrix
        matrix_hash = hash(topology.routing_matrix.cpu().numpy().tobytes())
        params_hash = hash((
            topology.routing_params.temperature,
            topology.routing_params.top_k,
            topology.routing_params.load_balancing_weight
        ))
        return f"{matrix_hash}_{params_hash}"
    
    def _evaluate_accuracy(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: str
    ) -> float:
        """Evaluate model accuracy on validation data."""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if batch_idx >= 10:  # Limit evaluation for speed
                    break
                    
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                if outputs.dim() > 2:  # Handle sequence outputs
                    outputs = outputs.view(-1, outputs.size(-1))
                    if targets.dim() > 1:
                        targets = targets.view(-1)
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def _measure_latency(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: str,
        num_trials: int = 5
    ) -> float:
        """Measure model latency."""
        model.eval()
        latencies = []
        
        # Get sample batch
        sample_batch = next(iter(data_loader))
        inputs = sample_batch[0][:4].to(device)  # Small batch for speed
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(inputs)
        
        # Measure
        with torch.no_grad():
            for _ in range(num_trials):
                torch.cuda.synchronize() if device.startswith('cuda') else None
                start_time = time.time()
                
                _ = model(inputs)
                
                torch.cuda.synchronize() if device.startswith('cuda') else None
                end_time = time.time()
                
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        return np.mean(latencies)
    
    def _estimate_memory_usage(self, topology: TopologyGenome, model: nn.Module) -> float:
        """Estimate memory usage of topology."""
        # Count active connections
        active_connections = topology.routing_matrix.sum().item()
        total_possible = topology.routing_matrix.numel()
        
        # Estimate memory based on sparsity and model size
        if hasattr(model, 'expert_pool'):
            base_memory = model.expert_pool.get_memory_usage()['total']
            # Adjust based on routing sparsity
            routing_memory = base_memory * (active_connections / total_possible)
        else:
            # Rough estimate
            routing_memory = active_connections * 1000  # Rough estimate
        
        return routing_memory
    
    def _compute_routing_diversity(self, topology: TopologyGenome) -> float:
        """Compute diversity of routing patterns."""
        routing_matrix = topology.routing_matrix.cpu().numpy()
        
        # Compute entropy of expert usage across tokens
        expert_usage = routing_matrix.sum(axis=0)
        expert_usage = expert_usage / expert_usage.sum()
        
        # Avoid log(0)
        expert_usage = expert_usage + 1e-8
        entropy = -np.sum(expert_usage * np.log(expert_usage))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(expert_usage))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _compute_load_balance(
        self,
        topology: TopologyGenome,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: str
    ) -> float:
        """Compute load balance across experts."""
        # Simple version: just look at routing matrix balance
        routing_matrix = topology.routing_matrix.cpu().numpy()
        expert_loads = routing_matrix.sum(axis=0)
        
        if expert_loads.sum() == 0:
            return 0.0
        
        # Compute coefficient of variation (lower is better)
        mean_load = expert_loads.mean()
        if mean_load == 0:
            return 0.0
        
        std_load = expert_loads.std()
        cv = std_load / mean_load
        
        # Convert to 0-1 scale (higher is better)
        load_balance_score = 1.0 / (1.0 + cv)
        
        return load_balance_score
    
    def _compute_constraint_penalty(
        self,
        topology: TopologyGenome,
        metrics: Dict[str, float]
    ) -> float:
        """Compute penalty for constraint violations."""
        penalty = 0.0
        
        # Active experts constraint
        active_experts = (topology.routing_matrix.sum(dim=0) > 0).sum().item()
        if active_experts > self.config.max_active_experts:
            penalty += (active_experts - self.config.max_active_experts) * 0.1
        
        # Sparsity constraint
        sparsity = metrics.get('sparsity', 0.0)
        if sparsity < self.config.target_sparsity:
            penalty += (self.config.target_sparsity - sparsity) * 0.5
        
        # Memory constraint
        if self.config.memory_budget is not None:
            memory_usage = metrics.get('memory', 0.0)
            if memory_usage > self.config.memory_budget:
                penalty += (memory_usage - self.config.memory_budget) / 1e9 * 2.0
        
        # Latency constraint
        if self.config.latency_budget is not None:
            latency = metrics.get('latency', 0.0)
            if latency > self.config.latency_budget:
                penalty += (latency - self.config.latency_budget) * 0.01
        
        return penalty


class EvolvingMoERouter:
    """
    Main class for evolutionary optimization of MoE routing topologies.
    
    Implements genetic algorithms to discover optimal sparse routing patterns
    that balance task performance with computational efficiency.
    """
    
    def __init__(
        self,
        expert_pool: ExpertPool,
        config: Optional[EvolutionConfig] = None,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        device: str = "cpu"
    ):
        """
        Initialize evolutionary MoE router.
        
        Args:
            expert_pool: Pool of expert networks
            config: Evolution configuration
            fitness_evaluator: Custom fitness evaluator
            device: Device for computations
        """
        self.expert_pool = expert_pool
        self.config = config or EvolutionConfig()
        self.device = device
        
        # Initialize fitness evaluator
        self.fitness_evaluator = fitness_evaluator or FitnessEvaluator(self.config)
        
        # Evolution state
        self.population: List[TopologyGenome] = []
        self.fitness_scores: List[float] = []
        self.fitness_history: List[Dict[str, Any]] = []
        self.generation = 0
        self.best_topology: Optional[TopologyGenome] = None
        self.best_fitness = float('-inf')
        
        # Convergence tracking
        self.no_improvement_count = 0
        self.last_best_fitness = float('-inf')
        
        # Setup logging (before population initialization)
        self.logger = logging.getLogger(__name__)
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize random population of routing topologies."""
        self.population = []
        
        # Determine topology dimensions
        # For simplicity, assume token count equals hidden dimension
        num_tokens = self.expert_pool.expert_dim
        num_experts = self.expert_pool.num_experts
        
        for _ in range(self.config.population_size):
            # Vary sparsity across population
            sparsity = random.uniform(0.5, 0.95)
            
            topology = TopologyGenome(
                num_tokens=num_tokens,
                num_experts=num_experts,
                sparsity=sparsity,
                device=self.device
            )
            
            self.population.append(topology)
        
        self.logger.info(f"Initialized population of {len(self.population)} topologies")
    
    def evolve(
        self,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader,
        generations: Optional[int] = None
    ) -> TopologyGenome:
        """
        Run evolutionary optimization.
        
        Args:
            model: MoE model to optimize routing for
            train_data: Training data loader
            val_data: Validation data loader
            generations: Number of generations (if None, uses config)
            
        Returns:
            Best topology found
        """
        if generations is None:
            generations = self.config.generations
        
        self.logger.info(f"Starting evolution for {generations} generations")
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate population
            self._evaluate_population(model, train_data, val_data)
            
            # Check for improvement
            current_best = max(self.fitness_scores)
            if current_best > self.best_fitness:
                self.best_fitness = current_best
                best_idx = self.fitness_scores.index(current_best)
                self.best_topology = copy.deepcopy(self.population[best_idx])
                self.no_improvement_count = 0
                
                self.logger.info(f"Generation {gen}: New best fitness = {current_best:.6f}")
            else:
                self.no_improvement_count += 1
            
            # Check convergence
            if self.no_improvement_count >= self.config.patience:
                self.logger.info(f"Converged after {gen + 1} generations")
                break
            
            # Create next generation
            self._create_next_generation()
            
            # Log progress
            if gen % 10 == 0:
                avg_fitness = np.mean(self.fitness_scores)
                std_fitness = np.std(self.fitness_scores)
                self.logger.info(
                    f"Generation {gen}: avg={avg_fitness:.4f}, std={std_fitness:.4f}, "
                    f"best={current_best:.4f}"
                )
        
        self.logger.info(f"Evolution completed. Best fitness: {self.best_fitness:.6f}")
        return self.best_topology
    
    def _evaluate_population(
        self,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader
    ):
        """Evaluate fitness of entire population."""
        self.fitness_scores = []
        generation_metrics = []
        
        for i, topology in enumerate(self.population):
            fitness, metrics = self.fitness_evaluator.evaluate(
                topology, model, train_data, val_data, self.device
            )
            self.fitness_scores.append(fitness)
            generation_metrics.append(metrics)
        
        # Store generation statistics
        self.fitness_history.append({
            'generation': self.generation,
            'fitness_scores': self.fitness_scores.copy(),
            'metrics': generation_metrics,
            'best_fitness': max(self.fitness_scores),
            'avg_fitness': np.mean(self.fitness_scores),
            'std_fitness': np.std(self.fitness_scores)
        })
    
    def _create_next_generation(self):
        """Create next generation through selection, crossover, and mutation."""
        new_population = []
        
        # Elitism - keep best individuals
        num_elite = max(1, int(self.config.population_size * self.config.elitism_ratio))
        elite_indices = np.argsort(self.fitness_scores)[-num_elite:]
        
        for idx in elite_indices:
            elite = copy.deepcopy(self.population[idx])
            new_population.append(elite)
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = copy.deepcopy(parent1)
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child = child.mutate(self.config.mutation_rate)
            
            new_population.append(child)
        
        self.population = new_population[:self.config.population_size]
    
    def _select_parent(self) -> TopologyGenome:
        """Select parent for reproduction."""
        if self.config.selection_method == "tournament":
            return self._tournament_selection()
        elif self.config.selection_method == "roulette":
            return self._roulette_selection()
        elif self.config.selection_method == "rank":
            return self._rank_selection()
        else:
            raise ValueError(f"Unknown selection method: {self.config.selection_method}")
    
    def _tournament_selection(self) -> TopologyGenome:
        """Tournament selection."""
        tournament_size = min(self.config.tournament_size, len(self.population))
        tournament_indices = random.sample(range(len(self.population)), tournament_size)
        
        best_idx = max(tournament_indices, key=lambda i: self.fitness_scores[i])
        return copy.deepcopy(self.population[best_idx])
    
    def _roulette_selection(self) -> TopologyGenome:
        """Roulette wheel selection."""
        # Ensure all fitness scores are positive
        min_fitness = min(self.fitness_scores)
        adjusted_scores = [score - min_fitness + 1e-8 for score in self.fitness_scores]
        
        total_fitness = sum(adjusted_scores)
        if total_fitness == 0:
            return random.choice(self.population)
        
        # Spin the wheel
        wheel_pos = random.uniform(0, total_fitness)
        cumulative = 0
        
        for i, score in enumerate(adjusted_scores):
            cumulative += score
            if cumulative >= wheel_pos:
                return copy.deepcopy(self.population[i])
        
        # Fallback
        return copy.deepcopy(self.population[-1])
    
    def _rank_selection(self) -> TopologyGenome:
        """Rank-based selection."""
        ranked_indices = np.argsort(self.fitness_scores)
        
        # Linear ranking probabilities
        n = len(ranked_indices)
        probabilities = [(i + 1) / (n * (n + 1) / 2) for i in range(n)]
        
        # Select based on rank probabilities
        selected_rank = np.random.choice(n, p=probabilities)
        selected_idx = ranked_indices[selected_rank]
        
        return copy.deepcopy(self.population[selected_idx])
    
    def get_best_topology(self) -> Optional[TopologyGenome]:
        """Get the best topology found so far."""
        return self.best_topology
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get detailed evolution statistics."""
        if not self.fitness_history:
            return {}
        
        return {
            'generations_run': len(self.fitness_history),
            'best_fitness': self.best_fitness,
            'convergence_generation': len(self.fitness_history) - self.no_improvement_count,
            'fitness_history': [gen['best_fitness'] for gen in self.fitness_history],
            'avg_fitness_history': [gen['avg_fitness'] for gen in self.fitness_history],
            'final_population_diversity': np.std(self.fitness_scores) if self.fitness_scores else 0,
            'best_topology_summary': self.best_topology.get_topology_summary() if self.best_topology else None
        }
    
    def save_evolution_state(self, filepath: str):
        """Save complete evolution state."""
        state = {
            'config': self.config,
            'generation': self.generation,
            'population': [topology.get_topology_summary() for topology in self.population],
            'fitness_scores': self.fitness_scores,
            'fitness_history': self.fitness_history,
            'best_fitness': self.best_fitness,
            'best_topology': self.best_topology.get_topology_summary() if self.best_topology else None,
            'no_improvement_count': self.no_improvement_count
        }
        
        torch.save(state, filepath)
        
        # Save best topology separately
        if self.best_topology:
            best_path = str(Path(filepath).with_suffix('.best.pt'))
            self.best_topology.save_topology(best_path)
    
    @classmethod
    def load_evolution_state(
        cls,
        filepath: str,
        expert_pool: ExpertPool,
        device: str = "cpu"
    ) -> 'EvolvingMoERouter':
        """Load evolution state from file."""
        state = torch.load(filepath, map_location=device, weights_only=False)
        
        # Create router
        router = cls(
            expert_pool=expert_pool,
            config=state['config'],
            device=device
        )
        
        # Restore state
        router.generation = state['generation']
        router.fitness_scores = state['fitness_scores']
        router.fitness_history = state['fitness_history']
        router.best_fitness = state['best_fitness']
        router.no_improvement_count = state['no_improvement_count']
        
        # Load best topology
        best_path = str(Path(filepath).with_suffix('.best.pt'))
        if Path(best_path).exists():
            router.best_topology = TopologyGenome.load_topology(best_path, device)
        
        return router
    
    def continue_evolution(
        self,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader,
        additional_generations: int
    ) -> TopologyGenome:
        """Continue evolution from current state."""
        return self.evolve(model, train_data, val_data, additional_generations)
    
    def adaptive_mutation_rate(self):
        """Adaptively adjust mutation rate based on diversity."""
        if len(self.fitness_scores) < 2:
            return
        
        diversity = np.std(self.fitness_scores)
        avg_fitness = np.mean(self.fitness_scores)
        
        # If diversity is low, increase mutation rate
        if diversity < 0.1 * abs(avg_fitness):
            self.config.mutation_rate = min(0.5, self.config.mutation_rate * 1.1)
        else:
            self.config.mutation_rate = max(0.01, self.config.mutation_rate * 0.95)
    
    def multi_objective_selection(self) -> List[TopologyGenome]:
        """Perform multi-objective selection (Pareto front)."""
        if not self.fitness_history:
            return []
        
        latest_metrics = self.fitness_history[-1]['metrics']
        
        # Extract objectives (higher is better for all)
        objectives = []
        for metrics in latest_metrics:
            obj = [
                metrics.get('accuracy', 0),
                -metrics.get('latency', 0),  # Negative because we want to minimize
                -metrics.get('memory', 0),
                metrics.get('sparsity', 0),
                metrics.get('diversity', 0)
            ]
            objectives.append(obj)
        
        # Find Pareto front
        pareto_indices = []
        for i, obj_i in enumerate(objectives):
            is_dominated = False
            for j, obj_j in enumerate(objectives):
                if i != j and self._dominates(obj_j, obj_i):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_indices.append(i)
        
        return [self.population[i] for i in pareto_indices]
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2 (Pareto dominance)."""
        return all(x >= y for x, y in zip(obj1, obj2)) and any(x > y for x, y in zip(obj1, obj2))