"""
Multi-objective fitness evaluation for topology optimization.

This module implements sophisticated fitness functions that optimize
multiple objectives simultaneously, including accuracy, efficiency,
diversity, and robustness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import math

from ..routing.topology import TopologyGenome
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    MEMORY = "memory"
    SPARSITY = "sparsity"
    LOAD_BALANCE = "load_balance"
    DIVERSITY = "diversity"
    ROBUSTNESS = "robustness"
    NOVELTY = "novelty"
    STABILITY = "stability"


@dataclass
class ObjectiveConfig:
    """Configuration for individual objectives."""
    name: str
    weight: float = 1.0
    target: Optional[float] = None
    minimize: bool = False
    tolerance: float = 0.1
    adaptive_weight: bool = False
    constraint: bool = False
    constraint_threshold: Optional[float] = None


@dataclass
class FitnessConfig:
    """Configuration for multi-objective fitness evaluation."""
    objectives: List[ObjectiveConfig]
    aggregation_method: str = "weighted_sum"  # weighted_sum, pareto, lexicographic
    constraint_penalty: float = 1000.0
    normalization_method: str = "min_max"  # min_max, z_score, rank
    
    # Performance evaluation settings
    max_eval_batches: int = 10
    accuracy_samples: int = 1000
    latency_trials: int = 5
    
    # Advanced settings
    use_pareto_dominance: bool = False
    diversity_window: int = 50
    novelty_archive_size: int = 100
    stability_generations: int = 5


class MultiObjectiveFitnessEvaluator:
    """Advanced fitness evaluator with multiple objectives and constraints."""
    
    def __init__(self, config: FitnessConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # Fitness history for normalization and analysis
        self.fitness_history: Dict[str, List[float]] = {obj.name: [] for obj in config.objectives}
        self.pareto_archive: List[Dict[str, float]] = []
        self.novelty_archive: List[TopologyGenome] = []
        
        # Evaluation cache for efficiency
        self.evaluation_cache: Dict[str, Dict[str, float]] = {}
        
        # Adaptive weights
        self.adaptive_weights: Dict[str, float] = {obj.name: obj.weight for obj in config.objectives}
        
        logger.info(f"Initialized multi-objective fitness with {len(config.objectives)} objectives")
        
    def evaluate(
        self,
        topology: TopologyGenome,
        model: nn.Module,
        data_loader,
        generation: int = 0,
        population: Optional[List[TopologyGenome]] = None
    ) -> Dict[str, float]:
        """
        Comprehensive fitness evaluation across multiple objectives.
        
        Args:
            topology: Topology to evaluate
            model: MoE model
            data_loader: Evaluation data
            generation: Current generation number
            population: Current population (for diversity metrics)
            
        Returns:
            Dictionary of objective scores and combined fitness
        """
        # Create cache key
        cache_key = self._get_cache_key(topology)
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        scores = {}
        
        # Apply topology to model
        model.set_routing_topology(topology)
        model.to(self.device)
        model.eval()
        
        # Evaluate each objective
        with torch.no_grad():
            for obj_config in self.config.objectives:
                try:
                    if obj_config.name == ObjectiveType.ACCURACY.value:
                        scores[obj_config.name] = self._evaluate_accuracy(model, data_loader)
                    elif obj_config.name == ObjectiveType.LATENCY.value:
                        scores[obj_config.name] = self._evaluate_latency(model, data_loader)
                    elif obj_config.name == ObjectiveType.MEMORY.value:
                        scores[obj_config.name] = self._evaluate_memory(model, topology)
                    elif obj_config.name == ObjectiveType.SPARSITY.value:
                        scores[obj_config.name] = self._evaluate_sparsity(topology)
                    elif obj_config.name == ObjectiveType.LOAD_BALANCE.value:
                        scores[obj_config.name] = self._evaluate_load_balance(topology)
                    elif obj_config.name == ObjectiveType.DIVERSITY.value:
                        scores[obj_config.name] = self._evaluate_diversity(topology, population)
                    elif obj_config.name == ObjectiveType.ROBUSTNESS.value:
                        scores[obj_config.name] = self._evaluate_robustness(model, data_loader, topology)
                    elif obj_config.name == ObjectiveType.NOVELTY.value:
                        scores[obj_config.name] = self._evaluate_novelty(topology)
                    elif obj_config.name == ObjectiveType.STABILITY.value:
                        scores[obj_config.name] = self._evaluate_stability(topology, generation)
                    else:
                        logger.warning(f"Unknown objective: {obj_config.name}")
                        scores[obj_config.name] = 0.0
                        
                except Exception as e:
                    logger.warning(f"Evaluation failed for {obj_config.name}: {e}")
                    scores[obj_config.name] = 0.0 if not obj_config.minimize else float('inf')
        
        # Update fitness history
        for obj_name, score in scores.items():
            if obj_name in self.fitness_history:
                self.fitness_history[obj_name].append(score)
                # Keep only recent history
                if len(self.fitness_history[obj_name]) > 1000:
                    self.fitness_history[obj_name] = self.fitness_history[obj_name][-500:]
        
        # Compute combined fitness
        combined_fitness = self._aggregate_objectives(scores, generation)
        scores['combined_fitness'] = combined_fitness
        
        # Cache result
        self.evaluation_cache[cache_key] = scores
        
        # Update Pareto archive if using Pareto optimization
        if self.config.use_pareto_dominance:
            self._update_pareto_archive(scores)
        
        return scores
    
    def _get_cache_key(self, topology: TopologyGenome) -> str:
        """Generate cache key for topology."""
        matrix_hash = hash(topology.routing_matrix.data.tobytes())
        params_hash = hash((
            topology.routing_params.temperature,
            topology.routing_params.top_k,
            topology.routing_params.load_balancing_weight,
            topology.routing_params.diversity_weight
        ))
        return f"{matrix_hash}_{params_hash}"
    
    # OBJECTIVE EVALUATION METHODS
    
    def _evaluate_accuracy(self, model: nn.Module, data_loader) -> float:
        """Evaluate model accuracy on the dataset."""
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_idx >= self.config.max_eval_batches:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
            
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            
            if total_samples >= self.config.accuracy_samples:
                break
        
        return total_correct / max(total_samples, 1)
    
    def _evaluate_latency(self, model: nn.Module, data_loader) -> float:
        """Evaluate model inference latency."""
        latencies = []
        
        # Get a representative batch
        for inputs, _ in data_loader:
            inputs = inputs.to(self.device)
            break
        
        # Warmup
        for _ in range(2):
            _ = model(inputs)
        
        # Measure latency
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        for _ in range(self.config.latency_trials):
            start_time = time.time()
            
            if self.device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                
            _ = model(inputs)
            
            if self.device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                latency = start_event.elapsed_time(end_event)  # milliseconds
            else:
                latency = (time.time() - start_time) * 1000  # convert to ms
                
            latencies.append(latency)
        
        return np.mean(latencies)
    
    def _evaluate_memory(self, model: nn.Module, topology: TopologyGenome) -> float:
        """Evaluate memory usage."""
        if self.device == "cuda":
            memory_before = torch.cuda.memory_allocated(self.device)
            
            # Dummy forward pass to measure peak memory
            dummy_input = torch.randn(1, 64, model.expert_pool.expert_config.hidden_dim, device=self.device)
            _ = model(dummy_input)
            
            memory_after = torch.cuda.max_memory_allocated(self.device)
            memory_usage = (memory_after - memory_before) / (1024 ** 2)  # MB
            
            torch.cuda.reset_peak_memory_stats(self.device)
            return memory_usage
        else:
            # Estimate memory usage based on active parameters
            active_connections = topology.routing_matrix.sum().item()
            total_connections = topology.routing_matrix.numel()
            estimated_memory = (active_connections / total_connections) * 100  # Rough estimate in MB
            return estimated_memory
    
    def _evaluate_sparsity(self, topology: TopologyGenome) -> float:
        """Evaluate routing sparsity."""
        return topology.compute_sparsity()
    
    def _evaluate_load_balance(self, topology: TopologyGenome) -> float:
        """Evaluate expert load balance."""
        expert_loads = topology.routing_matrix.sum(dim=0).float()
        if expert_loads.sum() == 0:
            return 0.0
            
        expert_probs = expert_loads / expert_loads.sum()
        entropy = -(expert_probs * torch.log(expert_probs + 1e-8)).sum()
        max_entropy = math.log(topology.num_experts)
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _evaluate_diversity(self, topology: TopologyGenome, population: Optional[List[TopologyGenome]]) -> float:
        """Evaluate routing pattern diversity."""
        if population is None or len(population) < 2:
            return 0.0
        
        # Intra-topology diversity (diversity within the topology)
        routing_patterns = topology.routing_matrix.float()
        if routing_patterns.size(0) <= 1:
            intra_diversity = 0.0
        else:
            # Calculate pairwise distances between token routing patterns
            distances = []
            for i in range(routing_patterns.size(0)):
                for j in range(i + 1, routing_patterns.size(0)):
                    dist = torch.norm(routing_patterns[i] - routing_patterns[j])
                    distances.append(dist.item())
            intra_diversity = np.mean(distances) if distances else 0.0
        
        # Inter-topology diversity (diversity relative to population)
        inter_diversity = 0.0
        for other_topology in population:
            if other_topology is not topology:
                diff = (topology.routing_matrix != other_topology.routing_matrix).float()
                distance = diff.mean().item()
                inter_diversity += distance
        
        inter_diversity /= max(len(population) - 1, 1)
        
        # Combine both diversity measures
        return (intra_diversity + inter_diversity) / 2
    
    def _evaluate_robustness(self, model: nn.Module, data_loader, topology: TopologyGenome) -> float:
        """Evaluate model robustness to input perturbations."""
        total_robustness = 0.0
        num_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_idx >= min(3, self.config.max_eval_batches):  # Limited for efficiency
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Original predictions
            original_outputs = model(inputs)
            original_preds = original_outputs.argmax(dim=-1)
            
            # Add small perturbations
            noise_std = 0.01 * inputs.std()
            perturbed_inputs = inputs + torch.randn_like(inputs) * noise_std
            perturbed_outputs = model(perturbed_inputs)
            perturbed_preds = perturbed_outputs.argmax(dim=-1)
            
            # Compute robustness as prediction consistency
            consistency = (original_preds == perturbed_preds).float().mean().item()
            total_robustness += consistency
            num_samples += 1
        
        return total_robustness / max(num_samples, 1)
    
    def _evaluate_novelty(self, topology: TopologyGenome) -> float:
        """Evaluate novelty compared to archive of previous topologies."""
        if not self.novelty_archive:
            # First topology is maximally novel
            self.novelty_archive.append(topology)
            return 1.0
        
        # Compute distance to k-nearest neighbors in archive
        distances = []
        for archived_topology in self.novelty_archive:
            diff = (topology.routing_matrix != archived_topology.routing_matrix).float()
            distance = diff.mean().item()
            distances.append(distance)
        
        # Average distance to k nearest neighbors (novelty)
        k = min(5, len(distances))
        distances.sort()
        novelty = np.mean(distances[:k])
        
        # Update novelty archive
        self.novelty_archive.append(topology)
        if len(self.novelty_archive) > self.config.novelty_archive_size:
            # Remove oldest entries
            self.novelty_archive = self.novelty_archive[-self.config.novelty_archive_size:]
        
        return novelty
    
    def _evaluate_stability(self, topology: TopologyGenome, generation: int) -> float:
        """Evaluate stability of routing patterns over generations."""
        # This is a placeholder - in practice you'd track topology evolution
        # For now, use the variance in recent fitness values as a proxy
        
        stability_score = 1.0  # Default to stable
        
        # Look at fitness variance over recent generations
        if len(self.fitness_history[ObjectiveType.ACCURACY.value]) > self.config.stability_generations:
            recent_accuracy = self.fitness_history[ObjectiveType.ACCURACY.value][-self.config.stability_generations:]
            accuracy_variance = np.var(recent_accuracy)
            # Lower variance = higher stability
            stability_score = 1.0 / (1.0 + accuracy_variance * 10)
        
        return stability_score
    
    # FITNESS AGGREGATION METHODS
    
    def _aggregate_objectives(self, scores: Dict[str, float], generation: int) -> float:
        """Aggregate multiple objectives into a single fitness score."""
        if self.config.aggregation_method == "weighted_sum":
            return self._weighted_sum_aggregation(scores, generation)
        elif self.config.aggregation_method == "pareto":
            return self._pareto_aggregation(scores)
        elif self.config.aggregation_method == "lexicographic":
            return self._lexicographic_aggregation(scores)
        else:
            logger.warning(f"Unknown aggregation method: {self.config.aggregation_method}")
            return self._weighted_sum_aggregation(scores, generation)
    
    def _weighted_sum_aggregation(self, scores: Dict[str, float], generation: int) -> float:
        """Weighted sum aggregation with constraint handling."""
        fitness = 0.0
        constraint_violations = 0.0
        
        # Normalize scores if we have history
        normalized_scores = self._normalize_scores(scores)
        
        # Update adaptive weights
        self._update_adaptive_weights(normalized_scores, generation)
        
        for obj_config in self.config.objectives:
            obj_name = obj_config.name
            if obj_name not in scores:
                continue
                
            score = normalized_scores.get(obj_name, scores[obj_name])
            weight = self.adaptive_weights.get(obj_name, obj_config.weight)
            
            # Handle minimization objectives
            if obj_config.minimize:
                score = -score
            
            # Handle constraints
            if obj_config.constraint and obj_config.constraint_threshold is not None:
                if obj_config.minimize:
                    if scores[obj_name] > obj_config.constraint_threshold:
                        constraint_violations += (scores[obj_name] - obj_config.constraint_threshold)
                else:
                    if scores[obj_name] < obj_config.constraint_threshold:
                        constraint_violations += (obj_config.constraint_threshold - scores[obj_name])
            
            fitness += weight * score
        
        # Apply constraint penalties
        if constraint_violations > 0:
            fitness -= self.config.constraint_penalty * constraint_violations
        
        return fitness
    
    def _pareto_aggregation(self, scores: Dict[str, float]) -> float:
        """Pareto-based aggregation using domination count."""
        # Count how many solutions in the archive dominate this one
        domination_count = 0
        
        for archived_scores in self.pareto_archive:
            if self._dominates(archived_scores, scores):
                domination_count += 1
        
        # Fitness is inverse of domination count (less dominated = higher fitness)
        return 1.0 / (1.0 + domination_count)
    
    def _lexicographic_aggregation(self, scores: Dict[str, float]) -> float:
        """Lexicographic aggregation (prioritize objectives in order)."""
        # Sort objectives by weight (higher weight = higher priority)
        sorted_objectives = sorted(self.config.objectives, key=lambda x: x.weight, reverse=True)
        
        # Create lexicographic value
        fitness = 0.0
        multiplier = 1.0
        
        for obj_config in sorted_objectives:
            if obj_config.name in scores:
                score = scores[obj_config.name]
                if obj_config.minimize:
                    score = -score
                fitness += multiplier * score
                multiplier *= 0.001  # Each objective gets orders of magnitude less influence
        
        return fitness
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores based on historical data."""
        if self.config.normalization_method == "none":
            return scores
        
        normalized = {}
        
        for obj_name, score in scores.items():
            if obj_name in self.fitness_history and len(self.fitness_history[obj_name]) > 10:
                history = self.fitness_history[obj_name]
                
                if self.config.normalization_method == "min_max":
                    min_val, max_val = min(history), max(history)
                    if max_val > min_val:
                        normalized[obj_name] = (score - min_val) / (max_val - min_val)
                    else:
                        normalized[obj_name] = score
                
                elif self.config.normalization_method == "z_score":
                    mean_val, std_val = np.mean(history), np.std(history)
                    if std_val > 0:
                        normalized[obj_name] = (score - mean_val) / std_val
                    else:
                        normalized[obj_name] = score
                
                else:  # rank normalization
                    rank = sum(1 for h in history if h <= score)
                    normalized[obj_name] = rank / len(history)
            else:
                normalized[obj_name] = score
        
        return normalized
    
    def _update_adaptive_weights(self, scores: Dict[str, float], generation: int):
        """Update adaptive weights based on progress and convergence."""
        for obj_config in self.config.objectives:
            if not obj_config.adaptive_weight:
                continue
                
            obj_name = obj_config.name
            if obj_name in self.fitness_history and len(self.fitness_history[obj_name]) > 20:
                history = self.fitness_history[obj_name][-20:]  # Last 20 generations
                
                # Compute improvement rate
                if len(history) > 10:
                    recent_trend = np.polyfit(range(len(history)), history, 1)[0]
                    
                    # If objective is improving, slightly reduce its weight (focus on others)
                    # If objective is stagnating, increase its weight
                    if recent_trend > 0.001:  # Improving
                        self.adaptive_weights[obj_name] *= 0.99
                    elif abs(recent_trend) < 0.0001:  # Stagnating
                        self.adaptive_weights[obj_name] *= 1.01
                    
                    # Keep weights in reasonable bounds
                    self.adaptive_weights[obj_name] = max(0.1, min(2.0, self.adaptive_weights[obj_name]))
    
    def _dominates(self, scores1: Dict[str, float], scores2: Dict[str, float]) -> bool:
        """Check if scores1 dominates scores2 (Pareto dominance)."""
        better_in_any = False
        
        for obj_config in self.config.objectives:
            obj_name = obj_config.name
            if obj_name not in scores1 or obj_name not in scores2:
                continue
                
            score1, score2 = scores1[obj_name], scores2[obj_name]
            
            if obj_config.minimize:
                if score1 > score2:
                    return False  # scores1 is worse in this objective
                elif score1 < score2:
                    better_in_any = True
            else:
                if score1 < score2:
                    return False  # scores1 is worse in this objective
                elif score1 > score2:
                    better_in_any = True
        
        return better_in_any
    
    def _update_pareto_archive(self, scores: Dict[str, float]):
        """Update Pareto archive with non-dominated solutions."""
        # Remove solutions dominated by the new one
        self.pareto_archive = [
            archived for archived in self.pareto_archive
            if not self._dominates(scores, archived)
        ]
        
        # Add new solution if it's not dominated
        is_dominated = any(self._dominates(archived, scores) for archived in self.pareto_archive)
        if not is_dominated:
            self.pareto_archive.append(scores.copy())
            
        # Limit archive size
        if len(self.pareto_archive) > 100:
            # Keep most diverse solutions
            self.pareto_archive = self._select_diverse_solutions(self.pareto_archive, 100)
    
    def _select_diverse_solutions(self, solutions: List[Dict[str, float]], target_size: int) -> List[Dict[str, float]]:
        """Select diverse solutions from archive."""
        if len(solutions) <= target_size:
            return solutions
            
        # Simple diversity selection based on distance in objective space
        selected = [solutions[0]]  # Start with first solution
        remaining = solutions[1:]
        
        while len(selected) < target_size and remaining:
            # Find solution with maximum minimum distance to selected
            best_candidate = None
            best_min_distance = -1
            
            for candidate in remaining:
                min_distance = float('inf')
                for selected_sol in selected:
                    distance = self._objective_distance(candidate, selected_sol)
                    min_distance = min(min_distance, distance)
                
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def _objective_distance(self, sol1: Dict[str, float], sol2: Dict[str, float]) -> float:
        """Compute distance between solutions in objective space."""
        distance = 0.0
        count = 0
        
        for obj_config in self.config.objectives:
            obj_name = obj_config.name
            if obj_name in sol1 and obj_name in sol2:
                diff = sol1[obj_name] - sol2[obj_name]
                distance += diff * diff
                count += 1
        
        return math.sqrt(distance) / max(count, 1)
    
    def get_pareto_front(self) -> List[Dict[str, float]]:
        """Get current Pareto front."""
        return self.pareto_archive.copy()
    
    def get_fitness_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive fitness statistics."""
        stats = {}
        
        for obj_name, history in self.fitness_history.items():
            if history:
                stats[obj_name] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'latest': history[-1],
                    'trend': np.polyfit(range(len(history)), history, 1)[0] if len(history) > 1 else 0.0,
                    'samples': len(history)
                }
        
        stats['adaptive_weights'] = self.adaptive_weights.copy()
        stats['pareto_front_size'] = len(self.pareto_archive)
        stats['cache_size'] = len(self.evaluation_cache)
        
        return stats
    
    def clear_cache(self):
        """Clear evaluation cache to free memory."""
        self.evaluation_cache.clear()
        logger.info("Cleared fitness evaluation cache")