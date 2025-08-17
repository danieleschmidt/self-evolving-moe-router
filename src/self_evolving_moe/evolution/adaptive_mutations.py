"""
Adaptive Mutation Strategies for MoE Topology Evolution
TERRAGON NEXT-GEN v5.0 - Adaptive Evolutionary Operators

Enhanced adaptive mutation strategies that self-adjust based on:
- Population diversity and convergence patterns
- Fitness landscape characteristics and topology complexity
- Performance feedback and historical success rates
- Multi-scale mutation with hierarchical adaptation
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import deque, defaultdict
import random

logger = logging.getLogger(__name__)

class MutationType(Enum):
    """Types of adaptive mutations"""
    GAUSSIAN = "gaussian"
    CAUCHY = "cauchy"
    LEVY = "levy"
    UNIFORM = "uniform"
    SPARSE_FLIP = "sparse_flip"
    GRADIENT_BASED = "gradient_based"
    TOPOLOGY_AWARE = "topology_aware"
    HIERARCHICAL = "hierarchical"
    SELF_ADAPTIVE = "self_adaptive"

class AdaptationStrategy(Enum):
    """Adaptation strategies for mutation parameters"""
    SUCCESS_RATE = "success_rate"
    DIVERSITY_MAINTENANCE = "diversity_maintenance"
    FITNESS_IMPROVEMENT = "fitness_improvement"
    POPULATION_ENTROPY = "population_entropy"
    MULTI_OBJECTIVE = "multi_objective"
    COVARIANCE_MATRIX = "covariance_matrix"

@dataclass
class MutationMetrics:
    """Metrics for tracking mutation performance"""
    total_mutations: int = 0
    successful_mutations: int = 0
    fitness_improvements: int = 0
    diversity_contributions: int = 0
    average_improvement: float = 0.0
    success_rate: float = 0.0
    recent_success_rate: float = 0.0
    parameter_adaptations: int = 0
    
    def update_success(self, improvement: float):
        """Update metrics with a successful mutation"""
        self.total_mutations += 1
        self.successful_mutations += 1
        self.fitness_improvements += 1 if improvement > 0 else 0
        self.average_improvement = (self.average_improvement * (self.total_mutations - 1) + improvement) / self.total_mutations
        self.success_rate = self.successful_mutations / self.total_mutations
    
    def update_failure(self):
        """Update metrics with a failed mutation"""
        self.total_mutations += 1
        self.success_rate = self.successful_mutations / self.total_mutations

@dataclass
class AdaptiveMutationConfig:
    """Configuration for adaptive mutation system"""
    # Base mutation parameters
    initial_mutation_rate: float = 0.1
    initial_mutation_strength: float = 0.05
    min_mutation_rate: float = 0.001
    max_mutation_rate: float = 0.5
    min_mutation_strength: float = 0.001
    max_mutation_strength: float = 0.3
    
    # Adaptation parameters
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.SUCCESS_RATE
    adaptation_rate: float = 0.05
    success_threshold: float = 0.2
    diversity_threshold: float = 0.1
    adaptation_window: int = 20
    
    # Multi-scale parameters
    enable_hierarchical: bool = True
    hierarchy_levels: int = 3
    coarse_mutation_weight: float = 0.3
    fine_mutation_weight: float = 0.7
    
    # Advanced features
    enable_self_adaptation: bool = True
    enable_covariance_adaptation: bool = True
    enable_topology_awareness: bool = True
    
    # Performance tracking
    track_mutation_history: bool = True
    history_window: int = 100

class AdaptiveMutationOperator(ABC):
    """Base class for adaptive mutation operators"""
    
    def __init__(self, config: AdaptiveMutationConfig):
        self.config = config
        self.metrics = MutationMetrics()
        self.mutation_history: deque = deque(maxlen=config.history_window)
        self.parameter_history: deque = deque(maxlen=config.history_window)
        
        # Current adaptive parameters
        self.current_mutation_rate = config.initial_mutation_rate
        self.current_mutation_strength = config.initial_mutation_strength
        
    @abstractmethod
    def mutate(self, individual: np.ndarray, fitness: float, generation: int, 
               population: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply adaptive mutation to an individual"""
        pass
    
    @abstractmethod
    def adapt_parameters(self, population: List[np.ndarray], fitness_scores: np.ndarray, 
                        generation: int) -> Dict[str, float]:
        """Adapt mutation parameters based on population state"""
        pass

class SelfAdaptiveMutation(AdaptiveMutationOperator):
    """
    Self-adaptive mutation that evolves its own parameters
    
    Implements the classic (μ + λ)-ES self-adaptation where mutation
    parameters are part of the genome and evolve alongside the solution
    """
    
    def __init__(self, config: AdaptiveMutationConfig):
        super().__init__(config)
        
        # Self-adaptation parameters
        self.tau = 1.0 / math.sqrt(2 * math.sqrt(64))  # Global learning rate
        self.tau_prime = 1.0 / math.sqrt(2 * 64)        # Individual learning rate
        
        # Strategy parameters for each individual (evolved alongside solution)
        self.strategy_parameters: Dict[str, Dict[str, float]] = {}
        self.global_step_size = 1.0
        
    def mutate(self, individual: np.ndarray, fitness: float, generation: int, 
               population: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Self-adaptive mutation with parameter evolution"""
        
        # Get or initialize strategy parameters for this individual
        individual_id = self._get_individual_id(individual)
        if individual_id not in self.strategy_parameters:
            self.strategy_parameters[individual_id] = {
                'step_sizes': np.full(individual.size, self.current_mutation_strength),
                'global_step_size': self.global_step_size,
                'success_count': 0,
                'generation_created': generation
            }
        
        strategy_params = self.strategy_parameters[individual_id]
        
        # Evolve strategy parameters first (self-adaptation)
        new_strategy_params = self._evolve_strategy_parameters(strategy_params, generation)
        self.strategy_parameters[individual_id] = new_strategy_params
        
        # Apply mutation using evolved strategy parameters
        mutated_individual, mutation_info = self._apply_self_adaptive_mutation(
            individual, new_strategy_params, generation
        )
        
        # Track mutation performance
        mutation_info.update({
            'mutation_type': 'self_adaptive',
            'global_step_size': new_strategy_params['global_step_size'],
            'average_step_size': np.mean(new_strategy_params['step_sizes']),
            'strategy_age': generation - new_strategy_params['generation_created']
        })
        
        return mutated_individual, mutation_info
    
    def _evolve_strategy_parameters(self, strategy_params: Dict[str, Any], generation: int) -> Dict[str, Any]:
        """Evolve the strategy parameters using self-adaptation rules"""
        
        # Evolve global step size
        global_noise = np.random.normal(0, 1)
        new_global_step_size = strategy_params['global_step_size'] * math.exp(self.tau_prime * global_noise)
        
        # Evolve individual step sizes
        individual_noise = np.random.normal(0, 1, strategy_params['step_sizes'].shape)
        new_step_sizes = strategy_params['step_sizes'] * np.exp(
            self.tau * global_noise + self.tau_prime * individual_noise
        )
        
        # Apply bounds to prevent degenerate step sizes
        new_step_sizes = np.clip(new_step_sizes, self.config.min_mutation_strength, self.config.max_mutation_strength)
        new_global_step_size = max(self.config.min_mutation_strength, 
                                  min(self.config.max_mutation_strength, new_global_step_size))
        
        return {
            'step_sizes': new_step_sizes,
            'global_step_size': new_global_step_size,
            'success_count': strategy_params['success_count'],
            'generation_created': strategy_params['generation_created']
        }
    
    def _apply_self_adaptive_mutation(self, individual: np.ndarray, strategy_params: Dict[str, Any], 
                                    generation: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply mutation using self-adapted parameters"""
        
        flat_individual = individual.flatten()
        step_sizes = strategy_params['step_sizes']
        
        # Generate mutation vector with individual step sizes
        mutation_vector = np.random.normal(0, step_sizes)
        
        # Apply mutation
        mutated_flat = flat_individual + mutation_vector
        
        # Reshape back to original shape
        mutated_individual = mutated_flat.reshape(individual.shape)
        
        # Apply topology constraints (sparsity preservation)
        mutated_individual = self._apply_topology_constraints(mutated_individual, individual)
        
        mutation_info = {
            'mutation_strength': np.mean(step_sizes),
            'mutation_variance': np.var(step_sizes),
            'elements_mutated': np.sum(np.abs(mutation_vector) > 1e-6),
            'max_mutation': np.max(np.abs(mutation_vector))
        }
        
        return mutated_individual, mutation_info
    
    def adapt_parameters(self, population: List[np.ndarray], fitness_scores: np.ndarray, 
                        generation: int) -> Dict[str, float]:
        """Adapt global parameters based on population success rates"""
        
        # Calculate population-level success metrics
        population_diversity = self._calculate_population_diversity(population)
        fitness_improvement = self._calculate_fitness_improvement(fitness_scores)
        
        # Adapt global parameters
        adaptation_info = {
            'population_diversity': population_diversity,
            'fitness_improvement': fitness_improvement,
            'active_strategies': len(self.strategy_parameters),
            'average_global_step_size': np.mean([sp['global_step_size'] for sp in self.strategy_parameters.values()]) if self.strategy_parameters else self.global_step_size
        }
        
        # Update tau values based on population performance
        if fitness_improvement < 0.01 and population_diversity < 0.1:
            # Increase exploration
            self.tau *= 1.05
            self.tau_prime *= 1.05
        elif fitness_improvement > 0.1 and population_diversity > 0.3:
            # Increase exploitation
            self.tau *= 0.95
            self.tau_prime *= 0.95
        
        # Keep tau values in reasonable bounds
        self.tau = max(0.01, min(0.5, self.tau))
        self.tau_prime = max(0.01, min(0.5, self.tau_prime))
        
        adaptation_info.update({
            'tau': self.tau,
            'tau_prime': self.tau_prime
        })
        
        return adaptation_info
    
    def _get_individual_id(self, individual: np.ndarray) -> str:
        """Generate unique ID for individual to track strategy parameters"""
        return str(hash(individual.data.tobytes()))[:16]
    
    def _apply_topology_constraints(self, mutated: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Apply topology-specific constraints to maintain valid routing matrices"""
        
        # Clip to valid range
        constrained = np.clip(mutated, 0, 1)
        
        # Preserve sparsity pattern similar to original
        original_sparsity = np.mean(original > 0)
        target_sparsity = max(0.1, min(0.5, original_sparsity))  # Keep sparsity in reasonable range
        
        # Threshold to maintain sparsity
        threshold = np.percentile(constrained, (1 - target_sparsity) * 100)
        constrained = (constrained > threshold).astype(np.float32)
        
        return constrained

class DiversityMaintainingMutation(AdaptiveMutationOperator):
    """
    Mutation operator that adapts to maintain population diversity
    
    Monitors population diversity and increases mutation strength
    when diversity drops below threshold
    """
    
    def __init__(self, config: AdaptiveMutationConfig):
        super().__init__(config)
        
        self.diversity_history: deque = deque(maxlen=50)
        self.min_diversity_threshold = config.diversity_threshold
        self.diversity_adaptation_rate = config.adaptation_rate * 2
        
        # Diversity-based adaptation parameters
        self.diversity_pressure = 1.0
        self.novelty_archive: List[np.ndarray] = []
        self.archive_size = 100
        
    def mutate(self, individual: np.ndarray, fitness: float, generation: int, 
               population: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Diversity-maintaining adaptive mutation"""
        
        # Calculate current diversity if population provided
        current_diversity = 0.0
        novelty_score = 0.0
        
        if population:
            current_diversity = self._calculate_population_diversity(population)
            novelty_score = self._calculate_novelty_score(individual, population)
        
        # Adapt mutation strength based on diversity
        adaptive_strength = self._calculate_adaptive_strength(current_diversity, novelty_score)
        adaptive_rate = self._calculate_adaptive_rate(current_diversity)
        
        # Apply diversity-maintaining mutation
        mutated_individual = self._apply_diversity_mutation(
            individual, adaptive_strength, adaptive_rate, generation
        )
        
        # Update novelty archive
        self._update_novelty_archive(individual)
        
        mutation_info = {
            'mutation_type': 'diversity_maintaining',
            'adaptive_strength': adaptive_strength,
            'adaptive_rate': adaptive_rate,
            'population_diversity': current_diversity,
            'individual_novelty': novelty_score,
            'diversity_pressure': self.diversity_pressure
        }
        
        return mutated_individual, mutation_info
    
    def _calculate_adaptive_strength(self, population_diversity: float, novelty_score: float) -> float:
        """Calculate adaptive mutation strength based on diversity metrics"""
        
        base_strength = self.current_mutation_strength
        
        # Increase strength if diversity is low
        if population_diversity < self.min_diversity_threshold:
            diversity_boost = (self.min_diversity_threshold - population_diversity) * 2.0
            base_strength *= (1.0 + diversity_boost)
        
        # Increase strength for highly novel individuals (explore further)
        if novelty_score > 0.7:
            novelty_boost = (novelty_score - 0.7) * 1.5
            base_strength *= (1.0 + novelty_boost)
        # Decrease strength for common individuals (exploit locally)
        elif novelty_score < 0.3:
            novelty_reduction = (0.3 - novelty_score) * 0.5
            base_strength *= (1.0 - novelty_reduction)
        
        # Apply bounds
        return max(self.config.min_mutation_strength, 
                  min(self.config.max_mutation_strength, base_strength))
    
    def _calculate_adaptive_rate(self, population_diversity: float) -> float:
        """Calculate adaptive mutation rate based on diversity"""
        
        base_rate = self.current_mutation_rate
        
        # Increase rate if diversity is critically low
        if population_diversity < self.min_diversity_threshold * 0.5:
            rate_boost = 2.0 * (self.min_diversity_threshold * 0.5 - population_diversity)
            base_rate *= (1.0 + rate_boost)
        
        return max(self.config.min_mutation_rate,
                  min(self.config.max_mutation_rate, base_rate))
    
    def _calculate_novelty_score(self, individual: np.ndarray, population: List[np.ndarray]) -> float:
        """Calculate novelty score of individual relative to population and archive"""
        
        all_references = population + self.novelty_archive
        
        if not all_references:
            return 1.0  # Maximally novel if no references
        
        # Calculate distances to all reference individuals
        distances = []
        for ref_individual in all_references:
            if ref_individual.shape == individual.shape:
                distance = np.mean(np.abs(individual - ref_individual))
                distances.append(distance)
        
        if not distances:
            return 1.0
        
        # Novelty is average distance to k-nearest neighbors
        k = min(5, len(distances))
        distances.sort()
        novelty = np.mean(distances[:k])
        
        # Normalize to [0, 1] range
        return min(1.0, novelty * 2.0)  # Scale factor for typical topology distances
    
    def _apply_diversity_mutation(self, individual: np.ndarray, strength: float, 
                                 rate: float, generation: int) -> np.ndarray:
        """Apply mutation with diversity preservation bias"""
        
        # Multiple mutation strategies for diversity
        mutated = individual.copy()
        
        # 1. Gaussian mutation for local exploration
        gaussian_mask = np.random.random(individual.shape) < rate * 0.6
        gaussian_noise = np.random.normal(0, strength * 0.5, individual.shape)
        mutated = np.where(gaussian_mask, mutated + gaussian_noise, mutated)
        
        # 2. Large jump mutation for diversity
        jump_mask = np.random.random(individual.shape) < rate * 0.2
        jump_noise = np.random.uniform(-strength * 2, strength * 2, individual.shape)
        mutated = np.where(jump_mask, mutated + jump_noise, mutated)
        
        # 3. Sparse flip mutation for topology diversity
        flip_mask = np.random.random(individual.shape) < rate * 0.2
        flip_values = np.random.choice([0, 1], size=individual.shape)
        mutated = np.where(flip_mask, flip_values.astype(np.float32), mutated)
        
        # Apply constraints
        mutated = self._apply_topology_constraints(mutated, individual)
        
        return mutated
    
    def _update_novelty_archive(self, individual: np.ndarray):
        """Update novelty archive with individual if sufficiently novel"""
        
        if len(self.novelty_archive) < self.archive_size:
            self.novelty_archive.append(individual.copy())
        else:
            # Replace least novel individual if current is more novel
            novelty_scores = []
            for archived in self.novelty_archive:
                other_archive = [ind for ind in self.novelty_archive if not np.array_equal(ind, archived)]
                archived_novelty = self._calculate_novelty_score(archived, other_archive)
                novelty_scores.append(archived_novelty)
            
            min_novelty_idx = np.argmin(novelty_scores)
            individual_novelty = self._calculate_novelty_score(individual, self.novelty_archive)
            
            if individual_novelty > novelty_scores[min_novelty_idx]:
                self.novelty_archive[min_novelty_idx] = individual.copy()
    
    def adapt_parameters(self, population: List[np.ndarray], fitness_scores: np.ndarray, 
                        generation: int) -> Dict[str, float]:
        """Adapt parameters based on diversity metrics"""
        
        # Calculate current population diversity
        current_diversity = self._calculate_population_diversity(population)
        self.diversity_history.append(current_diversity)
        
        # Calculate diversity trend
        diversity_trend = 0.0
        if len(self.diversity_history) >= 5:
            recent_diversity = list(self.diversity_history)[-5:]
            diversity_trend = np.polyfit(range(len(recent_diversity)), recent_diversity, 1)[0]
        
        # Adapt parameters based on diversity state
        if current_diversity < self.min_diversity_threshold:
            # Low diversity - increase mutation
            self.current_mutation_strength *= (1.0 + self.diversity_adaptation_rate)
            self.current_mutation_rate *= (1.0 + self.diversity_adaptation_rate * 0.5)
            self.diversity_pressure *= 1.1
        elif current_diversity > self.min_diversity_threshold * 2:
            # High diversity - focus mutations
            self.current_mutation_strength *= (1.0 - self.diversity_adaptation_rate * 0.5)
            self.current_mutation_rate *= (1.0 - self.diversity_adaptation_rate * 0.3)
            self.diversity_pressure *= 0.95
        
        # Apply bounds
        self.current_mutation_strength = max(self.config.min_mutation_strength,
                                           min(self.config.max_mutation_strength, self.current_mutation_strength))
        self.current_mutation_rate = max(self.config.min_mutation_rate,
                                       min(self.config.max_mutation_rate, self.current_mutation_rate))
        
        return {
            'current_diversity': current_diversity,
            'diversity_trend': diversity_trend,
            'adapted_strength': self.current_mutation_strength,
            'adapted_rate': self.current_mutation_rate,
            'diversity_pressure': self.diversity_pressure,
            'novelty_archive_size': len(self.novelty_archive)
        }
    
    def _apply_topology_constraints(self, mutated: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Apply topology constraints while preserving diversity"""
        
        # Clip values
        constrained = np.clip(mutated, 0, 1)
        
        # Preserve sparsity with some flexibility for diversity
        original_sparsity = np.mean(original > 0)
        sparsity_range = 0.1  # Allow ±10% sparsity variation
        min_sparsity = max(0.1, original_sparsity - sparsity_range)
        max_sparsity = min(0.5, original_sparsity + sparsity_range)
        
        # Choose sparsity within allowed range
        target_sparsity = np.random.uniform(min_sparsity, max_sparsity)
        threshold = np.percentile(constrained, (1 - target_sparsity) * 100)
        constrained = (constrained > threshold).astype(np.float32)
        
        return constrained

class HierarchicalAdaptiveMutation(AdaptiveMutationOperator):
    """
    Hierarchical mutation that operates at multiple scales
    
    Applies mutations at different levels of the topology hierarchy:
    - Coarse: Global topology structure
    - Medium: Expert group patterns  
    - Fine: Individual connections
    """
    
    def __init__(self, config: AdaptiveMutationConfig):
        super().__init__(config)
        
        self.hierarchy_levels = config.hierarchy_levels
        self.level_weights = np.array([0.2, 0.3, 0.5])  # Coarse, medium, fine
        self.level_adaptations = np.ones(self.hierarchy_levels)
        
        # Scale-specific parameters
        self.scale_mutation_rates = np.full(self.hierarchy_levels, config.initial_mutation_rate)
        self.scale_mutation_strengths = np.full(self.hierarchy_levels, config.initial_mutation_strength)
        
        # Performance tracking per scale
        self.scale_metrics = [MutationMetrics() for _ in range(self.hierarchy_levels)]
        
    def mutate(self, individual: np.ndarray, fitness: float, generation: int, 
               population: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Hierarchical adaptive mutation across multiple scales"""
        
        mutated_individual = individual.copy()
        scale_contributions = {}
        
        # Apply mutations at each hierarchical level
        for level in range(self.hierarchy_levels):
            if np.random.random() < self.level_weights[level]:
                level_mutated, level_info = self._apply_scale_mutation(
                    mutated_individual, level, generation
                )
                
                # Blend with current state
                blend_factor = self.level_weights[level] * self.level_adaptations[level]
                mutated_individual = self._blend_mutations(mutated_individual, level_mutated, blend_factor)
                
                scale_contributions[f'level_{level}'] = level_info
        
        # Apply final constraints
        mutated_individual = self._apply_topology_constraints(mutated_individual, individual)
        
        mutation_info = {
            'mutation_type': 'hierarchical_adaptive',
            'scales_applied': len(scale_contributions),
            'scale_contributions': scale_contributions,
            'level_weights': self.level_weights.tolist(),
            'level_adaptations': self.level_adaptations.tolist()
        }
        
        return mutated_individual, mutation_info
    
    def _apply_scale_mutation(self, individual: np.ndarray, level: int, generation: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply mutation at specific hierarchical level"""
        
        rows, cols = individual.shape
        mutated = individual.copy()
        
        if level == 0:  # Coarse scale - global patterns
            mutated = self._apply_coarse_mutation(mutated, rows, cols)
            mutation_info = {'scale': 'coarse', 'pattern': 'global'}
            
        elif level == 1:  # Medium scale - expert groups
            mutated = self._apply_medium_mutation(mutated, rows, cols)
            mutation_info = {'scale': 'medium', 'pattern': 'expert_groups'}
            
        else:  # Fine scale - individual connections
            mutated = self._apply_fine_mutation(mutated, rows, cols)
            mutation_info = {'scale': 'fine', 'pattern': 'connections'}
        
        mutation_info.update({
            'level': level,
            'mutation_rate': self.scale_mutation_rates[level],
            'mutation_strength': self.scale_mutation_strengths[level]
        })
        
        return mutated, mutation_info
    
    def _apply_coarse_mutation(self, individual: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """Apply coarse-scale mutations affecting global topology structure"""
        
        mutated = individual.copy()
        
        # Global density adjustment
        if np.random.random() < self.scale_mutation_rates[0]:
            current_density = np.mean(individual)
            density_change = np.random.normal(0, self.scale_mutation_strengths[0])
            target_density = max(0.1, min(0.5, current_density + density_change))
            
            # Adjust overall connectivity to target density
            if target_density > current_density:
                # Increase connections
                add_mask = (individual == 0) & (np.random.random(individual.shape) < 0.1)
                mutated[add_mask] = 1.0
            else:
                # Decrease connections
                remove_mask = (individual > 0) & (np.random.random(individual.shape) < 0.1)
                mutated[remove_mask] = 0.0
        
        # Global pattern shifts
        if np.random.random() < self.scale_mutation_rates[0] * 0.5:
            # Shift entire connectivity pattern
            shift_rows = np.random.randint(-2, 3)
            shift_cols = np.random.randint(-2, 3)
            mutated = np.roll(mutated, shift_rows, axis=0)
            mutated = np.roll(mutated, shift_cols, axis=1)
        
        return mutated
    
    def _apply_medium_mutation(self, individual: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """Apply medium-scale mutations affecting expert group patterns"""
        
        mutated = individual.copy()
        
        # Expert group size (assuming groups of 2-4 experts)
        group_size = max(2, min(4, rows // 4))
        
        # Mutate expert group connectivity patterns
        for group_start in range(0, rows, group_size):
            group_end = min(group_start + group_size, rows)
            
            if np.random.random() < self.scale_mutation_rates[1]:
                # Modify group connectivity pattern
                group_slice = mutated[group_start:group_end, :]
                
                # Apply group-level changes
                if np.random.random() < 0.5:
                    # Increase intra-group connectivity
                    group_slice += np.random.normal(0, self.scale_mutation_strengths[1], group_slice.shape)
                else:
                    # Specialize group to certain tokens
                    token_start = np.random.randint(0, max(1, cols - group_size))
                    token_end = min(token_start + group_size * 2, cols)
                    
                    # Boost connectivity to specific token range
                    group_slice[:, token_start:token_end] *= (1.0 + self.scale_mutation_strengths[1])
                
                mutated[group_start:group_end, :] = group_slice
        
        return mutated
    
    def _apply_fine_mutation(self, individual: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """Apply fine-scale mutations affecting individual connections"""
        
        mutated = individual.copy()
        
        # Individual connection mutations
        mutation_mask = np.random.random(individual.shape) < self.scale_mutation_rates[2]
        
        # Gaussian noise for existing connections
        gaussian_noise = np.random.normal(0, self.scale_mutation_strengths[2], individual.shape)
        mutated += np.where(mutation_mask, gaussian_noise, 0)
        
        # Random connection flips
        flip_mask = np.random.random(individual.shape) < self.scale_mutation_rates[2] * 0.1
        mutated = np.where(flip_mask, 1.0 - mutated, mutated)
        
        return mutated
    
    def _blend_mutations(self, base: np.ndarray, mutated: np.ndarray, blend_factor: float) -> np.ndarray:
        """Blend mutations from different scales"""
        
        return base * (1.0 - blend_factor) + mutated * blend_factor
    
    def adapt_parameters(self, population: List[np.ndarray], fitness_scores: np.ndarray, 
                        generation: int) -> Dict[str, float]:
        """Adapt hierarchical parameters based on scale-specific performance"""
        
        # Calculate overall population metrics
        population_diversity = self._calculate_population_diversity(population)
        fitness_improvement = self._calculate_fitness_improvement(fitness_scores)
        
        # Adapt each scale independently
        scale_adaptations = {}
        
        for level in range(self.hierarchy_levels):
            scale_metrics = self.scale_metrics[level]
            
            # Adapt based on scale performance
            if scale_metrics.total_mutations > 10:
                success_rate = scale_metrics.success_rate
                
                if success_rate > 0.3:
                    # Good performance - slightly increase influence
                    self.level_adaptations[level] *= 1.02
                    self.scale_mutation_rates[level] *= 1.01
                elif success_rate < 0.1:
                    # Poor performance - decrease influence
                    self.level_adaptations[level] *= 0.98
                    self.scale_mutation_rates[level] *= 0.99
            
            # Apply bounds
            self.level_adaptations[level] = max(0.1, min(2.0, self.level_adaptations[level]))
            self.scale_mutation_rates[level] = max(self.config.min_mutation_rate,
                                                  min(self.config.max_mutation_rate, self.scale_mutation_rates[level]))
            
            scale_adaptations[f'level_{level}'] = {
                'adaptation': self.level_adaptations[level],
                'mutation_rate': self.scale_mutation_rates[level],
                'success_rate': scale_metrics.success_rate,
                'total_mutations': scale_metrics.total_mutations
            }
        
        # Normalize level weights
        self.level_weights = self.level_adaptations / np.sum(self.level_adaptations)
        
        return {
            'population_diversity': population_diversity,
            'fitness_improvement': fitness_improvement,
            'scale_adaptations': scale_adaptations,
            'normalized_weights': self.level_weights.tolist()
        }
    
    def _apply_topology_constraints(self, mutated: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Apply hierarchical topology constraints"""
        
        # Clip values
        constrained = np.clip(mutated, 0, 1)
        
        # Maintain reasonable sparsity across scales
        original_sparsity = np.mean(original > 0)
        target_sparsity = max(0.15, min(0.45, original_sparsity))
        
        # Apply sparsity constraint
        threshold = np.percentile(constrained, (1 - target_sparsity) * 100)
        constrained = (constrained > threshold).astype(np.float32)
        
        return constrained

class AdaptiveMutationEngine:
    """
    Unified engine managing multiple adaptive mutation strategies
    """
    
    def __init__(self, config: AdaptiveMutationConfig):
        self.config = config
        
        # Initialize mutation operators
        self.operators = {}
        
        if config.enable_self_adaptation:
            self.operators['self_adaptive'] = SelfAdaptiveMutation(config)
        
        self.operators['diversity_maintaining'] = DiversityMaintainingMutation(config)
        
        if config.enable_hierarchical:
            self.operators['hierarchical'] = HierarchicalAdaptiveMutation(config)
        
        # Operator selection and performance tracking
        self.operator_weights = {name: 1.0 for name in self.operators.keys()}
        self.operator_performance = {name: MutationMetrics() for name in self.operators.keys()}
        
        # Engine state
        self.generation_count = 0
        self.adaptation_history: List[Dict[str, Any]] = []
        
    def evolve_population(self, population: List[np.ndarray], fitness_scores: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Evolve population using adaptive mutation strategies
        """
        
        self.generation_count += 1
        mutated_population = []
        generation_metrics = {
            'generation': self.generation_count,
            'population_size': len(population),
            'operator_usage': defaultdict(int),
            'operator_performance': {},
            'total_mutations': 0,
            'successful_mutations': 0
        }
        
        # Adapt operator weights based on recent performance
        self._adapt_operator_weights()
        
        # Apply mutations to each individual
        for i, individual in enumerate(population):
            fitness = fitness_scores[i] if i < len(fitness_scores) else 0.0
            
            # Select mutation operator based on weights
            selected_operator = self._select_operator()
            generation_metrics['operator_usage'][selected_operator] += 1
            
            # Apply mutation
            try:
                mutated_individual, mutation_info = self.operators[selected_operator].mutate(
                    individual, fitness, self.generation_count, population
                )
                
                mutated_population.append(mutated_individual)
                generation_metrics['total_mutations'] += 1
                
                # Track success (simplified - in practice you'd evaluate fitness)
                mutation_success = self._evaluate_mutation_success(individual, mutated_individual, mutation_info)
                if mutation_success:
                    generation_metrics['successful_mutations'] += 1
                    self.operator_performance[selected_operator].update_success(0.1)  # Placeholder improvement
                else:
                    self.operator_performance[selected_operator].update_failure()
                
            except Exception as e:
                logger.warning(f"Mutation failed with {selected_operator}: {e}")
                mutated_population.append(individual.copy())  # Fallback to original
                self.operator_performance[selected_operator].update_failure()
        
        # Adapt parameters for all operators
        for operator_name, operator in self.operators.items():
            adaptation_info = operator.adapt_parameters(population, fitness_scores, self.generation_count)
            generation_metrics['operator_performance'][operator_name] = adaptation_info
        
        # Record adaptation history
        self.adaptation_history.append(generation_metrics)
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-50:]
        
        generation_metrics['success_rate'] = generation_metrics['successful_mutations'] / max(generation_metrics['total_mutations'], 1)
        
        return mutated_population, generation_metrics
    
    def _select_operator(self) -> str:
        """Select mutation operator based on adaptive weights"""
        
        operators = list(self.operator_weights.keys())
        weights = list(self.operator_weights.values())
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            probabilities = [w / total_weight for w in weights]
        else:
            probabilities = [1.0 / len(operators)] * len(operators)
        
        return np.random.choice(operators, p=probabilities)
    
    def _adapt_operator_weights(self):
        """Adapt operator selection weights based on performance"""
        
        for operator_name in self.operators.keys():
            performance = self.operator_performance[operator_name]
            
            if performance.total_mutations > 5:
                # Adjust weight based on success rate
                success_rate = performance.success_rate
                
                if success_rate > 0.3:
                    self.operator_weights[operator_name] *= 1.05
                elif success_rate < 0.1:
                    self.operator_weights[operator_name] *= 0.95
                
                # Keep weights in reasonable bounds
                self.operator_weights[operator_name] = max(0.1, min(3.0, self.operator_weights[operator_name]))
    
    def _evaluate_mutation_success(self, original: np.ndarray, mutated: np.ndarray, 
                                  mutation_info: Dict[str, Any]) -> bool:
        """Evaluate if mutation was successful (simplified heuristic)"""
        
        # Simple success criteria - in practice you'd use actual fitness evaluation
        difference = np.mean(np.abs(original - mutated))
        
        # Success if mutation is neither too small nor too large
        return 0.01 < difference < 0.3
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics"""
        
        stats = {
            'generation_count': self.generation_count,
            'total_operators': len(self.operators),
            'operator_weights': self.operator_weights.copy(),
            'operator_performance': {}
        }
        
        # Compile operator performance statistics
        for operator_name, performance in self.operator_performance.items():
            stats['operator_performance'][operator_name] = {
                'total_mutations': performance.total_mutations,
                'successful_mutations': performance.successful_mutations,
                'success_rate': performance.success_rate,
                'average_improvement': performance.average_improvement,
                'current_weight': self.operator_weights[operator_name]
            }
        
        # Recent adaptation trends
        if len(self.adaptation_history) >= 5:
            recent_history = self.adaptation_history[-5:]
            stats['recent_trends'] = {
                'average_success_rate': np.mean([h['success_rate'] for h in recent_history]),
                'operator_usage_trend': {},
                'adaptation_stability': np.std([h['success_rate'] for h in recent_history])
            }
            
            # Calculate operator usage trends
            for operator_name in self.operators.keys():
                usage_counts = [h['operator_usage'].get(operator_name, 0) for h in recent_history]
                stats['recent_trends']['operator_usage_trend'][operator_name] = np.mean(usage_counts)
        
        return stats

# Utility functions shared across mutation operators
def _calculate_population_diversity(population: List[np.ndarray]) -> float:
    """Calculate population diversity metric"""
    
    if len(population) < 2:
        return 0.0
    
    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distance = np.mean(np.abs(population[i] - population[j]))
            distances.append(distance)
    
    return np.mean(distances) if distances else 0.0

def _calculate_fitness_improvement(fitness_scores: np.ndarray) -> float:
    """Calculate recent fitness improvement trend"""
    
    if len(fitness_scores) < 2:
        return 0.0
    
    # Simple improvement metric - difference between best and average
    best_fitness = np.max(fitness_scores)
    avg_fitness = np.mean(fitness_scores)
    
    return best_fitness - avg_fitness

# Example usage and testing
if __name__ == "__main__":
    # Test adaptive mutation system
    config = AdaptiveMutationConfig(
        initial_mutation_rate=0.1,
        initial_mutation_strength=0.05,
        enable_hierarchical=True,
        enable_self_adaptation=True,
        enable_topology_awareness=True
    )
    
    # Initialize mutation engine
    mutation_engine = AdaptiveMutationEngine(config)
    
    # Create sample population
    population_size = 20
    topology_shape = (8, 16)
    
    sample_population = []
    for _ in range(population_size):
        topology = np.random.random(topology_shape)
        threshold = np.percentile(topology, 70)  # 30% sparsity
        topology = (topology > threshold).astype(np.float32)
        sample_population.append(topology)
    
    # Sample fitness scores
    sample_fitness = np.random.uniform(-1.0, -0.1, population_size)
    
    # Run adaptive evolution
    print("Testing Adaptive Mutation Engine...")
    
    for generation in range(5):
        print(f"\n=== Generation {generation + 1} ===")
        
        # Evolve population
        new_population, metrics = mutation_engine.evolve_population(sample_population, sample_fitness)
        
        print(f"Population size: {len(new_population)}")
        print(f"Success rate: {metrics['success_rate']:.3f}")
        print(f"Operator usage: {dict(metrics['operator_usage'])}")
        
        # Update for next generation
        sample_population = new_population
        
        # Show adaptation statistics
        if generation % 2 == 1:
            stats = mutation_engine.get_adaptation_statistics()
            print(f"Adaptation stats: {stats}")
    
    print("\nAdaptive mutation testing completed!")