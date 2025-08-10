"""
Advanced mutation operators for topology evolution.

This module implements sophisticated mutation strategies that go beyond
simple random modifications to enable more intelligent exploration
of the topology search space.
"""

import random
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

from ..routing.topology import TopologyGenome
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MutationType(Enum):
    """Types of mutations available."""
    STRUCTURAL = "structural"
    PARAMETRIC = "parametric"
    ARCHITECTURAL = "architectural"
    ADAPTIVE = "adaptive"


@dataclass
class MutationConfig:
    """Configuration for mutation operators."""
    structural_rate: float = 0.1
    parametric_rate: float = 0.05
    architectural_rate: float = 0.02
    adaptive_rate: float = 0.03
    
    # Structural mutation parameters
    connection_flip_rate: float = 0.05
    rewire_probability: float = 0.02
    subgraph_mutation_rate: float = 0.01
    
    # Parametric mutation parameters
    temperature_mutation_std: float = 0.1
    top_k_change_prob: float = 0.1
    
    # Architectural parameters
    expert_specialization_rate: float = 0.01
    hierarchical_mutation_rate: float = 0.005


class AdvancedMutationOperator:
    """Advanced mutation operator with multiple strategies."""
    
    def __init__(self, config: MutationConfig):
        self.config = config
        self.mutation_history: List[Dict[str, Any]] = []
        self.success_rates: Dict[str, float] = {}
        
        # Register mutation functions
        self.mutations = {
            MutationType.STRUCTURAL: [
                self.connection_flip_mutation,
                self.rewire_expert_mutation,
                self.subgraph_transplant_mutation,
                self.sparsity_gradient_mutation,
                self.cluster_based_mutation
            ],
            MutationType.PARAMETRIC: [
                self.temperature_perturbation,
                self.top_k_adaptation,
                self.load_balance_adjustment,
                self.diversity_enhancement
            ],
            MutationType.ARCHITECTURAL: [
                self.expert_specialization_mutation,
                self.hierarchical_structure_mutation,
                self.attention_pattern_mutation
            ],
            MutationType.ADAPTIVE: [
                self.fitness_guided_mutation,
                self.history_based_mutation,
                self.population_diversity_mutation
            ]
        }
        
        logger.info(f"Initialized advanced mutation operator with {len(self.mutations)} mutation types")
    
    def mutate(self, topology: TopologyGenome, generation: int, fitness_history: List[float]) -> TopologyGenome:
        """
        Apply advanced mutations to topology.
        
        Args:
            topology: Topology to mutate
            generation: Current generation number
            fitness_history: Historical fitness values
            
        Returns:
            Mutated topology
        """
        mutated = topology.mutate(self.config.structural_rate)  # Start with base mutation
        
        # Apply different mutation types based on probability
        applied_mutations = []
        
        if random.random() < self.config.structural_rate:
            mutation_fn = random.choice(self.mutations[MutationType.STRUCTURAL])
            mutated = mutation_fn(mutated)
            applied_mutations.append(mutation_fn.__name__)
        
        if random.random() < self.config.parametric_rate:
            mutation_fn = random.choice(self.mutations[MutationType.PARAMETRIC])
            mutated = mutation_fn(mutated)
            applied_mutations.append(mutation_fn.__name__)
        
        if random.random() < self.config.architectural_rate:
            mutation_fn = random.choice(self.mutations[MutationType.ARCHITECTURAL])
            mutated = mutation_fn(mutated)
            applied_mutations.append(mutation_fn.__name__)
        
        if random.random() < self.config.adaptive_rate:
            mutation_fn = random.choice(self.mutations[MutationType.ADAPTIVE])
            mutated = mutation_fn(mutated, generation, fitness_history)
            applied_mutations.append(mutation_fn.__name__)
        
        # Record mutation history
        self.mutation_history.append({
            'generation': generation,
            'applied_mutations': applied_mutations,
            'original_sparsity': topology.compute_sparsity(),
            'mutated_sparsity': mutated.compute_sparsity()
        })
        
        return mutated
    
    # STRUCTURAL MUTATIONS
    
    def connection_flip_mutation(self, topology: TopologyGenome) -> TopologyGenome:
        """Intelligently flip connections based on current usage patterns."""
        # Identify underutilized experts and overutilized tokens
        expert_usage = topology.routing_matrix.sum(dim=0)  # Usage per expert
        token_usage = topology.routing_matrix.sum(dim=1)   # Connections per token
        
        # Find candidates for connection changes
        underused_experts = (expert_usage < expert_usage.mean() * 0.5).nonzero().flatten()
        overused_experts = (expert_usage > expert_usage.mean() * 1.5).nonzero().flatten()
        
        num_flips = max(1, int(topology.routing_matrix.numel() * self.config.connection_flip_rate))
        
        for _ in range(num_flips):
            if len(underused_experts) > 0 and len(overused_experts) > 0:
                # Transfer connection from overused to underused expert
                token_idx = random.randint(0, topology.num_tokens - 1)
                overused_expert = random.choice(overused_experts).item()
                underused_expert = random.choice(underused_experts).item()
                
                if topology.routing_matrix[token_idx, overused_expert] == 1:
                    topology.routing_matrix[token_idx, overused_expert] = 0
                    topology.routing_matrix[token_idx, underused_expert] = 1
            else:
                # Random flip
                token_idx = random.randint(0, topology.num_tokens - 1)
                expert_idx = random.randint(0, topology.num_experts - 1)
                topology.routing_matrix[token_idx, expert_idx] = 1 - topology.routing_matrix[token_idx, expert_idx]
        
        return topology
    
    def rewire_expert_mutation(self, topology: TopologyGenome) -> TopologyGenome:
        """Rewire entire expert connections based on similarity patterns."""
        if random.random() < self.config.rewire_probability:
            expert_idx = random.randint(0, topology.num_experts - 1)
            
            # Clear existing connections
            topology.routing_matrix[:, expert_idx] = 0
            
            # Create new connections based on clustering
            num_new_connections = max(1, int(topology.num_tokens * topology.sparsity))
            
            # Use token position-based clustering for more structure
            if topology.num_tokens > 10:
                # Create clustered connections (nearby tokens)
                start_token = random.randint(0, topology.num_tokens - num_new_connections)
                for i in range(num_new_connections):
                    token_idx = (start_token + i) % topology.num_tokens
                    topology.routing_matrix[token_idx, expert_idx] = 1
            else:
                # Random connections for small topologies
                token_indices = random.sample(range(topology.num_tokens), num_new_connections)
                for token_idx in token_indices:
                    topology.routing_matrix[token_idx, expert_idx] = 1
        
        return topology
    
    def subgraph_transplant_mutation(self, topology: TopologyGenome) -> TopologyGenome:
        """Transplant routing subgraphs between different regions."""
        if topology.num_tokens < 4 or topology.num_experts < 2:
            return topology
        
        # Define source and target regions
        region_size = max(2, topology.num_tokens // 4)
        source_start = random.randint(0, topology.num_tokens - region_size)
        target_start = random.randint(0, topology.num_tokens - region_size)
        
        if source_start != target_start:
            # Copy routing pattern from source to target region
            source_pattern = topology.routing_matrix[source_start:source_start + region_size].clone()
            topology.routing_matrix[target_start:target_start + region_size] = source_pattern
        
        return topology
    
    def sparsity_gradient_mutation(self, topology: TopologyGenome) -> TopologyGenome:
        """Create sparsity gradients across the topology."""
        # Calculate current sparsity per region
        region_size = max(1, topology.num_tokens // 4)
        
        for i in range(0, topology.num_tokens, region_size):
            end_idx = min(i + region_size, topology.num_tokens)
            region = topology.routing_matrix[i:end_idx]
            current_sparsity = 1.0 - region.float().mean()
            
            # Adjust sparsity towards gradient
            target_sparsity = topology.sparsity * (1.0 + 0.5 * np.sin(i / topology.num_tokens * np.pi))
            
            if current_sparsity < target_sparsity:
                # Remove connections
                ones = region.nonzero()
                if len(ones) > 0:
                    remove_idx = random.choice(range(len(ones)))
                    region[tuple(ones[remove_idx])] = 0
            elif current_sparsity > target_sparsity:
                # Add connections
                zeros = (region == 0).nonzero()
                if len(zeros) > 0:
                    add_idx = random.choice(range(len(zeros)))
                    region[tuple(zeros[add_idx])] = 1
        
        return topology
    
    def cluster_based_mutation(self, topology: TopologyGenome) -> TopologyGenome:
        """Create clustered connection patterns for improved locality."""
        # Create expert clusters
        num_clusters = max(2, topology.num_experts // 3)
        cluster_size = topology.num_experts // num_clusters
        
        for cluster_id in range(num_clusters):
            cluster_start = cluster_id * cluster_size
            cluster_end = min((cluster_id + 1) * cluster_size, topology.num_experts)
            
            # Find tokens that should connect to this cluster
            token_start = cluster_id * (topology.num_tokens // num_clusters)
            token_end = min((cluster_id + 1) * (topology.num_tokens // num_clusters), topology.num_tokens)
            
            # Increase connectivity within cluster
            if random.random() < 0.3:  # 30% chance to apply clustering
                for token_idx in range(token_start, token_end):
                    expert_idx = random.randint(cluster_start, cluster_end - 1)
                    topology.routing_matrix[token_idx, expert_idx] = 1
        
        return topology
    
    # PARAMETRIC MUTATIONS
    
    def temperature_perturbation(self, topology: TopologyGenome) -> TopologyGenome:
        """Intelligently adjust routing temperature based on sparsity."""
        current_sparsity = topology.compute_sparsity()
        
        # Higher sparsity -> lower temperature (more decisive routing)
        # Lower sparsity -> higher temperature (more exploration)
        if current_sparsity > topology.sparsity:
            # Too sparse, increase temperature for more exploration
            topology.routing_params.temperature *= (1 + self.config.temperature_mutation_std)
        else:
            # Not sparse enough, decrease temperature for more decisive routing
            topology.routing_params.temperature *= (1 - self.config.temperature_mutation_std)
        
        # Clamp to reasonable bounds
        topology.routing_params.temperature = max(0.1, min(5.0, topology.routing_params.temperature))
        
        return topology
    
    def top_k_adaptation(self, topology: TopologyGenome) -> TopologyGenome:
        """Adapt top-k based on expert utilization patterns."""
        if random.random() < self.config.top_k_change_prob:
            expert_usage = topology.routing_matrix.sum(dim=0)
            avg_usage = expert_usage.mean()
            
            # If usage is very uneven, increase top_k for more diversity
            usage_std = expert_usage.std()
            if usage_std > avg_usage * 0.5:
                topology.routing_params.top_k = min(topology.num_experts, topology.routing_params.top_k + 1)
            else:
                topology.routing_params.top_k = max(1, topology.routing_params.top_k - 1)
        
        return topology
    
    def load_balance_adjustment(self, topology: TopologyGenome) -> TopologyGenome:
        """Adjust load balancing weight based on current balance."""
        expert_usage = topology.routing_matrix.sum(dim=0).float()
        if expert_usage.sum() > 0:
            expert_probs = expert_usage / expert_usage.sum()
            entropy = -(expert_probs * torch.log(expert_probs + 1e-8)).sum()
            max_entropy = np.log(topology.num_experts)
            balance_score = entropy / max_entropy
            
            # If poorly balanced, increase load balancing weight
            if balance_score < 0.8:
                topology.routing_params.load_balancing_weight *= 1.2
            else:
                topology.routing_params.load_balancing_weight *= 0.9
            
            # Clamp to reasonable bounds
            topology.routing_params.load_balancing_weight = max(0.0, min(1.0, topology.routing_params.load_balancing_weight))
        
        return topology
    
    def diversity_enhancement(self, topology: TopologyGenome) -> TopologyGenome:
        """Enhance routing diversity through parameter adjustment."""
        # Calculate current routing diversity
        routing_patterns = topology.routing_matrix.float()
        if routing_patterns.size(0) > 1:
            # Compute pairwise similarities
            similarities = []
            for i in range(routing_patterns.size(0)):
                for j in range(i + 1, routing_patterns.size(0)):
                    similarity = F.cosine_similarity(
                        routing_patterns[i].unsqueeze(0),
                        routing_patterns[j].unsqueeze(0)
                    ).item()
                    similarities.append(similarity)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            
            # If patterns are too similar, adjust diversity weight
            if avg_similarity > 0.7:
                topology.routing_params.diversity_weight *= 1.3
            else:
                topology.routing_params.diversity_weight *= 0.9
            
            # Clamp bounds
            topology.routing_params.diversity_weight = max(0.0, min(1.0, topology.routing_params.diversity_weight))
        
        return topology
    
    # ARCHITECTURAL MUTATIONS
    
    def expert_specialization_mutation(self, topology: TopologyGenome) -> TopologyGenome:
        """Create specialized expert patterns for different sequence regions."""
        if random.random() < self.config.expert_specialization_rate:
            # Choose an expert to specialize
            expert_idx = random.randint(0, topology.num_experts - 1)
            
            # Clear current connections
            topology.routing_matrix[:, expert_idx] = 0
            
            # Specialize to specific regions or patterns
            specialization_type = random.choice(['positional', 'pattern', 'frequency'])
            
            if specialization_type == 'positional':
                # Specialize to specific positions (e.g., beginning, middle, end)
                region = random.choice(['start', 'middle', 'end'])
                if region == 'start':
                    start, end = 0, topology.num_tokens // 3
                elif region == 'middle':
                    start, end = topology.num_tokens // 3, 2 * topology.num_tokens // 3
                else:  # end
                    start, end = 2 * topology.num_tokens // 3, topology.num_tokens
                
                # Connect to this region with higher probability
                for token_idx in range(start, end):
                    if random.random() < 0.6:  # 60% connection probability
                        topology.routing_matrix[token_idx, expert_idx] = 1
            
            elif specialization_type == 'pattern':
                # Specialize to regular patterns (every nth token)
                pattern_step = random.choice([2, 3, 4, 5])
                start_offset = random.randint(0, pattern_step - 1)
                for token_idx in range(start_offset, topology.num_tokens, pattern_step):
                    topology.routing_matrix[token_idx, expert_idx] = 1
            
            elif specialization_type == 'frequency':
                # Specialize to high or low frequency tokens
                num_connections = max(1, int(topology.num_tokens * topology.sparsity))
                if random.choice([True, False]):  # High frequency
                    # Connect to first/last tokens (typically more important)
                    for i in range(min(num_connections, topology.num_tokens // 2)):
                        topology.routing_matrix[i, expert_idx] = 1
                        if topology.num_tokens - 1 - i >= 0:
                            topology.routing_matrix[topology.num_tokens - 1 - i, expert_idx] = 1
                else:  # Sparse connections to middle tokens
                    middle_start = topology.num_tokens // 4
                    middle_end = 3 * topology.num_tokens // 4
                    for _ in range(num_connections):
                        token_idx = random.randint(middle_start, middle_end - 1)
                        topology.routing_matrix[token_idx, expert_idx] = 1
        
        return topology
    
    def hierarchical_structure_mutation(self, topology: TopologyGenome) -> TopologyGenome:
        """Create hierarchical routing structures."""
        if random.random() < self.config.hierarchical_mutation_rate and topology.num_experts >= 4:
            # Create two-level hierarchy: primary and secondary experts
            num_primary = max(1, topology.num_experts // 3)
            primary_experts = random.sample(range(topology.num_experts), num_primary)
            
            # Primary experts get broad token coverage
            for expert_idx in primary_experts:
                topology.routing_matrix[:, expert_idx] = 0  # Clear first
                num_connections = max(2, int(topology.num_tokens * 0.3))  # 30% coverage
                token_indices = random.sample(range(topology.num_tokens), num_connections)
                for token_idx in token_indices:
                    topology.routing_matrix[token_idx, expert_idx] = 1
            
            # Secondary experts get specialized coverage
            secondary_experts = [e for e in range(topology.num_experts) if e not in primary_experts]
            for expert_idx in secondary_experts:
                topology.routing_matrix[:, expert_idx] = 0  # Clear first
                num_connections = max(1, int(topology.num_tokens * 0.1))  # 10% coverage
                token_indices = random.sample(range(topology.num_tokens), num_connections)
                for token_idx in token_indices:
                    topology.routing_matrix[token_idx, expert_idx] = 1
        
        return topology
    
    def attention_pattern_mutation(self, topology: TopologyGenome) -> TopologyGenome:
        """Create attention-like connectivity patterns."""
        if topology.num_tokens >= 8:
            # Implement attention-like patterns (tokens attending to previous tokens)
            attention_range = min(8, topology.num_tokens // 2)
            
            for token_idx in range(attention_range, topology.num_tokens):
                # This token can attend to previous tokens
                for prev_token in range(max(0, token_idx - attention_range), token_idx):
                    if random.random() < 0.2:  # 20% attention probability
                        # Map both tokens to the same expert for attention-like behavior
                        expert_idx = random.randint(0, topology.num_experts - 1)
                        topology.routing_matrix[token_idx, expert_idx] = 1
                        topology.routing_matrix[prev_token, expert_idx] = 1
        
        return topology
    
    # ADAPTIVE MUTATIONS
    
    def fitness_guided_mutation(self, topology: TopologyGenome, generation: int, fitness_history: List[float]) -> TopologyGenome:
        """Guide mutations based on fitness trends."""
        if len(fitness_history) < 5:
            return topology
        
        recent_fitness = fitness_history[-5:]
        fitness_trend = np.polyfit(range(len(recent_fitness)), recent_fitness, 1)[0]
        
        if fitness_trend < 0:  # Declining fitness
            # Increase exploration - more aggressive mutations
            self.config.structural_rate *= 1.2
            self.config.parametric_rate *= 1.2
        elif fitness_trend > 0.001:  # Good improvement
            # Fine-tune - reduce mutation rates
            self.config.structural_rate *= 0.9
            self.config.parametric_rate *= 0.9
        
        # Apply bounds
        self.config.structural_rate = max(0.01, min(0.3, self.config.structural_rate))
        self.config.parametric_rate = max(0.005, min(0.2, self.config.parametric_rate))
        
        return topology
    
    def history_based_mutation(self, topology: TopologyGenome, generation: int, fitness_history: List[float]) -> TopologyGenome:
        """Use mutation history to prefer successful mutation types."""
        if len(self.mutation_history) < 10:
            return topology
        
        # Analyze which mutations have been most successful
        # (This would require tracking fitness improvements after mutations)
        # For now, implement a simple version that varies mutation types over time
        
        cycle_length = 20
        cycle_position = generation % cycle_length
        
        if cycle_position < 5:
            # Focus on structural changes early in cycle
            topology = self.connection_flip_mutation(topology)
        elif cycle_position < 10:
            # Focus on parametric changes
            topology = self.temperature_perturbation(topology)
        elif cycle_position < 15:
            # Focus on architectural changes
            topology = self.expert_specialization_mutation(topology)
        else:
            # Focus on fine-tuning
            topology = self.load_balance_adjustment(topology)
        
        return topology
    
    def population_diversity_mutation(self, topology: TopologyGenome, generation: int, fitness_history: List[float]) -> TopologyGenome:
        """Adjust mutation based on population diversity needs."""
        # This would ideally take population diversity as input
        # For now, implement time-based diversity enhancement
        
        if generation % 10 == 0:  # Every 10 generations
            # Inject diversity through radical mutations
            if random.random() < 0.3:
                topology = self.subgraph_transplant_mutation(topology)
                topology = self.expert_specialization_mutation(topology)
        
        return topology
    
    def get_mutation_statistics(self) -> Dict[str, Any]:
        """Get statistics about mutation applications."""
        if not self.mutation_history:
            return {}
        
        mutation_counts = {}
        for record in self.mutation_history:
            for mutation in record['applied_mutations']:
                mutation_counts[mutation] = mutation_counts.get(mutation, 0) + 1
        
        return {
            'total_mutations': len(self.mutation_history),
            'mutation_counts': mutation_counts,
            'current_rates': {
                'structural': self.config.structural_rate,
                'parametric': self.config.parametric_rate,
                'architectural': self.config.architectural_rate,
                'adaptive': self.config.adaptive_rate
            }
        }