"""
Routing topology representation and genetic operations.

This module implements the TopologyGenome class that represents evolvable
routing patterns for Mixture of Experts models.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class RoutingParams:
    """Parameters controlling routing behavior."""
    temperature: float = 1.0
    top_k: int = 2
    load_balancing_weight: float = 0.01
    diversity_weight: float = 0.1
    sparsity_target: float = 0.1


class TopologyGenome:
    """
    Represents an evolvable routing pattern for MoE models.
    
    The topology consists of:
    - Binary routing matrix (tokens × experts) defining possible connections
    - Expert connectivity graph for hierarchical routing
    - Routing function parameters for fine-tuning behavior
    """
    
    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        sparsity: float = 0.1,
        routing_params: Optional[RoutingParams] = None,
        device: str = "cpu"
    ):
        """
        Initialize routing topology.
        
        Args:
            num_tokens: Number of input tokens/positions
            num_experts: Number of available experts
            sparsity: Target sparsity level (0.0 = dense, 1.0 = completely sparse)
            routing_params: Routing behavior parameters
            device: Device for tensor operations
        """
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.sparsity = sparsity
        self.device = device
        self.routing_params = routing_params or RoutingParams()
        
        # Initialize sparse routing matrix
        self.routing_matrix = self._init_sparse_matrix()
        
        # Expert connectivity graph (expert-to-expert connections)
        self.expert_graph = self._init_expert_connections()
        
        # Fitness tracking
        self.fitness_history: List[float] = []
        self.generation = 0
        
    def _init_sparse_matrix(self) -> torch.Tensor:
        """Initialize sparse binary routing matrix."""
        matrix = torch.zeros(self.num_tokens, self.num_experts, device=self.device)
        
        # Calculate number of connections based on sparsity
        total_possible = self.num_tokens * self.num_experts
        num_connections = int(total_possible * (1 - self.sparsity))
        
        # Ensure each token has at least one connection
        for token_idx in range(self.num_tokens):
            expert_idx = random.randint(0, self.num_experts - 1)
            matrix[token_idx, expert_idx] = 1
            num_connections -= 1
        
        # Add remaining connections randomly
        remaining_positions = []
        for t in range(self.num_tokens):
            for e in range(self.num_experts):
                if matrix[t, e] == 0:
                    remaining_positions.append((t, e))
        
        if num_connections > 0 and remaining_positions:
            selected_positions = random.sample(
                remaining_positions, 
                min(num_connections, len(remaining_positions))
            )
            for t, e in selected_positions:
                matrix[t, e] = 1
                
        return matrix
    
    def _init_expert_connections(self) -> torch.Tensor:
        """Initialize expert-to-expert connectivity graph."""
        # Sparse connectivity between experts for hierarchical routing
        graph = torch.zeros(self.num_experts, self.num_experts, device=self.device)
        
        # Each expert connects to 2-4 other experts on average
        connections_per_expert = random.randint(2, min(4, self.num_experts - 1))
        
        for expert_idx in range(self.num_experts):
            # Select random other experts to connect to
            other_experts = list(range(self.num_experts))
            other_experts.remove(expert_idx)
            
            if other_experts:
                connected_experts = random.sample(
                    other_experts,
                    min(connections_per_expert, len(other_experts))
                )
                for connected_expert in connected_experts:
                    graph[expert_idx, connected_expert] = 1
                    
        return graph
    
    def mutate(self, mutation_rate: float = 0.1) -> 'TopologyGenome':
        """
        Apply mutations to create a new topology variant.
        
        Args:
            mutation_rate: Probability of applying each mutation type
            
        Returns:
            New mutated topology instance
        """
        # Create a copy for mutation
        mutated = TopologyGenome(
            self.num_tokens,
            self.num_experts,
            self.sparsity,
            RoutingParams(
                temperature=self.routing_params.temperature,
                top_k=self.routing_params.top_k,
                load_balancing_weight=self.routing_params.load_balancing_weight,
                diversity_weight=self.routing_params.diversity_weight,
                sparsity_target=self.routing_params.sparsity_target
            ),
            self.device
        )
        
        mutated.routing_matrix = self.routing_matrix.clone()
        mutated.expert_graph = self.expert_graph.clone()
        
        # Structural mutations
        if random.random() < mutation_rate:
            mutated._mutate_routing_matrix()
            
        # Parameter mutations
        if random.random() < mutation_rate:
            mutated._mutate_routing_params()
            
        # Connection mutations
        if random.random() < mutation_rate:
            mutated._mutate_expert_connections()
            
        mutated.generation = self.generation + 1
        return mutated
    
    def _mutate_routing_matrix(self):
        """Apply structural mutations to routing matrix."""
        mutation_type = random.choice(['add_connection', 'remove_connection', 'rewire_token'])
        
        if mutation_type == 'add_connection':
            # Add new token-expert connection
            zero_positions = (self.routing_matrix == 0).nonzero()
            if len(zero_positions) > 0:
                idx = random.randint(0, len(zero_positions) - 1)
                t, e = zero_positions[idx]
                self.routing_matrix[t, e] = 1
                
        elif mutation_type == 'remove_connection':
            # Remove existing connection (but ensure each token has at least one)
            one_positions = (self.routing_matrix == 1).nonzero()
            if len(one_positions) > self.num_tokens:  # Keep at least one per token
                # Find connections that can be safely removed
                removable = []
                for t, e in one_positions:
                    if self.routing_matrix[t].sum() > 1:  # Token has multiple connections
                        removable.append((t, e))
                        
                if removable:
                    t, e = random.choice(removable)
                    self.routing_matrix[t, e] = 0
                    
        elif mutation_type == 'rewire_token':
            # Rewire all connections of a random token
            token_idx = random.randint(0, self.num_tokens - 1)
            self.routing_matrix[token_idx] = 0
            
            # Add new random connections
            num_connections = max(1, int(self.num_experts * (1 - self.sparsity) / self.num_tokens))
            expert_indices = random.sample(range(self.num_experts), num_connections)
            for expert_idx in expert_indices:
                self.routing_matrix[token_idx, expert_idx] = 1
    
    def _mutate_routing_params(self):
        """Apply parameter mutations."""
        param_to_mutate = random.choice(['temperature', 'top_k', 'load_balancing_weight', 'diversity_weight'])
        
        if param_to_mutate == 'temperature':
            # Multiply by random factor between 0.8 and 1.2
            self.routing_params.temperature *= random.uniform(0.8, 1.2)
            self.routing_params.temperature = max(0.1, min(10.0, self.routing_params.temperature))
            
        elif param_to_mutate == 'top_k':
            # Adjust top_k by ±1
            delta = random.choice([-1, 1])
            self.routing_params.top_k = max(1, min(self.num_experts, self.routing_params.top_k + delta))
            
        elif param_to_mutate == 'load_balancing_weight':
            self.routing_params.load_balancing_weight *= random.uniform(0.5, 2.0)
            self.routing_params.load_balancing_weight = max(0.0, min(1.0, self.routing_params.load_balancing_weight))
            
        elif param_to_mutate == 'diversity_weight':
            self.routing_params.diversity_weight *= random.uniform(0.5, 2.0)
            self.routing_params.diversity_weight = max(0.0, min(1.0, self.routing_params.diversity_weight))
    
    def _mutate_expert_connections(self):
        """Apply mutations to expert connectivity graph."""
        mutation_type = random.choice(['add_expert_connection', 'remove_expert_connection'])
        
        if mutation_type == 'add_expert_connection':
            # Add new expert-expert connection
            zero_positions = (self.expert_graph == 0).nonzero()
            # Remove self-connections
            zero_positions = [(i, j) for i, j in zero_positions if i != j]
            
            if zero_positions:
                i, j = random.choice(zero_positions)
                self.expert_graph[i, j] = 1
                
        elif mutation_type == 'remove_expert_connection':
            # Remove existing expert connection
            one_positions = (self.expert_graph == 1).nonzero()
            if len(one_positions) > 0:
                idx = random.randint(0, len(one_positions) - 1)
                i, j = one_positions[idx]
                self.expert_graph[i, j] = 0
    
    def crossover(self, other: 'TopologyGenome') -> 'TopologyGenome':
        """
        Perform crossover with another topology to create offspring.
        
        Args:
            other: Another TopologyGenome to crossover with
            
        Returns:
            New topology combining features from both parents
        """
        if self.num_tokens != other.num_tokens or self.num_experts != other.num_experts:
            raise ValueError("Cannot crossover topologies with different dimensions")
        
        child = TopologyGenome(
            self.num_tokens,
            self.num_experts,
            sparsity=(self.sparsity + other.sparsity) / 2,
            device=self.device
        )
        
        # Uniform crossover for routing matrix
        mask = torch.rand_like(self.routing_matrix) > 0.5
        child.routing_matrix = torch.where(
            mask,
            self.routing_matrix,
            other.routing_matrix
        )
        
        # Ensure each token has at least one connection
        for token_idx in range(self.num_tokens):
            if child.routing_matrix[token_idx].sum() == 0:
                # Copy from parent with connection
                if self.routing_matrix[token_idx].sum() > 0:
                    child.routing_matrix[token_idx] = self.routing_matrix[token_idx]
                else:
                    child.routing_matrix[token_idx] = other.routing_matrix[token_idx]
        
        # Crossover expert graph
        expert_mask = torch.rand_like(self.expert_graph) > 0.5
        child.expert_graph = torch.where(
            expert_mask,
            self.expert_graph,
            other.expert_graph
        )
        
        # Average routing parameters
        child.routing_params = RoutingParams(
            temperature=(self.routing_params.temperature + other.routing_params.temperature) / 2,
            top_k=int((self.routing_params.top_k + other.routing_params.top_k) / 2),
            load_balancing_weight=(self.routing_params.load_balancing_weight + other.routing_params.load_balancing_weight) / 2,
            diversity_weight=(self.routing_params.diversity_weight + other.routing_params.diversity_weight) / 2,
            sparsity_target=(self.routing_params.sparsity_target + other.routing_params.sparsity_target) / 2
        )
        
        child.generation = max(self.generation, other.generation) + 1
        return child
    
    def get_routing_weights(self, token_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing weights for given token embeddings.
        
        Args:
            token_embeddings: Input token embeddings [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tuple of (routing_weights, selected_experts) tensors
        """
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        
        # Simple learned routing based on topology structure
        # In practice, this would use learned parameters
        base_weights = self.routing_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply temperature scaling
        scaled_weights = base_weights / self.routing_params.temperature
        
        # Apply softmax to get probabilities
        routing_probs = torch.softmax(scaled_weights, dim=-1)
        
        # Apply top-k selection
        top_k = min(self.routing_params.top_k, self.num_experts)
        routing_weights, selected_experts = routing_probs.topk(top_k, dim=-1)
        
        # Renormalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, selected_experts
    
    def compute_sparsity(self) -> float:
        """Compute actual sparsity of routing matrix."""
        total_connections = self.routing_matrix.sum().item()
        total_possible = self.routing_matrix.numel()
        return 1.0 - (total_connections / total_possible)
    
    def compute_load_balance_loss(self, expert_usage: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage even expert utilization.
        
        Args:
            expert_usage: Tensor of expert usage frequencies
            
        Returns:
            Load balancing loss
        """
        # Encourage uniform distribution across experts
        uniform_target = torch.ones_like(expert_usage) / self.num_experts
        return torch.nn.functional.mse_loss(expert_usage, uniform_target)
    
    def get_topology_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the topology."""
        return {
            'num_tokens': self.num_tokens,
            'num_experts': self.num_experts,
            'sparsity': self.compute_sparsity(),
            'total_connections': self.routing_matrix.sum().item(),
            'avg_connections_per_token': self.routing_matrix.sum(dim=1).float().mean().item(),
            'avg_connections_per_expert': self.routing_matrix.sum(dim=0).float().mean().item(),
            'expert_graph_density': self.expert_graph.sum().item() / (self.num_experts * (self.num_experts - 1)),
            'generation': self.generation,
            'routing_params': {
                'temperature': self.routing_params.temperature,
                'top_k': self.routing_params.top_k,
                'load_balancing_weight': self.routing_params.load_balancing_weight,
                'diversity_weight': self.routing_params.diversity_weight,
                'sparsity_target': self.routing_params.sparsity_target
            }
        }
    
    def save_topology(self, filepath: str):
        """Save topology to file."""
        state = {
            'routing_matrix': self.routing_matrix.cpu(),
            'expert_graph': self.expert_graph.cpu(),
            'routing_params': self.routing_params,
            'num_tokens': self.num_tokens,
            'num_experts': self.num_experts,
            'sparsity': self.sparsity,
            'generation': self.generation,
            'fitness_history': self.fitness_history
        }
        torch.save(state, filepath)
    
    @classmethod
    def load_topology(cls, filepath: str, device: str = "cpu") -> 'TopologyGenome':
        """Load topology from file."""
        state = torch.load(filepath, map_location=device, weights_only=False)
        
        topology = cls(
            num_tokens=state['num_tokens'],
            num_experts=state['num_experts'],
            sparsity=state['sparsity'],
            routing_params=state['routing_params'],
            device=device
        )
        
        topology.routing_matrix = state['routing_matrix'].to(device)
        topology.expert_graph = state['expert_graph'].to(device)
        topology.generation = state.get('generation', 0)
        topology.fitness_history = state.get('fitness_history', [])
        
        return topology
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert topology to dictionary for serialization."""
        return {
            'routing_matrix': self.routing_matrix.cpu().numpy().tolist(),
            'expert_graph': self.expert_graph.cpu().numpy().tolist(),
            'routing_params': {
                'temperature': self.routing_params.temperature,
                'top_k': self.routing_params.top_k,
                'load_balancing_weight': self.routing_params.load_balancing_weight,
                'diversity_weight': self.routing_params.diversity_weight,
                'sparsity_target': self.routing_params.sparsity_target
            },
            'num_tokens': self.num_tokens,
            'num_experts': self.num_experts,
            'sparsity': self.sparsity,
            'generation': self.generation,
            'fitness_history': self.fitness_history,
            'device': str(self.device)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str = "cpu") -> 'TopologyGenome':
        """Create topology from dictionary."""
        import numpy as np
        
        routing_params = RoutingParams(**data['routing_params'])
        
        topology = cls(
            num_tokens=data['num_tokens'],
            num_experts=data['num_experts'],
            sparsity=data['sparsity'],
            routing_params=routing_params,
            device=device
        )
        
        # Load matrices
        topology.routing_matrix = torch.tensor(data['routing_matrix'], dtype=torch.float32, device=device)
        topology.expert_graph = torch.tensor(data['expert_graph'], dtype=torch.float32, device=device)
        topology.generation = data.get('generation', 0)
        topology.fitness_history = data.get('fitness_history', [])
        
        return topology
    
    def get_routing_mask(self, seq_len: int) -> torch.Tensor:
        """
        Get routing mask for the given sequence length.
        
        Args:
            seq_len: Target sequence length
            
        Returns:
            Routing mask tensor [seq_len, num_experts]
        """
        if seq_len == self.num_tokens:
            # Exact match - return routing matrix directly
            return self.routing_matrix.clone()
        elif seq_len < self.num_tokens:
            # Truncate routing matrix
            return self.routing_matrix[:seq_len, :].clone()
        else:
            # Extend routing matrix by repeating patterns
            # Use modular indexing to repeat the pattern
            indices = torch.arange(seq_len, device=self.device) % self.num_tokens
            return self.routing_matrix[indices, :].clone()