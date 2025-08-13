#!/usr/bin/env python3
"""
Working Generation 1 Demo: Self-Evolving MoE Router
Simple but functional implementation demonstrating core evolutionary MoE capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleMoEModel(nn.Module):
    """Simple MoE model for demonstration."""
    
    def __init__(self, input_dim: int = 128, num_experts: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Simple experts (MLPs)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        
        # Simple router
        self.router = nn.Linear(input_dim, num_experts)
        
        # Track current topology
        self.current_topology = None
        
    def set_routing_topology(self, topology):
        """Set routing topology."""
        self.current_topology = topology
        
    def forward(self, x: torch.Tensor, top_k: int = 2) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with expert routing."""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Router logits
        router_logits = self.router(x)  # [batch, seq, num_experts]
        
        # Apply topology mask if available
        if self.current_topology is not None:
            mask = self.current_topology.get_routing_mask(seq_len)
            if mask.device != x.device:
                mask = mask.to(x.device)
            router_logits = router_logits * mask + (1 - mask) * (-1e9)
        
        # Top-k selection
        routing_weights, selected_experts = torch.topk(router_logits, top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Route through experts
        output = torch.zeros_like(x)
        aux_info = {'expert_usage': torch.zeros(self.num_experts)}
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx)
            
            if expert_mask.any():
                # Extract tokens for this expert
                token_positions = expert_mask.nonzero()
                if len(token_positions) > 0:
                    batch_indices = token_positions[:, 0]
                    seq_indices = token_positions[:, 1]
                    expert_rank_indices = token_positions[:, 2]
                    
                    expert_input = x[batch_indices, seq_indices]
                    weights = routing_weights[batch_indices, seq_indices, expert_rank_indices].unsqueeze(-1)
                    
                    # Forward through expert
                    expert_output = self.experts[expert_idx](expert_input)
                    weighted_output = expert_output * weights
                    
                    # Add to output
                    output[batch_indices, seq_indices] += weighted_output
                    
                    # Track usage
                    aux_info['expert_usage'][expert_idx] += len(token_positions)
        
        return output, aux_info


class SimpleTopology:
    """Simplified topology for Generation 1."""
    
    def __init__(self, num_tokens: int, num_experts: int, sparsity: float = 0.1):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.sparsity = sparsity
        
        # Initialize random sparse routing matrix
        self.routing_matrix = torch.zeros(num_tokens, num_experts)
        
        # Ensure each token has at least one connection
        for token_idx in range(num_tokens):
            expert_idx = np.random.randint(0, num_experts)
            self.routing_matrix[token_idx, expert_idx] = 1
        
        # Add additional random connections based on sparsity
        total_possible = num_tokens * num_experts
        num_connections = int(total_possible * (1 - sparsity))
        
        for _ in range(num_connections - num_tokens):  # Already have num_tokens connections
            token_idx = np.random.randint(0, num_tokens)
            expert_idx = np.random.randint(0, num_experts)
            self.routing_matrix[token_idx, expert_idx] = 1
    
    def get_routing_mask(self, seq_len: int) -> torch.Tensor:
        """Get routing mask for given sequence length."""
        if seq_len <= self.num_tokens:
            return self.routing_matrix[:seq_len, :]
        else:
            # Repeat pattern for longer sequences
            repetitions = (seq_len + self.num_tokens - 1) // self.num_tokens
            repeated = self.routing_matrix.repeat(repetitions, 1)
            return repeated[:seq_len, :]
    
    def mutate(self, mutation_rate: float = 0.1):
        """Simple mutation operation."""
        # Add/remove connections
        if np.random.random() < mutation_rate:
            # Add connection
            zero_positions = (self.routing_matrix == 0).nonzero()
            if len(zero_positions) > 0:
                idx = np.random.randint(0, len(zero_positions))
                t, e = zero_positions[idx]
                self.routing_matrix[t, e] = 1
        
        if np.random.random() < mutation_rate:
            # Remove connection (but keep at least one per token)
            for token_idx in range(self.num_tokens):
                if self.routing_matrix[token_idx].sum() > 1:
                    expert_candidates = (self.routing_matrix[token_idx] == 1).nonzero().flatten()
                    if len(expert_candidates) > 1:
                        expert_to_remove = expert_candidates[np.random.randint(0, len(expert_candidates))]
                        self.routing_matrix[token_idx, expert_to_remove] = 0
                    break
    
    def crossover(self, other):
        """Simple crossover operation."""
        child = SimpleTopology(self.num_tokens, self.num_experts, self.sparsity)
        
        # Uniform crossover
        mask = torch.rand_like(self.routing_matrix) > 0.5
        child.routing_matrix = torch.where(mask, self.routing_matrix, other.routing_matrix)
        
        # Ensure each token has at least one connection
        for token_idx in range(self.num_tokens):
            if child.routing_matrix[token_idx].sum() == 0:
                expert_idx = np.random.randint(0, self.num_experts)
                child.routing_matrix[token_idx, expert_idx] = 1
        
        return child
    
    def compute_sparsity(self) -> float:
        """Compute actual sparsity."""
        total_connections = self.routing_matrix.sum().item()
        total_possible = self.routing_matrix.numel()
        return 1.0 - (total_connections / total_possible)


class SimpleEvolver:
    """Simplified evolutionary algorithm for Generation 1."""
    
    def __init__(self, num_tokens: int = 32, num_experts: int = 8, population_size: int = 20):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.population_size = population_size
        
        # Initialize population
        self.population = [
            SimpleTopology(num_tokens, num_experts, sparsity=np.random.uniform(0.05, 0.3))
            for _ in range(population_size)
        ]
        
        self.generation = 0
        self.best_topology = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
    
    def evaluate_fitness(self, topology: SimpleTopology, model: SimpleMoEModel, data_loader) -> float:
        """Evaluate topology fitness."""
        model.set_routing_topology(topology)
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        expert_usage = torch.zeros(self.num_experts)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if batch_idx >= 5:  # Limit evaluation for speed
                    break
                
                outputs, aux_info = model(inputs)
                
                # Simple reconstruction loss
                loss = F.mse_loss(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                
                expert_usage += aux_info['expert_usage']
        
        # Calculate metrics
        avg_loss = total_loss / max(total_samples, 1)
        sparsity = topology.compute_sparsity()
        
        # Load balance score (higher is better)
        if expert_usage.sum() > 0:
            usage_probs = expert_usage / expert_usage.sum()
            entropy = -(usage_probs * torch.log(usage_probs + 1e-8)).sum()
            max_entropy = np.log(self.num_experts)
            load_balance = entropy / max_entropy
        else:
            load_balance = 0.0
        
        # Combine objectives (lower loss is better, higher sparsity and balance is better)
        fitness = -avg_loss + 0.1 * sparsity + 0.2 * load_balance
        
        return fitness
    
    def evolve_one_generation(self, model: SimpleMoEModel, data_loader) -> Dict:
        """Execute one generation of evolution."""
        logger.info(f"Starting generation {self.generation}")
        
        # Evaluate fitness
        fitness_scores = []
        for i, topology in enumerate(self.population):
            fitness = self.evaluate_fitness(topology, model, data_loader)
            fitness_scores.append(fitness)
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_topology = topology
                logger.info(f"New best fitness: {fitness:.4f}")
        
        self.fitness_history.append(fitness_scores)
        
        # Create next generation
        new_population = []
        
        # Elitism - keep best 2
        best_indices = np.argsort(fitness_scores)[-2:]
        for idx in best_indices:
            new_population.append(self.population[idx])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select(fitness_scores)
            parent2 = self._tournament_select(fitness_scores)
            
            # Crossover
            child = parent1.crossover(parent2)
            
            # Mutation
            child.mutate(0.1)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        stats = {
            'generation': self.generation - 1,
            'best_fitness': max(fitness_scores),
            'avg_fitness': np.mean(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'population_size': len(self.population)
        }
        
        logger.info(f"Generation {self.generation-1}: Best={stats['best_fitness']:.4f}, "
                   f"Avg={stats['avg_fitness']:.4f}")
        
        return stats
    
    def _tournament_select(self, fitness_scores: List[float], tournament_size: int = 3):
        """Tournament selection."""
        tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return self.population[best_idx]
    
    def evolve(self, model: SimpleMoEModel, data_loader, generations: int = 50):
        """Run full evolution process."""
        logger.info(f"Starting evolution for {generations} generations")
        
        all_stats = []
        for gen in range(generations):
            stats = self.evolve_one_generation(model, data_loader)
            all_stats.append(stats)
            
            # Early stopping if converged
            if len(self.fitness_history) >= 10:
                recent_best = [max(scores) for scores in self.fitness_history[-10:]]
                if max(recent_best) - min(recent_best) < 0.001:
                    logger.info("Converged - stopping early")
                    break
        
        logger.info(f"Evolution complete! Best fitness: {self.best_fitness:.4f}")
        return all_stats


def create_demo_data(batch_size: int = 8, seq_len: int = 32, hidden_dim: int = 128, num_batches: int = 20):
    """Create synthetic demo data."""
    data = []
    for _ in range(num_batches):
        # Create input data with some structure
        inputs = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Create targets (simple reconstruction task)
        targets = inputs + 0.1 * torch.randn_like(inputs)
        
        data.append((inputs, targets))
    
    return data


def run_working_demo():
    """Run the working Generation 1 demo."""
    logger.info("🚀 Starting Self-Evolving MoE Router - Generation 1 Demo")
    
    # Configuration
    config = {
        'input_dim': 128,
        'num_experts': 8,
        'hidden_dim': 256,
        'num_tokens': 32,
        'population_size': 20,
        'generations': 30,
        'batch_size': 8,
        'seq_len': 32
    }
    
    logger.info(f"Configuration: {config}")
    
    # Create model
    model = SimpleMoEModel(
        input_dim=config['input_dim'],
        num_experts=config['num_experts'],
        hidden_dim=config['hidden_dim']
    )
    
    logger.info(f"Created MoE model with {config['num_experts']} experts")
    
    # Create demo data
    data_loader = create_demo_data(
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
        hidden_dim=config['input_dim']
    )
    
    logger.info(f"Created demo dataset with {len(data_loader)} batches")
    
    # Create evolver
    evolver = SimpleEvolver(
        num_tokens=config['num_tokens'],
        num_experts=config['num_experts'],
        population_size=config['population_size']
    )
    
    logger.info(f"Initialized evolver with population size {config['population_size']}")
    
    # Run evolution
    start_time = time.time()
    stats = evolver.evolve(model, data_loader, config['generations'])
    end_time = time.time()
    
    logger.info(f"Evolution completed in {end_time - start_time:.2f} seconds")
    
    # Apply best topology
    if evolver.best_topology:
        model.set_routing_topology(evolver.best_topology)
        logger.info(f"Applied best topology with sparsity: {evolver.best_topology.compute_sparsity():.3f}")
        
        # Test with best topology
        test_input = torch.randn(4, config['seq_len'], config['input_dim'])
        with torch.no_grad():
            output, aux_info = model(test_input)
            logger.info(f"Test output shape: {output.shape}")
            logger.info(f"Expert usage: {aux_info['expert_usage']}")
    
    # Save results
    results = {
        'config': config,
        'evolution_stats': stats,
        'final_fitness': evolver.best_fitness,
        'total_generations': evolver.generation,
        'computation_time': end_time - start_time,
        'best_topology_sparsity': evolver.best_topology.compute_sparsity() if evolver.best_topology else None
    }
    
    # Save to file
    results_path = Path("evolution_results")
    results_path.mkdir(exist_ok=True)
    
    with open(results_path / "generation1_demo_results.json", 'w') as f:
        # Convert tensors to lists for JSON serialization
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                serializable_results[k] = v.tolist()
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                # Handle stats list
                serializable_results[k] = v
            else:
                serializable_results[k] = v
        
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}/generation1_demo_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 GENERATION 1 DEMO COMPLETE!")
    print("="*60)
    print(f"✅ Successfully evolved MoE routing topology")
    print(f"📊 Final fitness score: {evolver.best_fitness:.4f}")
    print(f"🧬 Total generations: {evolver.generation}")
    print(f"⚡ Computation time: {end_time - start_time:.2f}s")
    if evolver.best_topology:
        print(f"🕸️  Best topology sparsity: {evolver.best_topology.compute_sparsity():.3f}")
    print(f"💾 Results saved to: {results_path}/generation1_demo_results.json")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = run_working_demo()