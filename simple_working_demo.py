#!/usr/bin/env python3
"""
Simple Working Demo: Self-Evolving MoE Router (CPU-only, minimal dependencies)
Generation 1 - Core functionality demonstration
"""

import numpy as np
import random
import json
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleTopology:
    """Simplified routing topology for CPU-only demo."""
    
    def __init__(self, num_tokens: int, num_experts: int, sparsity: float = 0.1):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.sparsity = sparsity
        
        # Initialize sparse routing matrix as numpy array
        self.routing_matrix = np.zeros((num_tokens, num_experts))
        
        # Ensure each token connects to at least one expert
        for token_idx in range(num_tokens):
            expert_idx = random.randint(0, num_experts - 1)
            self.routing_matrix[token_idx, expert_idx] = 1
        
        # Add additional connections based on sparsity
        total_possible = num_tokens * num_experts
        target_connections = int(total_possible * (1 - sparsity))
        current_connections = np.sum(self.routing_matrix)
        
        # Add more connections if needed
        needed_connections = max(0, target_connections - int(current_connections))
        for _ in range(needed_connections):
            token_idx = random.randint(0, num_tokens - 1)
            expert_idx = random.randint(0, num_experts - 1)
            self.routing_matrix[token_idx, expert_idx] = 1
    
    def get_routing_mask(self, seq_len: int) -> np.ndarray:
        """Get routing mask for given sequence length."""
        if seq_len <= self.num_tokens:
            return self.routing_matrix[:seq_len, :]
        else:
            # Repeat pattern for longer sequences
            repetitions = (seq_len + self.num_tokens - 1) // self.num_tokens
            repeated = np.tile(self.routing_matrix, (repetitions, 1))
            return repeated[:seq_len, :]
    
    def mutate(self, mutation_rate: float = 0.1):
        """Apply mutations to the topology."""
        # Add connection
        if random.random() < mutation_rate:
            zero_positions = np.where(self.routing_matrix == 0)
            if len(zero_positions[0]) > 0:
                idx = random.randint(0, len(zero_positions[0]) - 1)
                t, e = zero_positions[0][idx], zero_positions[1][idx]
                self.routing_matrix[t, e] = 1
        
        # Remove connection (but keep at least one per token)
        if random.random() < mutation_rate:
            for token_idx in range(self.num_tokens):
                if np.sum(self.routing_matrix[token_idx]) > 1:
                    expert_candidates = np.where(self.routing_matrix[token_idx] == 1)[0]
                    if len(expert_candidates) > 1:
                        expert_to_remove = random.choice(expert_candidates)
                        self.routing_matrix[token_idx, expert_to_remove] = 0
                    break
    
    def crossover(self, other: 'SimpleTopology') -> 'SimpleTopology':
        """Create offspring topology through crossover."""
        child = SimpleTopology(self.num_tokens, self.num_experts, self.sparsity)
        
        # Uniform crossover
        mask = np.random.random((self.num_tokens, self.num_experts)) > 0.5
        child.routing_matrix = np.where(mask, self.routing_matrix, other.routing_matrix)
        
        # Ensure each token has at least one connection
        for token_idx in range(self.num_tokens):
            if np.sum(child.routing_matrix[token_idx]) == 0:
                expert_idx = random.randint(0, self.num_experts - 1)
                child.routing_matrix[token_idx, expert_idx] = 1
        
        return child
    
    def compute_sparsity(self) -> float:
        """Compute actual sparsity of the topology."""
        total_connections = np.sum(self.routing_matrix)
        total_possible = self.routing_matrix.size
        return 1.0 - (total_connections / total_possible)


class SimpleMoEModel:
    """Simplified MoE model using numpy."""
    
    def __init__(self, input_dim: int = 64, num_experts: int = 8, hidden_dim: int = 128):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Initialize expert weights (simplified)
        self.expert_weights = []
        for _ in range(num_experts):
            # Simple 2-layer MLP weights
            w1 = np.random.randn(input_dim, hidden_dim) * 0.1
            b1 = np.zeros(hidden_dim)
            w2 = np.random.randn(hidden_dim, input_dim) * 0.1
            b2 = np.zeros(input_dim)
            self.expert_weights.append((w1, b1, w2, b2))
        
        # Router weights
        self.router_weights = np.random.randn(input_dim, num_experts) * 0.1
        self.router_bias = np.zeros(num_experts)
        
        self.current_topology = None
    
    def set_routing_topology(self, topology: Optional[SimpleTopology]):
        """Set the current routing topology."""
        self.current_topology = topology
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def forward(self, x: np.ndarray, top_k: int = 2) -> Tuple[np.ndarray, Dict]:
        """Forward pass through the MoE model."""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute router logits
        router_logits = np.dot(x, self.router_weights) + self.router_bias
        
        # Apply topology mask if available
        if self.current_topology is not None:
            mask = self.current_topology.get_routing_mask(seq_len)
            # Broadcast mask to match router_logits shape
            mask_expanded = np.expand_dims(mask, 0)  # Add batch dimension
            router_logits = router_logits * mask_expanded + (1 - mask_expanded) * (-1e9)
        
        # Top-k selection (simplified)
        routing_weights = self.softmax(router_logits)
        
        # Get top-k experts for each token
        top_k_indices = np.argsort(routing_weights, axis=-1)[..., -top_k:]
        
        # Initialize output and auxiliary info
        output = np.zeros_like(x)
        expert_usage = np.zeros(self.num_experts)
        
        # Route through experts
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                token_input = x[batch_idx, seq_idx]
                token_output = np.zeros(self.input_dim)
                
                for k in range(top_k):
                    expert_idx = top_k_indices[batch_idx, seq_idx, k]
                    weight = routing_weights[batch_idx, seq_idx, expert_idx]
                    
                    # Forward through expert
                    w1, b1, w2, b2 = self.expert_weights[expert_idx]
                    hidden = self.relu(np.dot(token_input, w1) + b1)
                    expert_out = np.dot(hidden, w2) + b2
                    
                    # Weight and accumulate
                    token_output += weight * expert_out
                    expert_usage[expert_idx] += weight
                
                output[batch_idx, seq_idx] = token_output
        
        aux_info = {'expert_usage': expert_usage}
        return output, aux_info


class SimpleEvolver:
    """Simplified evolutionary algorithm."""
    
    def __init__(self, num_tokens: int = 16, num_experts: int = 8, population_size: int = 15):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.population_size = population_size
        
        # Initialize population
        self.population = [
            SimpleTopology(num_tokens, num_experts, sparsity=random.uniform(0.1, 0.4))
            for _ in range(population_size)
        ]
        
        self.generation = 0
        self.best_topology = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
    
    def evaluate_fitness(self, topology: SimpleTopology, model: SimpleMoEModel, 
                        data_batches: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate fitness of a topology."""
        model.set_routing_topology(topology)
        
        total_loss = 0.0
        total_samples = 0
        expert_usage = np.zeros(self.num_experts)
        
        # Evaluate on subset of data for efficiency
        for batch_idx, (inputs, targets) in enumerate(data_batches):
            if batch_idx >= 3:  # Limit evaluation batches
                break
            
            outputs, aux_info = model.forward(inputs)
            
            # Simple MSE loss
            loss = np.mean((outputs - targets) ** 2)
            total_loss += loss * inputs.shape[0]
            total_samples += inputs.shape[0]
            
            expert_usage += aux_info['expert_usage']
        
        # Calculate metrics
        avg_loss = total_loss / max(total_samples, 1)
        sparsity = topology.compute_sparsity()
        
        # Load balance score
        if np.sum(expert_usage) > 0:
            usage_probs = expert_usage / np.sum(expert_usage)
            # Entropy calculation
            entropy = -np.sum(usage_probs * np.log(usage_probs + 1e-8))
            max_entropy = np.log(self.num_experts)
            load_balance = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            load_balance = 0.0
        
        # Combined fitness (minimize loss, maximize sparsity and balance)
        fitness = -avg_loss + 0.2 * sparsity + 0.3 * load_balance
        
        return fitness
    
    def tournament_select(self, fitness_scores: List[float], tournament_size: int = 3) -> SimpleTopology:
        """Tournament selection."""
        tournament_indices = random.sample(range(len(self.population)), min(tournament_size, len(self.population)))
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return self.population[best_idx]
    
    def evolve_one_generation(self, model: SimpleMoEModel, data_batches: List) -> Dict:
        """Execute one generation of evolution."""
        logger.info(f"Generation {self.generation}")
        
        # Evaluate fitness
        fitness_scores = []
        for i, topology in enumerate(self.population):
            fitness = self.evaluate_fitness(topology, model, data_batches)
            fitness_scores.append(fitness)
            
            # Track best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_topology = topology
                logger.info(f"New best fitness: {fitness:.4f}")
        
        self.fitness_history.append(fitness_scores)
        
        # Create next generation
        new_population = []
        
        # Elitism - keep best 2
        best_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[-2:]
        for idx in best_indices:
            new_population.append(self.population[idx])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_select(fitness_scores)
            parent2 = self.tournament_select(fitness_scores)
            
            # Crossover
            child = parent1.crossover(parent2)
            
            # Mutation
            child.mutate(0.15)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        stats = {
            'generation': self.generation - 1,
            'best_fitness': max(fitness_scores),
            'avg_fitness': np.mean(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'std_fitness': np.std(fitness_scores)
        }
        
        logger.info(f"Gen {stats['generation']}: Best={stats['best_fitness']:.4f}, "
                   f"Avg={stats['avg_fitness']:.4f}")
        
        return stats
    
    def evolve(self, model: SimpleMoEModel, data_batches: List, generations: int = 25) -> List[Dict]:
        """Run complete evolution."""
        logger.info(f"Starting evolution for {generations} generations")
        
        all_stats = []
        for gen in range(generations):
            stats = self.evolve_one_generation(model, data_batches)
            all_stats.append(stats)
            
            # Early stopping if converged
            if len(self.fitness_history) >= 8:
                recent_best = [max(scores) for scores in self.fitness_history[-8:]]
                if max(recent_best) - min(recent_best) < 0.005:
                    logger.info("Converged early!")
                    break
        
        logger.info(f"Evolution complete! Best fitness: {self.best_fitness:.4f}")
        return all_stats


def create_demo_data(batch_size: int = 4, seq_len: int = 16, hidden_dim: int = 64, 
                    num_batches: int = 12) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create synthetic demo data."""
    data = []
    for _ in range(num_batches):
        # Create structured input data
        inputs = np.random.randn(batch_size, seq_len, hidden_dim)
        
        # Create targets (simple transformation)
        targets = inputs * 0.8 + 0.1 * np.random.randn(batch_size, seq_len, hidden_dim)
        
        data.append((inputs, targets))
    
    return data


def run_simple_demo():
    """Run the simplified CPU-only demo."""
    logger.info("🚀 Starting Simple Self-Evolving MoE Router Demo")
    
    # Configuration
    config = {
        'input_dim': 64,
        'num_experts': 8,
        'hidden_dim': 128,
        'num_tokens': 16,
        'population_size': 15,
        'generations': 25,
        'batch_size': 4,
        'seq_len': 16
    }
    
    logger.info(f"Configuration: {config}")
    
    # Create model
    model = SimpleMoEModel(
        input_dim=config['input_dim'],
        num_experts=config['num_experts'],
        hidden_dim=config['hidden_dim']
    )
    
    logger.info(f"Created simple MoE model with {config['num_experts']} experts")
    
    # Create data
    data_batches = create_demo_data(
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
        hidden_dim=config['input_dim']
    )
    
    logger.info(f"Generated {len(data_batches)} demo batches")
    
    # Create evolver
    evolver = SimpleEvolver(
        num_tokens=config['num_tokens'],
        num_experts=config['num_experts'],
        population_size=config['population_size']
    )
    
    logger.info(f"Initialized evolver with population {config['population_size']}")
    
    # Run evolution
    start_time = time.time()
    evolution_stats = evolver.evolve(model, data_batches, config['generations'])
    end_time = time.time()
    
    logger.info(f"Evolution completed in {end_time - start_time:.2f} seconds")
    
    # Test best topology
    if evolver.best_topology:
        model.set_routing_topology(evolver.best_topology)
        logger.info(f"Best topology sparsity: {evolver.best_topology.compute_sparsity():.3f}")
        
        # Test inference
        test_input = np.random.randn(2, config['seq_len'], config['input_dim'])
        output, aux_info = model.forward(test_input)
        logger.info(f"Test output shape: {output.shape}")
        logger.info(f"Expert usage: {aux_info['expert_usage']}")
    
    # Prepare results
    results = {
        'config': config,
        'evolution_stats': evolution_stats,
        'final_fitness': float(evolver.best_fitness),
        'total_generations': evolver.generation,
        'computation_time': end_time - start_time,
        'best_topology_sparsity': float(evolver.best_topology.compute_sparsity()) if evolver.best_topology else None,
        'convergence_generation': len(evolution_stats)
    }
    
    # Save results
    results_dir = Path("evolution_results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "simple_demo_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_dir}/simple_demo_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 SIMPLE DEMO COMPLETE!")
    print("="*60)
    print(f"✅ Evolution successful with {config['num_experts']} experts")
    print(f"📊 Final fitness: {evolver.best_fitness:.4f}")
    print(f"🧬 Generations: {evolver.generation}")
    print(f"⚡ Runtime: {end_time - start_time:.1f}s")
    if evolver.best_topology:
        print(f"🕸️  Topology sparsity: {evolver.best_topology.compute_sparsity():.3f}")
    print(f"💾 Results: {results_dir}/simple_demo_results.json")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = run_simple_demo()