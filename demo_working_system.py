#!/usr/bin/env python3
"""
Working demonstration of Self-Evolving MoE Router
Generation 1: MAKE IT WORK - Basic functionality implementation

This is a simplified working version that demonstrates core concepts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple working versions of core components

@dataclass
class SimpleConfig:
    """Simple configuration for demonstration."""
    num_experts: int = 8
    hidden_dim: int = 256
    seq_len: int = 64
    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.1
    sparsity: float = 0.1
    device: str = "cpu"

class SimpleExpert(nn.Module):
    """Simple expert implementation."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SimpleTopology:
    """Simple topology genome for evolution."""
    
    def __init__(self, num_tokens: int, num_experts: int, sparsity: float = 0.1):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.sparsity = sparsity
        
        # Create sparse routing matrix
        self.routing_matrix = self._create_sparse_matrix()
        self.fitness = 0.0
        
    def _create_sparse_matrix(self) -> torch.Tensor:
        """Create sparse binary routing matrix."""
        matrix = torch.zeros(self.num_tokens, self.num_experts)
        
        # Ensure each token connects to at least one expert
        for token_idx in range(self.num_tokens):
            expert_idx = random.randint(0, self.num_experts - 1)
            matrix[token_idx, expert_idx] = 1
        
        # Add additional connections based on sparsity
        total_connections = int(self.num_tokens * self.num_experts * (1 - self.sparsity))
        current_connections = matrix.sum().item()
        
        for _ in range(int(total_connections - current_connections)):
            token_idx = random.randint(0, self.num_tokens - 1)
            expert_idx = random.randint(0, self.num_experts - 1)
            matrix[token_idx, expert_idx] = 1
            
        return matrix
    
    def mutate(self) -> 'SimpleTopology':
        """Create a mutated version."""
        new_topology = SimpleTopology(self.num_tokens, self.num_experts, self.sparsity)
        new_topology.routing_matrix = self.routing_matrix.clone()
        
        # Random mutation: flip some connections
        num_mutations = max(1, int(self.routing_matrix.numel() * 0.05))
        for _ in range(num_mutations):
            token_idx = random.randint(0, self.num_tokens - 1)
            expert_idx = random.randint(0, self.num_experts - 1)
            new_topology.routing_matrix[token_idx, expert_idx] = 1 - new_topology.routing_matrix[token_idx, expert_idx]
        
        # Ensure each token has at least one connection
        for token_idx in range(self.num_tokens):
            if new_topology.routing_matrix[token_idx].sum() == 0:
                expert_idx = random.randint(0, self.num_experts - 1)
                new_topology.routing_matrix[token_idx, expert_idx] = 1
                
        return new_topology
    
    def crossover(self, other: 'SimpleTopology') -> 'SimpleTopology':
        """Create offspring through crossover."""
        child = SimpleTopology(self.num_tokens, self.num_experts, self.sparsity)
        
        # Uniform crossover
        mask = torch.rand_like(self.routing_matrix) > 0.5
        child.routing_matrix = torch.where(mask, self.routing_matrix, other.routing_matrix)
        
        return child

class SimpleMoEModel(nn.Module):
    """Simple MoE model for demonstration."""
    
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            SimpleExpert(config.hidden_dim) for _ in range(config.num_experts)
        ])
        self.router = nn.Linear(config.hidden_dim, config.num_experts)
        self.classifier = nn.Linear(config.hidden_dim, 10)  # 10 classes
        self.current_topology = None
        
    def set_routing_topology(self, topology: SimpleTopology):
        """Set the routing topology."""
        self.current_topology = topology
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        
        # Simple routing based on topology if available
        if self.current_topology is not None:
            # Use topology-based routing
            output = torch.zeros_like(x)
            for token_idx in range(min(seq_len, self.current_topology.num_tokens)):
                for expert_idx in range(self.config.num_experts):
                    if self.current_topology.routing_matrix[token_idx, expert_idx] > 0:
                        expert_out = self.experts[expert_idx](x[:, token_idx])
                        output[:, token_idx] += expert_out * 0.5  # Simple weighting
        else:
            # Learned routing
            router_weights = F.softmax(self.router(x), dim=-1)
            output = torch.zeros_like(x)
            
            for expert_idx, expert in enumerate(self.experts):
                expert_out = expert(x)
                weights = router_weights[..., expert_idx].unsqueeze(-1)
                output += expert_out * weights
        
        # Classification (use average pooling)
        pooled = output.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

class SimpleEvolutionEngine:
    """Simple evolutionary algorithm for topology discovery."""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.population = []
        self.generation = 0
        self.best_topology = None
        self.best_fitness = -float('inf')
        
        # Initialize population
        for _ in range(config.population_size):
            topology = SimpleTopology(config.seq_len, config.num_experts, config.sparsity)
            self.population.append(topology)
            
        logger.info(f"Initialized population with {len(self.population)} topologies")
    
    def evaluate_fitness(self, topology: SimpleTopology, model: SimpleMoEModel, data_loader) -> float:
        """Evaluate topology fitness."""
        model.set_routing_topology(topology)
        model.eval()
        
        total_correct = 0
        total_samples = 0
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if batch_idx > 5:  # Limit evaluation for speed
                    break
                    
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                predictions = outputs.argmax(dim=-1)
                
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)
                total_loss += loss.item()
        
        accuracy = total_correct / max(total_samples, 1)
        avg_loss = total_loss / max(batch_idx + 1, 1)
        
        # Combine accuracy and efficiency (sparsity)
        sparsity_score = 1.0 - topology.routing_matrix.float().mean()
        fitness = accuracy + 0.1 * sparsity_score - 0.01 * avg_loss
        
        return fitness
    
    def evolve_one_generation(self, model: SimpleMoEModel, data_loader) -> Dict:
        """Execute one evolution generation."""
        logger.info(f"Evolution Generation {self.generation}")
        
        # Evaluate fitness for all topologies
        fitness_scores = []
        for i, topology in enumerate(self.population):
            fitness = self.evaluate_fitness(topology, model, data_loader)
            topology.fitness = fitness
            fitness_scores.append(fitness)
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_topology = topology
                logger.info(f"New best fitness: {fitness:.4f}")
        
        # Selection and reproduction
        new_population = []
        
        # Elitism - keep best 20%
        elite_count = max(1, int(0.2 * len(self.population)))
        sorted_population = sorted(self.population, key=lambda t: t.fitness, reverse=True)
        new_population.extend(sorted_population[:elite_count])
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            child = parent1.crossover(parent2)
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child = child.mutate()
                
            new_population.append(child)
        
        self.population = new_population[:self.config.population_size]
        self.generation += 1
        
        stats = {
            'generation': self.generation - 1,
            'best_fitness': max(fitness_scores),
            'avg_fitness': np.mean(fitness_scores),
            'population_diversity': self._calculate_diversity()
        }
        
        logger.info(f"Gen {self.generation-1}: Best={stats['best_fitness']:.4f}, "
                   f"Avg={stats['avg_fitness']:.4f}")
        
        return stats
    
    def _tournament_selection(self, tournament_size: int = 3) -> SimpleTopology:
        """Tournament selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda t: t.fitness)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Hamming distance
                diff = (self.population[i].routing_matrix != self.population[j].routing_matrix).float()
                distance = diff.mean().item()
                distances.append(distance)
        
        return np.mean(distances)
    
    def evolve(self, model: SimpleMoEModel, data_loader, generations: int = None) -> SimpleTopology:
        """Run full evolution process."""
        target_gens = generations or self.config.generations
        
        logger.info(f"Starting evolution for {target_gens} generations")
        
        for gen in range(target_gens):
            stats = self.evolve_one_generation(model, data_loader)
            
            if gen % 10 == 0 or gen == target_gens - 1:
                logger.info(f"Generation {gen} completed - "
                           f"Best: {stats['best_fitness']:.4f}")
        
        logger.info(f"Evolution complete! Best fitness: {self.best_fitness:.4f}")
        return self.best_topology

def create_sample_data(config: SimpleConfig, num_samples: int = 320):
    """Create sample dataset for demonstration."""
    from torch.utils.data import TensorDataset, DataLoader
    
    # Generate synthetic sequence data
    inputs = torch.randn(num_samples, config.seq_len, config.hidden_dim)
    targets = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return data_loader

def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Self-Evolving MoE Router Demonstration")
    
    # Configuration
    config = SimpleConfig(
        num_experts=8,
        hidden_dim=256,
        seq_len=64,
        population_size=20,
        generations=30,
        mutation_rate=0.15,
        sparsity=0.1,
        device="cpu"
    )
    
    logger.info(f"Configuration: {config}")
    
    # Create model
    model = SimpleMoEModel(config)
    logger.info(f"Created MoE model with {config.num_experts} experts")
    
    # Create sample data
    data_loader = create_sample_data(config)
    logger.info("Created sample dataset")
    
    # Initialize evolution engine
    evolution_engine = SimpleEvolutionEngine(config)
    
    # Run evolution
    try:
        best_topology = evolution_engine.evolve(model, data_loader)
        
        logger.info("🎯 Evolution Complete!")
        logger.info(f"Best Fitness Achieved: {evolution_engine.best_fitness:.4f}")
        logger.info(f"Best Topology Sparsity: {1 - best_topology.routing_matrix.float().mean():.3f}")
        
        # Save results
        results_dir = Path("evolution_results")
        results_dir.mkdir(exist_ok=True)
        
        results = {
            'best_fitness': evolution_engine.best_fitness,
            'generations': evolution_engine.generation,
            'final_sparsity': (1 - best_topology.routing_matrix.float().mean()).item(),
            'config': {
                'num_experts': config.num_experts,
                'population_size': config.population_size,
                'generations': config.generations,
                'mutation_rate': config.mutation_rate
            }
        }
        
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save topology matrix
        torch.save(best_topology.routing_matrix, results_dir / "best_topology.pt")
        
        logger.info(f"Results saved to {results_dir}")
        
        # Test the evolved model
        logger.info("Testing evolved model...")
        model.set_routing_topology(best_topology)
        model.eval()
        
        test_accuracy = 0
        test_samples = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                predictions = outputs.argmax(dim=-1)
                test_accuracy += (predictions == targets).sum().item()
                test_samples += targets.size(0)
                break  # Just test one batch
        
        final_accuracy = test_accuracy / test_samples
        logger.info(f"Final test accuracy: {final_accuracy:.3f}")
        
        print("\n" + "="*60)
        print("SELF-EVOLVING MOE ROUTER - GENERATION 1 COMPLETE")
        print("="*60)
        print(f"✅ Evolution Successful!")
        print(f"📊 Best Fitness: {evolution_engine.best_fitness:.4f}")
        print(f"🎯 Test Accuracy: {final_accuracy:.3f}")
        print(f"🔬 Topology Sparsity: {1 - best_topology.routing_matrix.float().mean():.3f}")
        print(f"🧬 Generations: {evolution_engine.generation}")
        print(f"📁 Results: {results_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        raise

if __name__ == "__main__":
    main()