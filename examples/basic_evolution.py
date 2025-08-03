#!/usr/bin/env python3
"""
Basic example of evolving MoE routing topology.

This example demonstrates the core evolution loop for discovering
optimal routing patterns in a simple classification task.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from self_evolving_moe import EvolvingMoERouter, ExpertPool, TopologyGenome
from self_evolving_moe.evolution.router import EvolutionConfig


class SimpleMoEModel(nn.Module):
    """Simple MoE model for demonstration."""
    
    def __init__(self, expert_pool: ExpertPool, num_classes: int = 10):
        super().__init__()
        self.expert_pool = expert_pool
        self.router = nn.Linear(expert_pool.expert_dim, expert_pool.num_experts)
        self.classifier = nn.Linear(expert_pool.expert_dim, num_classes)
        self.current_topology = None
        
    def set_routing_topology(self, topology: TopologyGenome):
        """Set routing topology."""
        self.current_topology = topology
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, input_dim = x.shape
        
        if self.current_topology is not None:
            # Use evolved routing
            routing_weights, selected_experts = self.current_topology.get_routing_weights(
                x.unsqueeze(1)  # Add sequence dimension
            )
            
            # Route through experts (simplified)
            expert_outputs = []
            for expert_idx in range(self.expert_pool.num_experts):
                expert_output = self.expert_pool.experts[expert_idx](x.unsqueeze(1))
                expert_outputs.append(expert_output.squeeze(1))
            
            # Combine expert outputs
            output = torch.zeros_like(x)
            for i, expert_output in enumerate(expert_outputs):
                weight = routing_weights[:, 0, :].sum(dim=1, keepdim=True)  # Simplified weighting
                if weight.sum() > 0:
                    output += expert_output * (weight / max(weight.sum(), 1e-8))
            
        else:
            # Use learned routing
            routing_logits = self.router(x)
            routing_weights = torch.softmax(routing_logits, dim=-1)
            
            # Route through experts
            output = torch.zeros_like(x)
            for i, expert in enumerate(self.expert_pool.experts):
                expert_output = expert(x.unsqueeze(1)).squeeze(1)
                output += expert_output * routing_weights[:, i:i+1]
        
        # Classification
        return self.classifier(output)


def create_dummy_data(num_samples=1000, input_dim=128, num_classes=10):
    """Create dummy classification dataset."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def main():
    """Run basic evolution example."""
    print("Setting up basic evolution example...")
    
    # Configuration
    config = EvolutionConfig(
        population_size=20,  # Smaller for demo
        generations=50,      # Fewer generations for demo
        mutation_rate=0.1,
        accuracy_weight=1.0,
        latency_weight=-0.1,
        sparsity_weight=0.2
    )
    
    # Create expert pool
    expert_pool = ExpertPool(
        num_experts=8,
        expert_dim=128,
        expert_type="mlp",
        device="cpu"
    )
    
    print(f"Created expert pool with {expert_pool.num_experts} experts")
    
    # Create model
    model = SimpleMoEModel(expert_pool, num_classes=10)
    
    # Create dummy dataset
    train_data = create_dummy_data(num_samples=1000, input_dim=128)
    val_data = create_dummy_data(num_samples=200, input_dim=128)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    print(f"Created datasets: {len(train_data)} train, {len(val_data)} val")
    
    # Initialize evolutionary router
    evolver = EvolvingMoERouter(
        expert_pool=expert_pool,
        config=config,
        device="cpu"
    )
    
    print(f"Initialized evolution with population size {config.population_size}")
    print("Starting evolution...")
    
    # Run evolution
    best_topology = evolver.evolve(
        model=model,
        train_data=train_loader,
        val_data=val_loader,
        generations=config.generations
    )
    
    # Get evolution statistics
    stats = evolver.get_evolution_stats()
    
    print("\n" + "="*50)
    print("EVOLUTION RESULTS")
    print("="*50)
    print(f"Generations run: {stats['generations_run']}")
    print(f"Best fitness achieved: {stats['best_fitness']:.6f}")
    print(f"Convergence generation: {stats.get('convergence_generation', 'N/A')}")
    
    if best_topology:
        summary = best_topology.get_topology_summary()
        print(f"\nBest topology summary:")
        print(f"  Sparsity: {summary['sparsity']:.3f}")
        print(f"  Total connections: {summary['total_connections']}")
        print(f"  Avg connections per token: {summary['avg_connections_per_token']:.2f}")
        print(f"  Avg connections per expert: {summary['avg_connections_per_expert']:.2f}")
        print(f"  Generation: {summary['generation']}")
        
        # Test the evolved topology
        print(f"\nTesting evolved topology...")
        model.set_routing_topology(best_topology)
        model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        print(f"Final accuracy with evolved topology: {accuracy:.4f}")
        
        # Save the best topology
        best_topology.save_topology("best_evolved_topology.pt")
        print("Saved best topology to 'best_evolved_topology.pt'")
    
    print("\nEvolution completed successfully!")


if __name__ == "__main__":
    main()