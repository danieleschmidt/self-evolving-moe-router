#!/usr/bin/env python3
"""
Basic example of evolving MoE routing topology.

This example demonstrates the core evolution loop for discovering
optimal routing patterns in a simple classification task.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# TODO: Import actual implementation once available
# from self_evolving_moe import EvolvingMoERouter, ExpertPool


def create_dummy_data(num_samples=1000, input_dim=128, num_classes=10):
    """Create dummy classification dataset."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def evaluate_topology(topology, model, data_loader):
    """Evaluate routing topology performance."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Apply topology to model
            model.set_routing_topology(topology)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return correct / total


def main():
    """Run basic evolution example."""
    print("Setting up basic evolution example...")
    
    # Configuration
    config = {
        "num_experts": 8,
        "expert_dim": 128,
        "population_size": 50,
        "generations": 100,
        "mutation_rate": 0.1,
    }
    
    # Create dummy dataset
    train_data = create_dummy_data(num_samples=5000)
    val_data = create_dummy_data(num_samples=1000)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    print(f"Created datasets: {len(train_data)} train, {len(val_data)} val")
    
    # TODO: Implement actual evolution once classes are available
    print("Evolution would run here with the following configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Placeholder evolution loop
    best_fitness = 0.0
    for generation in range(config["generations"]):
        # This would be the actual evolution step
        fitness = 0.5 + 0.4 * (generation / config["generations"])  # Dummy improvement
        if fitness > best_fitness:
            best_fitness = fitness
            print(f"Generation {generation}: New best fitness = {fitness:.4f}")
    
    print(f"Evolution completed. Best fitness: {best_fitness:.4f}")


if __name__ == "__main__":
    main()