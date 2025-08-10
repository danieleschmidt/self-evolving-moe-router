#!/usr/bin/env python3
"""
Advanced Self-Evolving MoE Router - Generation 1 Implementation
Integrates with the existing sophisticated codebase while ensuring it works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader, TensorDataset

# Import the sophisticated components
from src.self_evolving_moe.evolution.router import EvolvingMoERouter, EvolutionConfig
from src.self_evolving_moe.experts.pool import ExpertPool, ExpertConfig
from src.self_evolving_moe.routing.topology import TopologyGenome
from src.self_evolving_moe.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)

class WorkingMoEModel(nn.Module):
    """A working MoE model that integrates with the sophisticated components."""
    
    def __init__(self, expert_pool: ExpertPool, num_classes: int = 10):
        super().__init__()
        self.expert_pool = expert_pool
        self.classifier = nn.Linear(expert_pool.expert_config.hidden_dim, num_classes)
        self.current_topology = None
        
    def set_routing_topology(self, topology: TopologyGenome):
        """Set the routing topology."""
        self.current_topology = topology
        self.expert_pool.set_routing_topology(topology)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Pass through expert pool
        expert_output, aux_losses = self.expert_pool(x)
        
        # Global average pooling and classification
        pooled_output = expert_output.mean(dim=1)  # [batch_size, hidden_dim]
        logits = self.classifier(pooled_output)
        
        return logits

def create_synthetic_dataset(
    batch_size: int = 32,
    seq_len: int = 128,
    hidden_dim: int = 768,
    num_classes: int = 10,
    num_samples: int = 640
):
    """Create a synthetic dataset for testing."""
    
    # Generate synthetic sequence data that resembles transformer inputs
    inputs = torch.randn(num_samples, seq_len, hidden_dim)
    
    # Create realistic targets with some structure
    # Make certain input patterns more likely to correspond to certain classes
    pattern_features = inputs[:, :10, :10].mean(dim=(1, 2))  # Use first 10 tokens, first 10 dims
    targets = (pattern_features.sum(dim=1) * num_classes / 2).long().clamp(0, num_classes - 1)
    
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    logger.info(f"Created synthetic dataset: {num_samples} samples, "
               f"{seq_len} seq_len, {hidden_dim} hidden_dim, {num_classes} classes")
    
    return data_loader

def run_sophisticated_evolution():
    """Run evolution using the sophisticated codebase components."""
    
    logger.info("🧬 Starting Sophisticated Self-Evolving MoE Router")
    
    # Configuration
    expert_config = ExpertConfig(
        hidden_dim=256,  # Smaller for demo
        intermediate_dim=512,
        num_attention_heads=8,
        expert_type="transformer"
    )
    
    evolution_config = EvolutionConfig(
        population_size=30,
        generations=25,
        mutation_rate=0.12,
        crossover_rate=0.7,
        elitism_rate=0.15,
        selection_method="tournament",
        target_sparsity=0.15,
        accuracy_weight=1.0,
        latency_weight=-0.05,
        sparsity_weight=0.2
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create expert pool
    expert_pool = ExpertPool(
        num_experts=12,
        expert_config=expert_config,
        top_k=3,
        routing_temperature=1.5,
        load_balancing_weight=0.02,
        diversity_weight=0.15
    )
    
    # Create model
    model = WorkingMoEModel(expert_pool, num_classes=10)
    model.to(device)
    
    # Create dataset
    data_loader = create_synthetic_dataset(
        batch_size=24,
        seq_len=64,
        hidden_dim=expert_config.hidden_dim,
        num_samples=480
    )
    
    # Move data to device
    def move_data_to_device(loader, device):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            yield inputs, targets
    
    # Initialize evolutionary router
    router = EvolvingMoERouter(
        num_experts=expert_pool.num_experts,
        num_tokens=64,  # Match sequence length
        config=evolution_config,
        device=device
    )
    
    logger.info(f"Initialized evolution with {evolution_config.population_size} topologies")
    
    try:
        # Run evolution
        logger.info(f"🚀 Starting evolution for {evolution_config.generations} generations")
        
        # Convert data loader for device compatibility
        device_data_loader = [(inputs.to(device), targets.to(device)) 
                             for inputs, targets in data_loader]
        
        class DeviceDataLoader:
            def __init__(self, data):
                self.data = data
            def __iter__(self):
                return iter(self.data)
            def __len__(self):
                return len(self.data)
        
        device_loader = DeviceDataLoader(device_data_loader)
        
        best_topology = router.evolve(model, device_loader)
        
        # Results
        logger.info("🎉 Evolution Complete!")
        logger.info(f"📊 Best Fitness: {router.best_fitness:.4f}")
        logger.info(f"🧬 Generations: {router.generation}")
        logger.info(f"🔬 Best Topology Sparsity: {best_topology.compute_sparsity():.3f}")
        
        # Save comprehensive results
        results_dir = Path("evolution_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save evolution state
        router.save_evolution_state(results_dir / "evolution_state.json")
        
        # Save best topology
        best_topology.save_topology(str(results_dir / "best_topology.pt"))
        
        # Save detailed metrics
        evolution_metrics = router.get_evolution_metrics()
        with open(results_dir / "detailed_metrics.json", 'w') as f:
            json.dump(evolution_metrics, f, indent=2, default=str)
        
        # Save configuration
        config_data = {
            'expert_config': expert_config.__dict__,
            'evolution_config': evolution_config.__dict__,
            'device': device,
            'model_info': {
                'num_experts': expert_pool.num_experts,
                'expert_type': expert_config.expert_type,
                'total_parameters': expert_pool.get_total_parameters(),
                'active_parameters': expert_pool.get_active_parameters()
            }
        }
        
        with open(results_dir / "run_config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Test the evolved model
        logger.info("🧪 Testing evolved model performance...")
        model.set_routing_topology(best_topology)
        model.eval()
        
        test_correct = 0
        test_total = 0
        test_loss = 0
        
        with torch.no_grad():
            for inputs, targets in device_loader:
                if test_total > 200:  # Limit test samples
                    break
                    
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                predictions = outputs.argmax(dim=-1)
                
                test_correct += (predictions == targets).sum().item()
                test_total += targets.size(0)
                test_loss += loss.item()
        
        test_accuracy = test_correct / test_total
        avg_test_loss = test_loss / (test_total / 24)  # Average per batch
        
        # Expert utilization analysis
        expert_util = expert_pool.get_expert_utilization()
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("SOPHISTICATED SELF-EVOLVING MOE ROUTER - GENERATION 1 SUCCESS")
        print("="*80)
        print("🎯 EVOLUTION RESULTS:")
        print(f"   ✅ Evolution Completed Successfully")
        print(f"   📊 Best Fitness Achieved: {router.best_fitness:.6f}")
        print(f"   🧬 Total Generations: {router.generation}")
        print(f"   🏆 Population Size: {evolution_config.population_size}")
        print()
        print("🔬 TOPOLOGY ANALYSIS:")
        print(f"   📈 Sparsity Level: {best_topology.compute_sparsity():.4f}")
        print(f"   🔗 Active Connections: {best_topology.routing_matrix.sum().item():.0f}")
        print(f"   📊 Routing Temperature: {best_topology.routing_params.temperature:.3f}")
        print(f"   🎯 Top-K Selection: {best_topology.routing_params.top_k}")
        print()
        print("🤖 MODEL PERFORMANCE:")
        print(f"   🎯 Test Accuracy: {test_accuracy:.4f}")
        print(f"   📉 Test Loss: {avg_test_loss:.4f}")
        print(f"   👥 Active Experts: {expert_util['active_experts']}")
        print(f"   ⚖️  Load Balance Score: {expert_util['load_balance_score']:.4f}")
        print(f"   💾 Total Parameters: {config_data['model_info']['total_parameters']:,}")
        print()
        print("💾 SAVED ARTIFACTS:")
        print(f"   📁 Results Directory: {results_dir}")
        print(f"   🏷️  Evolution State: evolution_state.json")
        print(f"   🧬 Best Topology: best_topology.pt")
        print(f"   📊 Detailed Metrics: detailed_metrics.json")
        print(f"   ⚙️  Run Configuration: run_config.json")
        print("="*80)
        print("🌟 GENERATION 1: BASIC FUNCTIONALITY - COMPLETE! 🌟")
        print("="*80)
        
        return {
            'best_fitness': router.best_fitness,
            'test_accuracy': test_accuracy,
            'sparsity': best_topology.compute_sparsity(),
            'generations': router.generation,
            'results_dir': str(results_dir)
        }
        
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """Main execution function."""
    try:
        results = run_sophisticated_evolution()
        logger.info("🎉 Self-Evolving MoE Router Generation 1 completed successfully!")
        return results
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise

if __name__ == "__main__":
    main()