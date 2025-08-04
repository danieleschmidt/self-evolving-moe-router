#!/usr/bin/env python3
"""
Command-line interface for Self-Evolving MoE-Router.

This module provides a CLI for running evolution experiments, benchmarking
models, and managing evolved topologies.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .evolution.router import EvolvingMoERouter, EvolutionConfig
from .experts.pool import ExpertPool
from .experts.slimmable import SlimmableMoE
from .routing.topology import TopologyGenome


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


class SimpleMoEModel(nn.Module):
    """Simple MoE model for CLI demonstrations."""
    
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
            
            # Route through expert pool
            expert_outputs = self.expert_pool.forward(
                x.unsqueeze(1),
                selected_experts,
                routing_weights
            ).squeeze(1)
            
        else:
            # Use learned routing
            routing_logits = self.router(x)
            routing_weights = torch.softmax(routing_logits, dim=-1)
            
            # Route through experts
            expert_outputs = torch.zeros_like(x)
            for i, expert in enumerate(self.expert_pool.experts):
                if self.expert_pool.expert_type == "transformer_block":
                    expert_output = expert(x.unsqueeze(1)).squeeze(1)
                else:
                    expert_output = expert(x)
                expert_outputs += expert_output * routing_weights[:, i:i+1]
        
        # Classification
        return self.classifier(expert_outputs)


def create_dummy_data(num_samples: int, input_dim: int, num_classes: int, device: str = "cpu"):
    """Create dummy classification dataset."""
    X = torch.randn(num_samples, input_dim, device=device)
    y = torch.randint(0, num_classes, (num_samples,), device=device)
    return TensorDataset(X, y)


def evolve_command(args):
    """Run evolution experiment."""
    print("🧬 Starting Self-Evolving MoE-Router Evolution")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = EvolutionConfig(**config_dict)
    else:
        config = EvolutionConfig(
            population_size=args.population_size,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            accuracy_weight=1.0,
            latency_weight=-0.1,
            sparsity_weight=0.2
        )
    
    print(f"Evolution config: {config.population_size} pop, {config.generations} gen")
    
    # Create expert pool
    expert_pool = ExpertPool(
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        expert_type=args.expert_type,
        device=device
    )
    
    print(f"Created expert pool: {args.num_experts} × {args.expert_type} experts")
    
    # Create model
    model = SimpleMoEModel(expert_pool, num_classes=args.num_classes)
    model.to(device)
    
    # Create or load dataset
    if args.dataset:
        # TODO: Implement dataset loading
        raise NotImplementedError("Custom dataset loading not yet implemented")
    else:
        # Use dummy data
        train_data = create_dummy_data(args.train_samples, args.expert_dim, args.num_classes, device)
        val_data = create_dummy_data(args.val_samples, args.expert_dim, args.num_classes, device)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    print(f"Dataset: {len(train_data)} train, {len(val_data)} val samples")
    
    # Initialize evolutionary router
    evolver = EvolvingMoERouter(
        expert_pool=expert_pool,
        config=config,
        device=device
    )
    
    print("🚀 Starting evolution...")
    start_time = time.time()
    
    # Run evolution
    best_topology = evolver.evolve(
        model=model,
        train_data=train_loader,
        val_data=val_loader,
        generations=config.generations
    )
    
    evolution_time = time.time() - start_time
    
    # Get and display results
    stats = evolver.get_evolution_stats()
    
    print("\n" + "🎯 EVOLUTION RESULTS" + "=" * 43)
    print(f"⏱️  Evolution time: {evolution_time:.2f}s")
    print(f"🔄 Generations run: {stats['generations_run']}")
    print(f"🏆 Best fitness: {stats['best_fitness']:.6f}")
    print(f"📈 Convergence gen: {stats.get('convergence_generation', 'N/A')}")
    
    if best_topology:
        summary = best_topology.get_topology_summary()
        print(f"\n📊 Best Topology Summary:")
        print(f"   Sparsity: {summary['sparsity']:.3f}")
        print(f"   Connections: {summary['total_connections']}")
        print(f"   Avg per token: {summary['avg_connections_per_token']:.2f}")
        print(f"   Avg per expert: {summary['avg_connections_per_expert']:.2f}")
        print(f"   Generation: {summary['generation']}")
        
        # Test evolved topology
        print(f"\n🧪 Testing evolved topology...")
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
        print(f"   Final accuracy: {accuracy:.4f}")
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save topology
        topology_path = output_dir / "best_topology.pt"
        best_topology.save_topology(str(topology_path))
        print(f"💾 Saved topology: {topology_path}")
        
        # Save evolution state
        state_path = output_dir / "evolution_state.pt"
        evolver.save_evolution_state(str(state_path))
        print(f"💾 Saved evolution state: {state_path}")
        
        # Save results summary
        results = {
            'evolution_time': evolution_time,
            'best_fitness': stats['best_fitness'],
            'generations_run': stats['generations_run'],
            'final_accuracy': accuracy,
            'topology_summary': summary,
            'config': config.__dict__
        }
        
        results_path = output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Saved results: {results_path}")
    
    print("\n✅ Evolution completed successfully!")


def benchmark_command(args):
    """Run benchmarking experiments."""
    print("📊 Benchmarking Slimmable MoE Performance")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Create expert pool
    expert_pool = ExpertPool(
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        expert_type=args.expert_type,
        device=device
    )
    
    # Create slimmable MoE
    width_configs = [args.expert_dim // 4, args.expert_dim // 2, 3 * args.expert_dim // 4, args.expert_dim]
    slimmable_moe = SlimmableMoE(
        expert_pool=expert_pool,
        width_configs=width_configs
    )
    slimmable_moe.to(device)
    
    # Create sample input
    sample_input = torch.randn(args.batch_size, args.seq_len, args.expert_dim, device=device)
    
    print(f"Benchmarking with input shape: {sample_input.shape}")
    print(f"Width configurations: {width_configs}")
    
    # Benchmark each width
    results = {}
    for width in width_configs:
        print(f"\n🔬 Benchmarking width {width}...")
        
        metrics = slimmable_moe.benchmark_width(
            sample_input, 
            width=width, 
            num_trials=args.num_trials
        )
        
        results[width] = metrics
        print(f"   Latency: {metrics['latency_ms']:.2f}ms")
        print(f"   Parameters: {metrics['parameter_count']:,}")
        print(f"   Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/sec")
    
    # Save benchmark results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    benchmark_path = output_dir / "benchmark_results.json"
    with open(benchmark_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved benchmark results: {benchmark_path}")
    
    # Generate efficiency report
    efficiency_report = slimmable_moe.get_efficiency_report()
    report_path = output_dir / "efficiency_report.json"
    with open(report_path, 'w') as f:
        json.dump(efficiency_report, f, indent=2, default=str)
    
    print(f"💾 Saved efficiency report: {report_path}")
    print("\n✅ Benchmarking completed!")


def load_topology_command(args):
    """Load and inspect a saved topology."""
    print(f"📖 Loading topology from: {args.topology_path}")
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    
    # Load topology
    topology = TopologyGenome.load_topology(args.topology_path, device=device)
    
    # Display summary
    summary = topology.get_topology_summary()
    print("\n📊 Topology Summary:")
    print(f"   Tokens: {summary['num_tokens']}")
    print(f"   Experts: {summary['num_experts']}")
    print(f"   Sparsity: {summary['sparsity']:.3f}")
    print(f"   Connections: {summary['total_connections']}")
    print(f"   Avg per token: {summary['avg_connections_per_token']:.2f}")
    print(f"   Avg per expert: {summary['avg_connections_per_expert']:.2f}")
    print(f"   Generation: {summary['generation']}")
    print(f"   Routing params: {summary['routing_params']}")
    
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Plot routing matrix
            matrix = topology.routing_matrix.cpu().numpy()
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(matrix, cmap='Blues', cbar=True)
            plt.title('Routing Topology Matrix')
            plt.xlabel('Expert Index')
            plt.ylabel('Token Index')
            
            output_path = args.output_dir / "topology_visualization.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved visualization: {output_path}")
            
        except ImportError:
            print("⚠️  Visualization requires matplotlib and seaborn")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Self-Evolving MoE-Router CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage even if CUDA available')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for results')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evolution command
    evolve_parser = subparsers.add_parser('evolve', help='Run evolution experiment')
    evolve_parser.add_argument('--config', type=str,
                              help='Evolution configuration JSON file')
    evolve_parser.add_argument('--population-size', type=int, default=50,
                              help='Evolution population size')
    evolve_parser.add_argument('--generations', type=int, default=100,
                              help='Number of evolution generations')
    evolve_parser.add_argument('--mutation-rate', type=float, default=0.1,
                              help='Mutation rate')
    evolve_parser.add_argument('--num-experts', type=int, default=16,
                              help='Number of experts')
    evolve_parser.add_argument('--expert-dim', type=int, default=256,
                              help='Expert hidden dimension')
    evolve_parser.add_argument('--expert-type', type=str, default='mlp',
                              choices=['mlp', 'transformer_block'],
                              help='Type of expert architecture')
    evolve_parser.add_argument('--num-classes', type=int, default=10,
                              help='Number of output classes')
    evolve_parser.add_argument('--train-samples', type=int, default=10000,
                              help='Number of training samples (dummy data)')
    evolve_parser.add_argument('--val-samples', type=int, default=2000,
                              help='Number of validation samples (dummy data)')
    evolve_parser.add_argument('--batch-size', type=int, default=64,
                              help='Batch size')
    evolve_parser.add_argument('--dataset', type=str,
                              help='Custom dataset path (not implemented)')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark slimmable MoE')
    benchmark_parser.add_argument('--num-experts', type=int, default=16,
                                 help='Number of experts')
    benchmark_parser.add_argument('--expert-dim', type=int, default=256,
                                 help='Expert hidden dimension')
    benchmark_parser.add_argument('--expert-type', type=str, default='mlp',
                                 choices=['mlp', 'transformer_block'],
                                 help='Type of expert architecture')
    benchmark_parser.add_argument('--batch-size', type=int, default=32,
                                 help='Batch size for benchmarking')
    benchmark_parser.add_argument('--seq-len', type=int, default=128,
                                 help='Sequence length')
    benchmark_parser.add_argument('--num-trials', type=int, default=10,
                                 help='Number of benchmark trials')
    
    # Load topology command
    load_parser = subparsers.add_parser('load-topology', help='Load and inspect topology')
    load_parser.add_argument('topology_path', type=str,
                            help='Path to saved topology file')
    load_parser.add_argument('--visualize', action='store_true',
                            help='Generate topology visualization')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup
    setup_logging("DEBUG" if args.verbose else "INFO")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run command
    if args.command == 'evolve':
        evolve_command(args)
    elif args.command == 'benchmark':
        benchmark_command(args)
    elif args.command == 'load-topology':
        load_topology_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()