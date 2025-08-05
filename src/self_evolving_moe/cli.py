#!/usr/bin/env python3
"""
Command-line interface for Self-Evolving MoE-Router.

This module provides CLI commands for running evolution experiments,
managing models, and interacting with the MoE router system.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

import torch

from . import EvolvingMoERouter, ExpertPool, TopologyGenome
from .evolution.router import EvolutionConfig
from .data import TopologyRepository, ExperimentRepository, EvolutionCache, ModelStorage
from .data.repository import SQLiteRepository
from .data.storage import LocalStorageBackend


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('evolution.log')
        ]
    )


def create_expert_pool(args) -> ExpertPool:
    """Create expert pool from arguments."""
    return ExpertPool(
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        expert_type=args.expert_type,
        device=args.device,
        expert_config={
            'num_heads': args.num_heads,
            'dropout': args.dropout,
            'activation': args.activation,
            'num_layers': args.num_layers
        }
    )


def create_evolution_config(args) -> EvolutionConfig:
    """Create evolution configuration from arguments."""
    return EvolutionConfig(
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elitism_ratio=args.elitism_ratio,
        tournament_size=args.tournament_size,
        selection_method=args.selection_method,
        accuracy_weight=args.accuracy_weight,
        latency_weight=args.latency_weight,
        memory_weight=args.memory_weight,
        sparsity_weight=args.sparsity_weight,
        diversity_weight=args.diversity_weight,
        max_active_experts=args.max_active_experts,
        target_sparsity=args.target_sparsity,
        patience=args.patience,
        min_improvement=args.min_improvement,
        memory_budget=args.memory_budget,
        latency_budget=args.latency_budget
    )


def evolve_command(args):
    """Run evolution experiment."""
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting evolution experiment")
    
    # Create expert pool
    expert_pool = create_expert_pool(args)
    logger.info(f"Created expert pool with {expert_pool.num_experts} experts")
    
    # Create evolution configuration
    config = create_evolution_config(args)
    
    # Create evolutionary router
    evolver = EvolvingMoERouter(
        expert_pool=expert_pool,
        config=config,
        device=args.device
    )
    
    logger.info(f"Initialized evolution with population size {config.population_size}")
    
    # Create dummy model for this example
    class SimpleMoEModel(torch.nn.Module):
        def __init__(self, expert_pool):
            super().__init__()
            self.expert_pool = expert_pool
            self.router = torch.nn.Linear(expert_pool.expert_dim, expert_pool.num_experts)
            self.current_topology = None
            
        def set_routing_topology(self, topology):
            self.current_topology = topology
            
        def forward(self, x):
            # Simple forward pass for demonstration
            batch_size = x.shape[0]
            return torch.randn(batch_size, 10)  # Dummy classification output
    
    model = SimpleMoEModel(expert_pool)
    
    # Create dummy data
    train_data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(1000, args.expert_dim),
            torch.randint(0, 10, (1000,))
        ),
        batch_size=32,
        shuffle=True
    )
    
    val_data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(200, args.expert_dim),
            torch.randint(0, 10, (200,))
        ),
        batch_size=32,
        shuffle=False
    )
    
    # Run evolution
    best_topology = evolver.evolve(
        model=model,
        train_data=train_data,
        val_data=val_data,
        generations=config.generations
    )
    
    # Get results
    stats = evolver.get_evolution_stats()
    
    logger.info("Evolution completed!")
    logger.info(f"Best fitness: {stats['best_fitness']:.6f}")
    logger.info(f"Generations run: {stats['generations_run']}")
    
    if best_topology:
        summary = best_topology.get_topology_summary()
        logger.info(f"Best topology sparsity: {summary['sparsity']:.3f}")
        logger.info(f"Total connections: {summary['total_connections']}")
        
        # Save topology
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            topology_path = output_path / "best_topology.pt"
            best_topology.save_topology(str(topology_path))
            logger.info(f"Saved best topology to {topology_path}")
            
            # Save evolution state
            evolution_path = output_path / "evolution_state.pt"
            evolver.save_evolution_state(str(evolution_path))
            logger.info(f"Saved evolution state to {evolution_path}")


def benchmark_command(args):
    """Run benchmark experiment."""
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting benchmark experiment")
    
    # Create expert pool
    expert_pool = create_expert_pool(args)
    
    # Create sample input
    sample_input = torch.randn(args.batch_size, args.sequence_length, args.expert_dim)
    
    # Benchmark different configurations
    results = []
    
    for num_experts in [2, 4, 8, 16]:
        if num_experts > expert_pool.num_experts:
            continue
            
        expert_pool.set_active_experts(list(range(num_experts)))
        
        # Simple routing test
        import time
        start_time = time.time()
        
        for _ in range(args.num_trials):
            # Simulate expert routing
            for expert_idx in range(num_experts):
                expert = expert_pool.experts[expert_idx]
                _ = expert(sample_input)
        
        end_time = time.time()
        avg_latency = (end_time - start_time) / args.num_trials * 1000
        
        results.append({
            'num_experts': num_experts,
            'latency_ms': avg_latency,
            'throughput': args.batch_size / (avg_latency / 1000)
        })
        
        logger.info(f"Experts: {num_experts}, Latency: {avg_latency:.2f}ms, "
                   f"Throughput: {results[-1]['throughput']:.2f} samples/sec")
    
    logger.info("Benchmark completed!")


def info_command(args):
    """Display system information."""
    print("Self-Evolving MoE-Router")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 'CPU only'}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Self-Evolving MoE-Router CLI")
    
    # Global arguments
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evolve command
    evolve_parser = subparsers.add_parser('evolve', help='Run evolution experiment')
    
    # Expert pool arguments
    evolve_parser.add_argument('--num-experts', type=int, default=8, help='Number of experts')
    evolve_parser.add_argument('--expert-dim', type=int, default=128, help='Expert dimension')
    evolve_parser.add_argument('--expert-type', default='mlp', choices=['mlp', 'transformer_block'])
    evolve_parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    evolve_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    evolve_parser.add_argument('--activation', default='gelu', choices=['gelu', 'relu'])
    evolve_parser.add_argument('--num-layers', type=int, default=2, help='Number of MLP layers')
    
    # Evolution arguments
    evolve_parser.add_argument('--population-size', type=int, default=50, help='Population size')
    evolve_parser.add_argument('--generations', type=int, default=100, help='Number of generations')
    evolve_parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation rate')
    evolve_parser.add_argument('--crossover-rate', type=float, default=0.7, help='Crossover rate')
    evolve_parser.add_argument('--elitism-ratio', type=float, default=0.1, help='Elitism ratio')
    evolve_parser.add_argument('--tournament-size', type=int, default=3, help='Tournament size')
    evolve_parser.add_argument('--selection-method', default='tournament', 
                             choices=['tournament', 'roulette', 'rank'])
    
    # Objective weights
    evolve_parser.add_argument('--accuracy-weight', type=float, default=1.0)
    evolve_parser.add_argument('--latency-weight', type=float, default=-0.1)
    evolve_parser.add_argument('--memory-weight', type=float, default=-0.05)
    evolve_parser.add_argument('--sparsity-weight', type=float, default=0.1)
    evolve_parser.add_argument('--diversity-weight', type=float, default=0.05)
    
    # Constraints
    evolve_parser.add_argument('--max-active-experts', type=int, default=8)
    evolve_parser.add_argument('--target-sparsity', type=float, default=0.8)
    evolve_parser.add_argument('--patience', type=int, default=50)
    evolve_parser.add_argument('--min-improvement', type=float, default=1e-4)
    evolve_parser.add_argument('--memory-budget', type=int, help='Memory budget in bytes')
    evolve_parser.add_argument('--latency-budget', type=float, help='Latency budget in ms')
    
    # Output
    evolve_parser.add_argument('--output-dir', help='Output directory for results')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark')
    benchmark_parser.add_argument('--num-experts', type=int, default=16)
    benchmark_parser.add_argument('--expert-dim', type=int, default=256)
    benchmark_parser.add_argument('--expert-type', default='mlp', choices=['mlp', 'transformer_block'])
    benchmark_parser.add_argument('--batch-size', type=int, default=32)
    benchmark_parser.add_argument('--sequence-length', type=int, default=128)
    benchmark_parser.add_argument('--num-trials', type=int, default=10)
    benchmark_parser.add_argument('--num-heads', type=int, default=8)
    benchmark_parser.add_argument('--dropout', type=float, default=0.1)
    benchmark_parser.add_argument('--activation', default='gelu')
    benchmark_parser.add_argument('--num-layers', type=int, default=2)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display system information')
    
    args = parser.parse_args()
    
    if args.command == 'evolve':
        evolve_command(args)
    elif args.command == 'benchmark':
        benchmark_command(args)
    elif args.command == 'info':
        info_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()