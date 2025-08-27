"""
Command-line interface for Self-Evolving MoE Router.

This module provides the main CLI interface for the evolutionary MoE system,
including commands for evolution, benchmarking, and model management.
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

# Optional dependencies
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

try:
    from omegaconf import DictConfig, OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False
    DictConfig, OmegaConf = None, None
    class DictConfig(dict): pass
    class OmegaConf:
        """
        OmegaConf class.
        """
        """
        DictConfig class.
        """
        @staticmethod
        def create(data): return DictConfig(data)
        @staticmethod
        def to_yaml(data): return str(data)

from .evolution.router import EvolvingMoERouter, EvolutionConfig, FitnessFunction
from .experts.pool import ExpertPool, ExpertConfig
from .routing.topology import TopologyGenome
# from .monitoring.metrics import EvolutionMetrics  # Temporarily disabled
from .utils.logging import setup_logging, get_logger
from .utils.exceptions import EvolutionError
# from .utils.exceptions import ConfigError  # ConfigError not defined

class ConfigError(Exception):
    """
    To Yaml function.
    """
    """
    Create create.
    """
    """Configuration error."""
    pass

logger = get_logger(__name__)


class CLIConfig:
    """CLI configuration management."""

    def __init__(self):
        """
        Internal helper function.
        """
        self.config_dir = Path.home() / ".self_evolving_moe"
        self.config_file = self.config_dir / "config.yaml"
        self.ensure_config_dir()

    def ensure_config_dir(self):
        """Ensure configuration directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_path: Optional[Path] = None) -> DictConfig:
        """Load configuration from file."""
        if config_path is None:
            config_path = self.config_file

        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return OmegaConf.create(config_dict)
        else:
            return self.get_default_config()

    def save_config(self, config: DictConfig, config_path: Optional[Path] = None):
        """Save configuration to file."""
        if config_path is None:
            config_path = self.config_file

        with open(config_path, 'w') as f:
            yaml.dump(OmegaConf.to_yaml(config), f)

    def get_default_config(self) -> DictConfig:
        """Get default configuration."""
        return OmegaConf.create({
            'evolution': {
                'population_size': 50,
                'generations': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'selection_method': 'tournament',
                'target_sparsity': 0.1
            },
            'expert': {
                'num_experts': 8,
                'hidden_dim': 768,
                'intermediate_dim': 3072,
                'expert_type': 'transformer'
            },
            'training': {
                'device': 'auto',
                'batch_size': 32,
                'learning_rate': 1e-4,
                'num_epochs': 10
            },
            'logging': {
                'level': 'INFO',
                'file': None
            }
        })


def create_mock_data_loader(batch_size: int = 32, seq_len: int = 128, num_classes: int = 10, num_batches: int = 10):
    """Create mock data loader for demonstration."""
    from torch.utils.data import DataLoader, TensorDataset

    # Generate synthetic data
    inputs = torch.randn(num_batches * batch_size, seq_len, 768)  # Standard transformer dimension
    targets = torch.randint(0, num_classes, (num_batches * batch_size,))

    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def create_mock_model(num_experts: int, expert_config: ExpertConfig) -> torch.nn.Module:
    """
    Internal helper function.
    """
    """
    MockMoEModel class.
    """
    """Create a mock MoE model for demonstration."""

    class MockMoEModel(torch.nn.Module):
        """
        Forward function.
        """
        def __init__(self, expert_pool: ExpertPool):
            super().__init__()
            self.expert_pool = expert_pool
            self.classifier = torch.nn.Linear(expert_config.hidden_dim, 10)  # 10 classes
            self.current_topology = None

        def set_routing_topology(self, topology: TopologyGenome):
            """Set routing topology."""
            self.current_topology = topology
            self.expert_pool.set_routing_topology(topology)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Pass through expert pool
            expert_output, aux_losses = self.expert_pool(x)

            # Global average pooling and classification
            pooled_output = expert_output.mean(dim=1)  # [batch_size, hidden_dim]
            logits = self.classifier(pooled_output)  # [batch_size, num_classes]

            return logits

    # Create expert pool
    expert_pool = ExpertPool(num_experts=num_experts, expert_config=expert_config)

    # Create and return model
    model = MockMoEModel(expert_pool)
    return model


def evolve_command(args):
    """Execute evolution command."""
    logger.info("Starting evolution process")

    # Load configuration
    cli_config = CLIConfig()
    config = cli_config.load_config(Path(args.config) if args.config else None)

    # Override config with command line arguments
    if args.num_experts:
        config['expert']['num_experts'] = args.num_experts
    if args.generations:
        config['evolution']['generations'] = args.generations
    if args.population_size:
        config['evolution']['population_size'] = args.population_size

    # Set device
    device = args.device or config['training']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"Using device: {device}")

    # Create expert configuration
    expert_config = ExpertConfig(
        hidden_dim=config['expert']['hidden_dim'],
        intermediate_dim=config['expert']['intermediate_dim'],
        expert_type=config['expert']['expert_type']
    )

    # Create evolution configuration
    evolution_config = EvolutionConfig(
        population_size=config['evolution']['population_size'],
        generations=config['evolution']['generations'],
        mutation_rate=config['evolution']['mutation_rate'],
        crossover_rate=config['evolution']['crossover_rate'],
        selection_method=config['evolution']['selection_method'],
        target_sparsity=config['evolution']['target_sparsity']
    )

    # Initialize evolutionary router
    router = EvolvingMoERouter(
        num_experts=config['expert']['num_experts'],
        config=evolution_config,
        device=device
    )

    # Create mock model and data for demonstration
    model = create_mock_model(config['expert']['num_experts'], expert_config)
    data_loader = create_mock_data_loader(batch_size=config['training']['batch_size'])

    model.to(device)

    try:
        # Run evolution
        logger.info(f"Starting evolution with {evolution_config.generations} generations")
        best_topology = router.evolve(model, data_loader, generations=evolution_config.generations)

        # Save results
        output_dir = Path(args.output) if args.output else Path("evolution_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save best topology
        topology_file = output_dir / "best_topology.json"
        try:
            with open(topology_file, 'w') as f:
                json.dump(best_topology.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save topology: {e}")

        # Save evolution state
        state_file = output_dir / "evolution_state.json"
        try:
            router.save_evolution_state(state_file)
        except Exception as e:
            logger.warning(f"Failed to save evolution state: {e}")

        # Save metrics
        metrics_file = output_dir / "metrics.json"
        try:
            with open(metrics_file, 'w') as f:
                json.dump(router.get_evolution_metrics(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")

        logger.info(f"Evolution completed successfully!")
        logger.info(f"Best fitness: {router.best_fitness:.4f}")
        logger.info(f"Results saved to: {output_dir}")

        # Print summary
        print("\n" + "="*50)
        print("EVOLUTION SUMMARY")
        print("="*50)
        print(f"Best Fitness: {router.best_fitness:.4f}")
        print(f"Generations: {evolution_config.generations}")
        print(f"Population Size: {evolution_config.population_size}")
        print(f"Final Topology Sparsity: {best_topology.sparsity:.3f}")
        print(f"Results Directory: {output_dir}")
        print("="*50)

    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        sys.exit(1)


def benchmark_command(args):
    """Execute benchmark command."""
    logger.info("Starting benchmark")

    # Load configuration
    cli_config = CLIConfig()
    config = cli_config.load_config(Path(args.config) if args.config else None)

    # Create expert configuration
    expert_config = ExpertConfig(
        hidden_dim=config['expert']['hidden_dim'],
        intermediate_dim=config['expert']['intermediate_dim'],
        expert_type=config['expert']['expert_type']
    )

    # Create test model
    model = create_mock_model(config['expert']['num_experts'], expert_config)
    data_loader = create_mock_data_loader()

    device = args.device or config['training']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    # Benchmark different configurations
    results = {}

    for num_experts in [4, 8, 16, 32]:
        logger.info(f"Benchmarking with {num_experts} experts")

        # Create small evolution for quick benchmark
        router = EvolvingMoERouter(
            num_experts=num_experts,
            config=EvolutionConfig(population_size=10, generations=5),
            device=device
        )

        # Run quick evolution
        try:
            best_topology = router.evolve(model, data_loader, generations=5)

            results[f"{num_experts}_experts"] = {
                'num_experts': num_experts,
                'best_fitness': router.best_fitness,
                'sparsity': best_topology.sparsity,
                'parameters': model.expert_pool.get_total_parameters()
            }
        except Exception as e:
            logger.warning(f"Benchmark failed for {num_experts} experts: {e}")

    # Save benchmark results
    output_file = Path(args.output) if args.output else Path("benchmark_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    for config_name, result in results.items():
        print(f"{config_name}:")
        print(f"  Best Fitness: {result['best_fitness']:.4f}")
        print(f"  Sparsity: {result['sparsity']:.3f}")
        print(f"  Parameters: {result['parameters']:,}")
    print("="*50)

    logger.info(f"Benchmark results saved to: {output_file}")


def info_command(args):
    """Display system information."""
    print("\n" + "="*50)
    print("SELF-EVOLVING MOE ROUTER")
    print("="*50)
    print(f"Version: 0.1.0")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
    print("="*50)

    # Show configuration
    cli_config = CLIConfig()
    config = cli_config.load_config()

    print("\nCURRENT CONFIGURATION:")
    print("-" * 30)
    print(OmegaConf.to_yaml(config))


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Self-Evolving MoE Router CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  evolve-moe evolve --generations 100 --num-experts 16
  evolve-moe benchmark --output benchmark.json
  evolve-moe info
        """
    )

    # Global arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file path'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Computation device'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Evolve command
    evolve_parser = subparsers.add_parser('evolve', help='Run evolutionary algorithm')
    evolve_parser.add_argument(
        '--generations', '-g',
        type=int,
        help='Number of generations'
    )
    evolve_parser.add_argument(
        '--population-size', '-p',
        type=int,
        help='Population size'
    )
    evolve_parser.add_argument(
        '--num-experts', '-e',
        type=int,
        help='Number of experts'
    )
    evolve_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory'
    )
    evolve_parser.set_defaults(func=evolve_command)

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for results'
    )
    benchmark_parser.set_defaults(func=benchmark_command)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.set_defaults(func=info_command)

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    # Check if command was provided
    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        # Execute command
        args.func(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
