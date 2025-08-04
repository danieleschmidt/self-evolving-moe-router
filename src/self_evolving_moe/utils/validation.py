"""
Validation utilities for Self-Evolving MoE-Router components.

This module provides comprehensive validation functions for configurations,
topologies, expert pools, and other system components.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from .exceptions import (
    MoEValidationError,
    ConfigurationError,
    TopologyError,
    ExpertPoolError,
    ResourceConstraintError
)


def validate_config(config: Any) -> bool:
    """
    Validate evolution configuration parameters.
    
    Args:
        config: EvolutionConfig object to validate
        
    Returns:
        True if valid
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        # Population parameters
        if not isinstance(config.population_size, int) or config.population_size < 2:
            raise ConfigurationError(
                "Population size must be an integer >= 2",
                parameter="population_size",
                value=config.population_size
            )
        
        if config.population_size > 1000:
            raise ConfigurationError(
                "Population size too large (max 1000 for performance)",
                parameter="population_size", 
                value=config.population_size
            )
        
        # Generation parameters
        if not isinstance(config.generations, int) or config.generations < 1:
            raise ConfigurationError(
                "Generations must be an integer >= 1",
                parameter="generations",
                value=config.generations
            )
        
        # Rate parameters
        if not 0.0 <= config.mutation_rate <= 1.0:
            raise ConfigurationError(
                "Mutation rate must be between 0.0 and 1.0",
                parameter="mutation_rate",
                value=config.mutation_rate
            )
        
        if not 0.0 <= config.crossover_rate <= 1.0:
            raise ConfigurationError(
                "Crossover rate must be between 0.0 and 1.0", 
                parameter="crossover_rate",
                value=config.crossover_rate
            )
        
        if not 0.0 <= config.elitism_ratio <= 1.0:
            raise ConfigurationError(
                "Elitism ratio must be between 0.0 and 1.0",
                parameter="elitism_ratio",
                value=config.elitism_ratio
            )
        
        # Selection parameters
        if config.selection_method not in ["tournament", "roulette", "rank"]:
            raise ConfigurationError(
                f"Unknown selection method: {config.selection_method}",
                parameter="selection_method",
                value=config.selection_method
            )
        
        if config.tournament_size < 2:
            raise ConfigurationError(
                "Tournament size must be >= 2",
                parameter="tournament_size",
                value=config.tournament_size
            )
        
        # Constraint parameters
        if config.max_active_experts < 1:
            raise ConfigurationError(
                "Max active experts must be >= 1",
                parameter="max_active_experts",
                value=config.max_active_experts
            )
        
        if not 0.0 < config.min_expert_usage <= 1.0:
            raise ConfigurationError(
                "Min expert usage must be between 0.0 and 1.0",
                parameter="min_expert_usage",
                value=config.min_expert_usage
            )
        
        if not 0.0 <= config.target_sparsity <= 1.0:
            raise ConfigurationError(
                "Target sparsity must be between 0.0 and 1.0",
                parameter="target_sparsity", 
                value=config.target_sparsity
            )
        
        # Convergence parameters  
        if config.patience < 1:
            raise ConfigurationError(
                "Patience must be >= 1",
                parameter="patience",
                value=config.patience
            )
        
        if config.min_improvement < 0:
            raise ConfigurationError(
                "Min improvement must be >= 0",
                parameter="min_improvement",
                value=config.min_improvement
            )
        
        return True
        
    except AttributeError as e:
        raise ConfigurationError(f"Missing required configuration parameter: {e}")


def validate_topology(topology: Any) -> bool:
    """
    Validate routing topology for consistency and feasibility.
    
    Args:
        topology: TopologyGenome object to validate
        
    Returns:
        True if valid
        
    Raises:
        TopologyError: If topology is invalid
    """
    try:
        # Basic structure validation
        if not hasattr(topology, 'routing_matrix'):
            raise TopologyError("Topology missing routing_matrix")
        
        if not hasattr(topology, 'expert_graph'):
            raise TopologyError("Topology missing expert_graph")
        
        if not hasattr(topology, 'routing_params'):
            raise TopologyError("Topology missing routing_params")
        
        # Matrix dimensions
        if topology.routing_matrix.dim() != 2:
            raise TopologyError(
                f"Routing matrix must be 2D, got {topology.routing_matrix.dim()}D"
            )
        
        num_tokens, num_experts = topology.routing_matrix.shape
        
        if num_tokens != topology.num_tokens:
            raise TopologyError(
                f"Routing matrix token dimension mismatch: {num_tokens} vs {topology.num_tokens}"
            )
        
        if num_experts != topology.num_experts:
            raise TopologyError(
                f"Routing matrix expert dimension mismatch: {num_experts} vs {topology.num_experts}"
            )
        
        # Matrix values
        if not torch.all((topology.routing_matrix == 0) | (topology.routing_matrix == 1)):
            raise TopologyError("Routing matrix must be binary (0 or 1)")
        
        # Connectivity constraints
        token_connections = topology.routing_matrix.sum(dim=1)
        if torch.any(token_connections == 0):
            zero_tokens = torch.where(token_connections == 0)[0].tolist()
            raise TopologyError(f"Tokens {zero_tokens} have no expert connections")
        
        # Expert graph validation
        if topology.expert_graph.shape != (num_experts, num_experts):
            raise TopologyError(
                f"Expert graph shape mismatch: {topology.expert_graph.shape} vs ({num_experts}, {num_experts})"
            )
        
        # Self-connections check
        if torch.any(torch.diag(topology.expert_graph) != 0):
            raise TopologyError("Expert graph should not have self-connections")
        
        # Routing parameters validation
        params = topology.routing_params
        
        if params.temperature <= 0:
            raise TopologyError(f"Temperature must be positive, got {params.temperature}")
        
        if params.top_k < 1 or params.top_k > num_experts:
            raise TopologyError(
                f"Top-k must be between 1 and {num_experts}, got {params.top_k}"
            )
        
        if not 0.0 <= params.load_balancing_weight <= 1.0:
            raise TopologyError(
                f"Load balancing weight must be between 0 and 1, got {params.load_balancing_weight}"
            )
        
        if not 0.0 <= params.diversity_weight <= 1.0:
            raise TopologyError(
                f"Diversity weight must be between 0 and 1, got {params.diversity_weight}"
            )
        
        # Sparsity validation
        actual_sparsity = topology.compute_sparsity()
        if actual_sparsity < 0 or actual_sparsity > 1:
            raise TopologyError(f"Invalid sparsity value: {actual_sparsity}")
        
        return True
        
    except AttributeError as e:
        raise TopologyError(f"Invalid topology structure: {e}")


def validate_expert_pool(expert_pool: Any) -> bool:
    """
    Validate expert pool configuration and state.
    
    Args:
        expert_pool: ExpertPool object to validate
        
    Returns:
        True if valid
        
    Raises:
        ExpertPoolError: If expert pool is invalid
    """
    try:
        # Basic structure
        if not hasattr(expert_pool, 'experts'):
            raise ExpertPoolError("Expert pool missing experts list")
        
        if not hasattr(expert_pool, 'num_experts'):
            raise ExpertPoolError("Expert pool missing num_experts")
        
        if not hasattr(expert_pool, 'expert_dim'):
            raise ExpertPoolError("Expert pool missing expert_dim")
        
        # Size validation
        if expert_pool.num_experts < 1:
            raise ExpertPoolError(
                "Must have at least 1 expert",
                num_experts=expert_pool.num_experts
            )
        
        if expert_pool.num_experts > 1024:
            raise ExpertPoolError(
                "Too many experts (max 1024 for performance)",
                num_experts=expert_pool.num_experts
            )
        
        if expert_pool.expert_dim < 1:
            raise ExpertPoolError(f"Expert dimension must be positive, got {expert_pool.expert_dim}")
        
        # Expert list validation
        if len(expert_pool.experts) != expert_pool.num_experts:
            raise ExpertPoolError(
                f"Expert count mismatch: {len(expert_pool.experts)} vs {expert_pool.num_experts}",
                num_experts=expert_pool.num_experts
            )
        
        # Individual expert validation
        for i, expert in enumerate(expert_pool.experts):
            if not hasattr(expert, 'forward'):
                raise ExpertPoolError(f"Expert {i} missing forward method")
            
            if not hasattr(expert, 'expert_id'):
                raise ExpertPoolError(f"Expert {i} missing expert_id")
            
            if expert.expert_id != i:
                raise ExpertPoolError(f"Expert {i} has wrong ID: {expert.expert_id}")
        
        # Active experts validation
        if hasattr(expert_pool, 'active_experts'):
            for expert_id in expert_pool.active_experts:
                if not 0 <= expert_id < expert_pool.num_experts:
                    raise ExpertPoolError(f"Invalid active expert ID: {expert_id}")
        
        # Type validation
        if expert_pool.expert_type not in ["transformer_block", "mlp"]:
            raise ExpertPoolError(
                f"Unknown expert type: {expert_pool.expert_type}",
                expert_type=expert_pool.expert_type
            )
        
        return True
        
    except AttributeError as e:
        raise ExpertPoolError(f"Invalid expert pool structure: {e}")


def validate_device(device: str) -> str:
    """
    Validate and normalize device specification.
    
    Args:
        device: Device string (e.g., "cpu", "cuda", "cuda:0")
        
    Returns:
        Normalized device string
        
    Raises:
        MoEValidationError: If device is invalid
    """
    if not isinstance(device, str):
        raise MoEValidationError(f"Device must be string, got {type(device)}")
    
    device = device.lower().strip()
    
    if device == "cpu":
        return "cpu"
    
    if device.startswith("cuda"):
        # Check if CUDA is available
        try:
            import torch
            if not torch.cuda.is_available():
                raise MoEValidationError("CUDA requested but not available")
            
            if device == "cuda":
                return "cuda"
            
            # Validate specific CUDA device
            if device.startswith("cuda:"):
                device_id = device.split(":")[1]
                try:
                    device_id = int(device_id)
                    if device_id >= torch.cuda.device_count():
                        raise MoEValidationError(
                            f"CUDA device {device_id} not available. "
                            f"Only {torch.cuda.device_count()} devices available."
                        )
                    return f"cuda:{device_id}"
                except ValueError:
                    raise MoEValidationError(f"Invalid CUDA device ID: {device_id}")
            
        except ImportError:
            raise MoEValidationError("PyTorch not available for CUDA validation")
    
    raise MoEValidationError(f"Unknown device: {device}")


def validate_data_loader(data_loader: Any) -> bool:
    """
    Validate data loader for evolution training.
    
    Args:
        data_loader: PyTorch DataLoader to validate
        
    Returns:
        True if valid
        
    Raises:
        MoEValidationError: If data loader is invalid
    """
    if not hasattr(data_loader, '__iter__'):
        raise MoEValidationError("Data loader must be iterable")
    
    if not hasattr(data_loader, '__len__'):
        raise MoEValidationError("Data loader must have length")
    
    if len(data_loader) == 0:
        raise MoEValidationError("Data loader is empty")
    
    # Try to get a sample batch
    try:
        sample_batch = next(iter(data_loader))
        if len(sample_batch) != 2:
            raise MoEValidationError("Data loader must return (inputs, targets) tuples")
        
        inputs, targets = sample_batch
        
        # Basic tensor validation
        if not hasattr(inputs, 'shape'):
            raise MoEValidationError("Inputs must have shape attribute")
        
        if not hasattr(targets, 'shape'):
            raise MoEValidationError("Targets must have shape attribute")
        
        if len(inputs.shape) < 2:
            raise MoEValidationError("Inputs must be at least 2D (batch_size, features)")
        
        if inputs.shape[0] != targets.shape[0]:
            raise MoEValidationError(
                f"Batch size mismatch: inputs {inputs.shape[0]} vs targets {targets.shape[0]}"
            )
        
    except StopIteration:
        raise MoEValidationError("Data loader iterator is empty")
    except Exception as e:
        raise MoEValidationError(f"Error sampling from data loader: {e}")
    
    return True


def validate_resource_constraints(
    memory_limit: Optional[int] = None,
    latency_limit: Optional[float] = None,
    compute_budget: Optional[float] = None
) -> bool:
    """
    Validate resource constraint parameters.
    
    Args:
        memory_limit: Memory limit in bytes
        latency_limit: Latency limit in milliseconds  
        compute_budget: Compute budget (FLOPs)
        
    Returns:
        True if valid
        
    Raises:
        ResourceConstraintError: If constraints are invalid
    """
    if memory_limit is not None:
        if not isinstance(memory_limit, int) or memory_limit <= 0:
            raise ResourceConstraintError(
                "Memory limit must be positive integer (bytes)",
                resource_type="memory",
                limit=memory_limit
            )
        
        # Reasonable bounds check
        min_memory = 1024 * 1024  # 1MB
        max_memory = 1024 * 1024 * 1024 * 1024  # 1TB
        
        if memory_limit < min_memory:
            raise ResourceConstraintError(
                f"Memory limit too small (min {min_memory} bytes)",
                resource_type="memory",
                limit=memory_limit
            )
        
        if memory_limit > max_memory:
            raise ResourceConstraintError(
                f"Memory limit too large (max {max_memory} bytes)",
                resource_type="memory", 
                limit=memory_limit
            )
    
    if latency_limit is not None:
        if not isinstance(latency_limit, (int, float)) or latency_limit <= 0:
            raise ResourceConstraintError(
                "Latency limit must be positive number (ms)",
                resource_type="latency",
                limit=latency_limit
            )
        
        # Reasonable bounds
        if latency_limit < 0.1:  # 0.1ms
            raise ResourceConstraintError(
                "Latency limit too small (min 0.1ms)",
                resource_type="latency",
                limit=latency_limit
            )
        
        if latency_limit > 3600000:  # 1 hour
            raise ResourceConstraintError(
                "Latency limit too large (max 1 hour)",
                resource_type="latency",
                limit=latency_limit
            )
    
    if compute_budget is not None:
        if not isinstance(compute_budget, (int, float)) or compute_budget <= 0:
            raise ResourceConstraintError(
                "Compute budget must be positive number (FLOPs)",
                resource_type="compute",
                limit=compute_budget
            )
    
    return True


def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = True,
    create_parent: bool = False,
    expected_suffix: Optional[str] = None
) -> Path:
    """
    Validate file path and optionally create parent directories.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must already exist
        create_parent: Whether to create parent directories
        expected_suffix: Expected file extension
        
    Returns:
        Validated Path object
        
    Raises:
        MoEValidationError: If path is invalid
    """
    if not isinstance(file_path, (str, Path)):
        raise MoEValidationError(f"File path must be string or Path, got {type(file_path)}")
    
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise MoEValidationError(f"File does not exist: {path}")
    
    if must_exist and not path.is_file():
        raise MoEValidationError(f"Path is not a file: {path}")
    
    if expected_suffix and path.suffix.lower() != expected_suffix.lower():
        raise MoEValidationError(
            f"Expected file extension {expected_suffix}, got {path.suffix}"
        )
    
    if create_parent:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise MoEValidationError(f"Cannot create parent directory: {e}")
    
    return path