"""
Robust validation and error handling for Self-Evolving MoE Router.

This module implements comprehensive input validation, error recovery,
and robust operation handling to ensure production-ready reliability.
"""

import torch
import numpy as np
import traceback
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import functools
import time
from pathlib import Path
import json
import warnings

from .exceptions import (
    MoEValidationError, EvolutionError, TopologyError, ExpertPoolError,
    ResourceConstraintError, ConfigurationError, ModelLoadError, CompatibilityError
)
from .logging import get_logger

logger = get_logger(__name__)


class ValidationLevel(Enum):
    """Levels of validation strictness."""
    MINIMAL = "minimal"      # Basic checks only
    STANDARD = "standard"    # Standard production checks
    STRICT = "strict"        # Comprehensive validation
    PARANOID = "paranoid"    # Maximum validation and safety


class ErrorRecoveryStrategy(Enum):
    """Error recovery strategies."""
    FAIL_FAST = "fail_fast"           # Immediate failure
    RETRY = "retry"                   # Retry with backoff
    FALLBACK = "fallback"             # Use fallback values
    GRACEFUL_DEGRADATION = "graceful" # Continue with reduced functionality


@dataclass
class ValidationConfig:
    """Configuration for validation behavior."""
    level: ValidationLevel = ValidationLevel.STANDARD
    recovery_strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.FALLBACK
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_warnings: bool = True
    log_validation_details: bool = False
    strict_type_checking: bool = True
    memory_limit_mb: float = 8192.0
    computation_timeout: float = 300.0  # 5 minutes


class RobustValidator:
    """Comprehensive validation and error handling system."""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.validation_cache: Dict[str, bool] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.performance_stats: Dict[str, float] = {}
        
        logger.info(f"Initialized robust validator with {self.config.level.value} validation")
    
    def validate_topology(self, topology, population_context: Optional[List] = None) -> bool:
        """
        Comprehensive topology validation with context-aware checks.
        
        Args:
            topology: TopologyGenome to validate
            population_context: Optional population for context validation
            
        Returns:
            True if topology is valid
            
        Raises:
            TopologyError: If topology is invalid and recovery fails
        """
        validation_id = f"topology_{id(topology)}"
        
        try:
            with self._validation_timer("topology_validation"):
                # Basic structural validation
                self._validate_topology_structure(topology)
                
                # Advanced validation based on level
                if self.config.level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                    self._validate_topology_properties(topology)
                
                if self.config.level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                    self._validate_topology_consistency(topology)
                
                if self.config.level == ValidationLevel.PARANOID and population_context:
                    self._validate_topology_in_population(topology, population_context)
                
                # Cache successful validation
                self.validation_cache[validation_id] = True
                return True
                
        except Exception as e:
            self._handle_validation_error("topology_validation", e, topology)
            return False
    
    def _validate_topology_structure(self, topology) -> None:
        """Validate basic topology structure."""
        # Check required attributes
        required_attrs = ['routing_matrix', 'num_tokens', 'num_experts', 'routing_params']
        for attr in required_attrs:
            if not hasattr(topology, attr):
                raise TopologyError(f"Missing required attribute: {attr}")
        
        # Validate matrix dimensions
        if topology.routing_matrix.dim() != 2:
            raise TopologyError(f"Routing matrix must be 2D, got {topology.routing_matrix.dim()}D")
        
        expected_shape = (topology.num_tokens, topology.num_experts)
        if topology.routing_matrix.shape != expected_shape:
            raise TopologyError(f"Routing matrix shape {topology.routing_matrix.shape} doesn't match expected {expected_shape}")
        
        # Validate matrix contents
        if not torch.all((topology.routing_matrix == 0) | (topology.routing_matrix == 1)):
            raise TopologyError("Routing matrix must contain only 0s and 1s")
    
    def _validate_topology_properties(self, topology) -> None:
        """Validate advanced topology properties."""
        # Check for degenerate cases
        if topology.routing_matrix.sum() == 0:
            raise TopologyError("Topology has no connections (all zeros)")
        
        # Ensure each token connects to at least one expert
        token_connections = topology.routing_matrix.sum(dim=1)
        if torch.any(token_connections == 0):
            raise TopologyError("Some tokens have no expert connections")
        
        # Check sparsity bounds
        actual_sparsity = topology.compute_sparsity()
        if actual_sparsity < 0.0 or actual_sparsity > 1.0:
            raise TopologyError(f"Invalid sparsity value: {actual_sparsity}")
        
        # Validate routing parameters
        params = topology.routing_params
        if params.temperature <= 0:
            raise TopologyError(f"Invalid temperature: {params.temperature}")
        if params.top_k <= 0 or params.top_k > topology.num_experts:
            raise TopologyError(f"Invalid top_k: {params.top_k}")
    
    def _validate_topology_consistency(self, topology) -> None:
        """Validate topology consistency and detect anomalies."""
        # Check for unusual patterns that might indicate corruption
        routing_matrix = topology.routing_matrix.float()
        
        # Check for extreme load imbalance
        expert_loads = routing_matrix.sum(dim=0)
        if expert_loads.max() > expert_loads.mean() * 10:
            warnings.warn("Extreme expert load imbalance detected", UserWarning)
        
        # Check for unusual sparsity patterns
        row_sparsities = 1 - routing_matrix.mean(dim=1)
        if row_sparsities.std() > 0.5:
            warnings.warn("High variance in token routing sparsity", UserWarning)
        
        # Validate parameter consistency
        params = topology.routing_params
        if params.sparsity_target > 0 and abs(topology.compute_sparsity() - params.sparsity_target) > 0.3:
            warnings.warn(f"Large discrepancy between target and actual sparsity", UserWarning)
    
    def _validate_topology_in_population(self, topology, population: List) -> None:
        """Validate topology in context of population."""
        # Check for duplicates (which might indicate evolution issues)
        duplicates = 0
        for other in population:
            if hasattr(other, 'routing_matrix') and torch.equal(topology.routing_matrix, other.routing_matrix):
                duplicates += 1
        
        if duplicates > len(population) * 0.1:  # More than 10% duplicates
            warnings.warn("High duplicate rate in population detected", UserWarning)
    
    def validate_expert_pool(self, expert_pool, model_context: Optional[Any] = None) -> bool:
        """
        Validate expert pool configuration and state.
        
        Args:
            expert_pool: ExpertPool to validate
            model_context: Optional model context for compatibility checks
            
        Returns:
            True if expert pool is valid
        """
        try:
            with self._validation_timer("expert_pool_validation"):
                # Basic structure validation
                self._validate_expert_pool_structure(expert_pool)
                
                # Configuration validation
                if self.config.level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                    self._validate_expert_pool_config(expert_pool)
                
                # State validation
                if self.config.level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                    self._validate_expert_pool_state(expert_pool)
                
                # Model compatibility
                if self.config.level == ValidationLevel.PARANOID and model_context:
                    self._validate_expert_model_compatibility(expert_pool, model_context)
                
                return True
                
        except Exception as e:
            self._handle_validation_error("expert_pool_validation", e, expert_pool)
            return False
    
    def _validate_expert_pool_structure(self, expert_pool) -> None:
        """Validate expert pool basic structure."""
        required_attrs = ['experts', 'num_experts', 'expert_config', 'routing_layer']
        for attr in required_attrs:
            if not hasattr(expert_pool, attr):
                raise ExpertPoolError(f"Missing required attribute: {attr}")
        
        # Check expert count consistency
        if len(expert_pool.experts) != expert_pool.num_experts:
            raise ExpertPoolError(f"Expert count mismatch: {len(expert_pool.experts)} vs {expert_pool.num_experts}")
    
    def _validate_expert_pool_config(self, expert_pool) -> None:
        """Validate expert pool configuration."""
        config = expert_pool.expert_config
        
        # Validate dimensions
        if config.hidden_dim <= 0:
            raise ExpertPoolError(f"Invalid hidden dimension: {config.hidden_dim}")
        if config.intermediate_dim <= 0:
            raise ExpertPoolError(f"Invalid intermediate dimension: {config.intermediate_dim}")
        
        # Check reasonable ranges
        if config.hidden_dim > 8192:
            warnings.warn(f"Very large hidden dimension: {config.hidden_dim}")
        if config.dropout < 0 or config.dropout > 1:
            raise ExpertPoolError(f"Invalid dropout rate: {config.dropout}")
    
    def _validate_expert_pool_state(self, expert_pool) -> None:
        """Validate expert pool runtime state."""
        # Check for parameter initialization
        for i, expert in enumerate(expert_pool.experts):
            try:
                # Check if parameters are initialized (not all zeros)
                param_sum = sum(p.sum().item() for p in expert.parameters())
                if abs(param_sum) < 1e-6:
                    warnings.warn(f"Expert {i} may be uninitialized (all parameters near zero)")
            except Exception as e:
                logger.warning(f"Could not validate expert {i} parameters: {e}")
    
    def _validate_expert_model_compatibility(self, expert_pool, model) -> None:
        """Validate compatibility between expert pool and model."""
        # Check dimension compatibility
        try:
            if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
                expected_dim = expert_pool.expert_config.hidden_dim
                actual_dim = model.classifier.in_features
                if expected_dim != actual_dim:
                    raise CompatibilityError(
                        f"Dimension mismatch: expert pool {expected_dim} vs model {actual_dim}",
                        "expert_pool", "model"
                    )
        except Exception as e:
            logger.warning(f"Could not validate expert-model compatibility: {e}")
    
    def validate_evolution_config(self, config) -> bool:
        """Validate evolution configuration parameters."""
        try:
            with self._validation_timer("evolution_config_validation"):
                self._validate_evolution_config_structure(config)
                
                if self.config.level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                    self._validate_evolution_config_values(config)
                
                if self.config.level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                    self._validate_evolution_config_consistency(config)
                
                return True
                
        except Exception as e:
            self._handle_validation_error("evolution_config_validation", e, config)
            return False
    
    def _validate_evolution_config_structure(self, config) -> None:
        """Validate evolution config structure."""
        required_attrs = ['population_size', 'generations', 'mutation_rate', 'crossover_rate']
        for attr in required_attrs:
            if not hasattr(config, attr):
                raise ConfigurationError(f"Missing required attribute: {attr}", attr)
    
    def _validate_evolution_config_values(self, config) -> None:
        """Validate evolution config value ranges."""
        # Population size
        if config.population_size < 2:
            raise ConfigurationError("Population size must be at least 2", "population_size", config.population_size)
        if config.population_size > 1000:
            warnings.warn(f"Very large population size: {config.population_size}")
        
        # Generations
        if config.generations < 1:
            raise ConfigurationError("Must have at least 1 generation", "generations", config.generations)
        
        # Rates
        if not 0 <= config.mutation_rate <= 1:
            raise ConfigurationError("Mutation rate must be in [0,1]", "mutation_rate", config.mutation_rate)
        if not 0 <= config.crossover_rate <= 1:
            raise ConfigurationError("Crossover rate must be in [0,1]", "crossover_rate", config.crossover_rate)
    
    def _validate_evolution_config_consistency(self, config) -> None:
        """Validate evolution config internal consistency."""
        # Check for reasonable parameter combinations
        if config.mutation_rate + config.crossover_rate > 1.5:
            warnings.warn("Very high combined mutation and crossover rates")
        
        if hasattr(config, 'elitism_rate'):
            if config.elitism_rate > 0.5:
                warnings.warn("High elitism rate may reduce diversity")
    
    @functools.lru_cache(maxsize=128)
    def validate_tensor_properties(self, tensor: torch.Tensor, expected_shape: Optional[Tuple] = None,
                                  expected_dtype: Optional[torch.dtype] = None) -> bool:
        """
        Validate tensor properties with caching.
        
        Args:
            tensor: Tensor to validate
            expected_shape: Expected tensor shape
            expected_dtype: Expected tensor data type
            
        Returns:
            True if tensor is valid
        """
        try:
            # Basic tensor validation
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
            
            # Check for NaN/Inf
            if torch.isnan(tensor).any():
                raise ValueError("Tensor contains NaN values")
            if torch.isinf(tensor).any():
                raise ValueError("Tensor contains infinite values")
            
            # Shape validation
            if expected_shape and tensor.shape != expected_shape:
                raise ValueError(f"Shape mismatch: expected {expected_shape}, got {tensor.shape}")
            
            # Dtype validation
            if expected_dtype and tensor.dtype != expected_dtype:
                warnings.warn(f"Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
            
            # Memory validation
            tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            if tensor_size_mb > self.config.memory_limit_mb:
                raise ResourceConstraintError(
                    f"Tensor too large: {tensor_size_mb:.2f}MB > {self.config.memory_limit_mb}MB",
                    "memory", self.config.memory_limit_mb, tensor_size_mb
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Tensor validation failed: {e}")
            if self.config.recovery_strategy == ErrorRecoveryStrategy.FAIL_FAST:
                raise
            return False
    
    def validate_data_loader(self, data_loader, expected_batch_size: Optional[int] = None) -> bool:
        """
        Validate data loader properties and data quality.
        
        Args:
            data_loader: DataLoader to validate
            expected_batch_size: Expected batch size
            
        Returns:
            True if data loader is valid
        """
        try:
            with self._validation_timer("data_loader_validation"):
                # Basic structure validation
                if not hasattr(data_loader, '__iter__'):
                    raise ValueError("Data loader must be iterable")
                
                # Sample validation
                sample_batch = None
                for batch in data_loader:
                    sample_batch = batch
                    break
                
                if sample_batch is None:
                    raise ValueError("Data loader is empty")
                
                # Batch validation
                if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
                    inputs, targets = sample_batch[0], sample_batch[1]
                    
                    # Validate inputs
                    self.validate_tensor_properties(inputs)
                    
                    # Validate targets
                    if isinstance(targets, torch.Tensor):
                        self.validate_tensor_properties(targets)
                    
                    # Batch size validation
                    if expected_batch_size and inputs.shape[0] != expected_batch_size:
                        warnings.warn(f"Batch size mismatch: expected {expected_batch_size}, got {inputs.shape[0]}")
                
                return True
                
        except Exception as e:
            self._handle_validation_error("data_loader_validation", e, data_loader)
            return False
    
    def _handle_validation_error(self, validation_type: str, error: Exception, context: Any) -> None:
        """Handle validation errors with configured recovery strategy."""
        error_record = {
            'timestamp': time.time(),
            'validation_type': validation_type,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context_type': type(context).__name__,
            'recovery_strategy': self.config.recovery_strategy.value
        }
        
        self.error_history.append(error_record)
        
        if self.config.log_validation_details:
            logger.error(f"Validation error in {validation_type}: {error}")
            logger.debug(f"Error context: {context}")
        
        if self.config.recovery_strategy == ErrorRecoveryStrategy.FAIL_FAST:
            raise error
        elif self.config.recovery_strategy == ErrorRecoveryStrategy.RETRY:
            # Retry logic would be implemented here
            pass
        elif self.config.recovery_strategy == ErrorRecoveryStrategy.FALLBACK:
            # Fallback logic - continue with warnings
            if self.config.enable_warnings:
                warnings.warn(f"Validation failed in {validation_type}, using fallback behavior")
        elif self.config.recovery_strategy == ErrorRecoveryStrategy.GRACEFUL_DEGRADATION:
            # Graceful degradation - log and continue with reduced functionality
            logger.warning(f"Validation failed in {validation_type}, continuing with reduced functionality")
    
    def _validation_timer(self, operation_name: str):
        """Context manager for timing validation operations."""
        return ValidationTimer(operation_name, self.performance_stats)
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        return {
            'cache_size': len(self.validation_cache),
            'error_count': len(self.error_history),
            'performance_stats': self.performance_stats.copy(),
            'config': {
                'level': self.config.level.value,
                'recovery_strategy': self.config.recovery_strategy.value,
                'max_retries': self.config.max_retries
            },
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def clear_cache(self):
        """Clear validation cache and reset statistics."""
        self.validation_cache.clear()
        self.error_history = self.error_history[-100:]  # Keep last 100 errors
        logger.info("Cleared validation cache")


class ValidationTimer:
    """Context manager for timing validation operations."""
    
    def __init__(self, operation_name: str, stats_dict: Dict[str, float]):
        self.operation_name = operation_name
        self.stats_dict = stats_dict
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.stats_dict[self.operation_name] = duration


def validate_with_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for adding retry logic to validation functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delays
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Validation attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} validation attempts failed")
            
            raise last_exception
        return wrapper
    return decorator


def safe_operation(recovery_strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.FALLBACK,
                  default_return: Any = None):
    """
    Decorator for making operations safe with error recovery.
    
    Args:
        recovery_strategy: Strategy for handling errors
        default_return: Default return value on failure
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if recovery_strategy == ErrorRecoveryStrategy.FAIL_FAST:
                    raise
                elif recovery_strategy == ErrorRecoveryStrategy.FALLBACK:
                    logger.warning(f"Operation {func.__name__} failed, using fallback: {e}")
                    return default_return
                elif recovery_strategy == ErrorRecoveryStrategy.GRACEFUL_DEGRADATION:
                    logger.warning(f"Operation {func.__name__} failed, continuing with degraded functionality: {e}")
                    return default_return
                else:
                    raise
        return wrapper
    return decorator


# Global validator instance
_global_validator = None

def get_global_validator() -> RobustValidator:
    """Get the global validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = RobustValidator()
    return _global_validator


def set_validation_level(level: ValidationLevel):
    """Set global validation level."""
    validator = get_global_validator()
    validator.config.level = level
    logger.info(f"Set global validation level to {level.value}")


def validate_inputs(**validators):
    """
    Decorator for validating function inputs.
    
    Usage:
        @validate_inputs(x=lambda x: x > 0, name=lambda n: isinstance(n, str))
        def my_function(x, name):
            pass
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature for argument mapping
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each specified argument
            for arg_name, validator_func in validators.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    try:
                        if not validator_func(value):
                            raise ValueError(f"Validation failed for argument '{arg_name}': {value}")
                    except Exception as e:
                        raise ValueError(f"Validation error for argument '{arg_name}': {e}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator