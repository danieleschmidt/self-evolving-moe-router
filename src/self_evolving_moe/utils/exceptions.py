"""
Custom exceptions for Self-Evolving MoE-Router.

This module defines custom exception classes for better error handling
and debugging throughout the system.
"""

class MoEValidationError(ValueError):
    """Base exception for validation errors in MoE components."""
    pass


class EvolutionError(Exception):
    """Exception raised during evolution process."""
    
    def __init__(self, message: str, generation: int = None, population_size: int = None):
        super().__init__(message)
        self.generation = generation
        self.population_size = population_size


class TopologyError(Exception):
    """Exception raised for topology-related errors."""
    
    def __init__(self, message: str, topology_id: str = None, sparsity: float = None):
        super().__init__(message)
        self.topology_id = topology_id
        self.sparsity = sparsity


class ExpertPoolError(Exception):
    """Exception raised for expert pool management errors."""
    
    def __init__(self, message: str, num_experts: int = None, expert_type: str = None):
        super().__init__(message)
        self.num_experts = num_experts
        self.expert_type = expert_type


class ResourceConstraintError(Exception):
    """Exception raised when resource constraints are violated."""
    
    def __init__(self, message: str, resource_type: str = None, limit: float = None, actual: float = None):
        super().__init__(message)
        self.resource_type = resource_type
        self.limit = limit
        self.actual = actual


class ConfigurationError(MoEValidationError):
    """Exception raised for invalid configuration parameters."""
    
    def __init__(self, message: str, parameter: str = None, value = None):
        super().__init__(message)
        self.parameter = parameter
        self.value = value


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    
    def __init__(self, message: str, model_path: str = None, expected_format: str = None):
        super().__init__(message)
        self.model_path = model_path
        self.expected_format = expected_format


class CompatibilityError(Exception):
    """Exception raised for compatibility issues between components."""
    
    def __init__(self, message: str, component1: str = None, component2: str = None):
        super().__init__(message)
        self.component1 = component1
        self.component2 = component2