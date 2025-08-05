"""
Input validation for ensuring data integrity and type safety.

This module provides comprehensive validation for all inputs to the
MoE system including tensors, parameters, and configuration values.
"""

import torch
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple, Type
from dataclasses import dataclass
import logging


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class TensorValidationError(ValidationError):
    """Exception for tensor validation failures."""
    pass


class ParameterValidationError(ValidationError):
    """Exception for parameter validation failures."""
    pass


@dataclass
class TensorConstraints:
    """Constraints for tensor validation."""
    min_dims: Optional[int] = None
    max_dims: Optional[int] = None
    shape: Optional[Tuple[int, ...]] = None
    min_shape: Optional[Tuple[int, ...]] = None
    max_shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None
    device: Optional[Union[str, torch.device]] = None
    requires_grad: Optional[bool] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allow_nan: bool = False
    allow_inf: bool = False


@dataclass 
class ParameterConstraints:
    """Constraints for parameter validation."""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    data_type: Optional[Type] = None
    required: bool = True
    validator_func: Optional[callable] = None


class InputValidator:
    """
    Comprehensive input validation system.
    
    Provides validation for tensors, parameters, configurations,
    and complex data structures used throughout the MoE system.
    """
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(__name__)
    
    def validate_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        constraints: TensorConstraints
    ) -> torch.Tensor:
        """
        Validate a tensor against specified constraints.
        
        Args:
            tensor: Tensor to validate
            name: Name of tensor for error messages
            constraints: Validation constraints
            
        Returns:
            Validated tensor
            
        Raises:
            TensorValidationError: If validation fails
        """
        if not isinstance(tensor, torch.Tensor):
            raise TensorValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        
        # Dimension validation
        if constraints.min_dims is not None and tensor.ndim < constraints.min_dims:
            raise TensorValidationError(
                f"{name} must have at least {constraints.min_dims} dimensions, got {tensor.ndim}"
            )
        
        if constraints.max_dims is not None and tensor.ndim > constraints.max_dims:
            raise TensorValidationError(
                f"{name} must have at most {constraints.max_dims} dimensions, got {tensor.ndim}"
            )
        
        # Shape validation
        if constraints.shape is not None:
            # Allow -1 as wildcard in shape constraints
            expected_shape = constraints.shape
            actual_shape = tensor.shape
            
            if len(expected_shape) != len(actual_shape):
                raise TensorValidationError(
                    f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"
                )
            
            for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
                if expected != -1 and expected != actual:
                    raise TensorValidationError(
                        f"{name} shape mismatch at dimension {i}: expected {expected}, got {actual}"
                    )
        
        if constraints.min_shape is not None:
            for i, (min_dim, actual_dim) in enumerate(zip(constraints.min_shape, tensor.shape)):
                if actual_dim < min_dim:
                    raise TensorValidationError(
                        f"{name} dimension {i} too small: expected >= {min_dim}, got {actual_dim}"
                    )
        
        if constraints.max_shape is not None:
            for i, (max_dim, actual_dim) in enumerate(zip(constraints.max_shape, tensor.shape)):
                if actual_dim > max_dim:
                    raise TensorValidationError(
                        f"{name} dimension {i} too large: expected <= {max_dim}, got {actual_dim}"
                    )
        
        # Data type validation
        if constraints.dtype is not None and tensor.dtype != constraints.dtype:
            if self.strict_mode:
                raise TensorValidationError(
                    f"{name} dtype mismatch: expected {constraints.dtype}, got {tensor.dtype}"
                )
            else:
                self.logger.warning(
                    f"{name} dtype mismatch: expected {constraints.dtype}, got {tensor.dtype}. "
                    f"Converting..."
                )
                tensor = tensor.to(constraints.dtype)
        
        # Device validation
        if constraints.device is not None:
            expected_device = torch.device(constraints.device)
            if tensor.device != expected_device:
                if self.strict_mode:
                    raise TensorValidationError(
                        f"{name} device mismatch: expected {expected_device}, got {tensor.device}"
                    )
                else:
                    self.logger.warning(
                        f"{name} device mismatch: expected {expected_device}, got {tensor.device}. "
                        f"Moving..."
                    )
                    tensor = tensor.to(expected_device)
        
        # Gradient requirement validation
        if constraints.requires_grad is not None and tensor.requires_grad != constraints.requires_grad:
            if self.strict_mode:
                raise TensorValidationError(
                    f"{name} requires_grad mismatch: expected {constraints.requires_grad}, "
                    f"got {tensor.requires_grad}"
                )
            else:
                tensor = tensor.requires_grad_(constraints.requires_grad)
        
        # Value range validation
        if constraints.min_value is not None or constraints.max_value is not None:
            if tensor.dtype.is_floating_point:
                # Check for NaN and infinity
                if not constraints.allow_nan and torch.isnan(tensor).any():
                    raise TensorValidationError(f"{name} contains NaN values")
                
                if not constraints.allow_inf and torch.isinf(tensor).any():
                    raise TensorValidationError(f"{name} contains infinite values")
                
                # Filter out NaN/inf for range checking
                finite_mask = torch.isfinite(tensor)
                if finite_mask.any():
                    finite_values = tensor[finite_mask]
                    
                    if constraints.min_value is not None:
                        min_val = finite_values.min().item()
                        if min_val < constraints.min_value:
                            raise TensorValidationError(
                                f"{name} contains values below minimum: {min_val} < {constraints.min_value}"
                            )
                    
                    if constraints.max_value is not None:
                        max_val = finite_values.max().item()
                        if max_val > constraints.max_value:
                            raise TensorValidationError(
                                f"{name} contains values above maximum: {max_val} > {constraints.max_value}"
                            )
        
        return tensor
    
    def validate_parameter(
        self,
        value: Any,
        name: str,
        constraints: ParameterConstraints
    ) -> Any:
        """
        Validate a parameter against specified constraints.
        
        Args:
            value: Value to validate
            name: Parameter name for error messages
            constraints: Validation constraints
            
        Returns:
            Validated value
            
        Raises:
            ParameterValidationError: If validation fails
        """
        # Required check
        if constraints.required and value is None:
            raise ParameterValidationError(f"{name} is required but not provided")
        
        if value is None:
            return value
        
        # Data type validation
        if constraints.data_type is not None and not isinstance(value, constraints.data_type):
            if self.strict_mode:
                raise ParameterValidationError(
                    f"{name} must be of type {constraints.data_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            else:
                try:
                    value = constraints.data_type(value)
                except (ValueError, TypeError) as e:
                    raise ParameterValidationError(
                        f"Cannot convert {name} to {constraints.data_type.__name__}: {e}"
                    )
        
        # Value range validation
        if isinstance(value, (int, float)):
            if constraints.min_value is not None and value < constraints.min_value:
                raise ParameterValidationError(
                    f"{name} must be >= {constraints.min_value}, got {value}"
                )
            
            if constraints.max_value is not None and value > constraints.max_value:
                raise ParameterValidationError(
                    f"{name} must be <= {constraints.max_value}, got {value}"
                )
        
        # Allowed values validation
        if constraints.allowed_values is not None and value not in constraints.allowed_values:
            raise ParameterValidationError(
                f"{name} must be one of {constraints.allowed_values}, got {value}"
            )
        
        # Custom validator function
        if constraints.validator_func is not None:
            try:
                is_valid = constraints.validator_func(value)
                if not is_valid:
                    raise ParameterValidationError(f"{name} failed custom validation")
            except Exception as e:
                raise ParameterValidationError(f"{name} custom validation error: {e}")
        
        return value
    
    def validate_batch_input(
        self,
        x: torch.Tensor,
        expected_feature_dim: int,
        max_sequence_length: Optional[int] = None,
        max_batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Validate standard batch input tensor [batch_size, seq_len, feature_dim]."""
        constraints = TensorConstraints(
            min_dims=3,
            max_dims=3,
            min_shape=(1, 1, expected_feature_dim),
            dtype=torch.float32,
            allow_nan=False,
            allow_inf=False
        )
        
        if max_sequence_length is not None:
            constraints.max_shape = (max_batch_size or float('inf'), max_sequence_length, expected_feature_dim)
        
        return self.validate_tensor(x, "batch_input", constraints)
    
    def validate_expert_indices(
        self,
        indices: torch.Tensor,
        num_experts: int,
        batch_size: int,
        sequence_length: int
    ) -> torch.Tensor:
        """Validate expert selection indices."""
        constraints = TensorConstraints(
            min_dims=2,
            dtype=torch.long,
            min_value=0,
            max_value=num_experts - 1
        )
        
        # Check shape is compatible with batch and sequence dimensions
        indices = self.validate_tensor(indices, "expert_indices", constraints)
        
        if indices.shape[0] != batch_size:
            raise TensorValidationError(
                f"expert_indices batch dimension mismatch: expected {batch_size}, "
                f"got {indices.shape[0]}"
            )
        
        return indices
    
    def validate_routing_weights(
        self,
        weights: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Validate routing weights match expert indices."""
        constraints = TensorConstraints(
            dtype=torch.float32,
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_inf=False
        )
        
        weights = self.validate_tensor(weights, "routing_weights", constraints)
        
        # Check shape compatibility with indices
        if weights.shape != indices.shape:
            raise TensorValidationError(
                f"routing_weights shape {weights.shape} must match "
                f"expert_indices shape {indices.shape}"
            )
        
        # Check that weights sum approximately to 1 along last dimension
        weight_sums = weights.sum(dim=-1)
        tolerance = 1e-5
        
        if not torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=tolerance):
            if self.strict_mode:
                raise TensorValidationError(
                    f"routing_weights must sum to 1.0 along last dimension, "
                    f"got sums in range [{weight_sums.min():.6f}, {weight_sums.max():.6f}]"
                )
            else:
                # Normalize weights
                weights = weights / weight_sums.unsqueeze(-1)
                self.logger.warning("Normalized routing weights to sum to 1.0")
        
        return weights
    
    def validate_config_dict(
        self,
        config: Dict[str, Any],
        schema: Dict[str, ParameterConstraints]
    ) -> Dict[str, Any]:
        """
        Validate a configuration dictionary against a schema.
        
        Args:
            config: Configuration dictionary to validate
            schema: Schema defining constraints for each parameter
            
        Returns:
            Validated configuration dictionary
        """
        validated_config = {}
        
        # Validate provided parameters
        for key, value in config.items():
            if key in schema:
                validated_config[key] = self.validate_parameter(value, key, schema[key])
            elif self.strict_mode:
                raise ParameterValidationError(f"Unknown configuration parameter: {key}")
            else:
                validated_config[key] = value
                self.logger.warning(f"Unknown configuration parameter: {key}")
        
        # Check for missing required parameters
        for key, constraints in schema.items():
            if constraints.required and key not in config:
                raise ParameterValidationError(f"Required configuration parameter missing: {key}")
            elif key not in config and not constraints.required:
                # Use default value if available
                validated_config[key] = None
        
        return validated_config
    
    def validate_model_output(
        self,
        output: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Validate model output tensor."""
        constraints = TensorConstraints(
            dtype=expected_dtype,
            allow_nan=False,
            allow_inf=False
        )
        
        if expected_shape is not None:
            constraints.shape = expected_shape
        
        return self.validate_tensor(output, "model_output", constraints)


# Convenience functions for common validations
def validate_positive_int(value: int, name: str) -> int:
    """Validate positive integer."""
    validator = InputValidator()
    return validator.validate_parameter(
        value, name,
        ParameterConstraints(
            min_value=1,
            data_type=int
        )
    )


def validate_probability(value: float, name: str) -> float:
    """Validate probability value [0, 1]."""
    validator = InputValidator()
    return validator.validate_parameter(
        value, name,
        ParameterConstraints(
            min_value=0.0,
            max_value=1.0,
            data_type=float
        )
    )


def validate_device(device: Union[str, torch.device]) -> torch.device:
    """Validate and normalize device specification."""
    if isinstance(device, str):
        try:
            device = torch.device(device)
        except RuntimeError as e:
            raise ValidationError(f"Invalid device specification: {device}. Error: {e}")
    
    if not isinstance(device, torch.device):
        raise ValidationError(f"Device must be string or torch.device, got {type(device)}")
    
    # Check if CUDA device is available when requested
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValidationError(f"CUDA device requested but CUDA is not available")
    
    if device.type == 'cuda' and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise ValidationError(
                f"CUDA device {device.index} requested but only "
                f"{torch.cuda.device_count()} devices available"
            )
    
    return device