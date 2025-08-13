#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST
Self-Evolving MoE Router with comprehensive error handling, validation, and monitoring
"""

import numpy as np
import random
import json
import time
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import sys
import os


# Configure comprehensive logging
class ColoredFormatter(logging.Formatter):
    """Colored log formatter for better visibility."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_robust_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup comprehensive logging with error handling."""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")
    
    return logger


class ValidationError(Exception):
    """Custom validation error."""
    pass


class EvolutionError(Exception):
    """Custom evolution error."""
    pass


class TopologyError(Exception):
    """Custom topology error."""
    pass


class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class HealthMetrics:
    """System health metrics."""
    status: HealthStatus
    fitness_trend: float = 0.0
    convergence_rate: float = 0.0
    diversity_score: float = 0.0
    expert_utilization: float = 0.0
    memory_usage: float = 0.0
    computation_time: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    last_check: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        return result


class RobustValidator:
    """Comprehensive input validation and sanitization."""
    
    @staticmethod
    def validate_array_shape(array: np.ndarray, expected_shape: Tuple[int, ...], 
                           name: str = "array") -> np.ndarray:
        """Validate array shape with detailed error messages."""
        if not isinstance(array, np.ndarray):
            raise ValidationError(f"{name} must be numpy array, got {type(array)}")
        
        if len(array.shape) != len(expected_shape):
            raise ValidationError(
                f"{name} must have {len(expected_shape)} dimensions, "
                f"got {len(array.shape)}: {array.shape}"
            )
        
        for i, (actual, expected) in enumerate(zip(array.shape, expected_shape)):
            if expected != -1 and actual != expected:  # -1 means any size allowed
                raise ValidationError(
                    f"{name} dimension {i} must be {expected}, got {actual}"
                )
        
        return array
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], min_val: Optional[float] = None,
                             max_val: Optional[float] = None, name: str = "value") -> Union[int, float]:
        """Validate numeric value is within range."""
        if not isinstance(value, (int, float, np.number)):
            raise ValidationError(f"{name} must be numeric, got {type(value)}")
        
        if np.isnan(value) or np.isinf(value):
            raise ValidationError(f"{name} must be finite, got {value}")
        
        if min_val is not None and value < min_val:
            raise ValidationError(f"{name} must be >= {min_val}, got {value}")
        
        if max_val is not None and value > max_val:
            raise ValidationError(f"{name} must be <= {max_val}, got {value}")
        
        return value
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration dictionary."""
        required_keys = ['input_dim', 'num_experts', 'hidden_dim', 'num_tokens']
        
        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Missing required config key: {key}")
            
            if not isinstance(config[key], int) or config[key] <= 0:
                raise ValidationError(f"Config {key} must be positive integer")
        
        # Validate ranges
        if config['num_experts'] > 64:
            warnings.warn("Large number of experts may impact performance")
        
        if config['input_dim'] > 2048:
            warnings.warn("Large input dimension may impact memory usage")
        
        return config
    
    @staticmethod
    def sanitize_topology_matrix(matrix: np.ndarray) -> np.ndarray:
        """Sanitize and validate topology matrix."""
        # Check for NaN/inf values
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            raise ValidationError("Topology matrix contains NaN or infinite values")
        
        # Ensure binary values
        matrix = np.clip(matrix, 0, 1)
        matrix = (matrix > 0.5).astype(float)
        
        # Ensure each token connects to at least one expert
        for i in range(matrix.shape[0]):
            if np.sum(matrix[i]) == 0:
                # Connect to random expert
                expert_idx = random.randint(0, matrix.shape[1] - 1)
                matrix[i, expert_idx] = 1
                logging.warning(f"Fixed disconnected token {i} by connecting to expert {expert_idx}")
        
        return matrix


class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_history: List[HealthMetrics] = []
        self.error_count = 0
        self.warning_count = 0
        self.start_time = time.time()
        self.log_file = log_file
    
    def check_health(self, evolver, model) -> HealthMetrics:
        """Comprehensive health check."""
        try:
            current_time = time.time()
            
            # Calculate fitness trend
            fitness_trend = self._calculate_fitness_trend(evolver)
            
            # Calculate convergence rate
            convergence_rate = self._calculate_convergence_rate(evolver)
            
            # Calculate diversity
            diversity_score = self._calculate_diversity(evolver)
            
            # Calculate expert utilization
            expert_utilization = self._calculate_expert_utilization(model)
            
            # Memory usage
            memory_usage = self._estimate_memory_usage()
            
            # Computation time
            computation_time = current_time - self.start_time
            
            # Determine overall status
            status = self._determine_status(
                fitness_trend, convergence_rate, diversity_score, 
                expert_utilization, memory_usage
            )
            
            metrics = HealthMetrics(
                status=status,
                fitness_trend=fitness_trend,
                convergence_rate=convergence_rate,
                diversity_score=diversity_score,
                expert_utilization=expert_utilization,
                memory_usage=memory_usage,
                computation_time=computation_time,
                error_count=self.error_count,
                warning_count=self.warning_count,
                last_check=current_time
            )
            
            self.metrics_history.append(metrics)
            
            # Log health status
            self._log_health_status(metrics)
            
            return metrics
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Health check failed: {e}")
            return HealthMetrics(
                status=HealthStatus.FAILED,
                error_count=self.error_count,
                last_check=time.time()
            )
    
    def _calculate_fitness_trend(self, evolver) -> float:
        """Calculate fitness improvement trend."""
        if len(evolver.fitness_history) < 2:
            return 0.0
        
        recent_generations = min(5, len(evolver.fitness_history))
        recent_best = [max(scores) for scores in evolver.fitness_history[-recent_generations:]]
        
        if len(recent_best) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(recent_best))
        y = np.array(recent_best)
        slope = np.polyfit(x, y, 1)[0]
        
        return float(slope)
    
    def _calculate_convergence_rate(self, evolver) -> float:
        """Calculate convergence rate."""
        if len(evolver.fitness_history) < 3:
            return 0.0
        
        recent_generations = min(10, len(evolver.fitness_history))
        recent_best = [max(scores) for scores in evolver.fitness_history[-recent_generations:]]
        
        if len(recent_best) < 3:
            return 0.0
        
        # Standard deviation of recent best fitness scores
        std_dev = np.std(recent_best)
        convergence_rate = 1.0 / (1.0 + std_dev)  # Higher when more converged
        
        return float(convergence_rate)
    
    def _calculate_diversity(self, evolver) -> float:
        """Calculate population diversity."""
        if len(evolver.population) < 2:
            return 0.0
        
        try:
            diversity_sum = 0.0
            comparisons = 0
            
            for i in range(len(evolver.population)):
                for j in range(i + 1, len(evolver.population)):
                    # Calculate Hamming distance between routing matrices
                    matrix1 = evolver.population[i].routing_matrix
                    matrix2 = evolver.population[j].routing_matrix
                    
                    diff = np.mean(matrix1 != matrix2)
                    diversity_sum += diff
                    comparisons += 1
            
            return diversity_sum / max(comparisons, 1)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate diversity: {e}")
            return 0.0
    
    def _calculate_expert_utilization(self, model) -> float:
        """Calculate expert utilization balance."""
        try:
            if hasattr(model, 'last_expert_usage') and model.last_expert_usage is not None:
                usage = model.last_expert_usage
                if np.sum(usage) > 0:
                    usage_probs = usage / np.sum(usage)
                    entropy = -np.sum(usage_probs * np.log(usage_probs + 1e-8))
                    max_entropy = np.log(len(usage))
                    return entropy / max_entropy if max_entropy > 0 else 0.0
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Could not calculate expert utilization: {e}")
            return 0.0
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb
        except ImportError:
            return 0.0
        except Exception as e:
            self.logger.warning(f"Could not estimate memory usage: {e}")
            return 0.0
    
    def _determine_status(self, fitness_trend: float, convergence_rate: float,
                         diversity_score: float, expert_utilization: float,
                         memory_usage: float) -> HealthStatus:
        """Determine overall health status."""
        
        # Critical conditions
        if memory_usage > 8000:  # > 8GB
            return HealthStatus.CRITICAL
        
        if fitness_trend < -0.01:  # Fitness decreasing
            return HealthStatus.CRITICAL
        
        # Warning conditions
        if diversity_score < 0.1:  # Low diversity
            return HealthStatus.WARNING
        
        if expert_utilization < 0.3:  # Poor expert utilization
            return HealthStatus.WARNING
        
        if convergence_rate > 0.95:  # Potentially converged
            return HealthStatus.WARNING
        
        # Healthy conditions
        return HealthStatus.HEALTHY
    
    def _log_health_status(self, metrics: HealthMetrics):
        """Log health status with appropriate level."""
        status_msg = (
            f"Health: {metrics.status.value.upper()} | "
            f"Fitness trend: {metrics.fitness_trend:.4f} | "
            f"Diversity: {metrics.diversity_score:.3f} | "
            f"Expert util: {metrics.expert_utilization:.3f} | "
            f"Memory: {metrics.memory_usage:.1f}MB"
        )
        
        if metrics.status == HealthStatus.HEALTHY:
            self.logger.info(status_msg)
        elif metrics.status == HealthStatus.WARNING:
            self.logger.warning(status_msg)
        else:
            self.logger.error(status_msg)
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No health data available"}
        
        latest = self.metrics_history[-1]
        
        return {
            "current_status": latest.status.value,
            "metrics": latest.to_dict(),
            "trends": {
                "fitness_trend": [m.fitness_trend for m in self.metrics_history[-10:]],
                "diversity_trend": [m.diversity_score for m in self.metrics_history[-10:]],
                "memory_trend": [m.memory_usage for m in self.metrics_history[-10:]]
            },
            "total_errors": self.error_count,
            "total_warnings": self.warning_count,
            "uptime": time.time() - self.start_time
        }


class RobustTopology:
    """Robust topology with comprehensive error handling."""
    
    def __init__(self, num_tokens: int, num_experts: int, sparsity: float = 0.1,
                 validator: Optional[RobustValidator] = None):
        self.validator = validator or RobustValidator()
        
        # Validate inputs
        self.num_tokens = self.validator.validate_numeric_range(
            num_tokens, min_val=1, max_val=1024, name="num_tokens"
        )
        self.num_experts = self.validator.validate_numeric_range(
            num_experts, min_val=2, max_val=64, name="num_experts"
        )
        self.sparsity = self.validator.validate_numeric_range(
            sparsity, min_val=0.0, max_val=0.95, name="sparsity"
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize routing matrix with validation
        self.routing_matrix = self._initialize_robust_matrix()
        
        # Metadata
        self.generation = 0
        self.fitness_history = []
        self.mutation_count = 0
        self.crossover_count = 0
        
    def _initialize_robust_matrix(self) -> np.ndarray:
        """Initialize routing matrix with error handling."""
        try:
            matrix = np.zeros((self.num_tokens, self.num_experts))
            
            # Ensure each token connects to at least one expert
            for token_idx in range(self.num_tokens):
                expert_idx = random.randint(0, self.num_experts - 1)
                matrix[token_idx, expert_idx] = 1
            
            # Add additional connections based on sparsity
            total_possible = self.num_tokens * self.num_experts
            target_connections = max(
                self.num_tokens,  # At least one per token
                int(total_possible * (1 - self.sparsity))
            )
            
            current_connections = int(np.sum(matrix))
            needed_connections = target_connections - current_connections
            
            # Add random connections
            attempts = 0
            max_attempts = needed_connections * 3
            
            while needed_connections > 0 and attempts < max_attempts:
                token_idx = random.randint(0, self.num_tokens - 1)
                expert_idx = random.randint(0, self.num_experts - 1)
                
                if matrix[token_idx, expert_idx] == 0:
                    matrix[token_idx, expert_idx] = 1
                    needed_connections -= 1
                
                attempts += 1
            
            if attempts >= max_attempts:
                self.logger.warning(f"Could not achieve target sparsity, got {self.compute_sparsity():.3f}")
            
            # Final validation and sanitization
            matrix = self.validator.sanitize_topology_matrix(matrix)
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Failed to initialize routing matrix: {e}")
            # Fallback to simple initialization
            matrix = np.zeros((self.num_tokens, self.num_experts))
            for i in range(self.num_tokens):
                matrix[i, i % self.num_experts] = 1
            return matrix
    
    def get_routing_mask(self, seq_len: int) -> np.ndarray:
        """Get routing mask with robust handling of different sequence lengths."""
        try:
            seq_len = int(seq_len)
            if seq_len <= 0:
                raise ValidationError(f"seq_len must be positive, got {seq_len}")
            
            if seq_len <= self.num_tokens:
                return self.routing_matrix[:seq_len, :].copy()
            else:
                # Extend pattern for longer sequences
                repetitions = (seq_len + self.num_tokens - 1) // self.num_tokens
                repeated = np.tile(self.routing_matrix, (repetitions, 1))
                return repeated[:seq_len, :].copy()
                
        except Exception as e:
            self.logger.error(f"Failed to get routing mask: {e}")
            # Fallback to uniform routing
            return np.ones((seq_len, self.num_experts)) / self.num_experts
    
    def mutate(self, mutation_rate: float = 0.1) -> bool:
        """Robust mutation with comprehensive error handling."""
        try:
            mutation_rate = self.validator.validate_numeric_range(
                mutation_rate, min_val=0.0, max_val=1.0, name="mutation_rate"
            )
            
            original_matrix = self.routing_matrix.copy()
            mutated = False
            
            # Add connection mutation
            if random.random() < mutation_rate:
                zero_positions = np.where(self.routing_matrix == 0)
                if len(zero_positions[0]) > 0:
                    idx = random.randint(0, len(zero_positions[0]) - 1)
                    t, e = zero_positions[0][idx], zero_positions[1][idx]
                    self.routing_matrix[t, e] = 1
                    mutated = True
            
            # Remove connection mutation (with safety checks)
            if random.random() < mutation_rate:
                for token_idx in range(self.num_tokens):
                    token_connections = np.sum(self.routing_matrix[token_idx])
                    if token_connections > 1:  # Keep at least one connection
                        expert_candidates = np.where(self.routing_matrix[token_idx] == 1)[0]
                        if len(expert_candidates) > 1:
                            expert_to_remove = random.choice(expert_candidates)
                            self.routing_matrix[token_idx, expert_to_remove] = 0
                            mutated = True
                        break
            
            # Validate mutation result
            try:
                self.routing_matrix = self.validator.sanitize_topology_matrix(self.routing_matrix)
            except ValidationError as e:
                self.logger.warning(f"Mutation validation failed, reverting: {e}")
                self.routing_matrix = original_matrix
                return False
            
            if mutated:
                self.mutation_count += 1
                self.generation += 1
            
            return mutated
            
        except Exception as e:
            self.logger.error(f"Mutation failed: {e}")
            return False
    
    def crossover(self, other: 'RobustTopology') -> Optional['RobustTopology']:
        """Robust crossover with validation."""
        try:
            if (self.num_tokens != other.num_tokens or 
                self.num_experts != other.num_experts):
                raise ValidationError(
                    f"Cannot crossover topologies with different dimensions: "
                    f"({self.num_tokens}, {self.num_experts}) vs "
                    f"({other.num_tokens}, {other.num_experts})"
                )
            
            child = RobustTopology(
                num_tokens=self.num_tokens,
                num_experts=self.num_experts,
                sparsity=(self.sparsity + other.sparsity) / 2,
                validator=self.validator
            )
            
            # Uniform crossover
            mask = np.random.random((self.num_tokens, self.num_experts)) > 0.5
            child.routing_matrix = np.where(
                mask, self.routing_matrix, other.routing_matrix
            )
            
            # Validate and sanitize result
            child.routing_matrix = self.validator.sanitize_topology_matrix(child.routing_matrix)
            
            child.crossover_count = max(self.crossover_count, other.crossover_count) + 1
            child.generation = max(self.generation, other.generation) + 1
            
            return child
            
        except Exception as e:
            self.logger.error(f"Crossover failed: {e}")
            return None
    
    def compute_sparsity(self) -> float:
        """Compute actual sparsity with error handling."""
        try:
            total_connections = np.sum(self.routing_matrix)
            total_possible = self.routing_matrix.size
            return float(1.0 - (total_connections / total_possible))
        except Exception as e:
            self.logger.error(f"Failed to compute sparsity: {e}")
            return 0.0
    
    def validate_integrity(self) -> bool:
        """Validate topology integrity."""
        try:
            # Check matrix properties
            if self.routing_matrix.shape != (self.num_tokens, self.num_experts):
                return False
            
            # Check for invalid values
            if np.any(np.isnan(self.routing_matrix)) or np.any(np.isinf(self.routing_matrix)):
                return False
            
            # Check binary values
            unique_values = np.unique(self.routing_matrix)
            if not np.all(np.isin(unique_values, [0, 1])):
                return False
            
            # Check connectivity
            for token_idx in range(self.num_tokens):
                if np.sum(self.routing_matrix[token_idx]) == 0:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Integrity validation failed: {e}")
            return False


class RobustMoEModel:
    """Robust MoE model with comprehensive error handling."""
    
    def __init__(self, input_dim: int = 64, num_experts: int = 8, hidden_dim: int = 128,
                 validator: Optional[RobustValidator] = None):
        self.validator = validator or RobustValidator()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate and store configuration
        config = {
            'input_dim': input_dim,
            'num_experts': num_experts,
            'hidden_dim': hidden_dim,
            'num_tokens': 16  # Default
        }
        self.config = self.validator.validate_config(config)
        
        self.input_dim = self.config['input_dim']
        self.num_experts = self.config['num_experts']
        self.hidden_dim = self.config['hidden_dim']
        
        # Initialize model weights with error handling
        self.expert_weights = []
        self.router_weights = None
        self.router_bias = None
        self.current_topology = None
        self.last_expert_usage = None
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.error_count = 0
        
        try:
            self._initialize_weights()
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise EvolutionError(f"Model initialization failed: {e}")
    
    def _initialize_weights(self):
        """Initialize model weights with robust error handling."""
        try:
            # Initialize expert weights
            for i in range(self.num_experts):
                try:
                    w1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.1
                    b1 = np.zeros(self.hidden_dim)
                    w2 = np.random.randn(self.hidden_dim, self.input_dim) * 0.1
                    b2 = np.zeros(self.input_dim)
                    
                    # Validate weight shapes
                    assert w1.shape == (self.input_dim, self.hidden_dim)
                    assert b1.shape == (self.hidden_dim,)
                    assert w2.shape == (self.hidden_dim, self.input_dim)
                    assert b2.shape == (self.input_dim,)
                    
                    self.expert_weights.append((w1, b1, w2, b2))
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize expert {i}: {e}")
                    # Use identity transformation as fallback
                    w1 = np.eye(self.input_dim, self.hidden_dim) * 0.1
                    b1 = np.zeros(self.hidden_dim)
                    w2 = np.eye(self.hidden_dim, self.input_dim) * 0.1
                    b2 = np.zeros(self.input_dim)
                    self.expert_weights.append((w1, b1, w2, b2))
            
            # Initialize router
            self.router_weights = np.random.randn(self.input_dim, self.num_experts) * 0.1
            self.router_bias = np.zeros(self.num_experts)
            
            self.logger.info(f"Successfully initialized {len(self.expert_weights)} experts")
            
        except Exception as e:
            self.logger.error(f"Weight initialization failed: {e}")
            raise
    
    def set_routing_topology(self, topology: Optional[RobustTopology]):
        """Set routing topology with validation."""
        try:
            if topology is not None:
                if not isinstance(topology, RobustTopology):
                    raise ValidationError(f"Expected RobustTopology, got {type(topology)}")
                
                if not topology.validate_integrity():
                    raise ValidationError("Topology failed integrity check")
                
                if topology.num_experts != self.num_experts:
                    raise ValidationError(
                        f"Topology expert count {topology.num_experts} "
                        f"doesn't match model {self.num_experts}"
                    )
            
            self.current_topology = topology
            
            if topology:
                self.logger.debug(f"Set topology with sparsity {topology.compute_sparsity():.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to set topology: {e}")
            raise
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU with overflow protection."""
        try:
            # Clip to prevent overflow
            x = np.clip(x, -50, 50)
            return np.maximum(0, x)
        except Exception as e:
            self.logger.error(f"ReLU failed: {e}")
            return np.zeros_like(x)
    
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        try:
            # Clip inputs to prevent overflow
            x = np.clip(x, -50, 50)
            
            # Subtract max for numerical stability
            x_shifted = x - np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(x_shifted)
            
            # Check for NaN/inf
            if np.any(np.isnan(exp_x)) or np.any(np.isinf(exp_x)):
                self.logger.warning("Softmax produced NaN/inf, using uniform distribution")
                shape = list(x.shape)
                shape[axis] = 1
                return np.ones_like(x) / x.shape[axis]
            
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
            
        except Exception as e:
            self.logger.error(f"Softmax failed: {e}")
            # Fallback to uniform distribution
            return np.ones_like(x) / x.shape[axis]
    
    def forward(self, x: np.ndarray, top_k: int = 2) -> Tuple[np.ndarray, Dict]:
        """Robust forward pass with comprehensive error handling."""
        start_time = time.time()
        
        try:
            # Input validation
            if not isinstance(x, np.ndarray):
                raise ValidationError(f"Input must be numpy array, got {type(x)}")
            
            if len(x.shape) != 3:
                raise ValidationError(f"Input must have 3 dimensions, got {len(x.shape)}")
            
            batch_size, seq_len, input_dim = x.shape
            
            if input_dim != self.input_dim:
                raise ValidationError(
                    f"Input dimension {input_dim} doesn't match model {self.input_dim}"
                )
            
            # Validate top_k
            top_k = max(1, min(top_k, self.num_experts))
            
            # Check for NaN/inf in input
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                self.logger.warning("Input contains NaN/inf values, clipping")
                x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Compute router logits with error handling
            try:
                router_logits = np.dot(x, self.router_weights) + self.router_bias
            except Exception as e:
                self.logger.error(f"Router computation failed: {e}")
                # Fallback to uniform routing
                router_logits = np.ones((batch_size, seq_len, self.num_experts))
            
            # Apply topology constraints
            if self.current_topology is not None:
                try:
                    mask = self.current_topology.get_routing_mask(seq_len)
                    mask_expanded = np.expand_dims(mask, 0)
                    router_logits = router_logits * mask_expanded + (1 - mask_expanded) * (-1e9)
                except Exception as e:
                    self.logger.warning(f"Failed to apply topology mask: {e}")
            
            # Compute routing weights
            routing_weights = self.softmax(router_logits)
            
            # Top-k selection with error handling
            try:
                top_k_indices = np.argsort(routing_weights, axis=-1)[..., -top_k:]
            except Exception as e:
                self.logger.error(f"Top-k selection failed: {e}")
                # Fallback to first k experts
                top_k_indices = np.tile(
                    np.arange(top_k).reshape(1, 1, -1),
                    (batch_size, seq_len, 1)
                )
            
            # Initialize outputs
            output = np.zeros_like(x)
            expert_usage = np.zeros(self.num_experts)
            successful_forwards = 0
            failed_forwards = 0
            
            # Route through experts
            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    try:
                        token_input = x[batch_idx, seq_idx]
                        token_output = np.zeros(self.input_dim)
                        
                        for k in range(top_k):
                            expert_idx = top_k_indices[batch_idx, seq_idx, k]
                            weight = routing_weights[batch_idx, seq_idx, expert_idx]
                            
                            # Forward through expert with error handling
                            try:
                                w1, b1, w2, b2 = self.expert_weights[expert_idx]
                                
                                # First layer
                                hidden = np.dot(token_input, w1) + b1
                                hidden = self.relu(hidden)
                                
                                # Second layer
                                expert_out = np.dot(hidden, w2) + b2
                                
                                # Check for NaN/inf
                                if np.any(np.isnan(expert_out)) or np.any(np.isinf(expert_out)):
                                    self.logger.warning(f"Expert {expert_idx} output contains NaN/inf")
                                    expert_out = np.zeros_like(expert_out)
                                
                                # Apply weight and accumulate
                                token_output += weight * expert_out
                                expert_usage[expert_idx] += weight
                                
                            except Exception as e:
                                self.logger.warning(f"Expert {expert_idx} forward failed: {e}")
                                failed_forwards += 1
                                continue
                        
                        output[batch_idx, seq_idx] = token_output
                        successful_forwards += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Token forward failed at [{batch_idx}, {seq_idx}]: {e}")
                        failed_forwards += 1
                        continue
            
            # Update tracking
            self.inference_count += 1
            self.total_inference_time += time.time() - start_time
            self.last_expert_usage = expert_usage
            
            if failed_forwards > 0:
                self.error_count += 1
                self.logger.warning(f"Forward pass had {failed_forwards} failures")
            
            # Prepare auxiliary info
            aux_info = {
                'expert_usage': expert_usage,
                'successful_forwards': successful_forwards,
                'failed_forwards': failed_forwards,
                'inference_time': time.time() - start_time,
                'routing_weights_stats': {
                    'mean': float(np.mean(routing_weights)),
                    'std': float(np.std(routing_weights)),
                    'min': float(np.min(routing_weights)),
                    'max': float(np.max(routing_weights))
                }
            }
            
            return output, aux_info
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Forward pass failed completely: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return safe fallback
            try:
                fallback_output = np.zeros_like(x) if isinstance(x, np.ndarray) else np.zeros((1, 1, self.input_dim))
                fallback_aux = {
                    'expert_usage': np.zeros(self.num_experts),
                    'error': str(e),
                    'fallback_used': True
                }
                return fallback_output, fallback_aux
            except:
                raise EvolutionError(f"Forward pass failed critically: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get model performance statistics."""
        avg_inference_time = (
            self.total_inference_time / max(self.inference_count, 1)
        )
        
        return {
            'inference_count': self.inference_count,
            'total_inference_time': self.total_inference_time,
            'avg_inference_time': avg_inference_time,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.inference_count, 1),
            'last_expert_usage': self.last_expert_usage.tolist() if self.last_expert_usage is not None else None
        }


def run_robust_demo():
    """Run the robust Generation 2 demo."""
    
    # Setup logging
    log_file = "evolution_results/robust_demo.log"
    Path("evolution_results").mkdir(exist_ok=True)
    logger = setup_robust_logging("INFO", log_file)
    
    logger.info("🛡️  Starting Robust Self-Evolving MoE Router - Generation 2")
    
    try:
        # Configuration with validation
        config = {
            'input_dim': 64,
            'num_experts': 8,
            'hidden_dim': 128,
            'num_tokens': 16,
            'population_size': 12,
            'generations': 20,
            'batch_size': 4,
            'seq_len': 16
        }
        
        validator = RobustValidator()
        config = validator.validate_config(config)
        
        logger.info(f"Validated configuration: {config}")
        
        # Initialize health monitor
        health_monitor = HealthMonitor(log_file)
        
        # Create robust model
        model = RobustMoEModel(
            input_dim=config['input_dim'],
            num_experts=config['num_experts'],
            hidden_dim=config['hidden_dim'],
            validator=validator
        )
        
        logger.info(f"Created robust MoE model")
        
        # Create demo data with validation
        data_batches = []
        for i in range(10):
            try:
                inputs = np.random.randn(config['batch_size'], config['seq_len'], config['input_dim'])
                targets = inputs * 0.8 + 0.1 * np.random.randn(config['batch_size'], config['seq_len'], config['input_dim'])
                
                # Validate data
                validator.validate_array_shape(inputs, (config['batch_size'], config['seq_len'], config['input_dim']), "inputs")
                validator.validate_array_shape(targets, (config['batch_size'], config['seq_len'], config['input_dim']), "targets")
                
                data_batches.append((inputs, targets))
                
            except Exception as e:
                logger.error(f"Failed to create data batch {i}: {e}")
                continue
        
        logger.info(f"Created {len(data_batches)} validated data batches")
        
        # Create robust population
        population = []
        for i in range(config['population_size']):
            try:
                topology = RobustTopology(
                    num_tokens=config['num_tokens'],
                    num_experts=config['num_experts'],
                    sparsity=random.uniform(0.1, 0.4),
                    validator=validator
                )
                
                if topology.validate_integrity():
                    population.append(topology)
                else:
                    logger.warning(f"Topology {i} failed integrity check")
                    
            except Exception as e:
                logger.error(f"Failed to create topology {i}: {e}")
                continue
        
        logger.info(f"Created robust population of {len(population)} topologies")
        
        # Evolution loop with comprehensive monitoring
        generation = 0
        best_topology = None
        best_fitness = float('-inf')
        evolution_stats = []
        
        start_time = time.time()
        
        for gen in range(config['generations']):
            try:
                logger.info(f"=== Generation {gen} ===")
                
                # Health check
                health_metrics = health_monitor.check_health(
                    type('MockEvolver', (), {
                        'fitness_history': [[] for _ in range(gen + 1)],
                        'population': population
                    })(),
                    model
                )
                
                if health_metrics.status == HealthStatus.CRITICAL:
                    logger.critical("System health critical, stopping evolution")
                    break
                
                # Evaluate fitness with error handling
                fitness_scores = []
                successful_evaluations = 0
                
                for i, topology in enumerate(population):
                    try:
                        model.set_routing_topology(topology)
                        
                        total_loss = 0.0
                        total_samples = 0
                        expert_usage = np.zeros(config['num_experts'])
                        
                        # Evaluate on subset of data
                        for batch_idx, (inputs, targets) in enumerate(data_batches[:3]):
                            try:
                                outputs, aux_info = model.forward(inputs)
                                
                                # Compute loss with error handling
                                loss = np.mean((outputs - targets) ** 2)
                                if np.isnan(loss) or np.isinf(loss):
                                    logger.warning(f"Invalid loss for topology {i}")
                                    loss = 1.0  # Penalty for invalid loss
                                
                                total_loss += loss * inputs.shape[0]
                                total_samples += inputs.shape[0]
                                expert_usage += aux_info['expert_usage']
                                
                            except Exception as e:
                                logger.warning(f"Evaluation failed for topology {i}, batch {batch_idx}: {e}")
                                continue
                        
                        # Compute fitness
                        avg_loss = total_loss / max(total_samples, 1)
                        sparsity = topology.compute_sparsity()
                        
                        # Load balance
                        load_balance = 0.0
                        if np.sum(expert_usage) > 0:
                            usage_probs = expert_usage / np.sum(expert_usage)
                            entropy = -np.sum(usage_probs * np.log(usage_probs + 1e-8))
                            max_entropy = np.log(config['num_experts'])
                            load_balance = entropy / max_entropy if max_entropy > 0 else 0.0
                        
                        fitness = -avg_loss + 0.2 * sparsity + 0.3 * load_balance
                        fitness_scores.append(fitness)
                        
                        # Track best
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_topology = topology
                            logger.info(f"New best fitness: {fitness:.4f}")
                        
                        successful_evaluations += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to evaluate topology {i}: {e}")
                        fitness_scores.append(float('-inf'))
                        continue
                
                if successful_evaluations == 0:
                    logger.critical("No successful fitness evaluations, stopping")
                    break
                
                logger.info(f"Successful evaluations: {successful_evaluations}/{len(population)}")
                
                # Evolution statistics
                valid_scores = [s for s in fitness_scores if s != float('-inf')]
                stats = {
                    'generation': gen,
                    'best_fitness': max(valid_scores) if valid_scores else float('-inf'),
                    'avg_fitness': np.mean(valid_scores) if valid_scores else 0.0,
                    'worst_fitness': min(valid_scores) if valid_scores else float('-inf'),
                    'successful_evaluations': successful_evaluations,
                    'health_status': health_metrics.status.value
                }
                evolution_stats.append(stats)
                
                logger.info(f"Gen {gen}: Best={stats['best_fitness']:.4f}, Avg={stats['avg_fitness']:.4f}")
                
                # Create next generation with error handling
                new_population = []
                
                # Elitism
                if valid_scores:
                    best_indices = sorted(range(len(fitness_scores)), 
                                        key=lambda i: fitness_scores[i] if fitness_scores[i] != float('-inf') else float('-inf'))[-2:]
                    for idx in best_indices:
                        if fitness_scores[idx] != float('-inf'):
                            new_population.append(population[idx])
                
                # Generate offspring
                attempts = 0
                max_attempts = config['population_size'] * 3
                
                while len(new_population) < config['population_size'] and attempts < max_attempts:
                    try:
                        # Tournament selection
                        valid_indices = [i for i, score in enumerate(fitness_scores) if score != float('-inf')]
                        if len(valid_indices) < 2:
                            break
                        
                        parent1_idx = max(random.sample(valid_indices, min(3, len(valid_indices))),
                                        key=lambda i: fitness_scores[i])
                        parent2_idx = max(random.sample(valid_indices, min(3, len(valid_indices))),
                                        key=lambda i: fitness_scores[i])
                        
                        parent1 = population[parent1_idx]
                        parent2 = population[parent2_idx]
                        
                        # Crossover
                        child = parent1.crossover(parent2)
                        if child is None:
                            attempts += 1
                            continue
                        
                        # Mutation
                        child.mutate(0.15)
                        
                        # Validate child
                        if child.validate_integrity():
                            new_population.append(child)
                        else:
                            logger.warning("Generated invalid child, skipping")
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate offspring: {e}")
                    
                    attempts += 1
                
                # Fill remaining slots if needed
                while len(new_population) < config['population_size']:
                    try:
                        new_topology = RobustTopology(
                            num_tokens=config['num_tokens'],
                            num_experts=config['num_experts'],
                            sparsity=random.uniform(0.1, 0.4),
                            validator=validator
                        )
                        if new_topology.validate_integrity():
                            new_population.append(new_topology)
                    except Exception as e:
                        logger.error(f"Failed to create replacement topology: {e}")
                        break
                
                population = new_population
                generation = gen + 1
                
                logger.info(f"Generation {gen} complete, population size: {len(population)}")
                
            except Exception as e:
                logger.error(f"Generation {gen} failed: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                break
        
        end_time = time.time()
        
        logger.info(f"Evolution completed in {end_time - start_time:.2f} seconds")
        
        # Final testing
        if best_topology:
            try:
                model.set_routing_topology(best_topology)
                test_input = np.random.randn(2, config['seq_len'], config['input_dim'])
                output, aux_info = model.forward(test_input)
                
                logger.info(f"Final test successful:")
                logger.info(f"  Best topology sparsity: {best_topology.compute_sparsity():.3f}")
                logger.info(f"  Expert usage: {aux_info['expert_usage']}")
                logger.info(f"  Output shape: {output.shape}")
                
            except Exception as e:
                logger.error(f"Final test failed: {e}")
        
        # Comprehensive results
        results = {
            'config': config,
            'evolution_stats': evolution_stats,
            'final_fitness': float(best_fitness),
            'total_generations': generation,
            'computation_time': end_time - start_time,
            'best_topology_sparsity': float(best_topology.compute_sparsity()) if best_topology else None,
            'model_performance': model.get_performance_stats(),
            'health_report': health_monitor.get_health_report(),
            'success': best_topology is not None
        }
        
        # Save results
        results_path = Path("evolution_results/robust_demo_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("🛡️  GENERATION 2 ROBUST DEMO COMPLETE!")
        print("="*70)
        print(f"✅ Evolution completed with comprehensive error handling")
        print(f"📊 Final fitness: {best_fitness:.4f}")
        print(f"🧬 Generations: {generation}")
        print(f"⚡ Runtime: {end_time - start_time:.1f}s")
        print(f"🏥 Health status: {health_monitor.get_health_report()['current_status']}")
        if best_topology:
            print(f"🕸️  Best topology sparsity: {best_topology.compute_sparsity():.3f}")
        print(f"🔧 Model errors: {model.error_count}")
        print(f"💾 Results: {results_path}")
        print("="*70)
        
        return results
        
    except Exception as e:
        logger.critical(f"Demo failed critically: {e}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    try:
        results = run_robust_demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)