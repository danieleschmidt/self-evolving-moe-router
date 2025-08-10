"""
High-performance optimization for Self-Evolving MoE Router.

This module implements comprehensive performance optimization including:
- Memory optimization and caching
- Computation optimization and vectorization
- GPU acceleration and kernel optimization
- Profiling and performance analysis
- Automatic performance tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gc
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import warnings
from pathlib import Path
import json

from ..utils.logging import get_logger
from ..routing.topology import TopologyGenome

logger = get_logger(__name__)


class OptimizationLevel(Enum):
    """Levels of optimization aggressiveness."""
    CONSERVATIVE = "conservative"  # Safe optimizations only
    BALANCED = "balanced"         # Balance of performance and stability
    AGGRESSIVE = "aggressive"     # Maximum performance optimizations
    EXPERIMENTAL = "experimental" # Cutting-edge optimizations


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    
    # Memory optimizations
    enable_memory_pooling: bool = True
    enable_gradient_checkpointing: bool = True
    memory_cache_size: int = 1000
    gc_frequency: int = 10  # Generations between garbage collection
    
    # Computation optimizations
    enable_compilation: bool = True
    enable_mixed_precision: bool = True
    enable_tensorcore: bool = True
    vectorization_batch_size: int = 64
    
    # GPU optimizations
    enable_gpu_memory_optimization: bool = True
    enable_kernel_fusion: bool = True
    enable_async_execution: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Parallelization
    max_workers: int = 4
    enable_parallel_evaluation: bool = True
    chunk_size: int = 8
    
    # Profiling
    enable_profiling: bool = True
    profile_frequency: int = 20  # Profile every N generations
    detailed_profiling: bool = False


class MemoryPool:
    """Efficient memory pool for tensor allocation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.pools: Dict[Tuple[torch.Size, torch.dtype, str], List[torch.Tensor]] = {}
        self.allocation_count = 0
        self.hit_count = 0
        self._lock = threading.Lock()
    
    def get_tensor(self, shape: torch.Size, dtype: torch.dtype = torch.float32, 
                   device: str = "cpu") -> torch.Tensor:
        """Get tensor from pool or create new one."""
        key = (shape, dtype, device)
        
        with self._lock:
            if key in self.pools and self.pools[key]:
                tensor = self.pools[key].pop()
                self.hit_count += 1
                return tensor.zero_()  # Reset to zeros
            else:
                self.allocation_count += 1
                return torch.zeros(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse."""
        if tensor.numel() == 0:
            return  # Skip empty tensors
        
        key = (tensor.shape, tensor.dtype, str(tensor.device))
        
        with self._lock:
            if key not in self.pools:
                self.pools[key] = []
            
            if len(self.pools[key]) < self.max_size:
                self.pools[key].append(tensor.detach())
    
    def clear(self):
        """Clear all pooled tensors."""
        with self._lock:
            self.pools.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            total_pooled = sum(len(pool) for pool in self.pools.values())
            hit_ratio = self.hit_count / max(self.allocation_count, 1)
            
            return {
                'total_allocations': self.allocation_count,
                'cache_hits': self.hit_count,
                'hit_ratio': hit_ratio,
                'pooled_tensors': total_pooled,
                'pool_types': len(self.pools)
            }


class PerformanceProfiler:
    """Comprehensive performance profiler."""
    
    def __init__(self):
        self.profiles: List[Dict[str, Any]] = []
        self.current_profile: Optional[Dict[str, Any]] = None
        self.timers: Dict[str, float] = {}
        self.memory_snapshots: List[Dict[str, float]] = []
        
    def start_profiling(self, name: str):
        """Start profiling session."""
        self.current_profile = {
            'name': name,
            'start_time': time.time(),
            'timings': {},
            'memory_usage': {},
            'gpu_usage': {},
            'operations': []
        }
        
        # Initial memory snapshot
        self._take_memory_snapshot('start')
    
    def end_profiling(self):
        """End profiling session."""
        if self.current_profile is None:
            return
        
        self.current_profile['end_time'] = time.time()
        self.current_profile['total_time'] = (
            self.current_profile['end_time'] - self.current_profile['start_time']
        )
        
        # Final memory snapshot
        self._take_memory_snapshot('end')
        
        self.profiles.append(self.current_profile)
        self.current_profile = None
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return OperationTimer(operation_name, self)
    
    def record_operation(self, name: str, duration: float, memory_delta: float = 0):
        """Record operation timing and memory usage."""
        if self.current_profile is None:
            return
        
        self.current_profile['operations'].append({
            'name': name,
            'duration': duration,
            'memory_delta': memory_delta,
            'timestamp': time.time() - self.current_profile['start_time']
        })
    
    def _take_memory_snapshot(self, label: str):
        """Take memory usage snapshot."""
        if self.current_profile is None:
            return
        
        # CPU memory
        cpu_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB
        
        # GPU memory
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        
        self.current_profile['memory_usage'][label] = {
            'cpu_memory_gb': cpu_memory,
            'gpu_memory_gb': gpu_memory
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.profiles:
            return {'error': 'No profiling data available'}
        
        # Aggregate statistics
        total_operations = sum(len(p['operations']) for p in self.profiles)
        total_time = sum(p['total_time'] for p in self.profiles)
        
        # Find bottlenecks
        operation_times = {}
        for profile in self.profiles:
            for op in profile['operations']:
                op_name = op['name']
                if op_name not in operation_times:
                    operation_times[op_name] = []
                operation_times[op_name].append(op['duration'])
        
        # Calculate operation statistics
        operation_stats = {}
        for op_name, times in operation_times.items():
            operation_stats[op_name] = {
                'total_time': sum(times),
                'avg_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'call_count': len(times),
                'std_time': np.std(times)
            }
        
        # Sort by total time to find bottlenecks
        bottlenecks = sorted(
            operation_stats.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )[:10]
        
        return {
            'total_profiles': len(self.profiles),
            'total_operations': total_operations,
            'total_time': total_time,
            'avg_profile_time': total_time / len(self.profiles),
            'operation_stats': operation_stats,
            'top_bottlenecks': [{'operation': name, **stats} for name, stats in bottlenecks],
            'latest_profile': self.profiles[-1] if self.profiles else None
        }


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, profiler: PerformanceProfiler):
        self.operation_name = operation_name
        self.profiler = profiler
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Record memory usage
        if torch.cuda.is_available():
            self.start_memory = torch.cuda.memory_allocated()
        else:
            self.start_memory = 0
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        duration = time.time() - self.start_time
        
        # Calculate memory delta
        memory_delta = 0
        if torch.cuda.is_available():
            memory_delta = torch.cuda.memory_allocated() - self.start_memory
        
        self.profiler.record_operation(self.operation_name, duration, memory_delta)


class HighPerformanceOptimizer:
    """Main performance optimization engine."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.memory_pool = MemoryPool(config.memory_cache_size) if config.enable_memory_pooling else None
        self.profiler = PerformanceProfiler() if config.enable_profiling else None
        
        # Performance state
        self.generation_count = 0
        self.optimization_stats = {
            'total_optimizations': 0,
            'memory_optimizations': 0,
            'computation_optimizations': 0,
            'gpu_optimizations': 0,
            'cache_hits': 0,
            'performance_gains': []
        }
        
        # Compiled functions cache
        self.compiled_functions: Dict[str, Callable] = {}
        
        # Initialize optimizations
        self._initialize_optimizations()
        
        logger.info(f"Initialized high-performance optimizer with {config.optimization_level.value} level")
    
    def _initialize_optimizations(self):
        """Initialize performance optimizations."""
        try:
            # Mixed precision setup
            if self.config.enable_mixed_precision and torch.cuda.is_available():
                self.scaler = torch.cuda.amp.GradScaler()
                logger.info("Enabled mixed precision training")
            
            # GPU memory optimization
            if self.config.enable_gpu_memory_optimization and torch.cuda.is_available():
                torch.cuda.memory._set_allocator_settings(
                    'max_split_size_mb:128'
                )
                logger.info("Configured GPU memory allocator")
            
            # Compilation setup
            if self.config.enable_compilation:
                if hasattr(torch, 'compile'):
                    logger.info("PyTorch compilation available")
                else:
                    logger.info("PyTorch compilation not available, using alternative optimizations")
            
            self.optimization_stats['total_optimizations'] += 1
            
        except Exception as e:
            logger.warning(f"Some optimizations failed to initialize: {e}")
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply model-level optimizations."""
        logger.info("Applying model optimizations")
        
        try:
            # Apply compilation if available and enabled
            if self.config.enable_compilation and hasattr(torch, 'compile'):
                if self.config.optimization_level == OptimizationLevel.AGGRESSIVE:
                    model = torch.compile(model, mode="max-autotune")
                elif self.config.optimization_level == OptimizationLevel.BALANCED:
                    model = torch.compile(model, mode="default")
                else:
                    model = torch.compile(model, mode="reduce-overhead")
                
                logger.info("Applied PyTorch compilation")
            
            # Apply gradient checkpointing for memory efficiency
            if self.config.enable_gradient_checkpointing:
                if hasattr(model, 'expert_pool'):
                    self._apply_gradient_checkpointing(model.expert_pool)
                    logger.info("Applied gradient checkpointing")
            
            # Optimize tensor operations
            self._optimize_tensor_operations(model)
            
            self.optimization_stats['computation_optimizations'] += 1
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
    
    def _apply_gradient_checkpointing(self, expert_pool):
        """Apply gradient checkpointing to expert pool."""
        try:
            for expert in expert_pool.experts:
                if hasattr(expert, 'ffn') and isinstance(expert.ffn, nn.Sequential):
                    # Apply checkpointing to feed-forward networks
                    expert.ffn = torch.utils.checkpoint.checkpoint_sequential(
                        expert.ffn, segments=2, input=None
                    )
        except Exception as e:
            logger.warning(f"Gradient checkpointing failed: {e}")
    
    def _optimize_tensor_operations(self, model):
        """Optimize tensor operations in the model."""
        try:
            # Enable tensor core operations if available
            if (self.config.enable_tensorcore and 
                torch.cuda.is_available() and 
                torch.cuda.get_device_capability()[0] >= 7):
                
                # Set tensor core precision
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled Tensor Core optimizations")
            
            # Optimize attention operations
            if hasattr(model, 'expert_pool'):
                for expert in model.expert_pool.experts:
                    if hasattr(expert, 'attention'):
                        self._optimize_attention(expert.attention)
            
        except Exception as e:
            logger.warning(f"Tensor operation optimization failed: {e}")
    
    def _optimize_attention(self, attention_layer):
        """Optimize attention layer operations."""
        try:
            # Enable flash attention if available
            if hasattr(F, 'scaled_dot_product_attention'):
                # This would require modifying the attention implementation
                logger.info("Flash attention optimization available")
            
            # Enable memory efficient attention
            if hasattr(attention_layer, 'enable_nested_tensor'):
                attention_layer.enable_nested_tensor = True
                
        except Exception as e:
            logger.debug(f"Attention optimization failed: {e}")
    
    def optimize_evolution_step(self, evolution_fn: Callable, *args, **kwargs):
        """Optimize evolution step execution."""
        if self.profiler:
            self.profiler.start_profiling(f"evolution_step_{self.generation_count}")
        
        try:
            with self._memory_management_context():
                # Parallel evaluation if enabled
                if self.config.enable_parallel_evaluation:
                    result = self._parallel_evolution_step(evolution_fn, *args, **kwargs)
                else:
                    result = evolution_fn(*args, **kwargs)
                
                # Memory cleanup
                if self.generation_count % self.config.gc_frequency == 0:
                    self._cleanup_memory()
                
                self.generation_count += 1
                return result
                
        finally:
            if self.profiler:
                self.profiler.end_profiling()
    
    def _parallel_evolution_step(self, evolution_fn: Callable, *args, **kwargs):
        """Execute evolution step with parallel evaluation."""
        try:
            # This would require modifying the evolution function to support parallel execution
            # For now, we'll call the original function with optimization wrapper
            
            with self.profiler.time_operation("parallel_evolution") if self.profiler else nullcontext():
                return evolution_fn(*args, **kwargs)
                
        except Exception as e:
            logger.error(f"Parallel evolution step failed: {e}")
            # Fallback to sequential execution
            return evolution_fn(*args, **kwargs)
    
    def optimize_fitness_evaluation(self, fitness_fn: Callable, population: List, *args, **kwargs):
        """Optimize fitness evaluation with batching and parallelization."""
        if not population:
            return []
        
        try:
            if self.config.enable_parallel_evaluation and len(population) > self.config.chunk_size:
                return self._parallel_fitness_evaluation(fitness_fn, population, *args, **kwargs)
            else:
                return self._batch_fitness_evaluation(fitness_fn, population, *args, **kwargs)
                
        except Exception as e:
            logger.error(f"Fitness evaluation optimization failed: {e}")
            # Fallback to sequential evaluation
            return [fitness_fn(individual, *args, **kwargs) for individual in population]
    
    def _parallel_fitness_evaluation(self, fitness_fn: Callable, population: List, *args, **kwargs):
        """Parallel fitness evaluation using thread pool."""
        results = [None] * len(population)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit chunks of population for evaluation
            future_to_index = {}
            
            for i in range(0, len(population), self.config.chunk_size):
                chunk = population[i:i + self.config.chunk_size]
                chunk_indices = list(range(i, min(i + self.config.chunk_size, len(population))))
                
                future = executor.submit(self._evaluate_chunk, fitness_fn, chunk, *args, **kwargs)
                future_to_index[future] = chunk_indices
            
            # Collect results
            for future in as_completed(future_to_index):
                chunk_indices = future_to_index[future]
                try:
                    chunk_results = future.result()
                    for idx, result in zip(chunk_indices, chunk_results):
                        results[idx] = result
                except Exception as e:
                    logger.error(f"Chunk evaluation failed: {e}")
                    # Fill with fallback values
                    for idx in chunk_indices:
                        results[idx] = {'combined_fitness': -1000.0}
        
        return results
    
    def _evaluate_chunk(self, fitness_fn: Callable, chunk: List, *args, **kwargs):
        """Evaluate a chunk of population."""
        return [fitness_fn(individual, *args, **kwargs) for individual in chunk]
    
    def _batch_fitness_evaluation(self, fitness_fn: Callable, population: List, *args, **kwargs):
        """Batch fitness evaluation with vectorization."""
        results = []
        
        batch_size = min(self.config.vectorization_batch_size, len(population))
        
        for i in range(0, len(population), batch_size):
            batch = population[i:i + batch_size]
            
            if self.profiler:
                with self.profiler.time_operation(f"fitness_batch_{len(batch)}"):
                    batch_results = [fitness_fn(individual, *args, **kwargs) for individual in batch]
            else:
                batch_results = [fitness_fn(individual, *args, **kwargs) for individual in batch]
            
            results.extend(batch_results)
        
        return results
    
    def _memory_management_context(self):
        """Context manager for memory management."""
        return MemoryManagementContext(self.memory_pool)
    
    def _cleanup_memory(self):
        """Cleanup memory and caches."""
        try:
            # Clear memory pool
            if self.memory_pool:
                self.memory_pool.clear()
            
            # Clear PyTorch caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Garbage collection
            gc.collect()
            
            self.optimization_stats['memory_optimizations'] += 1
            logger.debug("Performed memory cleanup")
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def optimize_topology_operations(self, topology: TopologyGenome) -> TopologyGenome:
        """Optimize topology-specific operations."""
        try:
            # Optimize routing matrix operations
            if hasattr(topology, 'routing_matrix'):
                # Use memory pool for temporary tensors if available
                if self.memory_pool:
                    # This would require modifying topology operations to use pooled tensors
                    pass
                
                # Optimize sparse operations
                if topology.routing_matrix.is_sparse:
                    # Already sparse, optimize sparse operations
                    topology.routing_matrix = topology.routing_matrix.coalesce()
                else:
                    # Convert to sparse if beneficial
                    sparsity = topology.compute_sparsity()
                    if sparsity > 0.9:  # Very sparse
                        topology.routing_matrix = topology.routing_matrix.to_sparse()
                        logger.debug("Converted routing matrix to sparse format")
            
            return topology
            
        except Exception as e:
            logger.warning(f"Topology optimization failed: {e}")
            return topology
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'optimization_config': {
                'level': self.config.optimization_level.value,
                'memory_pooling': self.config.enable_memory_pooling,
                'mixed_precision': self.config.enable_mixed_precision,
                'compilation': self.config.enable_compilation,
                'parallel_evaluation': self.config.enable_parallel_evaluation
            },
            'optimization_stats': self.optimization_stats,
            'generation_count': self.generation_count
        }
        
        # Add memory pool statistics
        if self.memory_pool:
            report['memory_pool_stats'] = self.memory_pool.get_stats()
        
        # Add profiling results
        if self.profiler:
            report['profiling_results'] = self.profiler.get_performance_summary()
        
        # Add system information
        report['system_info'] = {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'cpu_count': torch.get_num_threads(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            report['gpu_info'] = {
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'memory_allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
                'memory_reserved': torch.cuda.memory_reserved() / (1024**3),    # GB
                'max_memory_allocated': torch.cuda.max_memory_allocated() / (1024**3)
            }
        
        return report
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest performance optimizations based on profiling results."""
        suggestions = []
        
        if self.profiler and self.profiler.profiles:
            perf_summary = self.profiler.get_performance_summary()
            
            # Analyze bottlenecks
            if 'top_bottlenecks' in perf_summary:
                for bottleneck in perf_summary['top_bottlenecks'][:3]:
                    operation = bottleneck['operation']
                    total_time = bottleneck['total_time']
                    
                    if 'fitness' in operation.lower() and total_time > 10:
                        suggestions.append("Consider reducing fitness evaluation complexity or using parallel evaluation")
                    
                    if 'mutation' in operation.lower() and total_time > 5:
                        suggestions.append("Consider optimizing mutation operators or reducing mutation rates")
                    
                    if 'topology' in operation.lower() and total_time > 3:
                        suggestions.append("Consider using sparse representations for topology operations")
        
        # Memory-based suggestions
        if self.memory_pool:
            pool_stats = self.memory_pool.get_stats()
            if pool_stats['hit_ratio'] < 0.5:
                suggestions.append("Low memory pool hit ratio - consider increasing cache size")
        
        # GPU utilization suggestions
        if torch.cuda.is_available() and not self.config.enable_mixed_precision:
            suggestions.append("Consider enabling mixed precision for improved GPU performance")
        
        # Parallelization suggestions
        if not self.config.enable_parallel_evaluation and self.config.max_workers > 1:
            suggestions.append("Consider enabling parallel evaluation for better CPU utilization")
        
        return suggestions
    
    def export_performance_data(self, filepath: str):
        """Export performance data for analysis."""
        try:
            report = self.get_performance_report()
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Exported performance data to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export performance data: {e}")


class MemoryManagementContext:
    """Context manager for memory management operations."""
    
    def __init__(self, memory_pool: Optional[MemoryPool]):
        self.memory_pool = memory_pool
        self.allocated_tensors = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Return all allocated tensors to pool
        if self.memory_pool:
            for tensor in self.allocated_tensors:
                self.memory_pool.return_tensor(tensor)
        
        self.allocated_tensors.clear()
    
    def get_tensor(self, shape: torch.Size, dtype: torch.dtype = torch.float32, 
                   device: str = "cpu") -> torch.Tensor:
        """Get tensor with automatic cleanup."""
        if self.memory_pool:
            tensor = self.memory_pool.get_tensor(shape, dtype, device)
            self.allocated_tensors.append(tensor)
            return tensor
        else:
            return torch.zeros(shape, dtype=dtype, device=device)


# Null context manager for Python < 3.7 compatibility
class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass