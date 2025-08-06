"""
Performance profiling and bottleneck analysis for MoE models.

This module provides comprehensive profiling tools to identify
performance bottlenecks and optimization opportunities.
"""

import torch
import torch.profiler
import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass
import json

from ..experts.pool import ExpertPool
from ..experts.slimmable import SlimmableMoE
from ..routing.topology import TopologyGenome
from ..utils.logging import get_logger


@dataclass
class ProfileMetrics:
    """Container for profiling metrics."""
    operation_name: str
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    memory_usage_mb: float
    gpu_memory_mb: float = 0.0
    flops: int = 0
    call_count: int = 1
    cpu_percent: float = 0.0


@dataclass
class BottleneckReport:
    """Report of identified performance bottlenecks."""
    operation: str
    severity: str  # "critical", "high", "medium", "low"
    time_percentage: float
    memory_percentage: float
    impact_score: float
    recommendations: List[str]
    optimization_potential: float  # 0.0 to 1.0


class ModelProfiler:
    """
    Comprehensive model profiler for MoE systems.
    
    Profiles CPU/GPU usage, memory consumption, operation timing,
    and identifies optimization opportunities.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        sample_inputs: torch.Tensor,
        warmup_steps: int = 10,
        profile_steps: int = 100,
        enable_gpu_profiling: bool = True
    ):
        self.model = model
        self.sample_inputs = sample_inputs
        self.warmup_steps = warmup_steps
        self.profile_steps = profile_steps
        self.enable_gpu_profiling = enable_gpu_profiling and torch.cuda.is_available()
        
        self.logger = get_logger(__name__)
        
        # Profiling results
        self.operation_metrics: Dict[str, ProfileMetrics] = {}
        self.memory_timeline: List[Dict[str, float]] = []
        self.timing_breakdown: Dict[str, List[float]] = defaultdict(list)
        
        # Profiler state
        self.is_profiling = False
        self.current_profile_step = 0
    
    def profile_inference(self, detailed: bool = True) -> Dict[str, Any]:
        """
        Profile model inference performance.
        
        Args:
            detailed: Whether to collect detailed operation-level metrics
            
        Returns:
            Dictionary containing profiling results
        """
        self.logger.info("Starting inference profiling...")
        
        # Warmup
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                _ = self.model(self.sample_inputs)
        
        if self.enable_gpu_profiling:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Profile with PyTorch profiler
        if detailed and hasattr(torch.profiler, 'profile'):
            return self._profile_with_pytorch_profiler()
        else:
            return self._profile_basic_timing()
    
    def _profile_with_pytorch_profiler(self) -> Dict[str, Any]:
        """Profile using PyTorch's built-in profiler."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.enable_gpu_profiling:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True
        ) as prof:
            
            with torch.no_grad():
                for step in range(self.profile_steps):
                    output = self.model(self.sample_inputs)
                    
                    # Record memory usage
                    if step % 10 == 0:
                        self._record_memory_snapshot()
        
        # Analyze profiler results
        return self._analyze_profiler_results(prof)
    
    def _profile_basic_timing(self) -> Dict[str, Any]:
        """Basic timing-based profiling."""
        timings = []
        memory_usage = []
        
        with torch.no_grad():
            for step in range(self.profile_steps):
                # Memory before
                if self.enable_gpu_profiling:
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated() / 1024 / 1024
                else:
                    mem_before = 0.0
                
                # Time inference
                start_time = time.perf_counter()
                
                if self.enable_gpu_profiling:
                    torch.cuda.synchronize()
                
                output = self.model(self.sample_inputs)
                
                if self.enable_gpu_profiling:
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                # Memory after
                if self.enable_gpu_profiling:
                    mem_after = torch.cuda.memory_allocated() / 1024 / 1024
                else:
                    mem_after = 0.0
                
                inference_time = (end_time - start_time) * 1000  # ms
                memory_used = mem_after - mem_before
                
                timings.append(inference_time)
                memory_usage.append(memory_used)
        
        # Calculate statistics
        avg_time = np.mean(timings)
        std_time = np.std(timings)
        min_time = np.min(timings)
        max_time = np.max(timings)
        
        avg_memory = np.mean(memory_usage)
        
        # Calculate throughput
        batch_size = self.sample_inputs.shape[0]
        throughput = (batch_size * 1000) / avg_time  # samples/sec
        
        return {
            'avg_latency_ms': avg_time,
            'std_latency_ms': std_time,
            'min_latency_ms': min_time,
            'max_latency_ms': max_time,
            'throughput_samples_per_sec': throughput,
            'avg_memory_mb': avg_memory,
            'p95_latency_ms': np.percentile(timings, 95),
            'p99_latency_ms': np.percentile(timings, 99),
            'total_profile_time_s': sum(timings) / 1000,
            'profile_steps': self.profile_steps
        }
    
    def _analyze_profiler_results(self, prof: torch.profiler.profile) -> Dict[str, Any]:
        """Analyze PyTorch profiler results."""
        # Get profiler table
        cpu_table = prof.key_averages().table(sort_by="cpu_time_total")
        
        if self.enable_gpu_profiling:
            gpu_table = prof.key_averages().table(sort_by="cuda_time_total")
        
        # Parse operation metrics
        operations = {}
        
        for event in prof.key_averages():
            if event.count > 0:
                op_name = event.key
                
                metrics = ProfileMetrics(
                    operation_name=op_name,
                    avg_time_ms=event.cpu_time_total / event.count / 1000,
                    min_time_ms=0.0,  # Not available from profiler
                    max_time_ms=0.0,  # Not available from profiler
                    std_time_ms=0.0,  # Not available from profiler
                    memory_usage_mb=event.cpu_memory_usage / 1024 / 1024 if event.cpu_memory_usage else 0.0,
                    gpu_memory_mb=event.cuda_memory_usage / 1024 / 1024 if hasattr(event, 'cuda_memory_usage') and event.cuda_memory_usage else 0.0,
                    flops=event.flops if hasattr(event, 'flops') and event.flops else 0,
                    call_count=event.count,
                    cpu_percent=0.0
                )
                
                operations[op_name] = metrics
        
        # Calculate total times
        total_cpu_time = sum(event.cpu_time_total for event in prof.key_averages())
        total_gpu_time = sum(getattr(event, 'cuda_time_total', 0) for event in prof.key_averages()) if self.enable_gpu_profiling else 0
        
        # Calculate batch metrics
        batch_size = self.sample_inputs.shape[0]
        avg_latency = total_cpu_time / self.profile_steps / 1000  # ms
        throughput = (batch_size * 1000) / avg_latency
        
        return {
            'operations': operations,
            'total_cpu_time_ms': total_cpu_time / 1000,
            'total_gpu_time_ms': total_gpu_time / 1000 if self.enable_gpu_profiling else 0,
            'avg_latency_ms': avg_latency,
            'throughput_samples_per_sec': throughput,
            'profile_steps': self.profile_steps,
            'profiler_tables': {
                'cpu': cpu_table,
                'gpu': gpu_table if self.enable_gpu_profiling else None
            }
        }
    
    def _record_memory_snapshot(self):
        """Record current memory usage."""
        memory_info = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system_memory_percent'] = system_memory.percent
        memory_info['system_memory_mb'] = system_memory.used / 1024 / 1024
        
        # GPU memory
        if self.enable_gpu_profiling:
            memory_info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        memory_info['timestamp'] = time.time()
        self.memory_timeline.append(memory_info)
    
    def profile_expert_utilization(self, expert_pool: ExpertPool, num_samples: int = 1000) -> Dict[str, Any]:
        """Profile expert utilization patterns."""
        if not isinstance(self.model, (SlimmableMoE,)) and not hasattr(self.model, 'expert_pool'):
            self.logger.warning("Model doesn't appear to be MoE - skipping expert utilization profiling")
            return {}
        
        expert_usage = defaultdict(int)
        routing_patterns = []
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_samples):
                # Generate random input
                sample = torch.randn_like(self.sample_inputs[:1])  # Single sample
                
                # Forward pass with usage tracking
                if hasattr(self.model, 'expert_pool'):
                    # Reset usage counters
                    for expert in self.model.expert_pool.experts:
                        expert.reset_usage()
                    
                    output = self.model(sample)
                    
                    # Record usage
                    for i, expert in enumerate(self.model.expert_pool.experts):
                        expert_usage[i] += expert.usage_count
                
                elif isinstance(self.model, SlimmableMoE):
                    # For slimmable MoE
                    output = self.model(sample)
                    # Usage tracking would need to be implemented in forward pass
        
        # Calculate utilization statistics
        total_usage = sum(expert_usage.values())
        utilization_balance = np.std(list(expert_usage.values())) / np.mean(list(expert_usage.values())) if expert_usage else 0
        
        # Expert efficiency
        expert_efficiency = {}
        for expert_id, usage in expert_usage.items():
            efficiency = usage / total_usage if total_usage > 0 else 0
            expert_efficiency[expert_id] = efficiency
        
        return {
            'expert_usage': dict(expert_usage),
            'expert_efficiency': expert_efficiency,
            'utilization_balance': utilization_balance,
            'total_expert_calls': total_usage,
            'avg_calls_per_expert': total_usage / len(expert_usage) if expert_usage else 0,
            'active_experts': len([u for u in expert_usage.values() if u > 0]),
            'inactive_experts': len([u for u in expert_usage.values() if u == 0])
        }
    
    def profile_scaling_behavior(self, batch_sizes: List[int]) -> Dict[str, Any]:
        """Profile model behavior across different batch sizes."""
        scaling_results = {}
        
        original_shape = self.sample_inputs.shape
        
        for batch_size in batch_sizes:
            self.logger.info(f"Profiling batch size {batch_size}")
            
            # Create input with target batch size
            test_input = torch.randn(batch_size, *original_shape[1:], device=self.sample_inputs.device)
            
            # Profile this batch size
            old_inputs = self.sample_inputs
            self.sample_inputs = test_input
            
            results = self._profile_basic_timing()
            
            # Calculate per-sample metrics
            results['latency_per_sample_ms'] = results['avg_latency_ms'] / batch_size
            results['memory_per_sample_mb'] = results['avg_memory_mb'] / batch_size
            results['batch_size'] = batch_size
            
            scaling_results[batch_size] = results
            
            # Restore original inputs
            self.sample_inputs = old_inputs
        
        # Analyze scaling efficiency
        base_batch_size = min(batch_sizes)
        base_latency_per_sample = scaling_results[base_batch_size]['latency_per_sample_ms']
        
        scaling_efficiency = {}
        for batch_size in batch_sizes:
            current_latency_per_sample = scaling_results[batch_size]['latency_per_sample_ms']
            efficiency = base_latency_per_sample / current_latency_per_sample
            scaling_efficiency[batch_size] = efficiency
        
        return {
            'batch_results': scaling_results,
            'scaling_efficiency': scaling_efficiency,
            'optimal_batch_size': max(scaling_efficiency.items(), key=lambda x: x[1])[0],
            'throughput_scaling': {
                bs: scaling_results[bs]['throughput_samples_per_sec'] 
                for bs in batch_sizes
            }
        }
    
    def export_profile_report(self, output_path: Path, format: str = "json") -> None:
        """Export comprehensive profiling report."""
        # Run comprehensive profiling
        inference_profile = self.profile_inference(detailed=True)
        
        # Expert utilization if applicable  
        expert_profile = {}
        if hasattr(self.model, 'expert_pool'):
            expert_profile = self.profile_expert_utilization(self.model.expert_pool)
        
        # Scaling behavior
        batch_sizes = [1, 4, 8, 16, 32] if self.sample_inputs.shape[0] >= 8 else [1, 2, 4]
        scaling_profile = self.profile_scaling_behavior(batch_sizes)
        
        # Compile full report
        report = {
            'model_info': {
                'model_type': type(self.model).__name__,
                'parameter_count': sum(p.numel() for p in self.model.parameters()),
                'input_shape': list(self.sample_inputs.shape),
                'device': str(self.sample_inputs.device)
            },
            'inference_profile': inference_profile,
            'expert_profile': expert_profile,
            'scaling_profile': scaling_profile,
            'memory_timeline': self.memory_timeline,
            'profiling_config': {
                'warmup_steps': self.warmup_steps,
                'profile_steps': self.profile_steps,
                'gpu_profiling_enabled': self.enable_gpu_profiling
            },
            'timestamp': time.time()
        }
        
        # Export in requested format
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported profiling report to {output_path}")


class BottleneckAnalyzer:
    """
    Analyzes profiling results to identify performance bottlenecks
    and suggest optimization strategies.
    """
    
    def __init__(self, profile_results: Dict[str, Any]):
        self.profile_results = profile_results
        self.logger = get_logger(__name__)
        
        # Bottleneck thresholds
        self.critical_time_threshold = 0.3  # 30% of total time
        self.high_time_threshold = 0.15     # 15% of total time
        self.critical_memory_threshold = 0.5  # 50% of total memory
        
    def analyze_bottlenecks(self) -> List[BottleneckReport]:
        """
        Analyze profiling results and identify bottlenecks.
        
        Returns:
            List of bottleneck reports sorted by severity
        """
        bottlenecks = []
        
        # Analyze operation timing bottlenecks
        if 'operations' in self.profile_results:
            bottlenecks.extend(self._analyze_timing_bottlenecks())
        
        # Analyze memory bottlenecks
        if 'memory_timeline' in self.profile_results:
            bottlenecks.extend(self._analyze_memory_bottlenecks())
        
        # Analyze expert utilization bottlenecks
        if 'expert_profile' in self.profile_results:
            bottlenecks.extend(self._analyze_expert_bottlenecks())
        
        # Analyze scaling bottlenecks
        if 'scaling_profile' in self.profile_results:
            bottlenecks.extend(self._analyze_scaling_bottlenecks())
        
        # Sort by impact score
        bottlenecks.sort(key=lambda x: x.impact_score, reverse=True)
        
        return bottlenecks
    
    def _analyze_timing_bottlenecks(self) -> List[BottleneckReport]:
        """Analyze timing-based bottlenecks."""
        bottlenecks = []
        operations = self.profile_results['operations']
        
        # Calculate total time
        total_time = sum(op.avg_time_ms * op.call_count for op in operations.values())
        
        for op_name, metrics in operations.items():
            op_total_time = metrics.avg_time_ms * metrics.call_count
            time_percentage = op_total_time / total_time if total_time > 0 else 0
            
            # Determine severity
            if time_percentage >= self.critical_time_threshold:
                severity = "critical"
            elif time_percentage >= self.high_time_threshold:
                severity = "high"
            elif time_percentage >= 0.05:  # 5%
                severity = "medium"
            else:
                continue  # Skip low-impact operations
            
            # Generate recommendations
            recommendations = self._get_timing_recommendations(op_name, metrics)
            
            # Calculate optimization potential
            optimization_potential = min(time_percentage, 0.8)  # Cap at 80%
            
            bottleneck = BottleneckReport(
                operation=op_name,
                severity=severity,
                time_percentage=time_percentage * 100,
                memory_percentage=0.0,  # Not calculated here
                impact_score=time_percentage * (1 + metrics.call_count / 1000),
                recommendations=recommendations,
                optimization_potential=optimization_potential
            )
            
            bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _analyze_memory_bottlenecks(self) -> List[BottleneckReport]:
        """Analyze memory-based bottlenecks."""
        bottlenecks = []
        
        if not self.profile_results.get('memory_timeline'):
            return bottlenecks
        
        # Find peak memory usage
        peak_memory = 0
        for snapshot in self.profile_results['memory_timeline']:
            if 'gpu_memory_allocated_mb' in snapshot:
                peak_memory = max(peak_memory, snapshot['gpu_memory_allocated_mb'])
            if 'system_memory_mb' in snapshot:
                peak_memory = max(peak_memory, snapshot['system_memory_mb'])
        
        # Check if memory usage is concerning
        if peak_memory > 1000:  # 1GB threshold
            severity = "high" if peak_memory > 4000 else "medium"
            
            recommendations = [
                "Consider model quantization to reduce memory usage",
                "Use gradient checkpointing during training",
                "Implement dynamic batching for inference",
                "Consider model pruning to remove unused parameters"
            ]
            
            bottleneck = BottleneckReport(
                operation="memory_usage",
                severity=severity,
                time_percentage=0.0,
                memory_percentage=100.0,  # Simplified
                impact_score=peak_memory / 1000,  # GB as impact score
                recommendations=recommendations,
                optimization_potential=0.3  # 30% memory reduction possible
            )
            
            bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _analyze_expert_bottlenecks(self) -> List[BottleneckReport]:
        """Analyze expert utilization bottlenecks."""
        bottlenecks = []
        expert_profile = self.profile_results.get('expert_profile', {})
        
        if not expert_profile:
            return bottlenecks
        
        # Check for imbalanced expert utilization
        utilization_balance = expert_profile.get('utilization_balance', 0)
        inactive_experts = expert_profile.get('inactive_experts', 0)
        total_experts = len(expert_profile.get('expert_usage', {}))
        
        if utilization_balance > 2.0:  # High imbalance
            recommendations = [
                "Improve routing algorithm to balance expert load",
                "Consider expert specialization training",
                "Implement load balancing loss during training",
                "Analyze routing patterns for bias"
            ]
            
            bottleneck = BottleneckReport(
                operation="expert_load_balancing",
                severity="high",
                time_percentage=0.0,
                memory_percentage=0.0,
                impact_score=utilization_balance,
                recommendations=recommendations,
                optimization_potential=0.4
            )
            
            bottlenecks.append(bottleneck)
        
        if inactive_experts > total_experts * 0.2:  # >20% inactive
            recommendations = [
                "Prune inactive experts to reduce model size",
                "Retrain routing to utilize all experts",
                "Consider reducing total number of experts",
                "Implement expert dropout during training"
            ]
            
            bottleneck = BottleneckReport(
                operation="unused_experts",
                severity="medium",
                time_percentage=0.0,
                memory_percentage=inactive_experts / total_experts * 100,
                impact_score=inactive_experts / total_experts,
                recommendations=recommendations,
                optimization_potential=inactive_experts / total_experts
            )
            
            bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _analyze_scaling_bottlenecks(self) -> List[BottleneckReport]:
        """Analyze scaling behavior bottlenecks."""
        bottlenecks = []
        scaling_profile = self.profile_results.get('scaling_profile', {})
        
        if not scaling_profile:
            return bottlenecks
        
        scaling_efficiency = scaling_profile.get('scaling_efficiency', {})
        
        if scaling_efficiency:
            batch_sizes = sorted(scaling_efficiency.keys())
            efficiencies = [scaling_efficiency[bs] for bs in batch_sizes]
            
            # Check if efficiency drops significantly with larger batches
            if len(efficiencies) > 1:
                efficiency_drop = max(efficiencies) - min(efficiencies)
                
                if efficiency_drop > 0.3:  # 30% efficiency drop
                    recommendations = [
                        "Optimize memory access patterns for larger batches",
                        "Consider batch size limits for optimal throughput",
                        "Implement gradient accumulation instead of large batches",
                        "Profile GPU utilization at different batch sizes"
                    ]
                    
                    bottleneck = BottleneckReport(
                        operation="batch_scaling",
                        severity="medium",
                        time_percentage=0.0,
                        memory_percentage=0.0,
                        impact_score=efficiency_drop,
                        recommendations=recommendations,
                        optimization_potential=efficiency_drop
                    )
                    
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _get_timing_recommendations(self, op_name: str, metrics: ProfileMetrics) -> List[str]:
        """Get optimization recommendations for timing bottlenecks."""
        recommendations = []
        
        # Operation-specific recommendations
        if "matmul" in op_name.lower() or "linear" in op_name.lower():
            recommendations.extend([
                "Consider using mixed precision (FP16) for matrix operations",
                "Optimize BLAS library (cuBLAS/MKL) configuration",
                "Implement matrix operation fusion where possible"
            ])
        
        elif "attention" in op_name.lower():
            recommendations.extend([
                "Use Flash Attention or similar optimized attention implementations",
                "Consider attention approximation techniques",
                "Implement key-value caching for inference"
            ])
        
        elif "softmax" in op_name.lower():
            recommendations.extend([
                "Use fused softmax operations",
                "Consider approximate softmax for large vocabularies"
            ])
        
        elif "conv" in op_name.lower():
            recommendations.extend([
                "Optimize convolution algorithms (Winograd, FFT)",
                "Use grouped or depthwise convolutions where applicable"
            ])
        
        # General recommendations based on metrics
        if metrics.call_count > 10000:
            recommendations.append("High call count - consider operation fusion")
        
        if metrics.avg_time_ms > 10:
            recommendations.append("Long execution time - consider parallelization")
        
        if not recommendations:
            recommendations = ["Profile operation in detail for specific optimizations"]
        
        return recommendations
    
    def generate_optimization_plan(self, bottlenecks: List[BottleneckReport]) -> Dict[str, Any]:
        """Generate comprehensive optimization plan."""
        plan = {
            'immediate_actions': [],
            'short_term_optimizations': [],
            'long_term_improvements': [],
            'estimated_speedup': 0.0,
            'estimated_memory_savings': 0.0
        }
        
        total_speedup_potential = 0.0
        total_memory_savings = 0.0
        
        for bottleneck in bottlenecks:
            if bottleneck.severity == "critical":
                plan['immediate_actions'].extend(bottleneck.recommendations[:2])
                total_speedup_potential += bottleneck.optimization_potential * 0.8
            elif bottleneck.severity == "high":
                plan['short_term_optimizations'].extend(bottleneck.recommendations[:2])
                total_speedup_potential += bottleneck.optimization_potential * 0.6
            else:
                plan['long_term_improvements'].extend(bottleneck.recommendations[:1])
                total_speedup_potential += bottleneck.optimization_potential * 0.3
            
            if bottleneck.memory_percentage > 0:
                total_memory_savings += bottleneck.optimization_potential * 0.5
        
        plan['estimated_speedup'] = min(total_speedup_potential, 2.0)  # Cap at 2x
        plan['estimated_memory_savings'] = min(total_memory_savings, 0.5)  # Cap at 50%
        
        # Remove duplicates
        plan['immediate_actions'] = list(set(plan['immediate_actions']))
        plan['short_term_optimizations'] = list(set(plan['short_term_optimizations']))
        plan['long_term_improvements'] = list(set(plan['long_term_improvements']))
        
        return plan