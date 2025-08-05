"""
Comprehensive metrics collection and performance profiling.

This module provides detailed metrics collection for evolution progress,
model performance, resource usage, and system behavior.
"""

import time
import psutil
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import numpy as np


@dataclass
class MetricValue:
    """Represents a single metric measurement."""
    value: Union[float, int, str]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific operation."""
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    gpu_utilization: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """
    Centralized metrics collection system.
    
    Collects and stores metrics for evolution progress, model performance,
    resource usage, and custom application metrics.
    """
    
    def __init__(
        self,
        max_history: int = 10000,
        flush_interval: int = 60,
        export_path: Optional[str] = None
    ):
        self.max_history = max_history
        self.flush_interval = flush_interval
        self.export_path = Path(export_path) if export_path else None
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background flush
        self._last_flush = time.time()
        self._start_background_flush()
        
        self.logger = logging.getLogger(__name__)
    
    def record_metric(
        self,
        name: str,
        value: Union[float, int, str],
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric value with optional tags."""
        with self.lock:
            metric = MetricValue(
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self.lock:
            self.counters[name] += value
            self.record_metric(f"{name}_total", self.counters[name], tags)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric to a specific value."""
        with self.lock:
            self.gauges[name] = value
            self.record_metric(name, value, tags)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a value for histogram metrics."""
        with self.lock:
            self.histograms[name].append(value)
            if len(self.histograms[name]) > self.max_history:
                self.histograms[name] = self.histograms[name][-self.max_history:]
            
            # Also record as regular metric
            self.record_metric(name, value, tags)
    
    def get_metric_history(self, name: str, limit: Optional[int] = None) -> List[MetricValue]:
        """Get historical values for a metric."""
        with self.lock:
            history = list(self.metrics[name])
            if limit:
                history = history[-limit:]
            return history
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get statistical summary of histogram data."""
        with self.lock:
            values = self.histograms.get(name, [])
            if not values:
                return {}
            
            values_array = np.array(values)
            return {
                'count': len(values),
                'mean': float(np.mean(values_array)),
                'median': float(np.median(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'p95': float(np.percentile(values_array, 95)),
                'p99': float(np.percentile(values_array, 99))
            }
    
    def get_evolution_metrics(self) -> Dict[str, Any]:
        """Get comprehensive evolution metrics."""
        with self.lock:
            fitness_stats = self.get_histogram_stats('evolution.fitness')
            generation_stats = self.get_histogram_stats('evolution.generation_time')
            
            return {
                'fitness_stats': fitness_stats,
                'generation_stats': generation_stats,
                'total_generations': self.counters.get('evolution.generations_completed', 0),
                'total_evaluations': self.counters.get('evolution.fitness_evaluations', 0),
                'convergence_rate': self.gauges.get('evolution.convergence_rate', 0.0),
                'diversity_score': self.gauges.get('evolution.diversity_score', 0.0),
                'best_fitness_ever': self.gauges.get('evolution.best_fitness', float('-inf'))
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        with self.lock:
            return {
                'latency_stats': self.get_histogram_stats('performance.latency_ms'),
                'throughput_stats': self.get_histogram_stats('performance.throughput'),
                'memory_stats': self.get_histogram_stats('performance.memory_mb'),
                'cpu_stats': self.get_histogram_stats('performance.cpu_percent'),
                'gpu_stats': self.get_histogram_stats('performance.gpu_utilization'),
                'current_memory_mb': self.gauges.get('performance.memory_mb', 0),
                'current_cpu_percent': self.gauges.get('performance.cpu_percent', 0)
            }
    
    def export_metrics(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Export all metrics to JSON format."""
        export_path = filepath or self.export_path
        
        with self.lock:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'evolution_metrics': self.get_evolution_metrics(),
                'performance_metrics': self.get_performance_metrics(),
                'metric_counts': {name: len(history) for name, history in self.metrics.items()}
            }
        
        if export_path:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported metrics to {export_path}")
        
        return export_data
    
    def _start_background_flush(self):
        """Start background thread for periodic metric flushing."""
        def flush_worker():
            while True:
                try:
                    time.sleep(self.flush_interval)
                    current_time = time.time()
                    
                    if current_time - self._last_flush >= self.flush_interval:
                        self.export_metrics()
                        self._last_flush = current_time
                        
                except Exception as e:
                    self.logger.error(f"Error in metrics flush worker: {e}")
        
        flush_thread = threading.Thread(target=flush_worker, daemon=True)
        flush_thread.start()
    
    def clear_metrics(self):
        """Clear all stored metrics."""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()


class PerformanceProfiler:
    """
    Context manager for profiling performance of operations.
    
    Automatically measures duration, memory usage, and CPU utilization
    for wrapped operations and records metrics.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        operation_name: str,
        tags: Optional[Dict[str, str]] = None,
        measure_gpu: bool = False
    ):
        self.metrics_collector = metrics_collector
        self.operation_name = operation_name
        self.tags = tags or {}
        self.measure_gpu = measure_gpu
        
        # Performance tracking
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.start_cpu: Optional[float] = None
        self.process = psutil.Process()
    
    def __enter__(self) -> 'PerformanceProfiler':
        """Start performance measurement."""
        self.start_time = time.time()
        
        # Memory measurement
        memory_info = self.process.memory_info()
        self.start_memory = memory_info.rss / 1024 / 1024  # Convert to MB
        
        # CPU measurement
        self.start_cpu = self.process.cpu_percent()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete performance measurement and record metrics."""
        if self.start_time is None:
            return
        
        # Calculate duration
        duration_ms = (time.time() - self.start_time) * 1000
        
        # Calculate memory usage
        memory_info = self.process.memory_info()
        end_memory = memory_info.rss / 1024 / 1024
        memory_usage = end_memory - self.start_memory
        
        # Calculate CPU usage
        end_cpu = self.process.cpu_percent()
        cpu_usage = max(0, end_cpu - self.start_cpu)  # Avoid negative values
        
        # GPU utilization (if available and requested)
        gpu_utilization = None
        if self.measure_gpu:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_utilization = gpus[0].load * 100
            except ImportError:
                pass
        
        # Record metrics
        operation_tags = {**self.tags, 'operation': self.operation_name}
        
        self.metrics_collector.record_histogram(
            'performance.latency_ms',
            duration_ms,
            operation_tags
        )
        
        self.metrics_collector.record_histogram(
            'performance.memory_mb',
            memory_usage,
            operation_tags
        )
        
        self.metrics_collector.record_histogram(
            'performance.cpu_percent',
            cpu_usage,
            operation_tags
        )
        
        if gpu_utilization is not None:
            self.metrics_collector.record_histogram(
                'performance.gpu_utilization',
                gpu_utilization,
                operation_tags
            )
        
        # Record operation completion
        self.metrics_collector.increment_counter(
            f'operations.{self.operation_name}.completed',
            tags=operation_tags
        )
        
        # Record operation-specific metrics
        self.metrics_collector.record_metric(
            f'operations.{self.operation_name}.duration_ms',
            duration_ms,
            operation_tags
        )
    
    def add_custom_metric(self, name: str, value: Union[float, int]):
        """Add a custom metric during profiling."""
        tags = {**self.tags, 'operation': self.operation_name}
        self.metrics_collector.record_metric(name, value, tags)


def profile_function(
    metrics_collector: MetricsCollector,
    operation_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    measure_gpu: bool = False
):
    """
    Decorator for automatically profiling function performance.
    
    Usage:
        @profile_function(metrics_collector, 'my_operation')
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            
            with PerformanceProfiler(
                metrics_collector=metrics_collector,
                operation_name=op_name,
                tags=tags,
                measure_gpu=measure_gpu
            ):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class ResourceMonitor:
    """Monitor system resource usage continuously."""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        interval: float = 5.0
    ):
        self.metrics_collector = metrics_collector
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start continuous resource monitoring."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self.logger.info("Started resource monitoring")
    
    def stop(self):
        """Stop resource monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.logger.info("Stopped resource monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # System-wide metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Record system metrics
                self.metrics_collector.set_gauge('system.cpu_percent', cpu_percent)
                self.metrics_collector.set_gauge('system.memory_percent', memory.percent)
                self.metrics_collector.set_gauge('system.memory_available_mb', memory.available / 1024 / 1024)
                self.metrics_collector.set_gauge('system.disk_percent', disk.percent)
                
                # Process-specific metrics
                process = psutil.Process()
                process_memory = process.memory_info().rss / 1024 / 1024
                process_cpu = process.cpu_percent()
                
                self.metrics_collector.set_gauge('process.memory_mb', process_memory)
                self.metrics_collector.set_gauge('process.cpu_percent', process_cpu)
                
                # GPU metrics (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        self.metrics_collector.set_gauge(
                            f'gpu.{i}.utilization',
                            gpu.load * 100
                        )
                        self.metrics_collector.set_gauge(
                            f'gpu.{i}.memory_percent',
                            (gpu.memoryUsed / gpu.memoryTotal) * 100
                        )
                        self.metrics_collector.set_gauge(
                            f'gpu.{i}.temperature',
                            gpu.temperature
                        )
                except ImportError:
                    pass
                
                time.sleep(self.interval)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.interval)