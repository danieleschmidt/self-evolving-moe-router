"""
System monitoring utilities for Self-Evolving MoE-Router.

This module provides real-time monitoring of system resources,
performance metrics, and evolution progress.
"""

import time
import threading
import psutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class SystemMetrics:
    """System resource metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0


@dataclass
class EvolutionMetrics:
    """Evolution process metrics snapshot."""
    timestamp: float
    generation: int
    best_fitness: float
    avg_fitness: float
    population_diversity: float
    mutation_rate: float
    convergence_rate: float
    active_experts: int
    topology_sparsity: float


@dataclass
class PerformanceMetrics:
    """Model performance metrics snapshot."""
    timestamp: float
    accuracy: float
    latency_ms: float
    throughput_samples_per_sec: float
    memory_usage_mb: float
    flops_per_sample: float = 0.0
    energy_consumption_j: float = 0.0


class SystemMonitor:
    """
    Real-time system resource monitoring.
    
    Continuously monitors CPU, memory, GPU, disk, and network usage
    with configurable sampling intervals and alert thresholds.
    """
    
    def __init__(
        self,
        sample_interval: float = 1.0,
        history_size: int = 1000,
        enable_gpu_monitoring: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring and TORCH_AVAILABLE
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'gpu_memory_percent': 90.0,
            'gpu_utilization': 95.0
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=history_size)
        self.alert_callbacks = []
        
        # Baseline measurements for delta calculations
        self.baseline_disk_io = None
        self.baseline_network_io = None
        
        # Initialize GPU monitoring
        if self.enable_gpu_monitoring:
            self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring if available."""
        try:
            if torch.cuda.is_available():
                self.gpu_device_count = torch.cuda.device_count()
                self.gpu_devices = list(range(self.gpu_device_count))
            else:
                self.enable_gpu_monitoring = False
        except Exception:
            self.enable_gpu_monitoring = False
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, alert_type: str, metrics: Dict[str, Any]):
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, metrics)
            except Exception as e:
                print(f"Alert callback error: {e}")
    
    def _get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if self.baseline_disk_io is None:
            self.baseline_disk_io = disk_io
            disk_read_mb = 0.0
            disk_write_mb = 0.0
        else:
            disk_read_mb = (disk_io.read_bytes - self.baseline_disk_io.read_bytes) / 1024 / 1024
            disk_write_mb = (disk_io.write_bytes - self.baseline_disk_io.write_bytes) / 1024 / 1024
        
        # Network I/O
        network_io = psutil.net_io_counters()
        if self.baseline_network_io is None:
            self.baseline_network_io = network_io
            network_sent_mb = 0.0
            network_recv_mb = 0.0
        else:
            network_sent_mb = (network_io.bytes_sent - self.baseline_network_io.bytes_sent) / 1024 / 1024
            network_recv_mb = (network_io.bytes_recv - self.baseline_network_io.bytes_recv) / 1024 / 1024
        
        # GPU metrics
        gpu_memory_used_mb = 0.0
        gpu_memory_total_mb = 0.0
        gpu_utilization = 0.0
        
        if self.enable_gpu_monitoring and torch.cuda.is_available():
            try:
                for device_id in self.gpu_devices:
                    torch.cuda.set_device(device_id)
                    gpu_memory_used_mb += torch.cuda.memory_allocated(device_id) / 1024 / 1024
                    gpu_memory_total_mb += torch.cuda.get_device_properties(device_id).total_memory / 1024 / 1024
                
                # GPU utilization (simplified - would need nvidia-ml-py for accurate data)
                gpu_utilization = (gpu_memory_used_mb / gpu_memory_total_mb * 100) if gpu_memory_total_mb > 0 else 0.0
                
            except Exception:
                pass  # GPU monitoring failed, use defaults
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_available_mb=memory.available / 1024 / 1024,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_utilization=gpu_utilization,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )
        
        # Check for alerts
        self._check_alerts(metrics)
        
        return metrics
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check if any metrics exceed alert thresholds."""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds.get('cpu_percent', 100):
            alerts.append(('cpu_high', {'cpu_percent': metrics.cpu_percent}))
        
        if metrics.memory_percent > self.alert_thresholds.get('memory_percent', 100):
            alerts.append(('memory_high', {'memory_percent': metrics.memory_percent}))
        
        if self.enable_gpu_monitoring:
            gpu_memory_percent = (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb * 100) if metrics.gpu_memory_total_mb > 0 else 0
            
            if gpu_memory_percent > self.alert_thresholds.get('gpu_memory_percent', 100):
                alerts.append(('gpu_memory_high', {'gpu_memory_percent': gpu_memory_percent}))
            
            if metrics.gpu_utilization > self.alert_thresholds.get('gpu_utilization', 100):
                alerts.append(('gpu_utilization_high', {'gpu_utilization': metrics.gpu_utilization}))
        
        # Trigger alerts
        for alert_type, alert_data in alerts:
            self._trigger_alert(alert_type, alert_data)
    
    def _monitor_loop(self):
        """Main monitoring loop (runs in separate thread)."""
        while self.is_monitoring:
            try:
                metrics = self._get_system_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.sample_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.sample_interval)
    
    def start_monitoring(self):
        """Start system monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent system metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, duration_minutes: Optional[float] = None) -> List[SystemMetrics]:
        """Get metrics history, optionally filtered by time duration."""
        if duration_minutes is None:
            return list(self.metrics_history)
        
        cutoff_time = time.time() - (duration_minutes * 60)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_resource_summary(self, duration_minutes: float = 5.0) -> Dict[str, Any]:
        """Get resource usage summary over specified duration."""
        history = self.get_metrics_history(duration_minutes)
        
        if not history:
            return {}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in history]
        memory_values = [m.memory_percent for m in history]
        gpu_memory_values = [m.gpu_memory_used_mb for m in history]
        
        return {
            'duration_minutes': duration_minutes,
            'sample_count': len(history),
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'current_used_mb': history[-1].memory_used_mb
            },
            'gpu_memory': {
                'avg_mb': sum(gpu_memory_values) / len(gpu_memory_values),
                'max_mb': max(gpu_memory_values),
                'current_mb': history[-1].gpu_memory_used_mb
            } if self.enable_gpu_monitoring else None
        }


class PerformanceTracker:
    """
    Tracks model performance metrics during evolution and inference.
    
    Monitors accuracy, latency, throughput, and resource efficiency
    across different model configurations and evolution generations.
    """
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.evolution_history = deque(maxlen=history_size)
        self.performance_history = deque(maxlen=history_size)
        self.generation_metrics = defaultdict(list)
        
        # Tracking state
        self.current_generation = 0
        self.generation_start_time = None
        self.best_performance = None
    
    def start_generation(self, generation: int):
        """Mark the start of a new evolution generation."""
        self.current_generation = generation
        self.generation_start_time = time.time()
    
    def record_evolution_metrics(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        population_diversity: float,
        mutation_rate: float,
        convergence_rate: float,
        active_experts: int,
        topology_sparsity: float
    ):
        """Record evolution metrics for current generation."""
        metrics = EvolutionMetrics(
            timestamp=time.time(),
            generation=generation,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            population_diversity=population_diversity,
            mutation_rate=mutation_rate,
            convergence_rate=convergence_rate,
            active_experts=active_experts,
            topology_sparsity=topology_sparsity
        )
        
        self.evolution_history.append(metrics)
        self.generation_metrics[generation].append(metrics)
    
    def record_performance_metrics(
        self,
        accuracy: float,
        latency_ms: float,
        throughput_samples_per_sec: float,
        memory_usage_mb: float,
        flops_per_sample: float = 0.0,
        energy_consumption_j: float = 0.0
    ):
        """Record model performance metrics."""
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            accuracy=accuracy,
            latency_ms=latency_ms,
            throughput_samples_per_sec=throughput_samples_per_sec,
            memory_usage_mb=memory_usage_mb,
            flops_per_sample=flops_per_sample,
            energy_consumption_j=energy_consumption_j
        )
        
        self.performance_history.append(metrics)
        
        # Track best performance
        if self.best_performance is None or accuracy > self.best_performance.accuracy:
            self.best_performance = metrics
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive evolution summary."""
        if not self.evolution_history:
            return {}
        
        # Calculate improvement metrics
        first_gen = self.evolution_history[0]
        last_gen = self.evolution_history[-1]
        
        fitness_improvement = last_gen.best_fitness - first_gen.best_fitness
        sparsity_improvement = last_gen.topology_sparsity - first_gen.topology_sparsity
        
        # Find convergence point
        convergence_generation = None
        best_fitness = max(m.best_fitness for m in self.evolution_history)
        
        for metrics in self.evolution_history:
            if metrics.best_fitness >= best_fitness * 0.95:  # 95% of best fitness
                convergence_generation = metrics.generation
                break
        
        return {
            'total_generations': last_gen.generation - first_gen.generation + 1,
            'fitness_improvement': fitness_improvement,
            'final_best_fitness': last_gen.best_fitness,
            'sparsity_improvement': sparsity_improvement,
            'final_sparsity': last_gen.topology_sparsity,
            'convergence_generation': convergence_generation,
            'final_active_experts': last_gen.active_experts,
            'avg_mutation_rate': sum(m.mutation_rate for m in self.evolution_history) / len(self.evolution_history)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_history:
            return {}
        
        # Calculate statistics
        accuracies = [m.accuracy for m in self.performance_history]
        latencies = [m.latency_ms for m in self.performance_history]
        throughputs = [m.throughput_samples_per_sec for m in self.performance_history]
        memory_usages = [m.memory_usage_mb for m in self.performance_history]
        
        def stats(values):
            return {
                'avg': sum(values) / len(values),
                'max': max(values),
                'min': min(values),
                'latest': values[-1]
            }
        
        return {
            'sample_count': len(self.performance_history),
            'accuracy': stats(accuracies),
            'latency_ms': stats(latencies),
            'throughput': stats(throughputs),
            'memory_usage_mb': stats(memory_usages),
            'best_performance': asdict(self.best_performance) if self.best_performance else None,
            'efficiency_score': self._calculate_efficiency_score()
        }
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score (accuracy / latency * throughput)."""
        if not self.performance_history:
            return 0.0
        
        latest = self.performance_history[-1]
        if latest.latency_ms == 0:
            return 0.0
        
        # Normalized efficiency score
        efficiency = (latest.accuracy * latest.throughput_samples_per_sec) / latest.latency_ms
        return efficiency
    
    def export_metrics(self, filepath: Path, format: str = "json"):
        """Export all tracked metrics to file."""
        if format == "json":
            data = {
                'evolution_history': [asdict(m) for m in self.evolution_history],
                'performance_history': [asdict(m) for m in self.performance_history],
                'evolution_summary': self.get_evolution_summary(),
                'performance_summary': self.get_performance_summary(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif format == "csv":
            import pandas as pd
            
            # Evolution metrics
            if self.evolution_history:
                evolution_df = pd.DataFrame([asdict(m) for m in self.evolution_history])
                evolution_df.to_csv(filepath.with_suffix('.evolution.csv'), index=False)
            
            # Performance metrics
            if self.performance_history:
                performance_df = pd.DataFrame([asdict(m) for m in self.performance_history])
                performance_df.to_csv(filepath.with_suffix('.performance.csv'), index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_pareto_frontier(self, x_metric: str = "latency_ms", y_metric: str = "accuracy") -> List[PerformanceMetrics]:
        """Get Pareto frontier for two performance metrics."""
        if not self.performance_history:
            return []
        
        points = [(getattr(m, x_metric), getattr(m, y_metric), m) for m in self.performance_history]
        
        # Sort by x metric (assume lower is better for x, higher for y)
        points.sort(key=lambda p: p[0])
        
        pareto_frontier = []
        max_y = float('-inf')
        
        for x, y, metrics in points:
            if y > max_y:
                max_y = y
                pareto_frontier.append(metrics)
        
        return pareto_frontier