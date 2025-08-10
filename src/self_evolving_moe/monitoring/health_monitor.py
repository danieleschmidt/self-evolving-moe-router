"""
Health monitoring and system diagnostics for Self-Evolving MoE Router.

This module implements comprehensive health monitoring, system diagnostics,
and automated recovery mechanisms for production deployment.
"""

import torch
import numpy as np
import time
import threading
import queue
import psutil
import os
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import warnings
from pathlib import Path

from ..utils.logging import get_logger
from ..utils.exceptions import ResourceConstraintError, EvolutionError

logger = get_logger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class MetricType(Enum):
    """Types of metrics to monitor."""
    PERFORMANCE = "performance"
    RESOURCE = "resource"  
    EVOLUTION = "evolution"
    MODEL = "model"
    SYSTEM = "system"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    timestamp: float
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemSnapshot:
    """Complete system health snapshot."""
    timestamp: float
    overall_status: HealthStatus
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, 
                 monitoring_interval: float = 30.0,
                 history_size: int = 1000,
                 enable_auto_recovery: bool = True):
        """
        Initialize health monitor.
        
        Args:
            monitoring_interval: Seconds between health checks
            history_size: Number of snapshots to keep in history
            enable_auto_recovery: Enable automatic recovery mechanisms
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_auto_recovery = enable_auto_recovery
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Health data
        self.current_snapshot: Optional[SystemSnapshot] = None
        self.history: deque = deque(maxlen=history_size)
        self.metric_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Thresholds and configuration
        self.thresholds = self._get_default_thresholds()
        self.recovery_actions: Dict[str, Callable] = {}
        
        # Performance tracking
        self.start_time = time.time()
        self.uptime_stats = {
            'total_uptime': 0.0,
            'healthy_uptime': 0.0,
            'warning_periods': 0,
            'critical_periods': 0,
            'recovery_count': 0
        }
        
        # Component references (set externally)
        self.evolution_engine: Optional[Any] = None
        self.expert_pool: Optional[Any] = None
        self.model: Optional[Any] = None
        
        logger.info(f"Initialized health monitor with {monitoring_interval}s interval")
    
    def _get_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get default health thresholds."""
        return {
            'cpu_usage': {'warning': 80.0, 'critical': 95.0},
            'memory_usage': {'warning': 85.0, 'critical': 95.0},
            'gpu_memory': {'warning': 90.0, 'critical': 98.0},
            'evolution_fitness_stagnation': {'warning': 20, 'critical': 50},  # generations
            'expert_utilization_imbalance': {'warning': 0.3, 'critical': 0.5},
            'routing_diversity': {'warning': 0.1, 'critical': 0.05},
            'validation_error_rate': {'warning': 0.05, 'critical': 0.2},
            'inference_latency': {'warning': 100.0, 'critical': 500.0},  # ms
            'temperature': {'warning': 70.0, 'critical': 85.0}  # CPU temp if available
        }
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Started health monitoring")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Health monitoring loop started")
        
        while not self.stop_event.wait(self.monitoring_interval):
            try:
                # Collect health snapshot
                snapshot = self._collect_health_snapshot()
                
                # Update current state
                self.current_snapshot = snapshot
                self.history.append(snapshot)
                
                # Check for alerts and take actions
                self._process_health_snapshot(snapshot)
                
                # Update uptime statistics
                self._update_uptime_stats(snapshot)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                # Continue monitoring even if individual checks fail
        
        logger.info("Health monitoring loop stopped")
    
    def _collect_health_snapshot(self) -> SystemSnapshot:
        """Collect comprehensive system health snapshot."""
        timestamp = time.time()
        metrics = {}
        
        try:
            # System resource metrics
            metrics.update(self._collect_system_metrics())
            
            # Evolution metrics (if available)
            if self.evolution_engine:
                metrics.update(self._collect_evolution_metrics())
            
            # Model metrics (if available)
            if self.model and self.expert_pool:
                metrics.update(self._collect_model_metrics())
            
            # Custom metrics
            metrics.update(self._collect_custom_metrics())
            
        except Exception as e:
            logger.error(f"Error collecting health metrics: {e}")
        
        # Determine overall status
        overall_status = self._determine_overall_status(metrics)
        
        # Generate alerts and recommendations
        alerts = self._generate_alerts(metrics)
        recommendations = self._generate_recommendations(metrics, overall_status)
        
        return SystemSnapshot(
            timestamp=timestamp,
            overall_status=overall_status,
            metrics=metrics,
            alerts=alerts,
            recommendations=recommendations
        )
    
    def _collect_system_metrics(self) -> Dict[str, HealthMetric]:
        """Collect system resource metrics."""
        metrics = {}
        
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            metrics['cpu_usage'] = HealthMetric(
                name='cpu_usage',
                value=cpu_usage,
                timestamp=time.time(),
                status=self._get_status_from_thresholds(cpu_usage, 'cpu_usage'),
                threshold_warning=self.thresholds['cpu_usage']['warning'],
                threshold_critical=self.thresholds['cpu_usage']['critical']
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            metrics['memory_usage'] = HealthMetric(
                name='memory_usage',
                value=memory_usage,
                timestamp=time.time(),
                status=self._get_status_from_thresholds(memory_usage, 'memory_usage'),
                threshold_warning=self.thresholds['memory_usage']['warning'],
                threshold_critical=self.thresholds['memory_usage']['critical'],
                metadata={'available_gb': memory.available / (1024**3)}
            )
            
            # GPU metrics (if available)
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory_used = torch.cuda.memory_allocated(i) / torch.cuda.max_memory_allocated(i) * 100
                    metrics[f'gpu_{i}_memory'] = HealthMetric(
                        name=f'gpu_{i}_memory',
                        value=gpu_memory_used,
                        timestamp=time.time(),
                        status=self._get_status_from_thresholds(gpu_memory_used, 'gpu_memory'),
                        threshold_warning=self.thresholds['gpu_memory']['warning'],
                        threshold_critical=self.thresholds['gpu_memory']['critical'],
                        metadata={'device_name': torch.cuda.get_device_name(i)}
                    )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            metrics['disk_usage'] = HealthMetric(
                name='disk_usage',
                value=disk_usage,
                timestamp=time.time(),
                status=HealthStatus.HEALTHY if disk_usage < 90 else HealthStatus.WARNING,
                metadata={'free_gb': disk.free / (1024**3)}
            )
            
            # System temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    max_temp = 0
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current and entry.current > max_temp:
                                max_temp = entry.current
                    
                    if max_temp > 0:
                        metrics['temperature'] = HealthMetric(
                            name='temperature',
                            value=max_temp,
                            timestamp=time.time(),
                            status=self._get_status_from_thresholds(max_temp, 'temperature'),
                            threshold_warning=self.thresholds['temperature']['warning'],
                            threshold_critical=self.thresholds['temperature']['critical']
                        )
            except Exception:
                pass  # Temperature monitoring not available
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def _collect_evolution_metrics(self) -> Dict[str, HealthMetric]:
        """Collect evolution-specific health metrics."""
        metrics = {}
        
        try:
            engine = self.evolution_engine
            
            # Fitness stagnation
            if hasattr(engine, 'evolution_history') and engine.evolution_history:
                recent_fitness = [stats.get('best_fitness', 0) for stats in engine.evolution_history[-20:]]
                if len(recent_fitness) > 5:
                    # Check for stagnation (no improvement)
                    recent_improvement = max(recent_fitness) - min(recent_fitness)
                    stagnation_score = 1.0 / (recent_improvement + 1e-6)  # Higher = more stagnant
                    
                    metrics['evolution_stagnation'] = HealthMetric(
                        name='evolution_stagnation',
                        value=stagnation_score,
                        timestamp=time.time(),
                        status=HealthStatus.HEALTHY if stagnation_score < 10 else HealthStatus.WARNING,
                        metadata={'recent_improvement': recent_improvement}
                    )
            
            # Population diversity
            if hasattr(engine, 'population') and engine.population:
                diversity = self._compute_population_diversity(engine.population)
                metrics['population_diversity'] = HealthMetric(
                    name='population_diversity',
                    value=diversity,
                    timestamp=time.time(),
                    status=self._get_status_from_thresholds(diversity, 'routing_diversity'),
                    threshold_warning=self.thresholds['routing_diversity']['warning'],
                    threshold_critical=self.thresholds['routing_diversity']['critical']
                )
            
            # Evolution rate
            if hasattr(engine, 'generation') and engine.generation > 0:
                evolution_time = time.time() - self.start_time
                generations_per_hour = (engine.generation / evolution_time) * 3600
                
                metrics['evolution_rate'] = HealthMetric(
                    name='evolution_rate',
                    value=generations_per_hour,
                    timestamp=time.time(),
                    status=HealthStatus.HEALTHY,
                    metadata={'total_generations': engine.generation}
                )
            
        except Exception as e:
            logger.error(f"Error collecting evolution metrics: {e}")
        
        return metrics
    
    def _collect_model_metrics(self) -> Dict[str, HealthMetric]:
        """Collect model-specific health metrics."""
        metrics = {}
        
        try:
            # Expert utilization balance
            if hasattr(self.expert_pool, 'get_expert_utilization'):
                util_stats = self.expert_pool.get_expert_utilization()
                load_balance_score = util_stats.get('load_balance_score', 1.0)
                
                # Invert score (higher imbalance = worse health)
                imbalance_score = 1.0 - load_balance_score
                
                metrics['expert_utilization_imbalance'] = HealthMetric(
                    name='expert_utilization_imbalance',
                    value=imbalance_score,
                    timestamp=time.time(),
                    status=self._get_status_from_thresholds(imbalance_score, 'expert_utilization_imbalance'),
                    threshold_warning=self.thresholds['expert_utilization_imbalance']['warning'],
                    threshold_critical=self.thresholds['expert_utilization_imbalance']['critical'],
                    metadata={'active_experts': util_stats.get('active_experts', 0)}
                )
            
            # Model parameter health
            if hasattr(self.model, 'parameters'):
                param_stats = self._analyze_model_parameters()
                metrics['parameter_health'] = HealthMetric(
                    name='parameter_health',
                    value=param_stats['health_score'],
                    timestamp=time.time(),
                    status=HealthStatus.HEALTHY if param_stats['health_score'] > 0.8 else HealthStatus.WARNING,
                    metadata=param_stats
                )
            
        except Exception as e:
            logger.error(f"Error collecting model metrics: {e}")
        
        return metrics
    
    def _collect_custom_metrics(self) -> Dict[str, HealthMetric]:
        """Collect custom application metrics."""
        metrics = {}
        
        try:
            # Uptime
            uptime = time.time() - self.start_time
            metrics['uptime'] = HealthMetric(
                name='uptime',
                value=uptime,
                timestamp=time.time(),
                status=HealthStatus.HEALTHY,
                metadata={'uptime_hours': uptime / 3600}
            )
            
            # Health check frequency
            if len(self.history) > 1:
                recent_intervals = []
                for i in range(1, min(10, len(self.history))):
                    interval = self.history[-i].timestamp - self.history[-i-1].timestamp
                    recent_intervals.append(interval)
                
                avg_interval = np.mean(recent_intervals) if recent_intervals else self.monitoring_interval
                metrics['monitoring_health'] = HealthMetric(
                    name='monitoring_health',
                    value=avg_interval,
                    timestamp=time.time(),
                    status=HealthStatus.HEALTHY if abs(avg_interval - self.monitoring_interval) < 10 else HealthStatus.WARNING,
                    metadata={'expected_interval': self.monitoring_interval}
                )
            
        except Exception as e:
            logger.error(f"Error collecting custom metrics: {e}")
        
        return metrics
    
    def _get_status_from_thresholds(self, value: float, threshold_key: str) -> HealthStatus:
        """Determine health status from value and thresholds."""
        if threshold_key not in self.thresholds:
            return HealthStatus.HEALTHY
        
        thresholds = self.thresholds[threshold_key]
        
        if value >= thresholds.get('critical', float('inf')):
            return HealthStatus.CRITICAL
        elif value >= thresholds.get('warning', float('inf')):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _determine_overall_status(self, metrics: Dict[str, HealthMetric]) -> HealthStatus:
        """Determine overall system health status."""
        if not metrics:
            return HealthStatus.FAILED
        
        status_counts = defaultdict(int)
        for metric in metrics.values():
            status_counts[metric.status] += 1
        
        # Priority: FAILED > CRITICAL > WARNING > HEALTHY
        if status_counts[HealthStatus.FAILED] > 0:
            return HealthStatus.FAILED
        elif status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] > len(metrics) * 0.3:  # More than 30% warnings
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _generate_alerts(self, metrics: Dict[str, HealthMetric]) -> List[str]:
        """Generate alerts based on metric values."""
        alerts = []
        
        for metric in metrics.values():
            if metric.status == HealthStatus.CRITICAL:
                alerts.append(f"CRITICAL: {metric.name} = {metric.value:.2f}")
            elif metric.status == HealthStatus.WARNING:
                alerts.append(f"WARNING: {metric.name} = {metric.value:.2f}")
        
        return alerts
    
    def _generate_recommendations(self, metrics: Dict[str, HealthMetric], 
                                overall_status: HealthStatus) -> List[str]:
        """Generate recommendations based on system health."""
        recommendations = []
        
        # Resource recommendations
        if 'memory_usage' in metrics and metrics['memory_usage'].status != HealthStatus.HEALTHY:
            recommendations.append("Consider reducing batch size or model size to reduce memory usage")
        
        if 'cpu_usage' in metrics and metrics['cpu_usage'].status != HealthStatus.HEALTHY:
            recommendations.append("High CPU usage detected - consider reducing population size or using GPU acceleration")
        
        # Evolution recommendations
        if 'evolution_stagnation' in metrics and metrics['evolution_stagnation'].status != HealthStatus.HEALTHY:
            recommendations.append("Evolution appears stagnant - consider increasing mutation rate or population diversity")
        
        if 'expert_utilization_imbalance' in metrics and metrics['expert_utilization_imbalance'].status != HealthStatus.HEALTHY:
            recommendations.append("Expert utilization is imbalanced - consider adjusting load balancing weights")
        
        # General recommendations
        if overall_status == HealthStatus.CRITICAL:
            recommendations.append("System is in critical state - consider stopping evolution and investigating issues")
        elif overall_status == HealthStatus.WARNING:
            recommendations.append("System performance is degraded - monitor closely and consider parameter adjustments")
        
        return recommendations
    
    def _process_health_snapshot(self, snapshot: SystemSnapshot):
        """Process health snapshot and take automated actions."""
        # Log status changes
        if len(self.history) > 0:
            prev_status = self.history[-1].overall_status
            if snapshot.overall_status != prev_status:
                logger.info(f"System health status changed: {prev_status.value} -> {snapshot.overall_status.value}")
        
        # Log alerts
        for alert in snapshot.alerts:
            logger.warning(f"Health Alert: {alert}")
        
        # Automated recovery actions
        if self.enable_auto_recovery:
            self._execute_recovery_actions(snapshot)
    
    def _execute_recovery_actions(self, snapshot: SystemSnapshot):
        """Execute automated recovery actions based on health status."""
        try:
            # Critical memory usage - trigger garbage collection
            if 'memory_usage' in snapshot.metrics:
                memory_metric = snapshot.metrics['memory_usage']
                if memory_metric.status == HealthStatus.CRITICAL:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("Executed memory cleanup due to critical memory usage")
                    self.uptime_stats['recovery_count'] += 1
            
            # Evolution stagnation - increase diversity
            if 'evolution_stagnation' in snapshot.metrics:
                stagnation_metric = snapshot.metrics['evolution_stagnation']
                if stagnation_metric.status == HealthStatus.CRITICAL and self.evolution_engine:
                    # This would require access to evolution engine's internal state
                    logger.info("Evolution stagnation detected - automated diversity boost recommended")
            
            # Execute custom recovery actions
            for action_name, action_func in self.recovery_actions.items():
                try:
                    action_func(snapshot)
                except Exception as e:
                    logger.error(f"Recovery action '{action_name}' failed: {e}")
                    
        except Exception as e:
            logger.error(f"Error executing recovery actions: {e}")
    
    def _update_uptime_stats(self, snapshot: SystemSnapshot):
        """Update uptime statistics."""
        current_time = time.time()
        interval = current_time - (self.history[-2].timestamp if len(self.history) >= 2 else self.start_time)
        
        self.uptime_stats['total_uptime'] = current_time - self.start_time
        
        if snapshot.overall_status == HealthStatus.HEALTHY:
            self.uptime_stats['healthy_uptime'] += interval
        elif snapshot.overall_status == HealthStatus.WARNING:
            self.uptime_stats['warning_periods'] += 1
        elif snapshot.overall_status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
            self.uptime_stats['critical_periods'] += 1
    
    def _compute_population_diversity(self, population: List) -> float:
        """Compute population diversity metric."""
        if len(population) < 2:
            return 0.0
        
        try:
            distances = []
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    if hasattr(population[i], 'routing_matrix') and hasattr(population[j], 'routing_matrix'):
                        diff = (population[i].routing_matrix != population[j].routing_matrix).float()
                        distance = diff.mean().item()
                        distances.append(distance)
            
            return np.mean(distances) if distances else 0.0
        except Exception:
            return 0.0
    
    def _analyze_model_parameters(self) -> Dict[str, Any]:
        """Analyze model parameter health."""
        try:
            stats = {
                'total_params': 0,
                'zero_params': 0,
                'nan_params': 0,
                'inf_params': 0,
                'grad_norm': 0.0,
                'health_score': 1.0
            }
            
            for param in self.model.parameters():
                stats['total_params'] += param.numel()
                stats['zero_params'] += (param == 0).sum().item()
                stats['nan_params'] += torch.isnan(param).sum().item()
                stats['inf_params'] += torch.isinf(param).sum().item()
                
                if param.grad is not None:
                    stats['grad_norm'] += param.grad.norm().item()
            
            # Calculate health score
            if stats['total_params'] > 0:
                nan_ratio = stats['nan_params'] / stats['total_params']
                inf_ratio = stats['inf_params'] / stats['total_params']
                zero_ratio = stats['zero_params'] / stats['total_params']
                
                # Penalize NaN/Inf parameters heavily, zero parameters moderately
                stats['health_score'] = 1.0 - (nan_ratio * 10 + inf_ratio * 10 + zero_ratio * 0.1)
                stats['health_score'] = max(0.0, min(1.0, stats['health_score']))
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing model parameters: {e}")
            return {'health_score': 0.5}
    
    def register_recovery_action(self, name: str, action: Callable):
        """Register custom recovery action."""
        self.recovery_actions[name] = action
        logger.info(f"Registered recovery action: {name}")
    
    def get_current_health(self) -> Optional[SystemSnapshot]:
        """Get current system health snapshot."""
        return self.current_snapshot
    
    def get_health_history(self, hours: float = 1.0) -> List[SystemSnapshot]:
        """Get health history for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        return [snapshot for snapshot in self.history if snapshot.timestamp >= cutoff_time]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        if not self.current_snapshot:
            return {'status': 'not_available'}
        
        current = self.current_snapshot
        
        # Compute availability metrics
        total_time = self.uptime_stats['total_uptime']
        healthy_ratio = self.uptime_stats['healthy_uptime'] / total_time if total_time > 0 else 0
        
        summary = {
            'current_status': current.overall_status.value,
            'timestamp': current.timestamp,
            'active_alerts': len(current.alerts),
            'recommendations_count': len(current.recommendations),
            'uptime_hours': total_time / 3600,
            'availability': healthy_ratio,
            'warning_periods': self.uptime_stats['warning_periods'],
            'critical_periods': self.uptime_stats['critical_periods'],
            'recovery_actions': self.uptime_stats['recovery_count'],
            'monitoring_active': self.is_monitoring
        }
        
        # Add key metrics
        key_metrics = ['cpu_usage', 'memory_usage', 'population_diversity', 'evolution_stagnation']
        for metric_name in key_metrics:
            if metric_name in current.metrics:
                metric = current.metrics[metric_name]
                summary[f'{metric_name}_value'] = metric.value
                summary[f'{metric_name}_status'] = metric.status.value
        
        return summary
    
    def export_health_data(self, filepath: str):
        """Export health monitoring data to file."""
        try:
            data = {
                'export_timestamp': time.time(),
                'monitoring_config': {
                    'interval': self.monitoring_interval,
                    'history_size': self.history_size,
                    'auto_recovery_enabled': self.enable_auto_recovery
                },
                'uptime_stats': self.uptime_stats,
                'thresholds': self.thresholds,
                'current_snapshot': self.current_snapshot.__dict__ if self.current_snapshot else None,
                'recent_history': [snapshot.__dict__ for snapshot in list(self.history)[-50:]]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported health data to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export health data: {e}")
    
    def __del__(self):
        """Cleanup when monitor is destroyed."""
        self.stop_monitoring()