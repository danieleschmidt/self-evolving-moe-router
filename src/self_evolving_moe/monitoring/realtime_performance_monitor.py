"""
Real-time Performance Monitoring and Auto-tuning System
TERRAGON NEXT-GEN v5.0 - Intelligent Performance Optimization

Advanced monitoring system that provides:
- Real-time performance metrics collection and analysis
- Automatic performance bottleneck detection
- Intelligent auto-tuning of system parameters
- Predictive performance modeling and optimization
- Adaptive resource allocation and scaling
"""

import asyncio
import time
import threading
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import logging
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
import queue

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    GPU_USAGE = "gpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    QUEUE_SIZE = "queue_size"
    FITNESS_SCORE = "fitness_score"
    CONVERGENCE_RATE = "convergence_rate"
    DIVERSITY_METRIC = "diversity_metric"
    EVOLUTION_EFFICIENCY = "evolution_efficiency"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class TuningAction(Enum):
    """Auto-tuning actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    ADJUST_PARAMETERS = "adjust_parameters"
    OPTIMIZE_MEMORY = "optimize_memory"
    INCREASE_PARALLELISM = "increase_parallelism"
    DECREASE_PARALLELISM = "decrease_parallelism"
    CLEAR_CACHE = "clear_cache"
    RESTART_COMPONENT = "restart_component"

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    timestamp: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert information"""
    alert_id: str
    metric_name: str
    level: AlertLevel
    message: str
    threshold_value: float
    actual_value: float
    timestamp: float
    suggested_actions: List[TuningAction] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TuningRecommendation:
    """Auto-tuning recommendation"""
    recommendation_id: str
    action: TuningAction
    target_component: str
    parameters: Dict[str, Any]
    confidence: float
    expected_improvement: float
    reasoning: str
    timestamp: float

@dataclass
class PerformanceProfile:
    """System performance profile"""
    profile_name: str
    metrics_summary: Dict[str, Dict[str, float]]  # metric_name -> {min, max, avg, std}
    performance_characteristics: Dict[str, Any]
    bottlenecks: List[str]
    optimization_opportunities: List[str]
    recommended_configuration: Dict[str, Any]
    creation_timestamp: float

class MetricCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.collection_thread = None
        self.running = False
        self.collection_interval = 1.0  # seconds
        
        # Custom metric collectors
        self.custom_collectors: Dict[str, Callable] = {}
        
    def start_collection(self):
        """Start metric collection in background thread"""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metric collection started")
    
    def stop_collection(self):
        """Stop metric collection"""
        self.running = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        logger.info("Metric collection stopped")
    
    def add_custom_collector(self, name: str, collector_func: Callable[[], float]):
        """Add custom metric collector function"""
        self.custom_collectors[name] = collector_func
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_custom_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metric collection: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        timestamp = time.time()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self._add_metric("cpu_usage", cpu_percent, timestamp, "%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        self._add_metric("memory_usage", memory.percent, timestamp, "%")
        self._add_metric("memory_available", memory.available / (1024**3), timestamp, "GB")
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self._add_metric("disk_read_rate", disk_io.read_bytes / (1024**2), timestamp, "MB/s")
            self._add_metric("disk_write_rate", disk_io.write_bytes / (1024**2), timestamp, "MB/s")
        
        # Network I/O
        network_io = psutil.net_io_counters()
        if network_io:
            self._add_metric("network_sent_rate", network_io.bytes_sent / (1024**2), timestamp, "MB/s")
            self._add_metric("network_recv_rate", network_io.bytes_recv / (1024**2), timestamp, "MB/s")
    
    def _collect_custom_metrics(self):
        """Collect custom application metrics"""
        timestamp = time.time()
        
        for name, collector_func in self.custom_collectors.items():
            try:
                value = collector_func()
                if isinstance(value, (int, float)) and not np.isnan(value):
                    self._add_metric(name, value, timestamp, "units")
            except Exception as e:
                logger.warning(f"Custom collector {name} failed: {e}")
    
    def _add_metric(self, name: str, value: float, timestamp: float, unit: str):
        """Add metric to buffer"""
        metric = PerformanceMetric(name, value, timestamp, unit)
        self.metrics_buffer[name].append(metric)
    
    def get_recent_metrics(self, metric_name: str, count: int = 10) -> List[PerformanceMetric]:
        """Get recent metrics for a specific metric"""
        if metric_name in self.metrics_buffer:
            return list(self.metrics_buffer[metric_name])[-count:]
        return []
    
    def get_metric_statistics(self, metric_name: str, window_seconds: int = 60) -> Dict[str, float]:
        """Get statistical summary of metric over time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        if metric_name not in self.metrics_buffer:
            return {}
        
        # Filter metrics in time window
        recent_metrics = [
            m for m in self.metrics_buffer[metric_name]
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'latest': values[-1],
            'trend': self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = list(range(len(values)))
        trend_coeff = np.polyfit(x, values, 1)[0] if len(values) > 1 else 0.0
        
        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, trend_coeff))

class PerformanceAnalyzer:
    """Analyzes performance data and detects patterns"""
    
    def __init__(self, metric_collector: MetricCollector):
        self.metric_collector = metric_collector
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': {'warning': 80.0, 'critical': 95.0},
            'memory_usage': {'warning': 85.0, 'critical': 95.0},
            'latency': {'warning': 100.0, 'critical': 500.0},  # ms
            'error_rate': {'warning': 0.01, 'critical': 0.05},  # 1%, 5%
            'cache_hit_rate': {'warning': 0.8, 'critical': 0.6},  # Below 80%, 60%
        }
        
        # Pattern detection
        self.anomaly_detection_window = 100
        self.pattern_history: Dict[str, List[float]] = defaultdict(list)
        
    def analyze_current_performance(self) -> Dict[str, Any]:
        """Analyze current system performance"""
        analysis = {
            'timestamp': time.time(),
            'overall_health': 'healthy',
            'metric_analysis': {},
            'detected_anomalies': [],
            'performance_patterns': {},
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Analyze each metric
        for metric_name in self.metric_collector.metrics_buffer.keys():
            metric_analysis = self._analyze_metric(metric_name)
            analysis['metric_analysis'][metric_name] = metric_analysis
            
            # Check for issues
            if metric_analysis.get('alert_level') == 'critical':
                analysis['overall_health'] = 'critical'
            elif metric_analysis.get('alert_level') == 'warning' and analysis['overall_health'] == 'healthy':
                analysis['overall_health'] = 'warning'
        
        # Detect system-wide patterns
        analysis['performance_patterns'] = self._detect_performance_patterns()
        
        # Identify bottlenecks
        analysis['bottlenecks'] = self._identify_bottlenecks()
        
        return analysis
    
    def _analyze_metric(self, metric_name: str) -> Dict[str, Any]:
        """Analyze individual metric"""
        stats = self.metric_collector.get_metric_statistics(metric_name, window_seconds=60)
        
        if not stats:
            return {'status': 'no_data'}
        
        analysis = {
            'current_value': stats['latest'],
            'statistics': stats,
            'status': 'normal',
            'alert_level': None,
            'anomaly_score': 0.0
        }
        
        # Check thresholds
        if metric_name in self.thresholds:
            thresholds = self.thresholds[metric_name]
            current_value = stats['latest']
            
            if current_value >= thresholds.get('critical', float('inf')):
                analysis['status'] = 'critical'
                analysis['alert_level'] = 'critical'
            elif current_value >= thresholds.get('warning', float('inf')):
                analysis['status'] = 'warning'
                analysis['alert_level'] = 'warning'
        
        # Detect anomalies
        analysis['anomaly_score'] = self._detect_anomaly(metric_name, stats['latest'])
        
        if analysis['anomaly_score'] > 0.8:
            analysis['status'] = 'anomalous'
            if not analysis['alert_level']:
                analysis['alert_level'] = 'warning'
        
        return analysis
    
    def _detect_anomaly(self, metric_name: str, current_value: float) -> float:
        """Detect if current value is anomalous"""
        history = self.pattern_history[metric_name]
        
        # Add current value to history
        history.append(current_value)
        if len(history) > self.anomaly_detection_window:
            history.pop(0)
        
        if len(history) < 10:
            return 0.0  # Not enough data
        
        # Simple statistical anomaly detection
        mean_val = np.mean(history[:-1])  # Exclude current value
        std_val = np.std(history[:-1])
        
        if std_val == 0:
            return 0.0
        
        # Z-score based anomaly detection
        z_score = abs(current_value - mean_val) / std_val
        
        # Convert to 0-1 anomaly score
        anomaly_score = min(1.0, z_score / 3.0)  # Values beyond 3 std devs are highly anomalous
        
        return anomaly_score
    
    def _detect_performance_patterns(self) -> Dict[str, Any]:
        """Detect system-wide performance patterns"""
        patterns = {
            'correlation_matrix': {},
            'cyclical_patterns': {},
            'trending_metrics': {},
            'performance_regime': 'normal'
        }
        
        # Calculate metric correlations
        metric_names = list(self.metric_collector.metrics_buffer.keys())
        
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i+1:]:
                correlation = self._calculate_metric_correlation(metric1, metric2)
                if abs(correlation) > 0.5:  # Significant correlation
                    patterns['correlation_matrix'][f"{metric1}_{metric2}"] = correlation
        
        # Detect trending metrics
        for metric_name in metric_names:
            stats = self.metric_collector.get_metric_statistics(metric_name)
            if stats and abs(stats.get('trend', 0)) > 0.1:
                patterns['trending_metrics'][metric_name] = stats['trend']
        
        return patterns
    
    def _calculate_metric_correlation(self, metric1: str, metric2: str, window_size: int = 50) -> float:
        """Calculate correlation between two metrics"""
        try:
            # Get recent values for both metrics
            recent1 = self.metric_collector.get_recent_metrics(metric1, window_size)
            recent2 = self.metric_collector.get_recent_metrics(metric2, window_size)
            
            if len(recent1) < 5 or len(recent2) < 5:
                return 0.0
            
            # Align timestamps and extract values
            values1, values2 = [], []
            
            # Simple alignment - match by index (assumes same collection frequency)
            min_length = min(len(recent1), len(recent2))
            for i in range(min_length):
                values1.append(recent1[i].value)
                values2.append(recent2[i].value)
            
            if len(values1) < 3:
                return 0.0
            
            correlation = np.corrcoef(values1, values2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.warning(f"Correlation calculation failed for {metric1} and {metric2}: {e}")
            return 0.0
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify system bottlenecks"""
        bottlenecks = []
        
        # Check common bottleneck indicators
        cpu_stats = self.metric_collector.get_metric_statistics('cpu_usage')
        memory_stats = self.metric_collector.get_metric_statistics('memory_usage')
        
        if cpu_stats and cpu_stats.get('mean', 0) > 80:
            bottlenecks.append('cpu_bound')
        
        if memory_stats and memory_stats.get('mean', 0) > 85:
            bottlenecks.append('memory_bound')
        
        # Check for I/O bottlenecks
        disk_read_stats = self.metric_collector.get_metric_statistics('disk_read_rate')
        disk_write_stats = self.metric_collector.get_metric_statistics('disk_write_rate')
        
        if disk_read_stats and disk_read_stats.get('mean', 0) > 100:  # > 100 MB/s sustained
            bottlenecks.append('disk_io_bound')
        
        return bottlenecks

class AutoTuner:
    """Automatic performance tuning system"""
    
    def __init__(self, analyzer: PerformanceAnalyzer):
        self.analyzer = analyzer
        self.tuning_history: List[TuningRecommendation] = []
        self.active_tunings: Dict[str, TuningRecommendation] = {}
        
        # Tuning strategies
        self.tuning_strategies = {
            'cpu_bound': self._tune_cpu_bound,
            'memory_bound': self._tune_memory_bound,
            'disk_io_bound': self._tune_disk_io,
            'high_latency': self._tune_latency,
            'low_throughput': self._tune_throughput
        }
        
        # Safety limits
        self.max_concurrent_tunings = 3
        self.min_tuning_interval = 300  # 5 minutes between tunings
        
    def generate_tuning_recommendations(self) -> List[TuningRecommendation]:
        """Generate automatic tuning recommendations"""
        analysis = self.analyzer.analyze_current_performance()
        recommendations = []
        
        current_time = time.time()
        
        # Check if we can make new recommendations
        if len(self.active_tunings) >= self.max_concurrent_tunings:
            logger.info("Maximum concurrent tunings reached, skipping new recommendations")
            return recommendations
        
        # Analyze bottlenecks and generate recommendations
        for bottleneck in analysis.get('bottlenecks', []):
            if bottleneck in self.tuning_strategies:
                # Check if we've recently tuned this area
                recent_tunings = [
                    t for t in self.tuning_history
                    if t.target_component == bottleneck and (current_time - t.timestamp) < self.min_tuning_interval
                ]
                
                if not recent_tunings:
                    recommendation = self.tuning_strategies[bottleneck](analysis)
                    if recommendation:
                        recommendations.append(recommendation)
        
        # Generate recommendations for specific metric issues
        for metric_name, metric_analysis in analysis.get('metric_analysis', {}).items():
            if metric_analysis.get('alert_level') in ['warning', 'critical']:
                recommendation = self._generate_metric_specific_recommendation(
                    metric_name, metric_analysis, analysis
                )
                if recommendation:
                    recommendations.append(recommendation)
        
        return recommendations
    
    def _tune_cpu_bound(self, analysis: Dict[str, Any]) -> Optional[TuningRecommendation]:
        """Generate CPU-bound tuning recommendation"""
        return TuningRecommendation(
            recommendation_id=f"cpu_tune_{int(time.time())}",
            action=TuningAction.INCREASE_PARALLELISM,
            target_component="cpu_bound",
            parameters={
                'parallel_workers': '+2',
                'thread_pool_size': '+4',
                'enable_cpu_affinity': True
            },
            confidence=0.8,
            expected_improvement=0.15,
            reasoning="High CPU usage detected, increase parallelism to distribute load",
            timestamp=time.time()
        )
    
    def _tune_memory_bound(self, analysis: Dict[str, Any]) -> Optional[TuningRecommendation]:
        """Generate memory-bound tuning recommendation"""
        return TuningRecommendation(
            recommendation_id=f"memory_tune_{int(time.time())}",
            action=TuningAction.OPTIMIZE_MEMORY,
            target_component="memory_bound",
            parameters={
                'enable_memory_pooling': True,
                'garbage_collection_frequency': 'increase',
                'cache_size_limit': '50%',
                'batch_size_reduction': 0.8
            },
            confidence=0.85,
            expected_improvement=0.20,
            reasoning="High memory usage detected, optimize memory allocation and caching",
            timestamp=time.time()
        )
    
    def _tune_disk_io(self, analysis: Dict[str, Any]) -> Optional[TuningRecommendation]:
        """Generate disk I/O tuning recommendation"""
        return TuningRecommendation(
            recommendation_id=f"disk_tune_{int(time.time())}",
            action=TuningAction.ADJUST_PARAMETERS,
            target_component="disk_io_bound",
            parameters={
                'enable_write_batching': True,
                'read_buffer_size': 'increase',
                'async_io': True,
                'compression': 'enable'
            },
            confidence=0.75,
            expected_improvement=0.25,
            reasoning="High disk I/O detected, optimize read/write patterns",
            timestamp=time.time()
        )
    
    def _tune_latency(self, analysis: Dict[str, Any]) -> Optional[TuningRecommendation]:
        """Generate latency tuning recommendation"""
        return TuningRecommendation(
            recommendation_id=f"latency_tune_{int(time.time())}",
            action=TuningAction.ADJUST_PARAMETERS,
            target_component="high_latency",
            parameters={
                'cache_warmup': True,
                'connection_pooling': True,
                'request_batching': True,
                'timeout_optimization': True
            },
            confidence=0.7,
            expected_improvement=0.30,
            reasoning="High latency detected, optimize request processing pipeline",
            timestamp=time.time()
        )
    
    def _tune_throughput(self, analysis: Dict[str, Any]) -> Optional[TuningRecommendation]:
        """Generate throughput tuning recommendation"""
        return TuningRecommendation(
            recommendation_id=f"throughput_tune_{int(time.time())}",
            action=TuningAction.SCALE_UP,
            target_component="low_throughput",
            parameters={
                'worker_processes': '+2',
                'queue_size': 'increase',
                'batch_processing': True,
                'pipeline_parallelism': True
            },
            confidence=0.8,
            expected_improvement=0.35,
            reasoning="Low throughput detected, scale up processing capacity",
            timestamp=time.time()
        )
    
    def _generate_metric_specific_recommendation(self, metric_name: str, metric_analysis: Dict[str, Any], 
                                               full_analysis: Dict[str, Any]) -> Optional[TuningRecommendation]:
        """Generate recommendation for specific metric issues"""
        
        current_value = metric_analysis.get('current_value', 0)
        alert_level = metric_analysis.get('alert_level')
        
        if metric_name == 'cache_hit_rate' and current_value < 0.7:
            return TuningRecommendation(
                recommendation_id=f"cache_tune_{int(time.time())}",
                action=TuningAction.ADJUST_PARAMETERS,
                target_component="cache_optimization",
                parameters={
                    'cache_size': 'increase',
                    'cache_policy': 'LRU_with_frequency',
                    'preload_patterns': True
                },
                confidence=0.9,
                expected_improvement=0.25,
                reasoning=f"Low cache hit rate ({current_value:.2f}), optimize caching strategy",
                timestamp=time.time()
            )
        
        elif metric_name == 'error_rate' and current_value > 0.01:
            return TuningRecommendation(
                recommendation_id=f"error_tune_{int(time.time())}",
                action=TuningAction.ADJUST_PARAMETERS,
                target_component="error_handling",
                parameters={
                    'retry_strategy': 'exponential_backoff',
                    'circuit_breaker': True,
                    'input_validation': 'strict',
                    'timeout_handling': 'improved'
                },
                confidence=0.85,
                expected_improvement=0.50,
                reasoning=f"High error rate ({current_value:.3f}), improve error handling",
                timestamp=time.time()
            )
        
        return None
    
    def apply_recommendation(self, recommendation: TuningRecommendation, 
                           apply_func: Optional[Callable] = None) -> bool:
        """Apply a tuning recommendation"""
        try:
            if apply_func:
                success = apply_func(recommendation)
            else:
                success = self._default_apply_recommendation(recommendation)
            
            if success:
                self.active_tunings[recommendation.recommendation_id] = recommendation
                self.tuning_history.append(recommendation)
                logger.info(f"Applied tuning recommendation: {recommendation.recommendation_id}")
                return True
            else:
                logger.warning(f"Failed to apply tuning recommendation: {recommendation.recommendation_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying tuning recommendation {recommendation.recommendation_id}: {e}")
            return False
    
    def _default_apply_recommendation(self, recommendation: TuningRecommendation) -> bool:
        """Default implementation for applying recommendations"""
        # This is a placeholder - in a real system, this would interface with
        # the actual system components to apply the tuning parameters
        
        logger.info(f"Simulating application of {recommendation.action.value} for {recommendation.target_component}")
        logger.info(f"Parameters: {recommendation.parameters}")
        
        # Simulate some processing time
        time.sleep(0.1)
        
        return True  # Assume success for simulation
    
    def get_tuning_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of applied tunings"""
        if not self.tuning_history:
            return {'status': 'no_tunings_applied'}
        
        # Group tunings by type
        tuning_groups = defaultdict(list)
        for tuning in self.tuning_history:
            tuning_groups[tuning.action.value].append(tuning)
        
        effectiveness = {
            'total_tunings': len(self.tuning_history),
            'tuning_types': {},
            'average_confidence': np.mean([t.confidence for t in self.tuning_history]),
            'average_expected_improvement': np.mean([t.expected_improvement for t in self.tuning_history])
        }
        
        for action_type, tunings in tuning_groups.items():
            effectiveness['tuning_types'][action_type] = {
                'count': len(tunings),
                'average_confidence': np.mean([t.confidence for t in tunings]),
                'average_expected_improvement': np.mean([t.expected_improvement for t in tunings]),
                'most_recent': max(tunings, key=lambda t: t.timestamp).timestamp
            }
        
        return effectiveness

class RealTimePerformanceMonitor:
    """Main real-time performance monitoring and auto-tuning system"""
    
    def __init__(self, auto_tune: bool = True, collection_interval: float = 1.0):
        self.auto_tune = auto_tune
        
        # Initialize components
        self.metric_collector = MetricCollector()
        self.metric_collector.collection_interval = collection_interval
        
        self.analyzer = PerformanceAnalyzer(self.metric_collector)
        self.auto_tuner = AutoTuner(self.analyzer) if auto_tune else None
        
        # Alert system
        self.alerts: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring state
        self.running = False
        self.monitor_thread = None
        self.monitor_interval = 10.0  # seconds
        
        # Performance profiles
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.metric_collector.start_collection()
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Real-time performance monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        
        self.metric_collector.stop_collection()
        
        logger.info("Real-time performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Analyze current performance
                analysis = self.analyzer.analyze_current_performance()
                
                # Generate alerts for critical issues
                self._process_alerts(analysis)
                
                # Generate auto-tuning recommendations if enabled
                if self.auto_tuner:
                    recommendations = self.auto_tuner.generate_tuning_recommendations()
                    
                    # Auto-apply high-confidence recommendations
                    for recommendation in recommendations:
                        if recommendation.confidence > 0.8:
                            self.auto_tuner.apply_recommendation(recommendation)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)
    
    def _process_alerts(self, analysis: Dict[str, Any]):
        """Process performance analysis and generate alerts"""
        for metric_name, metric_analysis in analysis.get('metric_analysis', {}).items():
            alert_level = metric_analysis.get('alert_level')
            
            if alert_level in ['warning', 'critical']:
                alert = PerformanceAlert(
                    alert_id=f"alert_{metric_name}_{int(time.time())}",
                    metric_name=metric_name,
                    level=AlertLevel(alert_level),
                    message=f"{metric_name} is {alert_level}: {metric_analysis.get('current_value', 'N/A')}",
                    threshold_value=self.analyzer.thresholds.get(metric_name, {}).get(alert_level, 0),
                    actual_value=metric_analysis.get('current_value', 0),
                    timestamp=time.time(),
                    context=metric_analysis
                )
                
                self.alerts.append(alert)
                self._notify_alert_callbacks(alert)
    
    def _notify_alert_callbacks(self, alert: PerformanceAlert):
        """Notify registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add alert notification callback"""
        self.alert_callbacks.append(callback)
    
    def add_custom_metric(self, name: str, collector_func: Callable[[], float]):
        """Add custom metric collector"""
        self.metric_collector.add_custom_collector(name, collector_func)
    
    def set_metric_threshold(self, metric_name: str, warning: float, critical: float):
        """Set alert thresholds for a metric"""
        self.analyzer.thresholds[metric_name] = {
            'warning': warning,
            'critical': critical
        }
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        analysis = self.analyzer.analyze_current_performance()
        
        status = {
            'monitoring_active': self.running,
            'auto_tuning_enabled': self.auto_tuner is not None,
            'overall_health': analysis.get('overall_health', 'unknown'),
            'active_alerts': len([a for a in self.alerts if time.time() - a.timestamp < 300]),  # Last 5 minutes
            'recent_tunings': len(self.auto_tuner.active_tunings) if self.auto_tuner else 0,
            'metrics_collected': len(self.metric_collector.metrics_buffer),
            'analysis': analysis
        }
        
        if self.auto_tuner:
            status['tuning_effectiveness'] = self.auto_tuner.get_tuning_effectiveness()
        
        return status
    
    def create_performance_profile(self, profile_name: str) -> PerformanceProfile:
        """Create performance profile snapshot"""
        analysis = self.analyzer.analyze_current_performance()
        
        # Gather metric summaries
        metrics_summary = {}
        for metric_name in self.metric_collector.metrics_buffer.keys():
            stats = self.metric_collector.get_metric_statistics(metric_name, window_seconds=300)  # 5 minutes
            if stats:
                metrics_summary[metric_name] = {
                    'min': stats['min'],
                    'max': stats['max'],
                    'avg': stats['mean'],
                    'std': stats['std']
                }
        
        profile = PerformanceProfile(
            profile_name=profile_name,
            metrics_summary=metrics_summary,
            performance_characteristics=analysis.get('performance_patterns', {}),
            bottlenecks=analysis.get('bottlenecks', []),
            optimization_opportunities=[],  # TODO: Implement optimization opportunity detection
            recommended_configuration={},  # TODO: Generate configuration recommendations
            creation_timestamp=time.time()
        )
        
        self.performance_profiles[profile_name] = profile
        return profile
    
    def export_metrics(self, metric_names: Optional[List[str]] = None, 
                      window_seconds: int = 3600) -> Dict[str, List[Dict[str, Any]]]:
        """Export metrics data"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        exported_data = {}
        
        target_metrics = metric_names or list(self.metric_collector.metrics_buffer.keys())
        
        for metric_name in target_metrics:
            if metric_name in self.metric_collector.metrics_buffer:
                # Filter metrics in time window
                recent_metrics = [
                    asdict(m) for m in self.metric_collector.metrics_buffer[metric_name]
                    if m.timestamp >= cutoff_time
                ]
                exported_data[metric_name] = recent_metrics
        
        return exported_data

# Example usage and integration
def create_evolution_performance_monitor() -> RealTimePerformanceMonitor:
    """Create performance monitor configured for evolution systems"""
    
    monitor = RealTimePerformanceMonitor(auto_tune=True, collection_interval=0.5)
    
    # Set evolution-specific thresholds
    monitor.set_metric_threshold('fitness_score', warning=-0.5, critical=-1.0)
    monitor.set_metric_threshold('convergence_rate', warning=0.001, critical=0.0001)
    monitor.set_metric_threshold('diversity_metric', warning=0.1, critical=0.05)
    monitor.set_metric_threshold('evolution_efficiency', warning=0.3, critical=0.1)
    
    # Add evolution-specific custom metrics
    def fitness_variance_collector():
        # Placeholder - would calculate from actual population
        return np.random.uniform(0.1, 0.5)
    
    def population_diversity_collector():
        # Placeholder - would calculate from actual population
        return np.random.uniform(0.2, 0.8)
    
    def convergence_rate_collector():
        # Placeholder - would calculate from fitness history
        return np.random.uniform(0.001, 0.01)
    
    monitor.add_custom_metric('fitness_variance', fitness_variance_collector)
    monitor.add_custom_metric('population_diversity', population_diversity_collector)
    monitor.add_custom_metric('convergence_rate', convergence_rate_collector)
    
    # Add alert callback
    def evolution_alert_handler(alert: PerformanceAlert):
        if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            print(f"CRITICAL EVOLUTION ALERT: {alert.message}")
    
    monitor.add_alert_callback(evolution_alert_handler)
    
    return monitor

# Testing and example usage
if __name__ == "__main__":
    # Create and test performance monitor
    monitor = create_evolution_performance_monitor()
    
    print("Starting performance monitoring test...")
    monitor.start_monitoring()
    
    # Simulate some workload and monitoring
    try:
        for i in range(30):
            time.sleep(2)
            
            if i % 10 == 0:
                status = monitor.get_current_status()
                print(f"\nMonitoring Status (iteration {i}):")
                print(f"Overall Health: {status['overall_health']}")
                print(f"Active Alerts: {status['active_alerts']}")
                print(f"Metrics Collected: {status['metrics_collected']}")
                
                if i == 20:
                    # Create performance profile
                    profile = monitor.create_performance_profile("test_profile")
                    print(f"Created performance profile: {profile.profile_name}")
    
    finally:
        monitor.stop_monitoring()
        print("Performance monitoring test completed")