"""Unit tests for monitoring utilities."""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from src.self_evolving_moe.utils.monitoring import (
    SystemMetrics,
    EvolutionMetrics,
    PerformanceMetrics,
    SystemMonitor,
    PerformanceTracker
)


class TestSystemMetrics:
    """Test SystemMetrics dataclass."""

    @pytest.mark.unit
    def test_system_metrics_creation(self):
        """Test SystemMetrics creation."""
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=45.5,
            memory_percent=60.2,
            memory_used_mb=2048.0,
            memory_available_mb=6144.0,
            gpu_memory_used_mb=512.0,
            gpu_memory_total_mb=8192.0,
            gpu_utilization=25.5
        )
        
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_percent == 60.2
        assert metrics.gpu_memory_used_mb == 512.0

    @pytest.mark.unit
    def test_system_metrics_defaults(self):
        """Test SystemMetrics with default values."""
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=70.0,
            memory_used_mb=1024.0,
            memory_available_mb=3072.0
        )
        
        # Default GPU values should be 0
        assert metrics.gpu_memory_used_mb == 0.0
        assert metrics.gpu_memory_total_mb == 0.0
        assert metrics.gpu_utilization == 0.0


class TestEvolutionMetrics:
    """Test EvolutionMetrics dataclass."""

    @pytest.mark.unit
    def test_evolution_metrics_creation(self):
        """Test EvolutionMetrics creation."""
        metrics = EvolutionMetrics(
            timestamp=time.time(),
            generation=10,
            best_fitness=0.85,
            avg_fitness=0.72,
            population_diversity=0.6,
            mutation_rate=0.1,
            convergence_rate=0.05,
            active_experts=16,
            topology_sparsity=0.9
        )
        
        assert metrics.generation == 10
        assert metrics.best_fitness == 0.85
        assert metrics.topology_sparsity == 0.9


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    @pytest.mark.unit
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation."""
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            accuracy=0.92,
            latency_ms=15.5,
            throughput_samples_per_sec=1000.0,
            memory_usage_mb=512.0,
            flops_per_sample=1e9,
            energy_consumption_j=25.5
        )
        
        assert metrics.accuracy == 0.92
        assert metrics.latency_ms == 15.5
        assert metrics.flops_per_sample == 1e9


class TestSystemMonitor:
    """Test SystemMonitor functionality."""

    @pytest.mark.unit
    def test_system_monitor_creation(self):
        """Test SystemMonitor creation with default parameters."""
        monitor = SystemMonitor()
        
        assert monitor.sample_interval == 1.0
        assert monitor.history_size == 1000
        assert not monitor.is_monitoring
        assert len(monitor.metrics_history) == 0
        assert len(monitor.alert_callbacks) == 0

    @pytest.mark.unit
    def test_system_monitor_custom_params(self):
        """Test SystemMonitor creation with custom parameters."""
        thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 90.0
        }
        
        monitor = SystemMonitor(
            sample_interval=0.5,
            history_size=500,
            enable_gpu_monitoring=False,
            alert_thresholds=thresholds
        )
        
        assert monitor.sample_interval == 0.5
        assert monitor.history_size == 500
        assert not monitor.enable_gpu_monitoring
        assert monitor.alert_thresholds['cpu_percent'] == 80.0

    @pytest.mark.unit
    @patch('src.self_evolving_moe.utils.monitoring.psutil')
    def test_get_system_metrics(self, mock_psutil):
        """Test system metrics collection."""
        # Mock psutil calls
        mock_psutil.cpu_percent.return_value = 45.5
        
        mock_memory = Mock()
        mock_memory.percent = 60.2
        mock_memory.used = 2048 * 1024 * 1024  # 2048 MB in bytes
        mock_memory.available = 6144 * 1024 * 1024  # 6144 MB in bytes
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk_io = Mock()
        mock_disk_io.read_bytes = 1024 * 1024
        mock_disk_io.write_bytes = 512 * 1024
        mock_psutil.disk_io_counters.return_value = mock_disk_io
        
        mock_network_io = Mock()
        mock_network_io.bytes_sent = 2048 * 1024
        mock_network_io.bytes_recv = 4096 * 1024
        mock_psutil.net_io_counters.return_value = mock_network_io
        
        monitor = SystemMonitor(enable_gpu_monitoring=False)
        metrics = monitor._get_system_metrics()
        
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_percent == 60.2
        assert metrics.memory_used_mb == 2048.0
        assert metrics.memory_available_mb == 6144.0

    @pytest.mark.unit
    def test_alert_callbacks(self):
        """Test alert callback functionality."""
        monitor = SystemMonitor(alert_thresholds={'cpu_percent': 50.0})
        
        # Add mock callback
        callback_mock = Mock()
        monitor.add_alert_callback(callback_mock)
        
        # Create high CPU metrics to trigger alert
        high_cpu_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=75.0,  # Above threshold
            memory_percent=40.0,
            memory_used_mb=1024.0,
            memory_available_mb=2048.0
        )
        
        monitor._check_alerts(high_cpu_metrics)
        
        # Callback should have been called
        callback_mock.assert_called()

    @pytest.mark.unit
    def test_alert_thresholds(self):
        """Test alert threshold checking."""
        monitor = SystemMonitor(alert_thresholds={
            'cpu_percent': 80.0,
            'memory_percent': 85.0
        })
        
        callback_mock = Mock()
        monitor.add_alert_callback(callback_mock)
        
        # Create metrics that exceed thresholds
        alert_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=90.0,  # Above 80% threshold
            memory_percent=90.0,  # Above 85% threshold
            memory_used_mb=1024.0,
            memory_available_mb=512.0
        )
        
        monitor._check_alerts(alert_metrics)
        
        # Should be called twice (CPU and memory alerts)
        assert callback_mock.call_count == 2

    @pytest.mark.unit
    def test_monitoring_start_stop(self):
        """Test monitoring start and stop functionality."""
        monitor = SystemMonitor(sample_interval=0.1)
        
        assert not monitor.is_monitoring
        assert monitor.monitor_thread is None
        
        monitor.start_monitoring()
        
        assert monitor.is_monitoring
        assert monitor.monitor_thread is not None
        assert monitor.monitor_thread.is_alive()
        
        # Let it run briefly
        time.sleep(0.2)
        
        monitor.stop_monitoring()
        
        assert not monitor.is_monitoring
        # Thread should stop
        time.sleep(0.1)
        assert not monitor.monitor_thread.is_alive()

    @pytest.mark.unit
    def test_metrics_history(self):
        """Test metrics history functionality."""
        monitor = SystemMonitor(history_size=5)
        
        # Manually add some metrics
        for i in range(7):  # More than history_size
            metrics = SystemMetrics(
                timestamp=time.time() + i,
                cpu_percent=float(i * 10),
                memory_percent=50.0,
                memory_used_mb=1024.0,
                memory_available_mb=2048.0
            )
            monitor.metrics_history.append(metrics)
        
        # Should only keep last 5 (history_size)
        assert len(monitor.metrics_history) == 5
        assert monitor.metrics_history[0].cpu_percent == 20.0  # Index 2
        assert monitor.metrics_history[-1].cpu_percent == 60.0  # Index 6

    @pytest.mark.unit
    def test_get_metrics_history_with_duration(self):
        """Test getting metrics history with time filter."""
        monitor = SystemMonitor()
        current_time = time.time()
        
        # Add metrics with different timestamps
        old_metrics = SystemMetrics(
            timestamp=current_time - 600,  # 10 minutes ago
            cpu_percent=30.0,
            memory_percent=50.0,
            memory_used_mb=1024.0,
            memory_available_mb=2048.0
        )
        
        recent_metrics = SystemMetrics(
            timestamp=current_time - 60,  # 1 minute ago
            cpu_percent=40.0,
            memory_percent=60.0,
            memory_used_mb=1024.0,
            memory_available_mb=2048.0
        )
        
        monitor.metrics_history.extend([old_metrics, recent_metrics])
        
        # Get history for last 5 minutes
        recent_history = monitor.get_metrics_history(duration_minutes=5.0)
        
        # Should only include recent metrics
        assert len(recent_history) == 1
        assert recent_history[0].cpu_percent == 40.0

    @pytest.mark.unit
    def test_resource_summary(self):
        """Test resource usage summary generation."""
        monitor = SystemMonitor()
        current_time = time.time()
        
        # Add sample metrics
        for i in range(5):
            metrics = SystemMetrics(
                timestamp=current_time - (i * 60),  # Spaced 1 minute apart
                cpu_percent=float(50 + i * 5),
                memory_percent=float(60 + i * 2),
                memory_used_mb=1024.0 + (i * 100),
                memory_available_mb=2048.0,
                gpu_memory_used_mb=512.0 + (i * 50),
                gpu_memory_total_mb=8192.0
            )
            monitor.metrics_history.append(metrics)
        
        summary = monitor.get_resource_summary(duration_minutes=10.0)
        
        assert summary['duration_minutes'] == 10.0
        assert summary['sample_count'] == 5
        assert 'cpu' in summary
        assert 'memory' in summary
        assert summary['cpu']['min'] == 50.0
        assert summary['cpu']['max'] == 70.0


class TestPerformanceTracker:
    """Test PerformanceTracker functionality."""

    @pytest.mark.unit
    def test_performance_tracker_creation(self):
        """Test PerformanceTracker creation."""
        tracker = PerformanceTracker(history_size=500)
        
        assert tracker.history_size == 500
        assert len(tracker.evolution_history) == 0
        assert len(tracker.performance_history) == 0
        assert tracker.current_generation == 0
        assert tracker.best_performance is None

    @pytest.mark.unit
    def test_record_evolution_metrics(self):
        """Test recording evolution metrics."""
        tracker = PerformanceTracker()
        
        tracker.record_evolution_metrics(
            generation=1,
            best_fitness=0.8,
            avg_fitness=0.7,
            population_diversity=0.6,
            mutation_rate=0.1,
            convergence_rate=0.05,
            active_experts=16,
            topology_sparsity=0.9
        )
        
        assert len(tracker.evolution_history) == 1
        metrics = tracker.evolution_history[0]
        assert metrics.generation == 1
        assert metrics.best_fitness == 0.8
        assert metrics.topology_sparsity == 0.9

    @pytest.mark.unit
    def test_record_performance_metrics(self):
        """Test recording performance metrics."""
        tracker = PerformanceTracker()
        
        tracker.record_performance_metrics(
            accuracy=0.85,
            latency_ms=12.5,
            throughput_samples_per_sec=800.0,
            memory_usage_mb=1024.0
        )
        
        assert len(tracker.performance_history) == 1
        metrics = tracker.performance_history[0]
        assert metrics.accuracy == 0.85
        assert metrics.latency_ms == 12.5
        
        # Should be set as best performance
        assert tracker.best_performance == metrics

    @pytest.mark.unit
    def test_best_performance_tracking(self):
        """Test best performance tracking."""
        tracker = PerformanceTracker()
        
        # Record initial performance
        tracker.record_performance_metrics(
            accuracy=0.8,
            latency_ms=15.0,
            throughput_samples_per_sec=600.0,
            memory_usage_mb=1024.0
        )
        
        # Record better performance
        tracker.record_performance_metrics(
            accuracy=0.9,  # Better accuracy
            latency_ms=12.0,
            throughput_samples_per_sec=800.0,
            memory_usage_mb=1024.0
        )
        
        # Record worse performance
        tracker.record_performance_metrics(
            accuracy=0.75,  # Worse accuracy
            latency_ms=10.0,
            throughput_samples_per_sec=900.0,
            memory_usage_mb=1024.0
        )
        
        # Best performance should be the one with highest accuracy
        assert tracker.best_performance.accuracy == 0.9

    @pytest.mark.unit
    def test_evolution_summary(self):
        """Test evolution summary generation."""
        tracker = PerformanceTracker()
        
        # Record multiple generations
        for gen in range(1, 6):
            tracker.record_evolution_metrics(
                generation=gen,
                best_fitness=0.5 + (gen * 0.05),  # Improving fitness
                avg_fitness=0.4 + (gen * 0.05),
                population_diversity=0.8 - (gen * 0.02),  # Decreasing diversity
                mutation_rate=0.1,
                convergence_rate=0.02 * gen,
                active_experts=16,
                topology_sparsity=0.85 + (gen * 0.02)
            )
        
        summary = tracker.get_evolution_summary()
        
        assert summary['total_generations'] == 5
        assert summary['final_best_fitness'] == 0.75  # 0.5 + (5 * 0.05)
        assert summary['fitness_improvement'] == 0.2   # 0.75 - 0.55
        assert summary['final_sparsity'] == 0.95       # 0.85 + (5 * 0.02)
        assert summary['convergence_generation'] is not None

    @pytest.mark.unit
    def test_performance_summary(self):
        """Test performance summary generation."""
        tracker = PerformanceTracker()
        
        # Record multiple performance measurements
        accuracies = [0.8, 0.85, 0.9, 0.87, 0.92]
        latencies = [15.0, 12.5, 10.0, 11.2, 9.8]
        
        for acc, lat in zip(accuracies, latencies):
            tracker.record_performance_metrics(
                accuracy=acc,
                latency_ms=lat,
                throughput_samples_per_sec=800.0,
                memory_usage_mb=1024.0
            )
        
        summary = tracker.get_performance_summary()
        
        assert summary['sample_count'] == 5
        assert summary['accuracy']['max'] == 0.92
        assert summary['accuracy']['min'] == 0.8
        assert summary['latency_ms']['min'] == 9.8
        assert summary['latency_ms']['max'] == 15.0
        assert summary['best_performance']['accuracy'] == 0.92

    @pytest.mark.unit
    def test_efficiency_score(self):
        """Test efficiency score calculation."""
        tracker = PerformanceTracker()
        
        tracker.record_performance_metrics(
            accuracy=0.9,
            latency_ms=10.0,
            throughput_samples_per_sec=800.0,
            memory_usage_mb=1024.0
        )
        
        summary = tracker.get_performance_summary()
        efficiency = summary['efficiency_score']
        
        # Efficiency = (accuracy * throughput) / latency
        expected_efficiency = (0.9 * 800.0) / 10.0
        assert abs(efficiency - expected_efficiency) < 1e-6

    @pytest.mark.unit
    def test_export_metrics_json(self, temp_dir):
        """Test exporting metrics to JSON."""
        tracker = PerformanceTracker()
        
        # Add some data
        tracker.record_evolution_metrics(1, 0.8, 0.7, 0.6, 0.1, 0.05, 16, 0.9)
        tracker.record_performance_metrics(0.85, 12.5, 800.0, 1024.0)
        
        export_path = temp_dir / "metrics.json"
        tracker.export_metrics(export_path, format="json")
        
        assert export_path.exists()
        
        with open(export_path) as f:
            data = json.load(f)
        
        assert 'evolution_history' in data
        assert 'performance_history' in data
        assert 'evolution_summary' in data
        assert 'performance_summary' in data
        assert len(data['evolution_history']) == 1
        assert len(data['performance_history']) == 1

    @pytest.mark.unit
    def test_pareto_frontier(self):
        """Test Pareto frontier calculation."""
        tracker = PerformanceTracker()
        
        # Add performance points with trade-offs
        performance_points = [
            (0.8, 20.0),   # Good accuracy, slow
            (0.85, 15.0),  # Better accuracy, faster
            (0.9, 12.0),   # Best accuracy, fast
            (0.75, 8.0),   # Lower accuracy, fastest
            (0.82, 18.0),  # Dominated point
        ]
        
        for accuracy, latency in performance_points:
            tracker.record_performance_metrics(
                accuracy=accuracy,
                latency_ms=latency,
                throughput_samples_per_sec=800.0,
                memory_usage_mb=1024.0
            )
        
        pareto_frontier = tracker.get_pareto_frontier(
            x_metric='latency_ms', 
            y_metric='accuracy'
        )
        
        # Pareto frontier should exclude dominated points
        assert len(pareto_frontier) < len(performance_points)
        
        # Verify Pareto optimality (for each point, no other point is better in both dimensions)
        for point in pareto_frontier:
            for other in tracker.performance_history:
                if other != point:
                    # No point should be better in both latency (lower) and accuracy (higher)
                    better_latency = other.latency_ms < point.latency_ms
                    better_accuracy = other.accuracy > point.accuracy
                    assert not (better_latency and better_accuracy)

    @pytest.mark.unit
    def test_generation_tracking(self):
        """Test generation tracking functionality."""
        tracker = PerformanceTracker()
        
        # Start generation
        tracker.start_generation(5)
        
        assert tracker.current_generation == 5
        assert tracker.generation_start_time is not None
        
        # Record metrics for this generation
        tracker.record_evolution_metrics(5, 0.85, 0.8, 0.7, 0.1, 0.02, 20, 0.88)
        
        # Check generation metrics
        assert 5 in tracker.generation_metrics
        assert len(tracker.generation_metrics[5]) == 1

    @pytest.mark.unit
    def test_invalid_export_format(self, temp_dir):
        """Test handling of invalid export format."""
        tracker = PerformanceTracker()
        export_path = temp_dir / "metrics.invalid"
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            tracker.export_metrics(export_path, format="invalid_format")