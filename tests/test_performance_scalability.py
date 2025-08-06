#!/usr/bin/env python3
"""
Performance and scalability tests for Self-Evolving MoE-Router.

This module tests system performance under various loads, memory constraints,
and scaling scenarios to ensure production readiness.
"""

import pytest
import torch
import numpy as np
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from self_evolving_moe.evolution.router import EvolvingMoERouter, EvolutionConfig
from self_evolving_moe.experts.pool import ExpertPool
from self_evolving_moe.experts.slimmable import SlimmableMoE
from self_evolving_moe.routing.topology import TopologyGenome
from self_evolving_moe.utils.monitoring import SystemMonitor, PerformanceTracker


class TestEvolutionPerformance:
    """Test evolution algorithm performance."""
    
    def test_evolution_speed_scaling(self):
        """Test evolution speed with different population sizes."""
        expert_pool = ExpertPool(num_experts=8, expert_dim=64, expert_type="mlp")
        
        population_sizes = [10, 25, 50, 100]
        times = []
        
        for pop_size in population_sizes:
            config = EvolutionConfig(
                population_size=pop_size,
                generations=3,  # Small number for speed
                mutation_rate=0.1
            )
            
            evolver = EvolvingMoERouter(expert_pool=expert_pool, config=config)
            
            # Mock model and data
            model = torch.nn.Linear(64, 10)
            data = [(torch.randn(16, 64), torch.randint(0, 10, (16,))) for _ in range(5)]
            
            class MockDataLoader:
                def __init__(self, data):
                    self.data = data
                def __iter__(self):
                    return iter(self.data)
                def __len__(self):
                    return len(self.data)
            
            train_loader = MockDataLoader(data)
            val_loader = MockDataLoader(data)
            
            start_time = time.time()
            evolver.evolve(model, train_loader, val_loader, generations=3)
            end_time = time.time()
            
            evolution_time = end_time - start_time
            times.append(evolution_time)
            
            print(f"Population {pop_size}: {evolution_time:.2f}s")
        
        # Evolution time should scale reasonably (not exponentially)
        # Allow for some variance but check it's not extremely bad scaling
        time_per_individual = [t/p for t, p in zip(times, population_sizes)]
        
        # Time per individual shouldn't increase too much with population size
        max_time_per_individual = max(time_per_individual)
        min_time_per_individual = min(time_per_individual)
        
        scaling_ratio = max_time_per_individual / min_time_per_individual
        assert scaling_ratio < 3.0, f"Poor scaling: {scaling_ratio:.2f}x slower per individual"
    
    def test_memory_usage_scaling(self):
        """Test memory usage with different model sizes."""
        process = psutil.Process()
        
        expert_dims = [32, 64, 128, 256]
        memory_usages = []
        
        for expert_dim in expert_dims:
            # Force garbage collection
            import gc
            gc.collect()
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            expert_pool = ExpertPool(
                num_experts=16,
                expert_dim=expert_dim,
                expert_type="mlp"
            )
            
            evolver = EvolvingMoERouter(
                expert_pool=expert_pool,
                config=EvolutionConfig(population_size=20)
            )
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = current_memory - initial_memory
            memory_usages.append(memory_used)
            
            print(f"Expert dim {expert_dim}: {memory_used:.1f}MB")
            
            # Clean up
            del evolver
            del expert_pool
            gc.collect()
        
        # Memory usage should scale reasonably with model size
        # Should be roughly quadratic with dimension (due to weight matrices)
        memory_ratios = [memory_usages[i] / memory_usages[0] for i in range(len(memory_usages))]
        dim_ratios = [expert_dims[i] / expert_dims[0] for i in range(len(expert_dims))]
        
        # Memory should not grow faster than O(dim^3)
        for i in range(1, len(memory_ratios)):
            expected_max_ratio = dim_ratios[i] ** 3
            assert memory_ratios[i] < expected_max_ratio * 2, f"Memory scaling too aggressive: {memory_ratios[i]:.2f}x"
    
    def test_topology_operations_performance(self):
        """Test performance of topology operations."""
        sizes = [(64, 8), (128, 16), (256, 32), (512, 64)]
        
        for num_tokens, num_experts in sizes:
            topology = TopologyGenome(
                num_tokens=num_tokens,
                num_experts=num_experts,
                sparsity=0.8
            )
            
            # Test mutation performance
            start_time = time.time()
            for _ in range(100):
                mutated = topology.mutate(0.1)
            mutation_time = time.time() - start_time
            
            # Test crossover performance
            topology2 = TopologyGenome(
                num_tokens=num_tokens,
                num_experts=num_experts,
                sparsity=0.8
            )
            
            start_time = time.time()
            for _ in range(100):
                child = topology.crossover(topology2)
            crossover_time = time.time() - start_time
            
            print(f"Size {num_tokens}x{num_experts}: mutation {mutation_time:.3f}s, crossover {crossover_time:.3f}s")
            
            # Operations should complete quickly
            assert mutation_time < 5.0, f"Mutation too slow: {mutation_time:.3f}s"
            assert crossover_time < 5.0, f"Crossover too slow: {crossover_time:.3f}s"
    
    def test_parallel_evaluation(self):
        """Test parallel fitness evaluation performance."""
        expert_pool = ExpertPool(num_experts=8, expert_dim=64, expert_type="mlp")
        config = EvolutionConfig(population_size=20, generations=1)
        
        evolver = EvolvingMoERouter(expert_pool=expert_pool, config=config)
        
        # Mock model and data
        model = torch.nn.Linear(64, 10)
        data = [(torch.randn(16, 64), torch.randint(0, 10, (16,))) for _ in range(10)]
        
        class MockDataLoader:
            def __init__(self, data):
                self.data = data
            def __iter__(self):
                return iter(self.data)
            def __len__(self):
                return len(self.data)
        
        data_loader = MockDataLoader(data)
        
        # Sequential evaluation
        start_time = time.time()
        evolver._evaluate_population(model, data_loader, data_loader)
        sequential_time = time.time() - start_time
        
        print(f"Sequential evaluation: {sequential_time:.3f}s")
        
        # For now, we don't have parallel evaluation implemented,
        # but we can test that evaluation scales reasonably
        assert sequential_time < 30.0, f"Fitness evaluation too slow: {sequential_time:.3f}s"


class TestSlimmableMoEPerformance:
    """Test SlimmableMoE performance characteristics."""
    
    def test_width_adaptation_speed(self):
        """Test speed of width adaptation."""
        expert_pool = ExpertPool(num_experts=8, expert_dim=256, expert_type="mlp")
        slimmable_moe = SlimmableMoE(expert_pool)
        
        input_tensor = torch.randn(32, 128, 256)
        widths = [64, 128, 192, 256]
        
        adaptation_times = []
        
        for width in widths:
            start_time = time.time()
            
            # Multiple forward passes to measure sustained performance
            for _ in range(10):
                output = slimmable_moe(input_tensor, width=width)
            
            adaptation_time = (time.time() - start_time) / 10
            adaptation_times.append(adaptation_time)
            
            print(f"Width {width}: {adaptation_time:.4f}s per forward pass")
        
        # Width adaptation should not add significant overhead
        # Larger widths should be proportionally slower, but not drastically
        relative_times = [t / adaptation_times[0] for t in adaptation_times]
        relative_widths = [w / widths[0] for w in widths]
        
        for i in range(len(relative_times)):
            # Time should scale no worse than quadratically with width
            max_expected_time = relative_widths[i] ** 2
            assert relative_times[i] < max_expected_time * 2, f"Width {widths[i]} scaling too poor"
    
    def test_memory_efficiency_by_width(self):
        """Test memory efficiency at different widths."""
        expert_pool = ExpertPool(num_experts=8, expert_dim=256, expert_type="mlp")
        slimmable_moe = SlimmableMoE(expert_pool)
        
        input_tensor = torch.randn(16, 64, 256)
        widths = [64, 128, 192, 256]
        
        memory_usages = []
        
        for width in widths:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()
            else:
                process = psutil.Process()
                initial_memory = process.memory_info().rss
            
            # Forward pass
            output = slimmable_moe(input_tensor, width=width)
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - initial_memory
                torch.cuda.reset_peak_memory_stats()
            else:
                current_memory = process.memory_info().rss
                memory_used = current_memory - initial_memory
            
            memory_usages.append(memory_used)
            print(f"Width {width}: {memory_used / 1024 / 1024:.1f}MB")
        
        # Memory usage should scale with width
        # Smaller widths should use less memory
        assert memory_usages[0] < memory_usages[-1], "Smaller widths should use less memory"
        
        # Memory scaling should be reasonable (not exponentially bad)
        memory_ratios = [m / memory_usages[0] for m in memory_usages]
        width_ratios = [w / widths[0] for w in widths]
        
        for i in range(len(memory_ratios)):
            # Memory should scale no worse than quadratically
            max_expected_ratio = width_ratios[i] ** 2
            assert memory_ratios[i] < max_expected_ratio * 3, f"Memory scaling too poor for width {widths[i]}"


class TestConcurrencyAndThreadSafety:
    """Test concurrent usage and thread safety."""
    
    def test_concurrent_evolution(self):
        """Test multiple evolution processes running concurrently."""
        def run_evolution(expert_dim, population_size):
            expert_pool = ExpertPool(
                num_experts=4,
                expert_dim=expert_dim,
                expert_type="mlp"
            )
            
            config = EvolutionConfig(
                population_size=population_size,
                generations=3
            )
            
            evolver = EvolvingMoERouter(expert_pool=expert_pool, config=config)
            
            # Mock simple evolution
            model = torch.nn.Linear(expert_dim, 5)
            data = [(torch.randn(8, expert_dim), torch.randint(0, 5, (8,))) for _ in range(3)]
            
            class MockDataLoader:
                def __init__(self, data):
                    self.data = data
                def __iter__(self):
                    return iter(self.data)
                def __len__(self):
                    return len(self.data)
            
            data_loader = MockDataLoader(data)
            
            best_topology = evolver.evolve(model, data_loader, data_loader)
            return best_topology is not None
        
        # Run multiple evolution processes concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            for i in range(3):
                future = executor.submit(run_evolution, 32 + i * 16, 5 + i * 2)
                futures.append(future)
            
            # All should complete successfully
            results = [future.result(timeout=60) for future in futures]
            assert all(results), "Some concurrent evolution processes failed"
    
    def test_system_monitor_thread_safety(self):
        """Test system monitor thread safety."""
        monitor = SystemMonitor(sample_interval=0.1, history_size=50)
        
        # Start monitoring
        monitor.start_monitoring()
        
        try:
            # Access metrics from multiple threads
            def access_metrics():
                for _ in range(20):
                    current = monitor.get_current_metrics()
                    history = monitor.get_metrics_history(duration_minutes=0.1)
                    summary = monitor.get_resource_summary(duration_minutes=0.1)
                    time.sleep(0.01)
                return True
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(access_metrics) for _ in range(3)]
                results = [future.result(timeout=10) for future in futures]
                assert all(results), "Thread safety test failed"
        
        finally:
            monitor.stop_monitoring()
    
    def test_performance_tracker_thread_safety(self):
        """Test performance tracker thread safety."""
        tracker = PerformanceTracker()
        
        def record_metrics(thread_id):
            for i in range(50):
                tracker.record_evolution_metrics(
                    generation=i,
                    best_fitness=0.5 + i * 0.01,
                    avg_fitness=0.4 + i * 0.005,
                    population_diversity=0.3,
                    mutation_rate=0.1,
                    convergence_rate=0.02,
                    active_experts=8,
                    topology_sparsity=0.8
                )
                
                tracker.record_performance_metrics(
                    accuracy=0.7 + i * 0.002,
                    latency_ms=10.0 + thread_id,
                    throughput_samples_per_sec=100 - i,
                    memory_usage_mb=500 + i * 2
                )
                
                if i % 10 == 0:
                    summary = tracker.get_evolution_summary()
                    perf_summary = tracker.get_performance_summary()
            
            return True
        
        # Record metrics from multiple threads
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(record_metrics, i) for i in range(3)]
            results = [future.result(timeout=30) for future in futures]
            assert all(results), "Performance tracker thread safety failed"
        
        # Check final state is consistent
        evolution_summary = tracker.get_evolution_summary()
        performance_summary = tracker.get_performance_summary()
        
        assert evolution_summary['total_generations'] > 0
        assert performance_summary['sample_count'] > 0


class TestStressAndLoadTesting:
    """Stress and load testing."""
    
    def test_large_population_evolution(self):
        """Test evolution with large population size."""
        expert_pool = ExpertPool(num_experts=16, expert_dim=128, expert_type="mlp")
        
        config = EvolutionConfig(
            population_size=200,  # Large population
            generations=5,
            mutation_rate=0.1
        )
        
        evolver = EvolvingMoERouter(expert_pool=expert_pool, config=config)
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Mock evolution with limited data to focus on population handling
        model = torch.nn.Linear(128, 10)
        data = [(torch.randn(16, 128), torch.randint(0, 10, (16,))) for _ in range(3)]
        
        class MockDataLoader:
            def __init__(self, data):
                self.data = data
            def __iter__(self):
                return iter(self.data)
            def __len__(self):
                return len(self.data)
        
        data_loader = MockDataLoader(data)
        
        start_time = time.time()
        best_topology = evolver.evolve(model, data_loader, data_loader, generations=3)
        evolution_time = time.time() - start_time
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory
        
        print(f"Large population evolution: {evolution_time:.2f}s, {memory_used:.1f}MB")
        
        # Should complete without crashing
        assert best_topology is not None
        assert evolution_time < 300, f"Evolution too slow: {evolution_time:.2f}s"
        assert memory_used < 2000, f"Memory usage too high: {memory_used:.1f}MB"
    
    def test_many_experts_performance(self):
        """Test performance with many experts."""
        expert_counts = [8, 16, 32, 64]
        times = []
        
        for num_experts in expert_counts:
            expert_pool = ExpertPool(
                num_experts=num_experts,
                expert_dim=64,
                expert_type="mlp"
            )
            
            config = EvolutionConfig(population_size=10, generations=2)
            evolver = EvolvingMoERouter(expert_pool=expert_pool, config=config)
            
            # Time a single generation evaluation
            model = torch.nn.Linear(64, 10)
            data = [(torch.randn(16, 64), torch.randint(0, 10, (16,))) for _ in range(5)]
            
            class MockDataLoader:
                def __init__(self, data):
                    self.data = data
                def __iter__(self):
                    return iter(self.data)
                def __len__(self):
                    return len(self.data)
            
            data_loader = MockDataLoader(data)
            
            start_time = time.time()
            evolver._evaluate_population(model, data_loader, data_loader)
            evaluation_time = time.time() - start_time
            times.append(evaluation_time)
            
            print(f"Experts {num_experts}: {evaluation_time:.3f}s")
        
        # Performance should scale reasonably with number of experts
        time_ratios = [t / times[0] for t in times]
        expert_ratios = [n / expert_counts[0] for n in expert_counts]
        
        for i in range(len(time_ratios)):
            # Time should scale no worse than quadratically with expert count
            max_expected_ratio = expert_ratios[i] ** 2
            assert time_ratios[i] < max_expected_ratio * 2, f"Poor scaling with {expert_counts[i]} experts"
    
    def test_long_running_stability(self):
        """Test stability during long-running evolution."""
        expert_pool = ExpertPool(num_experts=8, expert_dim=64, expert_type="mlp")
        config = EvolutionConfig(
            population_size=20,
            generations=50,  # Long run
            patience=20  # Allow for longer runs
        )
        
        evolver = EvolvingMoERouter(expert_pool=expert_pool, config=config)
        
        # Monitor system resources
        monitor = SystemMonitor(sample_interval=1.0)
        monitor.start_monitoring()
        
        try:
            model = torch.nn.Linear(64, 10)  
            data = [(torch.randn(16, 64), torch.randint(0, 10, (16,))) for _ in range(10)]
            
            class MockDataLoader:
                def __init__(self, data):
                    self.data = data
                def __iter__(self):
                    return iter(self.data)
                def __len__(self):
                    return len(self.data)
            
            data_loader = MockDataLoader(data)
            
            # Run long evolution
            start_time = time.time()
            best_topology = evolver.evolve(model, data_loader, data_loader)
            evolution_time = time.time() - start_time
            
            # Check system remained stable
            resource_summary = monitor.get_resource_summary(
                duration_minutes=evolution_time / 60
            )
            
            print(f"Long run: {evolution_time:.1f}s, {evolver.generation} generations")
            print(f"Peak memory: {resource_summary['memory']['max']:.1f}%")
            print(f"Avg CPU: {resource_summary['cpu']['avg']:.1f}%")
            
            # Should complete successfully
            assert best_topology is not None
            
            # System resources should be reasonable
            assert resource_summary['memory']['max'] < 95, "Memory usage too high"
            
        finally:
            monitor.stop_monitoring()


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])  # -s to see print outputs