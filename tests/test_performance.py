"""Performance and benchmarking tests."""

import pytest
import torch
import numpy as np
import time
import memory_profiler
from unittest.mock import Mock
import psutil
import os


class TestPerformance:
    """Performance tests for critical components."""

    @pytest.mark.benchmark
    def test_routing_matrix_operations_performance(self, large_config, device):
        """Benchmark routing matrix operations."""
        num_experts = large_config["num_experts"]
        seq_len = large_config["sequence_length"]
        batch_size = large_config["batch_size"]
        
        # Create large routing matrix
        routing_matrix = torch.rand((seq_len, num_experts), device=device)
        routing_matrix = (routing_matrix > 0.8).float()  # Make sparse
        
        input_data = torch.randn((batch_size, seq_len, 768), device=device)
        
        # Benchmark sparse matrix multiplication
        iterations = 100
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            # Simulate routing computation
            routing_weights = torch.softmax(
                torch.matmul(input_data, routing_matrix.T), dim=-1
            )
            selected_experts = torch.argmax(routing_weights, dim=-1)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Performance assertions
        assert avg_time < 0.1, f"Routing too slow: {avg_time:.4f}s"
        assert std_time < avg_time * 0.5, f"Too much variance: {std_time:.4f}s"
        
        print(f"Routing performance: {avg_time:.4f}±{std_time:.4f}s")

    @pytest.mark.benchmark
    @pytest.mark.memory_intensive
    def test_memory_usage_evolution(self, small_config, device):
        """Test memory usage during evolution."""
        population_size = small_config["population_size"]
        num_experts = small_config["num_experts"]
        seq_len = small_config["sequence_length"]
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create population
        population = []
        for _ in range(population_size):
            topology = {
                "routing_matrix": torch.rand((seq_len, num_experts), device=device),
                "routing_params": {
                    "temperature": 1.0,
                    "top_k": 2
                }
            }
            population.append(topology)
        
        # Measure memory after population creation
        after_population_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate evolution operations
        for generation in range(10):
            # Mock fitness evaluation
            fitness_scores = [np.random.random() for _ in population]
            
            # Mock selection and mutation
            new_population = []
            for individual in population:
                new_individual = {
                    "routing_matrix": individual["routing_matrix"].clone(),
                    "routing_params": individual["routing_params"].copy()
                }
                new_population.append(new_individual)
            
            population = new_population
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory growth should be reasonable
        memory_growth = final_memory - initial_memory
        per_individual_memory = memory_growth / population_size
        
        assert memory_growth < 500, f"Memory growth too high: {memory_growth:.1f}MB"
        assert per_individual_memory < 10, f"Per-individual memory too high: {per_individual_memory:.1f}MB"
        
        print(f"Memory usage - Initial: {initial_memory:.1f}MB, "
              f"After population: {after_population_memory:.1f}MB, "
              f"Final: {final_memory:.1f}MB")

    @pytest.mark.benchmark
    @pytest.mark.gpu
    def test_gpu_utilization_efficiency(self, large_config, device):
        """Test GPU utilization efficiency."""
        if device.type != 'cuda':
            pytest.skip("GPU test requires CUDA device")
        
        batch_size = large_config["batch_size"]
        seq_len = large_config["sequence_length"]
        num_experts = large_config["num_experts"]
        expert_dim = large_config["expert_dim"]
        
        # Create data on GPU
        input_data = torch.randn((batch_size, seq_len, expert_dim), device=device)
        expert_weights = torch.randn((num_experts, expert_dim, expert_dim), device=device)
        routing_matrix = torch.rand((seq_len, num_experts), device=device)
        
        # Warmup
        for _ in range(10):
            torch.matmul(input_data, expert_weights[0])
        torch.cuda.synchronize()
        
        # Benchmark GPU operations
        iterations = 50
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            # Simulate expert computation
            for expert_idx in range(num_experts):
                expert_output = torch.matmul(input_data, expert_weights[expert_idx])
            
            # Routing computation
            routing_weights = torch.softmax(routing_matrix, dim=-1)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_per_iteration = total_time / iterations
        
        # GPU should be reasonably fast
        assert avg_time_per_iteration < 0.05, f"GPU operations too slow: {avg_time_per_iteration:.4f}s"
        
        print(f"GPU performance: {avg_time_per_iteration:.4f}s per iteration")

    @pytest.mark.benchmark
    def test_evolution_convergence_speed(self, small_config, device):
        """Test evolution convergence speed."""
        population_size = small_config["population_size"]
        max_generations = 50
        
        # Mock population
        population = []
        for _ in range(population_size):
            individual = {
                "fitness": np.random.uniform(0.1, 0.6),
                "topology": torch.rand((16, 4), device=device)
            }
            population.append(individual)
        
        # Track convergence
        fitness_history = []
        convergence_generation = None
        
        for generation in range(max_generations):
            # Mock evolution step
            current_fitness = [ind["fitness"] for ind in population]
            best_fitness = max(current_fitness)
            avg_fitness = np.mean(current_fitness)
            
            fitness_history.append({
                "generation": generation,
                "best": best_fitness,
                "average": avg_fitness
            })
            
            # Check for convergence (improvement plateaus)
            if generation > 10:
                recent_improvement = (
                    fitness_history[-1]["best"] - fitness_history[-10]["best"]
                )
                if recent_improvement < 0.01 and convergence_generation is None:
                    convergence_generation = generation
            
            # Mock selection and mutation
            # Select top 50%
            population.sort(key=lambda x: x["fitness"], reverse=True)
            elite_size = population_size // 2
            
            # Generate new population
            new_population = population[:elite_size]  # Elitism
            
            while len(new_population) < population_size:
                parent = np.random.choice(population[:elite_size])
                child = {
                    "fitness": min(1.0, parent["fitness"] + np.random.normal(0, 0.05)),
                    "topology": parent["topology"].clone()
                }
                new_population.append(child)
            
            population = new_population
        
        # Convergence should happen within reasonable time
        if convergence_generation:
            assert convergence_generation < 40, f"Convergence too slow: {convergence_generation} generations"
        
        final_best = max(ind["fitness"] for ind in population)
        initial_best = fitness_history[0]["best"]
        improvement = final_best - initial_best
        
        assert improvement > 0.1, f"Insufficient improvement: {improvement:.3f}"
        
        print(f"Evolution converged at generation {convergence_generation}, "
              f"improvement: {improvement:.3f}")

    @pytest.mark.benchmark
    def test_batch_processing_scalability(self, device):
        """Test scalability with different batch sizes."""
        batch_sizes = [1, 4, 16, 64]
        if device.type == 'cuda':
            batch_sizes.extend([128, 256])
        
        seq_len = 128
        num_experts = 16
        expert_dim = 512
        
        performance_results = {}
        
        for batch_size in batch_sizes:
            # Create data
            input_data = torch.randn((batch_size, seq_len, expert_dim), device=device)
            routing_matrix = torch.rand((seq_len, num_experts), device=device)
            expert_weights = torch.randn((num_experts, expert_dim, expert_dim), device=device)
            
            # Benchmark
            iterations = 20
            times = []
            
            for _ in range(iterations):
                start_time = time.perf_counter()
                
                # Simulate MoE computation
                routing_weights = torch.softmax(routing_matrix, dim=-1)
                
                # Expert computation (simplified)
                expert_outputs = []
                for expert_idx in range(num_experts):
                    output = torch.matmul(input_data, expert_weights[expert_idx])
                    expert_outputs.append(output)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            throughput = batch_size / avg_time  # samples per second
            
            performance_results[batch_size] = {
                "avg_time": avg_time,
                "throughput": throughput
            }
        
        # Analyze scalability
        for i in range(1, len(batch_sizes)):
            prev_batch = batch_sizes[i-1]
            curr_batch = batch_sizes[i]
            
            prev_throughput = performance_results[prev_batch]["throughput"]
            curr_throughput = performance_results[curr_batch]["throughput"]
            
            # Throughput should generally increase with batch size
            # (allowing some variance)
            efficiency_ratio = curr_throughput / prev_throughput
            batch_ratio = curr_batch / prev_batch
            
            # Efficiency should be at least 60% of ideal linear scaling
            min_expected_efficiency = batch_ratio * 0.6
            assert efficiency_ratio >= min_expected_efficiency, \
                f"Poor scaling from batch {prev_batch} to {curr_batch}: {efficiency_ratio:.2f}"
        
        print("Batch processing scalability:")
        for batch_size, results in performance_results.items():
            print(f"  Batch {batch_size}: {results['throughput']:.1f} samples/sec")

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_long_running_evolution_stability(self, small_config, device):
        """Test stability of long-running evolution."""
        population_size = small_config["population_size"]
        long_generations = 200
        
        # Track memory and performance over time
        memory_samples = []
        performance_samples = []
        
        # Initial population
        population = []
        for _ in range(population_size):
            topology = torch.rand((16, 4), device=device)
            individual = {"topology": topology, "fitness": np.random.random()}
            population.append(individual)
        
        process = psutil.Process(os.getpid())
        
        for generation in range(long_generations):
            # Sample memory every 20 generations
            if generation % 20 == 0:
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)
            
            # Performance timing
            start_time = time.perf_counter()
            
            # Mock evolution step
            fitness_scores = [np.random.random() for _ in population]
            
            # Selection
            population.sort(key=lambda x: x["fitness"], reverse=True)
            elite = population[:population_size//2]
            
            # Generate new population
            new_population = []
            for _ in range(population_size):
                parent = np.random.choice(elite)
                child = {
                    "topology": parent["topology"].clone(),
                    "fitness": max(0, parent["fitness"] + np.random.normal(0, 0.02))
                }
                new_population.append(child)
            
            population = new_population
            
            end_time = time.perf_counter()
            generation_time = end_time - start_time
            
            if generation % 10 == 0:
                performance_samples.append(generation_time)
        
        # Analyze stability
        # Memory should not grow significantly
        if len(memory_samples) > 1:
            memory_growth = memory_samples[-1] - memory_samples[0]
            assert memory_growth < 100, f"Memory leak detected: {memory_growth:.1f}MB growth"
        
        # Performance should remain stable
        if len(performance_samples) > 10:
            early_performance = np.mean(performance_samples[:5])
            late_performance = np.mean(performance_samples[-5:])
            performance_degradation = late_performance / early_performance
            
            assert performance_degradation < 2.0, \
                f"Performance degraded significantly: {performance_degradation:.2f}x slower"
        
        print(f"Long evolution completed: {long_generations} generations")
        print(f"Memory growth: {memory_growth:.1f}MB" if len(memory_samples) > 1 else "Memory tracking skipped")
        print(f"Performance ratio: {performance_degradation:.2f}" if len(performance_samples) > 10 else "Performance tracking skipped")

    @pytest.mark.benchmark
    def test_topology_serialization_performance(self, large_config, temp_dir, device):
        """Test performance of topology serialization/deserialization."""
        num_topologies = 100
        seq_len = large_config["sequence_length"]
        num_experts = large_config["num_experts"]
        
        # Create topologies
        topologies = []
        for _ in range(num_topologies):
            topology = {
                "routing_matrix": torch.rand((seq_len, num_experts), device=device),
                "routing_params": {
                    "temperature": np.random.random(),
                    "top_k": np.random.choice([1, 2, 3]),
                    "load_balancing_weight": np.random.random() * 0.1
                },
                "metadata": {
                    "generation": np.random.randint(0, 1000),
                    "fitness": np.random.random(),
                    "sparsity": np.random.random()
                }
            }
            topologies.append(topology)
        
        # Test JSON serialization performance
        json_times = []
        for i in range(10):  # Test subset for speed
            topology = topologies[i]
            
            # Prepare for JSON (convert tensors to lists)
            json_data = {
                "routing_matrix": topology["routing_matrix"].cpu().numpy().tolist(),
                "routing_params": topology["routing_params"],
                "metadata": topology["metadata"]
            }
            
            start_time = time.perf_counter()
            
            import json
            json_path = temp_dir / f"topology_{i}.json"
            with open(json_path, 'w') as f:
                json.dump(json_data, f)
            
            # Read back
            with open(json_path, 'r') as f:
                loaded_data = json.load(f)
            
            end_time = time.perf_counter()
            json_times.append(end_time - start_time)
        
        # Test PyTorch serialization performance
        torch_times = []
        for i in range(10):
            topology = topologies[i]
            
            start_time = time.perf_counter()
            
            torch_path = temp_dir / f"topology_{i}.pt"
            torch.save(topology, torch_path)
            
            # Read back
            loaded_topology = torch.load(torch_path, map_location=device)
            
            end_time = time.perf_counter()
            torch_times.append(end_time - start_time)
        
        avg_json_time = np.mean(json_times)
        avg_torch_time = np.mean(torch_times)
        
        # Performance requirements
        assert avg_json_time < 0.1, f"JSON serialization too slow: {avg_json_time:.4f}s"
        assert avg_torch_time < 0.05, f"PyTorch serialization too slow: {avg_torch_time:.4f}s"
        
        print(f"Serialization performance:")
        print(f"  JSON: {avg_json_time:.4f}s")
        print(f"  PyTorch: {avg_torch_time:.4f}s")
        print(f"  PyTorch is {avg_json_time/avg_torch_time:.1f}x faster")