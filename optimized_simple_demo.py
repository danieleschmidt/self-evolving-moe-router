#!/usr/bin/env python3
"""
Generation 3: MAKE IT SCALE (Simplified but Optimized)
High-performance Self-Evolving MoE Router with caching and performance optimization
"""

import numpy as np
import random
import json
import time
import logging
import threading
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, deque
import hashlib


class PerformanceProfiler:
    """Simple but effective performance profiler."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.memory_snapshots = []
    
    def time_function(self, func, name: str = None):
        """Profile a function execution."""
        func_name = name or func.__name__
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()
        self.timings[func_name].append(end_time - start_time)
        return result
    
    def get_report(self) -> Dict[str, Any]:
        """Get performance report."""
        report = {}
        for name, times in self.timings.items():
            if times:
                report[name] = {
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'call_count': len(times)
                }
        return report


class SimpleCache:
    """Thread-safe cache with LRU eviction."""
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.cache = {}
        self.access_order = deque()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, obj) -> str:
        """Create cache key from object."""
        if isinstance(obj, np.ndarray):
            return hashlib.md5(obj.tobytes()).hexdigest()
        return str(hash(str(obj)))
    
    def get(self, key) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._make_key(key)
        
        with self.lock:
            if cache_key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(cache_key)
                self.access_order.append(cache_key)
                self.hits += 1
                return self.cache[cache_key]
            
            self.misses += 1
            return None
    
    def put(self, key, value):
        """Put value in cache."""
        cache_key = self._make_key(key)
        
        with self.lock:
            if cache_key in self.cache:
                # Update existing
                self.access_order.remove(cache_key)
            elif len(self.cache) >= self.max_size:
                # Evict LRU
                lru_key = self.access_order.popleft()
                del self.cache[lru_key]
            
            self.cache[cache_key] = value
            self.access_order.append(cache_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / max(total, 1),
            'size': len(self.cache),
            'max_size': self.max_size
        }


class OptimizedTopology:
    """Optimized topology with fast operations."""
    
    def __init__(self, num_tokens: int, num_experts: int, sparsity: float = 0.1):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.sparsity = sparsity
        
        # Use int8 for memory efficiency
        self.routing_matrix = self._init_matrix()
        self.generation = 0
        self.fitness_score = None
    
    def _init_matrix(self) -> np.ndarray:
        """Initialize sparse routing matrix efficiently."""
        matrix = np.zeros((self.num_tokens, self.num_experts), dtype=np.int8)
        
        # Ensure each token has one connection
        for i in range(self.num_tokens):
            matrix[i, random.randint(0, self.num_experts - 1)] = 1
        
        # Add additional connections
        total_possible = self.num_tokens * self.num_experts
        target_connections = int(total_possible * (1 - self.sparsity))
        current_connections = np.sum(matrix)
        
        # Vectorized addition of connections
        if target_connections > current_connections:
            zero_positions = np.where(matrix == 0)
            if len(zero_positions[0]) > 0:
                needed = min(target_connections - current_connections, len(zero_positions[0]))
                indices = np.random.choice(len(zero_positions[0]), needed, replace=False)
                matrix[zero_positions[0][indices], zero_positions[1][indices]] = 1
        
        return matrix
    
    def get_routing_mask(self, seq_len: int) -> np.ndarray:
        """Get routing mask efficiently."""
        if seq_len <= self.num_tokens:
            return self.routing_matrix[:seq_len].astype(np.float32)
        else:
            # Efficient tiling
            reps = (seq_len + self.num_tokens - 1) // self.num_tokens
            extended = np.tile(self.routing_matrix, (reps, 1))
            return extended[:seq_len].astype(np.float32)
    
    def mutate(self, mutation_rate: float = 0.1) -> bool:
        """Fast mutation with vectorized operations."""
        original_checksum = np.sum(self.routing_matrix)
        
        # Add connections
        if random.random() < mutation_rate:
            zero_pos = np.where(self.routing_matrix == 0)
            if len(zero_pos[0]) > 0:
                idx = random.randint(0, len(zero_pos[0]) - 1)
                self.routing_matrix[zero_pos[0][idx], zero_pos[1][idx]] = 1
        
        # Remove connections (keeping at least one per token)
        if random.random() < mutation_rate:
            for token_idx in range(self.num_tokens):
                if np.sum(self.routing_matrix[token_idx]) > 1:
                    one_pos = np.where(self.routing_matrix[token_idx] == 1)[0]
                    if len(one_pos) > 1:
                        remove_idx = random.choice(one_pos)
                        self.routing_matrix[token_idx, remove_idx] = 0
                    break
        
        self.generation += 1
        return np.sum(self.routing_matrix) != original_checksum
    
    def crossover(self, other: 'OptimizedTopology') -> 'OptimizedTopology':
        """Fast crossover with vectorized operations."""
        child = OptimizedTopology(self.num_tokens, self.num_experts, 
                                 (self.sparsity + other.sparsity) / 2)
        
        # Vectorized uniform crossover
        mask = np.random.random((self.num_tokens, self.num_experts)) > 0.5
        child.routing_matrix = np.where(mask, self.routing_matrix, other.routing_matrix)
        
        # Ensure connectivity
        for i in range(self.num_tokens):
            if np.sum(child.routing_matrix[i]) == 0:
                child.routing_matrix[i, random.randint(0, self.num_experts - 1)] = 1
        
        child.generation = max(self.generation, other.generation) + 1
        return child
    
    def compute_sparsity(self) -> float:
        """Fast sparsity computation."""
        return 1.0 - (np.sum(self.routing_matrix) / self.routing_matrix.size)
    
    def get_cache_key(self) -> str:
        """Get cache key for this topology."""
        return hashlib.md5(self.routing_matrix.tobytes()).hexdigest()


class OptimizedMoEModel:
    """Optimized MoE model with vectorized operations."""
    
    def __init__(self, input_dim: int = 64, num_experts: int = 8, hidden_dim: int = 128):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Pre-allocate and optimize weights
        self.expert_weights = self._init_weights()
        self.router_weights = np.random.randn(input_dim, num_experts).astype(np.float32) * 0.1
        self.router_bias = np.zeros(num_experts, dtype=np.float32)
        
        self.current_topology = None
        self.inference_count = 0
        self.total_inference_time = 0.0
    
    def _init_weights(self) -> List[Tuple[np.ndarray, ...]]:
        """Initialize expert weights with optimal data types."""
        weights = []
        for _ in range(self.num_experts):
            w1 = np.random.randn(self.input_dim, self.hidden_dim).astype(np.float32) * 0.1
            b1 = np.zeros(self.hidden_dim, dtype=np.float32)
            w2 = np.random.randn(self.hidden_dim, self.input_dim).astype(np.float32) * 0.1
            b2 = np.zeros(self.input_dim, dtype=np.float32)
            weights.append((w1, b1, w2, b2))
        return weights
    
    def set_routing_topology(self, topology: Optional[OptimizedTopology]):
        """Set routing topology."""
        self.current_topology = topology
    
    def forward_fast(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Optimized forward pass with vectorized operations."""
        start_time = time.perf_counter()
        
        try:
            batch_size, seq_len, input_dim = x.shape
            
            # Fast router computation
            router_logits = np.dot(x, self.router_weights) + self.router_bias
            
            # Apply topology mask
            if self.current_topology is not None:
                mask = self.current_topology.get_routing_mask(seq_len)
                router_logits = router_logits * mask[np.newaxis, :, :] + (1 - mask[np.newaxis, :, :]) * (-1e9)
            
            # Numerically stable softmax
            router_logits = np.clip(router_logits, -10, 10)
            exp_logits = np.exp(router_logits - np.max(router_logits, axis=-1, keepdims=True))
            routing_weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            # Top-2 routing for efficiency
            top_k = 2
            top_indices = np.argsort(routing_weights, axis=-1)[..., -top_k:]
            
            # Initialize output
            output = np.zeros_like(x)
            expert_usage = np.zeros(self.num_experts, dtype=np.float32)
            
            # Vectorized expert processing
            for k in range(top_k):
                expert_indices = top_indices[..., k]
                weights = np.take_along_axis(routing_weights, 
                                           expert_indices[..., np.newaxis], axis=-1).squeeze(-1)
                
                # Process each expert
                for expert_idx in range(self.num_experts):
                    mask = (expert_indices == expert_idx)
                    if not np.any(mask):
                        continue
                    
                    # Get positions and weights
                    positions = np.where(mask)
                    if len(positions[0]) == 0:
                        continue
                    
                    expert_input = x[positions]
                    expert_weights = weights[positions]
                    
                    # Fast expert computation
                    w1, b1, w2, b2 = self.expert_weights[expert_idx]
                    
                    # Vectorized forward pass
                    hidden = np.maximum(0, np.dot(expert_input, w1) + b1)  # ReLU
                    expert_out = np.dot(hidden, w2) + b2
                    
                    # Apply weights
                    weighted_out = expert_out * expert_weights[:, np.newaxis]
                    
                    # Accumulate output
                    output[positions] += weighted_out
                    expert_usage[expert_idx] += np.sum(expert_weights)
            
            # Update timing
            self.inference_count += 1
            self.total_inference_time += time.perf_counter() - start_time
            
            return output, {'expert_usage': expert_usage}
            
        except Exception as e:
            logging.warning(f"Fast forward failed: {e}")
            # Fallback
            return np.zeros_like(x), {'expert_usage': np.zeros(self.num_experts)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = self.total_inference_time / max(self.inference_count, 1)
        return {
            'inference_count': self.inference_count,
            'avg_inference_time_ms': avg_time * 1000,
            'total_time': self.total_inference_time,
            'throughput_inferences_per_sec': 1.0 / max(avg_time, 1e-6)
        }


class OptimizedEvolver:
    """Optimized evolver with caching and fast operations."""
    
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.fitness_cache = SimpleCache(max_size=1000)
        self.profiler = PerformanceProfiler()
        
        self.population = []
        self.generation = 0
        self.best_topology = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
    
    def initialize_population(self, num_tokens: int, num_experts: int):
        """Initialize population efficiently."""
        def create_pop():
            self.population = []
            for _ in range(self.population_size):
                topology = OptimizedTopology(
                    num_tokens=num_tokens,
                    num_experts=num_experts,
                    sparsity=random.uniform(0.1, 0.4)
                )
                self.population.append(topology)
        
        return self.profiler.time_function(create_pop, "population_init")
    
    def evaluate_fitness_cached(self, topology: OptimizedTopology, model: OptimizedMoEModel,
                               data_batches: List) -> float:
        """Evaluate fitness with caching."""
        # Check cache first
        cache_key = topology.get_cache_key()
        cached_fitness = self.fitness_cache.get(cache_key)
        
        if cached_fitness is not None:
            return cached_fitness
        
        # Compute fitness
        def evaluate():
            model.set_routing_topology(topology)
            
            total_loss = 0.0
            total_samples = 0
            expert_usage = np.zeros(model.num_experts)
            
            # Process limited batches for speed
            for batch_idx in range(min(3, len(data_batches))):
                inputs, targets = data_batches[batch_idx]
                
                outputs, aux_info = model.forward_fast(inputs)
                
                # Fast loss computation
                loss = np.mean((outputs - targets) ** 2)
                if np.isnan(loss) or np.isinf(loss):
                    loss = 1.0
                
                total_loss += loss * inputs.shape[0]
                total_samples += inputs.shape[0]
                expert_usage += aux_info['expert_usage']
            
            # Fast metrics
            avg_loss = total_loss / max(total_samples, 1)
            sparsity = topology.compute_sparsity()
            
            # Load balance (vectorized)
            if np.sum(expert_usage) > 0:
                usage_probs = expert_usage / np.sum(expert_usage)
                entropy = -np.sum(usage_probs * np.log(usage_probs + 1e-8))
                max_entropy = np.log(model.num_experts)
                load_balance = entropy / max_entropy
            else:
                load_balance = 0.0
            
            fitness = -avg_loss + 0.2 * sparsity + 0.3 * load_balance
            return fitness
        
        fitness = self.profiler.time_function(evaluate, "fitness_eval")
        
        # Cache result
        self.fitness_cache.put(cache_key, fitness)
        
        return fitness
    
    def evolve_generation(self, model: OptimizedMoEModel, data_batches: List) -> Dict[str, Any]:
        """Evolve one generation efficiently."""
        def evolve():
            # Parallel-style fitness evaluation (simulated with batch processing)
            fitness_scores = []
            
            for topology in self.population:
                fitness = self.evaluate_fitness_cached(topology, model, data_batches)
                fitness_scores.append(fitness)
                
                # Track best
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_topology = topology
            
            self.fitness_history.append(fitness_scores)
            
            # Fast next generation creation
            new_population = []
            
            # Elitism (top 20%)
            elite_count = max(2, self.population_size // 5)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(self.population[idx])
            
            # Fast offspring generation
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                candidates = random.sample(range(len(self.population)), 
                                         min(tournament_size, len(self.population)))
                parent1_idx = max(candidates, key=lambda i: fitness_scores[i])
                
                candidates = random.sample(range(len(self.population)), 
                                         min(tournament_size, len(self.population)))
                parent2_idx = max(candidates, key=lambda i: fitness_scores[i])
                
                # Crossover and mutation
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                child = parent1.crossover(parent2)
                if random.random() < 0.15:
                    child.mutate(0.15)
                
                new_population.append(child)
            
            self.population = new_population
            self.generation += 1
            
            # Statistics
            valid_scores = [s for s in fitness_scores if s != float('-inf')]
            return {
                'generation': self.generation - 1,
                'best_fitness': max(valid_scores) if valid_scores else float('-inf'),
                'avg_fitness': np.mean(valid_scores) if valid_scores else 0.0,
                'population_size': len(self.population),
                'cache_hit_rate': self.fitness_cache.get_stats()['hit_rate']
            }
        
        return self.profiler.time_function(evolve, f"generation_{self.generation}")


def run_optimized_simple_demo():
    """Run the optimized but simplified Generation 3 demo."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("⚡ Starting Optimized Simple MoE Router - Generation 3")
    
    try:
        # Optimized configuration
        config = {
            'input_dim': 64,
            'num_experts': 8,
            'hidden_dim': 128,
            'num_tokens': 16,
            'population_size': 24,
            'generations': 25,
            'batch_size': 8,
            'seq_len': 16
        }
        
        logger.info(f"Optimized config: {config}")
        
        # Create optimized model
        model = OptimizedMoEModel(
            input_dim=config['input_dim'],
            num_experts=config['num_experts'],
            hidden_dim=config['hidden_dim']
        )
        
        # Create optimized data
        logger.info("Creating optimized dataset...")
        data_batches = []
        for i in range(12):
            inputs = np.random.randn(config['batch_size'], config['seq_len'], 
                                   config['input_dim']).astype(np.float32)
            targets = (inputs * 0.8 + 0.1 * np.random.randn(config['batch_size'], 
                                                           config['seq_len'], 
                                                           config['input_dim'])).astype(np.float32)
            data_batches.append((inputs, targets))
        
        logger.info(f"Created {len(data_batches)} optimized batches")
        
        # Create optimized evolver
        evolver = OptimizedEvolver(population_size=config['population_size'])
        evolver.initialize_population(config['num_tokens'], config['num_experts'])
        
        logger.info(f"Initialized optimized population of {len(evolver.population)}")
        
        # High-performance evolution
        evolution_stats = []
        start_time = time.time()
        
        for generation in range(config['generations']):
            logger.info(f"=== Optimized Generation {generation} ===")
            
            stats = evolver.evolve_generation(model, data_batches)
            evolution_stats.append(stats)
            
            logger.info(
                f"Gen {generation}: Best={stats['best_fitness']:.4f}, "
                f"Avg={stats['avg_fitness']:.4f}, "
                f"Cache Hit Rate={stats['cache_hit_rate']:.2%}"
            )
            
            # Early stopping
            if generation > 8:
                recent_best = [s['best_fitness'] for s in evolution_stats[-5:]]
                if max(recent_best) - min(recent_best) < 0.003:
                    logger.info("Early convergence detected")
                    break
        
        total_time = time.time() - start_time
        
        # Performance benchmarking
        if evolver.best_topology:
            logger.info("Running performance benchmarks...")
            
            model.set_routing_topology(evolver.best_topology)
            
            # Warmup
            warmup_input = np.random.randn(4, config['seq_len'], config['input_dim']).astype(np.float32)
            for _ in range(3):
                model.forward_fast(warmup_input)
            
            # Benchmark
            benchmark_input = np.random.randn(16, config['seq_len'], config['input_dim']).astype(np.float32)
            
            benchmark_start = time.perf_counter()
            benchmark_runs = 50
            
            for _ in range(benchmark_runs):
                output, aux_info = model.forward_fast(benchmark_input)
            
            benchmark_time = time.perf_counter() - benchmark_start
            avg_inference_time = benchmark_time / benchmark_runs
            throughput = benchmark_input.shape[0] / avg_inference_time
            
            logger.info(f"Inference performance: {avg_inference_time*1000:.2f}ms per batch")
            logger.info(f"Throughput: {throughput:.1f} samples/sec")
        
        # Comprehensive results
        results = {
            'config': config,
            'evolution_stats': evolution_stats,
            'final_fitness': float(evolver.best_fitness),
            'total_generations': len(evolution_stats),
            'total_time': total_time,
            'best_topology_sparsity': float(evolver.best_topology.compute_sparsity()) if evolver.best_topology else None,
            'performance_metrics': {
                'avg_generation_time': total_time / len(evolution_stats),
                'avg_inference_time_ms': avg_inference_time * 1000 if 'avg_inference_time' in locals() else 0.0,
                'throughput_samples_per_sec': throughput if 'throughput' in locals() else 0.0,
                'model_performance': model.get_performance_stats(),
                'cache_stats': evolver.fitness_cache.get_stats(),
                'profiler_report': evolver.profiler.get_report()
            },
            'optimization_features': {
                'vectorized_operations': True,
                'fitness_caching': True,
                'fast_data_types': True,
                'optimized_algorithms': True
            }
        }
        
        # Save results
        results_dir = Path("evolution_results")
        results_dir.mkdir(exist_ok=True)
        results_path = results_dir / "optimized_simple_demo_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Summary
        print("\n" + "="*70)
        print("⚡ GENERATION 3 OPTIMIZED DEMO COMPLETE!")
        print("="*70)
        print(f"✅ High-performance evolution completed")
        print(f"📊 Final fitness: {evolver.best_fitness:.4f}")
        print(f"🧬 Generations: {len(evolution_stats)}")
        print(f"⚡ Total runtime: {total_time:.1f}s")
        print(f"🚀 Avg generation time: {total_time/len(evolution_stats):.2f}s")
        print(f"💾 Cache hit rate: {evolver.fitness_cache.get_stats()['hit_rate']:.1%}")
        if evolver.best_topology:
            print(f"🕸️  Best topology sparsity: {evolver.best_topology.compute_sparsity():.3f}")
        if 'avg_inference_time' in locals():
            print(f"⏱️  Inference speed: {avg_inference_time*1000:.2f}ms per batch")
            print(f"📈 Throughput: {throughput:.1f} samples/sec")
        print(f"🔧 Optimizations: Vectorized ✓, Cached ✓, Fast types ✓")
        print(f"💾 Results: {results_path}")
        print("="*70)
        
        return results
        
    except Exception as e:
        logger.error(f"Optimized demo failed: {e}")
        raise


if __name__ == "__main__":
    try:
        results = run_optimized_simple_demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()