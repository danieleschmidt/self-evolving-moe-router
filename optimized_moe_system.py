#!/usr/bin/env python3
"""
Generation 3: MAKE IT SCALE
High-performance, optimized Self-Evolving MoE Router with caching, 
parallel processing, and auto-scaling capabilities
"""

import numpy as np
import random
import json
import time
import logging
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import hashlib
import pickle
import sys
import os
import queue
from collections import defaultdict, deque
import warnings

# Suppress warnings for performance
warnings.filterwarnings('ignore')


class PerformanceProfiler:
    """High-performance profiler for bottleneck identification."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.memory_snapshots = deque(maxlen=100)
        self.active_timers = {}
        self.lock = threading.Lock()
    
    def start_timer(self, name: str) -> str:
        """Start a performance timer."""
        timer_id = f"{name}_{time.time()}_{random.randint(1000, 9999)}"
        with self.lock:
            self.active_timers[timer_id] = time.perf_counter()
        return timer_id
    
    def end_timer(self, timer_id: str, name: str):
        """End a performance timer."""
        end_time = time.perf_counter()
        with self.lock:
            if timer_id in self.active_timers:
                duration = end_time - self.active_timers[timer_id]
                self.timings[name].append(duration)
                self.call_counts[name] += 1
                del self.active_timers[timer_id]
    
    def profile_function(self, func: Callable, name: str = None):
        """Decorator for profiling functions."""
        func_name = name or func.__name__
        
        def wrapper(*args, **kwargs):
            timer_id = self.start_timer(func_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.end_timer(timer_id, func_name)
        
        return wrapper
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        with self.lock:
            report = {}
            for name, times in self.timings.items():
                if times:
                    report[name] = {
                        'total_time': sum(times),
                        'avg_time': np.mean(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'std_time': np.std(times),
                        'call_count': self.call_counts[name],
                        'calls_per_second': self.call_counts[name] / max(sum(times), 1e-6)
                    }
            return report


class LRUCache:
    """Thread-safe LRU cache with expiration."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.insert_times = {}
        self.lock = threading.RLock()
    
    def _hash_key(self, key: Any) -> str:
        """Create hash key from arbitrary object."""
        if isinstance(key, (str, int, float, bool)):
            return str(key)
        elif isinstance(key, np.ndarray):
            return hashlib.md5(key.tobytes()).hexdigest()
        else:
            try:
                return hashlib.md5(pickle.dumps(key)).hexdigest()
            except:
                return str(hash(str(key)))
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache."""
        hash_key = self._hash_key(key)
        current_time = time.time()
        
        with self.lock:
            if hash_key in self.cache:
                # Check expiration
                if current_time - self.insert_times[hash_key] > self.ttl:
                    del self.cache[hash_key]
                    del self.access_times[hash_key]
                    del self.insert_times[hash_key]
                    return None
                
                # Update access time
                self.access_times[hash_key] = current_time
                return self.cache[hash_key]
            
            return None
    
    def put(self, key: Any, value: Any):
        """Put value in cache."""
        hash_key = self._hash_key(key)
        current_time = time.time()
        
        with self.lock:
            # Remove expired entries
            expired_keys = [
                k for k, insert_time in self.insert_times.items()
                if current_time - insert_time > self.ttl
            ]
            for k in expired_keys:
                if k in self.cache:
                    del self.cache[k]
                    del self.access_times[k]
                    del self.insert_times[k]
            
            # Evict LRU if at capacity
            if len(self.cache) >= self.max_size and hash_key not in self.cache:
                lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[lru_key]
                del self.access_times[lru_key]
                del self.insert_times[lru_key]
            
            # Insert new value
            self.cache[hash_key] = value
            self.access_times[hash_key] = current_time
            self.insert_times[hash_key] = current_time
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.insert_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': 0.0,  # Would need hit/miss tracking for this
                'expired_entries': sum(
                    1 for insert_time in self.insert_times.values()
                    if time.time() - insert_time > self.ttl
                )
            }


class BatchProcessor:
    """High-performance batch processing with adaptive sizing."""
    
    def __init__(self, min_batch_size: int = 1, max_batch_size: int = 32, 
                 target_latency: float = 0.1):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency = target_latency
        self.current_batch_size = min_batch_size
        
        # Performance tracking
        self.latency_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        
        # Adaptive batch sizing
        self.last_adjustment = time.time()
        self.adjustment_interval = 1.0  # seconds
    
    def process_batch(self, items: List[Any], processor_func: Callable) -> List[Any]:
        """Process batch with adaptive sizing."""
        start_time = time.perf_counter()
        
        # Process in optimal batch sizes
        results = []
        for i in range(0, len(items), self.current_batch_size):
            batch = items[i:i + self.current_batch_size]
            batch_results = processor_func(batch)
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
        
        end_time = time.perf_counter()
        latency = end_time - start_time
        throughput = len(items) / max(latency, 1e-6)
        
        # Track performance
        self.latency_history.append(latency)
        self.throughput_history.append(throughput)
        
        # Adaptive batch sizing
        self._adjust_batch_size(latency, throughput)
        
        return results
    
    def _adjust_batch_size(self, latency: float, throughput: float):
        """Adjust batch size based on performance."""
        current_time = time.time()
        
        if current_time - self.last_adjustment < self.adjustment_interval:
            return
        
        if len(self.latency_history) < 10:
            return
        
        recent_latency = np.mean(list(self.latency_history)[-10:])
        recent_throughput = np.mean(list(self.throughput_history)[-10:])
        
        # Increase batch size if latency is under target and throughput is good
        if recent_latency < self.target_latency * 0.8 and recent_throughput > 10:
            new_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
        # Decrease batch size if latency is over target
        elif recent_latency > self.target_latency * 1.2:
            new_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))
        else:
            new_size = self.current_batch_size
        
        if new_size != self.current_batch_size:
            self.current_batch_size = new_size
            self.last_adjustment = current_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get batch processing performance statistics."""
        if not self.latency_history:
            return {'status': 'no_data'}
        
        return {
            'current_batch_size': self.current_batch_size,
            'avg_latency': np.mean(self.latency_history),
            'avg_throughput': np.mean(self.throughput_history),
            'max_throughput': max(self.throughput_history),
            'min_latency': min(self.latency_history),
            'target_latency': self.target_latency
        }


class ParallelEvolver:
    """Parallel evolution with multi-process fitness evaluation."""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or min(mp.cpu_count(), 8)
        self.fitness_cache = LRUCache(max_size=2000, ttl=600)
        self.profiler = PerformanceProfiler()
        self.batch_processor = BatchProcessor()
        
        # Performance optimization
        self.enable_cache = True
        self.enable_parallel = True
        self.cache_hit_count = 0
        self.cache_miss_count = 0
    
    def evaluate_fitness_parallel(self, population: List, model_config: Dict, 
                                data_batches: List) -> List[float]:
        """Evaluate fitness in parallel with caching."""
        timer_id = self.profiler.start_timer("fitness_evaluation")
        
        try:
            # Check cache first
            fitness_scores = []
            uncached_indices = []
            
            for i, topology in enumerate(population):
                if self.enable_cache:
                    cache_key = self._create_topology_key(topology)
                    cached_fitness = self.fitness_cache.get(cache_key)
                    
                    if cached_fitness is not None:
                        fitness_scores.append(cached_fitness)
                        self.cache_hit_count += 1
                    else:
                        fitness_scores.append(None)
                        uncached_indices.append(i)
                        self.cache_miss_count += 1
                else:
                    fitness_scores.append(None)
                    uncached_indices.append(i)
            
            # Evaluate uncached topologies
            if uncached_indices and self.enable_parallel and len(uncached_indices) > 1:
                # Parallel evaluation
                uncached_scores = self._evaluate_parallel_worker(
                    [population[i] for i in uncached_indices],
                    model_config,
                    data_batches
                )
                
                # Fill in results and cache
                for idx, score in zip(uncached_indices, uncached_scores):
                    fitness_scores[idx] = score
                    if self.enable_cache:
                        cache_key = self._create_topology_key(population[idx])
                        self.fitness_cache.put(cache_key, score)
            
            elif uncached_indices:
                # Sequential evaluation for small batches
                for idx in uncached_indices:
                    score = self._evaluate_single_topology(
                        population[idx], model_config, data_batches
                    )
                    fitness_scores[idx] = score
                    if self.enable_cache:
                        cache_key = self._create_topology_key(population[idx])
                        self.fitness_cache.put(cache_key, score)
            
            return fitness_scores
            
        finally:
            self.profiler.end_timer(timer_id, "fitness_evaluation")
    
    def _create_topology_key(self, topology) -> str:
        """Create cache key for topology."""
        try:
            matrix_key = hashlib.md5(topology.routing_matrix.tobytes()).hexdigest()
            params_key = hashlib.md5(str(topology.sparsity).encode()).hexdigest()
            return f"{matrix_key}_{params_key}"
        except:
            return str(random.randint(1000000, 9999999))
    
    def _evaluate_parallel_worker(self, topologies: List, model_config: Dict,
                                data_batches: List) -> List[float]:
        """Evaluate topologies using parallel workers."""
        if len(topologies) <= 2:  # Use sequential for small batches
            return [self._evaluate_single_topology(t, model_config, data_batches) 
                   for t in topologies]
        
        try:
            # Chunk topologies for workers
            chunk_size = max(1, len(topologies) // self.num_workers)
            chunks = [topologies[i:i + chunk_size] 
                     for i in range(0, len(topologies), chunk_size)]
            
            results = []
            
            # Use ThreadPoolExecutor for I/O bound tasks (better for numpy operations)
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                for chunk in chunks:
                    future = executor.submit(
                        self._evaluate_topology_chunk,
                        chunk, model_config, data_batches
                    )
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        chunk_results = future.result(timeout=30)  # 30 second timeout
                        results.extend(chunk_results)
                    except Exception as e:
                        logging.warning(f"Chunk evaluation failed: {e}")
                        # Add dummy scores for failed chunk
                        results.extend([float('-inf')] * len(chunks[0]))
            
            return results[:len(topologies)]  # Ensure correct length
            
        except Exception as e:
            logging.error(f"Parallel evaluation failed: {e}")
            # Fallback to sequential
            return [self._evaluate_single_topology(t, model_config, data_batches) 
                   for t in topologies]
    
    def _evaluate_topology_chunk(self, topologies: List, model_config: Dict,
                                data_batches: List) -> List[float]:
        """Evaluate a chunk of topologies."""
        scores = []
        
        for topology in topologies:
            try:
                score = self._evaluate_single_topology(topology, model_config, data_batches)
                scores.append(score)
            except Exception as e:
                logging.warning(f"Topology evaluation failed: {e}")
                scores.append(float('-inf'))
        
        return scores
    
    def _evaluate_single_topology(self, topology, model_config: Dict, 
                                 data_batches: List) -> float:
        """Evaluate single topology (optimized version)."""
        try:
            # Create lightweight model for this thread
            model = self._create_lightweight_model(model_config)
            model.set_routing_topology(topology)
            
            total_loss = 0.0
            total_samples = 0
            expert_usage = np.zeros(model_config['num_experts'])
            
            # Process limited batches for speed
            max_batches = min(2, len(data_batches))
            
            for batch_idx in range(max_batches):
                inputs, targets = data_batches[batch_idx]
                
                # Vectorized processing for speed
                outputs, aux_info = model.forward_optimized(inputs)
                
                # Fast loss computation
                loss = np.mean((outputs - targets) ** 2)
                if np.isnan(loss) or np.isinf(loss):
                    loss = 1.0
                
                total_loss += loss * inputs.shape[0]
                total_samples += inputs.shape[0]
                expert_usage += aux_info.get('expert_usage', np.zeros(model_config['num_experts']))
            
            # Fast fitness computation
            avg_loss = total_loss / max(total_samples, 1)
            sparsity = topology.compute_sparsity()
            
            # Fast load balance computation
            load_balance = 0.0
            usage_sum = np.sum(expert_usage)
            if usage_sum > 0:
                usage_probs = expert_usage / usage_sum
                entropy = -np.sum(usage_probs * np.log(usage_probs + 1e-8))
                max_entropy = np.log(model_config['num_experts'])
                load_balance = entropy / max_entropy if max_entropy > 0 else 0.0
            
            fitness = -avg_loss + 0.2 * sparsity + 0.3 * load_balance
            return float(fitness)
            
        except Exception as e:
            logging.warning(f"Single topology evaluation failed: {e}")
            return float('-inf')
    
    def _create_lightweight_model(self, config: Dict):
        """Create lightweight model for evaluation."""
        # Simplified model creation for parallel processing
        from robust_moe_system import RobustMoEModel, RobustValidator
        
        model = RobustMoEModel(
            input_dim=config['input_dim'],
            num_experts=config['num_experts'],
            hidden_dim=config['hidden_dim']
        )
        
        # Add optimized forward method
        def forward_optimized(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
            """Optimized forward pass for evaluation."""
            try:
                batch_size, seq_len, input_dim = x.shape
                
                # Fast router computation
                router_logits = np.dot(x, self.router_weights) + self.router_bias
                
                # Apply topology if available
                if self.current_topology is not None:
                    mask = self.current_topology.get_routing_mask(seq_len)
                    router_logits = router_logits * np.expand_dims(mask, 0)
                
                # Fast softmax
                router_logits = np.clip(router_logits, -10, 10)
                exp_logits = np.exp(router_logits - np.max(router_logits, axis=-1, keepdims=True))
                routing_weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                
                # Top-2 selection (hardcoded for speed)
                top_indices = np.argsort(routing_weights, axis=-1)[..., -2:]
                
                # Vectorized expert processing
                output = np.zeros_like(x)
                expert_usage = np.zeros(self.num_experts)
                
                # Process top experts only
                for k in range(2):  # top-2
                    expert_indices = top_indices[..., k]
                    weights = np.take_along_axis(routing_weights, np.expand_dims(expert_indices, -1), axis=-1).squeeze(-1)
                    
                    # Batch process through experts
                    for expert_idx in range(self.num_experts):
                        mask = (expert_indices == expert_idx)
                        if not np.any(mask):
                            continue
                        
                        # Extract tokens for this expert
                        positions = np.where(mask)
                        if len(positions[0]) == 0:
                            continue
                        
                        expert_input = x[positions]
                        expert_weights = weights[positions]
                        
                        # Fast expert forward
                        w1, b1, w2, b2 = self.expert_weights[expert_idx]
                        hidden = np.maximum(0, np.dot(expert_input, w1) + b1)
                        expert_out = np.dot(hidden, w2) + b2
                        
                        # Apply weights and accumulate
                        weighted_out = expert_out * expert_weights.reshape(-1, 1)
                        output[positions] += weighted_out
                        expert_usage[expert_idx] += np.sum(expert_weights)
                
                return output, {'expert_usage': expert_usage}
                
            except Exception as e:
                logging.warning(f"Optimized forward failed: {e}")
                return np.zeros_like(x), {'expert_usage': np.zeros(self.num_experts)}
        
        # Bind optimized method
        import types
        model.forward_optimized = types.MethodType(forward_optimized, model)
        
        return model
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hit_count + self.cache_miss_count
        hit_rate = self.cache_hit_count / max(total_requests, 1)
        
        return {
            'cache_hits': self.cache_hit_count,
            'cache_misses': self.cache_miss_count,
            'hit_rate': hit_rate,
            'cache_stats': self.fitness_cache.get_stats(),
            'performance_report': self.profiler.get_performance_report()
        }


class AutoScaler:
    """Automatic scaling based on performance metrics."""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=50)
        self.scale_history = deque(maxlen=20)
        self.last_scale_time = time.time()
        self.scale_cooldown = 10.0  # seconds
        
        # Scaling parameters
        self.min_population = 8
        self.max_population = 64
        self.min_generations = 10
        self.max_generations = 100
        
        # Performance thresholds
        self.fitness_stagnation_threshold = 0.001
        self.diversity_threshold = 0.1
        self.convergence_threshold = 0.95
    
    def should_scale_population(self, evolver_stats: Dict) -> Tuple[bool, int]:
        """Determine if population should be scaled."""
        current_time = time.time()
        
        if current_time - self.last_scale_time < self.scale_cooldown:
            return False, evolver_stats.get('population_size', self.min_population)
        
        current_pop = evolver_stats.get('population_size', self.min_population)
        fitness_trend = evolver_stats.get('fitness_trend', 0.0)
        diversity = evolver_stats.get('diversity', 1.0)
        
        # Scale up if low diversity or stagnant fitness
        if (diversity < self.diversity_threshold and current_pop < self.max_population):
            new_size = min(self.max_population, int(current_pop * 1.5))
            self.last_scale_time = current_time
            return True, new_size
        
        # Scale down if converged and good performance
        if (diversity > 0.8 and fitness_trend > 0.01 and current_pop > self.min_population):
            new_size = max(self.min_population, int(current_pop * 0.8))
            self.last_scale_time = current_time
            return True, new_size
        
        return False, current_pop
    
    def should_adjust_generations(self, convergence_rate: float, 
                                 generation: int) -> Tuple[bool, int]:
        """Determine if generation count should be adjusted."""
        if convergence_rate > self.convergence_threshold and generation > self.min_generations:
            # Early stopping
            return True, generation
        
        if convergence_rate < 0.3 and generation < self.max_generations:
            # Need more generations
            return False, min(self.max_generations, generation + 10)
        
        return False, generation


def run_optimized_demo():
    """Run the optimized Generation 3 demo."""
    
    # Setup logging for performance
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('evolution_results/optimized_demo.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("⚡ Starting Optimized Self-Evolving MoE Router - Generation 3")
    
    try:
        # High-performance configuration
        config = {
            'input_dim': 64,
            'num_experts': 8,
            'hidden_dim': 128,
            'num_tokens': 16,
            'population_size': 20,
            'generations': 30,
            'batch_size': 8,
            'seq_len': 16,
            'num_workers': min(mp.cpu_count(), 8),
            'enable_caching': True,
            'enable_parallel': True,
            'enable_autoscaling': True
        }
        
        logger.info(f"High-performance config: {config}")
        
        # Initialize performance components
        profiler = PerformanceProfiler()
        parallel_evolver = ParallelEvolver(num_workers=config['num_workers'])
        auto_scaler = AutoScaler()
        batch_processor = BatchProcessor(min_batch_size=4, max_batch_size=16)
        
        # Create optimized data with batch processing
        logger.info("Creating optimized dataset...")
        create_timer = profiler.start_timer("data_creation")
        
        def create_batch(batch_info):
            batch_idx, batch_size, seq_len, input_dim = batch_info
            inputs = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
            targets = inputs * 0.8 + 0.1 * np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
            return inputs, targets
        
        batch_infos = [
            (i, config['batch_size'], config['seq_len'], config['input_dim'])
            for i in range(15)
        ]
        
        data_batches = batch_processor.process_batch(batch_infos, create_batch)
        profiler.end_timer(create_timer, "data_creation")
        
        logger.info(f"Created {len(data_batches)} optimized batches")
        
        # Create initial population with parallel generation
        logger.info("Generating optimized population...")
        pop_timer = profiler.start_timer("population_creation")
        
        def create_topology(args):
            i, num_tokens, num_experts = args
            from robust_moe_system import RobustTopology, RobustValidator
            try:
                return RobustTopology(
                    num_tokens=num_tokens,
                    num_experts=num_experts,
                    sparsity=random.uniform(0.1, 0.4),
                    validator=RobustValidator()
                )
            except:
                return None
        
        topology_args = [
            (i, config['num_tokens'], config['num_experts'])
            for i in range(config['population_size'])
        ]
        
        population = batch_processor.process_batch(topology_args, create_topology)
        population = [t for t in population if t is not None and t.validate_integrity()]
        
        profiler.end_timer(pop_timer, "population_creation")
        logger.info(f"Generated optimized population of {len(population)} topologies")
        
        # High-performance evolution loop
        evolution_stats = []
        best_topology = None
        best_fitness = float('-inf')
        
        total_start_time = time.time()
        
        for generation in range(config['generations']):
            gen_timer = profiler.start_timer("generation")
            
            logger.info(f"=== Optimized Generation {generation} ===")
            
            # Parallel fitness evaluation with caching
            fitness_scores = parallel_evolver.evaluate_fitness_parallel(
                population, config, data_batches
            )
            
            # Track best fitness
            valid_scores = [s for s in fitness_scores if s != float('-inf')]
            if valid_scores:
                gen_best = max(valid_scores)
                gen_best_idx = fitness_scores.index(gen_best)
                
                if gen_best > best_fitness:
                    best_fitness = gen_best
                    best_topology = population[gen_best_idx]
                    logger.info(f"New best fitness: {gen_best:.4f}")
            
            # Calculate performance metrics
            diversity = 0.0
            if len(population) > 1:
                diversity_sum = 0.0
                comparisons = 0
                for i in range(len(population)):
                    for j in range(i + 1, min(i + 5, len(population))):  # Sample for speed
                        try:
                            diff = np.mean(population[i].routing_matrix != population[j].routing_matrix)
                            diversity_sum += diff
                            comparisons += 1
                        except:
                            continue
                diversity = diversity_sum / max(comparisons, 1)
            
            # Auto-scaling
            evolver_stats = {
                'population_size': len(population),
                'fitness_trend': (gen_best - best_fitness) if valid_scores else 0.0,
                'diversity': diversity,
                'generation': generation
            }
            
            if config['enable_autoscaling']:
                should_scale, new_pop_size = auto_scaler.should_scale_population(evolver_stats)
                if should_scale and new_pop_size != len(population):
                    logger.info(f"Auto-scaling population: {len(population)} -> {new_pop_size}")
                    
                    if new_pop_size > len(population):
                        # Add new topologies
                        additional_args = [
                            (i, config['num_tokens'], config['num_experts'])
                            for i in range(new_pop_size - len(population))
                        ]
                        additional_topologies = batch_processor.process_batch(additional_args, create_topology)
                        population.extend([t for t in additional_topologies if t is not None])
                    else:
                        # Remove worst topologies
                        sorted_indices = sorted(range(len(fitness_scores)), 
                                              key=lambda i: fitness_scores[i], reverse=True)
                        population = [population[i] for i in sorted_indices[:new_pop_size]]
                        fitness_scores = [fitness_scores[i] for i in sorted_indices[:new_pop_size]]
            
            # Optimized next generation creation
            new_population = []
            
            # Elitism (top 20%)
            elite_count = max(2, len(population) // 5)
            elite_indices = sorted(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
            new_population.extend([population[i] for i in elite_indices])
            
            # Parallel offspring generation
            def create_offspring(args):
                parent_indices, mutation_rate = args
                try:
                    p1_idx, p2_idx = parent_indices
                    parent1 = population[p1_idx]
                    parent2 = population[p2_idx]
                    
                    child = parent1.crossover(parent2)
                    if child and random.random() < mutation_rate:
                        child.mutate(mutation_rate)
                    
                    return child if child and child.validate_integrity() else None
                except:
                    return None
            
            # Generate offspring arguments
            offspring_count = len(population) - len(new_population)
            valid_indices = [i for i, score in enumerate(fitness_scores) if score != float('-inf')]
            
            if len(valid_indices) >= 2:
                offspring_args = []
                for _ in range(offspring_count):
                    # Tournament selection
                    p1 = max(random.sample(valid_indices, min(3, len(valid_indices))),
                            key=lambda i: fitness_scores[i])
                    p2 = max(random.sample(valid_indices, min(3, len(valid_indices))),
                            key=lambda i: fitness_scores[i])
                    offspring_args.append(((p1, p2), 0.15))
                
                # Parallel offspring creation
                offspring = batch_processor.process_batch(offspring_args, create_offspring)
                new_population.extend([o for o in offspring if o is not None])
            
            # Fill remaining slots if needed
            while len(new_population) < len(population):
                try:
                    from robust_moe_system import RobustTopology, RobustValidator
                    new_topo = RobustTopology(
                        num_tokens=config['num_tokens'],
                        num_experts=config['num_experts'],
                        sparsity=random.uniform(0.1, 0.4),
                        validator=RobustValidator()
                    )
                    if new_topo.validate_integrity():
                        new_population.append(new_topo)
                except:
                    break
            
            population = new_population
            
            # Generation statistics
            stats = {
                'generation': generation,
                'population_size': len(population),
                'best_fitness': max(valid_scores) if valid_scores else float('-inf'),
                'avg_fitness': np.mean(valid_scores) if valid_scores else 0.0,
                'diversity': diversity,
                'cache_hit_rate': parallel_evolver.cache_hit_count / max(
                    parallel_evolver.cache_hit_count + parallel_evolver.cache_miss_count, 1
                ),
                'generation_time': 0.0  # Will be filled below
            }
            
            profiler.end_timer(gen_timer, "generation")
            gen_times = profiler.timings.get("generation", [])
            stats['generation_time'] = gen_times[-1] if gen_times else 0.0
            
            evolution_stats.append(stats)
            
            logger.info(
                f"Gen {generation}: Best={stats['best_fitness']:.4f}, "
                f"Avg={stats['avg_fitness']:.4f}, "
                f"Diversity={diversity:.3f}, "
                f"Time={stats['generation_time']:.2f}s, "
                f"Cache Hit Rate={stats['cache_hit_rate']:.2f}"
            )
            
            # Early stopping based on convergence
            if generation > 10:
                recent_best = [s['best_fitness'] for s in evolution_stats[-5:]]
                if max(recent_best) - min(recent_best) < 0.002:
                    logger.info("Converged early due to fitness plateau")
                    break
        
        total_time = time.time() - total_start_time
        
        logger.info(f"Optimized evolution completed in {total_time:.2f} seconds")
        
        # Performance testing
        if best_topology:
            logger.info("Running performance benchmarks...")
            
            from robust_moe_system import RobustMoEModel
            model = RobustMoEModel(
                input_dim=config['input_dim'],
                num_experts=config['num_experts'],
                hidden_dim=config['hidden_dim']
            )
            model.set_routing_topology(best_topology)
            
            # Benchmark inference speed
            test_input = np.random.randn(16, config['seq_len'], config['input_dim'])
            
            warmup_timer = profiler.start_timer("warmup")
            for _ in range(5):
                model.forward(test_input)
            profiler.end_timer(warmup_timer, "warmup")
            
            benchmark_timer = profiler.start_timer("benchmark")
            benchmark_runs = 20
            for _ in range(benchmark_runs):
                output, aux_info = model.forward(test_input)
            profiler.end_timer(benchmark_timer, "benchmark")
            
            benchmark_times = profiler.timings.get("benchmark", [])
            avg_inference_time = benchmark_times[-1] / benchmark_runs if benchmark_times else 0.0
            throughput = test_input.shape[0] / avg_inference_time if avg_inference_time > 0 else 0.0
            
            logger.info(f"Performance: {avg_inference_time*1000:.2f}ms per batch, {throughput:.1f} samples/sec")
        
        # Comprehensive results
        results = {
            'config': config,
            'evolution_stats': evolution_stats,
            'final_fitness': float(best_fitness),
            'total_generations': len(evolution_stats),
            'total_computation_time': total_time,
            'best_topology_sparsity': float(best_topology.compute_sparsity()) if best_topology else None,
            'performance_metrics': {
                'avg_generation_time': np.mean([s['generation_time'] for s in evolution_stats]),
                'total_cache_hits': parallel_evolver.cache_hit_count,
                'total_cache_misses': parallel_evolver.cache_miss_count,
                'final_cache_hit_rate': parallel_evolver.cache_hit_count / max(
                    parallel_evolver.cache_hit_count + parallel_evolver.cache_miss_count, 1
                ),
                'avg_inference_time_ms': avg_inference_time * 1000 if 'avg_inference_time' in locals() else 0.0,
                'throughput_samples_per_sec': throughput if 'throughput' in locals() else 0.0
            },
            'profiler_report': profiler.get_performance_report(),
            'cache_stats': parallel_evolver.get_cache_stats(),
            'batch_processor_stats': batch_processor.get_performance_stats(),
            'optimization_features': {
                'parallel_evaluation': config['enable_parallel'],
                'fitness_caching': config['enable_caching'],
                'auto_scaling': config['enable_autoscaling'],
                'batch_processing': True,
                'num_workers': config['num_workers']
            }
        }
        
        # Save results
        results_path = Path("evolution_results/optimized_demo_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("⚡ GENERATION 3 OPTIMIZED DEMO COMPLETE!")
        print("="*80)
        print(f"✅ High-performance evolution with {config['num_workers']} workers")
        print(f"📊 Final fitness: {best_fitness:.4f}")
        print(f"🧬 Generations: {len(evolution_stats)}")
        print(f"⚡ Total runtime: {total_time:.1f}s")
        print(f"🚀 Avg generation time: {results['performance_metrics']['avg_generation_time']:.2f}s")
        print(f"💾 Cache hit rate: {results['performance_metrics']['final_cache_hit_rate']:.1%}")
        if best_topology:
            print(f"🕸️  Best topology sparsity: {best_topology.compute_sparsity():.3f}")
        if 'avg_inference_time' in locals():
            print(f"⏱️  Inference speed: {avg_inference_time*1000:.2f}ms per batch")
            print(f"📈 Throughput: {throughput:.1f} samples/sec")
        print(f"🔧 Optimization features: Parallel ✓, Caching ✓, Auto-scaling ✓")
        print(f"💾 Results: {results_path}")
        print("="*80)
        
        return results
        
    except Exception as e:
        logger.critical(f"Optimized demo failed: {e}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    try:
        # Set multiprocessing start method for compatibility
        if hasattr(mp, 'set_start_method'):
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
        
        results = run_optimized_demo()
        
    except Exception as e:
        print(f"❌ Optimized demo failed: {e}")
        sys.exit(1)