"""
Parallel and distributed evolution implementation.

This module provides distributed evolution capabilities using multiprocessing,
threading, and optional distributed computing frameworks like Ray or Dask.
"""

import torch
import torch.multiprocessing as mp
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import numpy as np
import time
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import pickle
import queue
import threading
from pathlib import Path

from ..evolution.router import EvolvingMoERouter, EvolutionConfig, FitnessEvaluator
from ..routing.topology import TopologyGenome
from ..experts.pool import ExpertPool
from ..utils.logging import get_evolution_logger
from ..utils.monitoring import PerformanceTracker


@dataclass
class DistributedConfig:
    """Configuration for distributed evolution."""
    num_workers: int = 4
    use_multiprocessing: bool = True
    batch_size_per_worker: int = 10
    communication_backend: str = "threading"  # "threading", "multiprocessing", "ray", "dask"
    worker_timeout: float = 300.0  # 5 minutes
    checkpoint_interval: int = 10  # Save every N generations
    load_balancing: bool = True
    fault_tolerance: bool = True


class WorkerTask:
    """Individual task for distributed workers."""
    
    def __init__(
        self,
        task_id: str,
        topologies: List[TopologyGenome],
        model_state: Dict[str, Any],
        expert_pool_config: Dict[str, Any],
        data_batch: List[Tuple[torch.Tensor, torch.Tensor]],
        config: EvolutionConfig
    ):
        self.task_id = task_id
        self.topologies = topologies
        self.model_state = model_state
        self.expert_pool_config = expert_pool_config
        self.data_batch = data_batch
        self.config = config


class WorkerResult:
    """Result from distributed worker."""
    
    def __init__(
        self,
        task_id: str,
        fitness_scores: List[float],
        detailed_metrics: List[Dict[str, Any]],
        execution_time: float,
        worker_id: str,
        success: bool = True,
        error_message: str = None
    ):
        self.task_id = task_id
        self.fitness_scores = fitness_scores
        self.detailed_metrics = detailed_metrics
        self.execution_time = execution_time
        self.worker_id = worker_id
        self.success = success
        self.error_message = error_message


def evaluate_topologies_worker(task: WorkerTask) -> WorkerResult:
    """
    Worker function for evaluating topologies in separate process.
    
    Args:
        task: WorkerTask containing topologies and data to evaluate
        
    Returns:
        WorkerResult with fitness scores and metrics
    """
    worker_id = f"worker_{mp.current_process().pid}"
    start_time = time.time()
    
    try:
        # Reconstruct components in worker process
        expert_pool = ExpertPool(**task.expert_pool_config)
        
        # Create simple model for evaluation
        model = torch.nn.Sequential(
            torch.nn.Linear(expert_pool.expert_dim, expert_pool.expert_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(expert_pool.expert_dim, 10)  # Assume 10 classes
        )
        
        # Load model state if provided
        if task.model_state:
            try:
                model.load_state_dict(task.model_state)
            except:
                pass  # Use default initialization if loading fails
        
        # Create fitness evaluator
        fitness_evaluator = FitnessEvaluator(task.config)
        
        # Create data loaders from batch
        class SimpleDataset:
            def __init__(self, data):
                self.data = data
            def __iter__(self):
                return iter(self.data)
            def __len__(self):
                return len(self.data)
        
        train_loader = SimpleDataset(task.data_batch)
        val_loader = SimpleDataset(task.data_batch)
        
        # Evaluate each topology
        fitness_scores = []
        detailed_metrics = []
        
        for topology in task.topologies:
            try:
                fitness, metrics = fitness_evaluator.evaluate(
                    topology, model, train_loader, val_loader
                )
                fitness_scores.append(fitness)
                detailed_metrics.append(metrics)
            except Exception as e:
                # Handle individual topology evaluation failure
                fitness_scores.append(float('-inf'))
                detailed_metrics.append({'error': str(e)})
        
        execution_time = time.time() - start_time
        
        return WorkerResult(
            task_id=task.task_id,
            fitness_scores=fitness_scores,
            detailed_metrics=detailed_metrics,
            execution_time=execution_time,
            worker_id=worker_id,
            success=True
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        return WorkerResult(
            task_id=task.task_id,
            fitness_scores=[],
            detailed_metrics=[],
            execution_time=execution_time,
            worker_id=worker_id,
            success=False,
            error_message=str(e)
        )


class ParallelFitnessEvaluator:
    """
    Parallel fitness evaluator that distributes topology evaluation across workers.
    
    Uses multiprocessing or threading to evaluate multiple topologies in parallel,
    significantly speeding up evolution for large populations.
    """
    
    def __init__(
        self,
        config: EvolutionConfig,
        distributed_config: DistributedConfig,
        expert_pool: ExpertPool
    ):
        self.config = config
        self.distributed_config = distributed_config
        self.expert_pool = expert_pool
        self.logger = get_evolution_logger(__name__)
        
        # Worker pool
        self.executor = None
        self.active_tasks = {}
        self.task_counter = 0
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
    
    def __enter__(self):
        """Context manager entry."""
        self._start_workers()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._stop_workers()
    
    def _start_workers(self):
        """Start worker pool."""
        if self.distributed_config.use_multiprocessing:
            self.executor = ProcessPoolExecutor(
                max_workers=self.distributed_config.num_workers,
                mp_context=mp.get_context('spawn')
            )
            self.logger.info(f"Started {self.distributed_config.num_workers} worker processes")
        else:
            self.executor = ThreadPoolExecutor(
                max_workers=self.distributed_config.num_workers
            )
            self.logger.info(f"Started {self.distributed_config.num_workers} worker threads")
    
    def _stop_workers(self):
        """Stop worker pool."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.logger.info("Stopped worker pool")
    
    def evaluate_population_parallel(
        self,
        population: List[TopologyGenome],
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Evaluate entire population in parallel.
        
        Args:
            population: List of topologies to evaluate
            model: Model to evaluate topologies with
            train_data: Training data loader
            val_data: Validation data loader
            
        Returns:
            Tuple of (fitness_scores, detailed_metrics)
        """
        start_time = time.time()
        
        # Prepare model state for serialization
        model_state = model.state_dict() if hasattr(model, 'state_dict') else {}
        
        # Expert pool configuration
        expert_pool_config = {
            'num_experts': self.expert_pool.num_experts,
            'expert_dim': self.expert_pool.expert_dim,
            'expert_type': self.expert_pool.expert_type,
            'ffn_dim': self.expert_pool.ffn_dim,
            'device': 'cpu'  # Workers use CPU
        }
        
        # Convert data loaders to batches for serialization
        train_batch = []
        val_batch = []
        
        # Sample limited data for worker efficiency
        max_samples = 50  # Limit for parallel evaluation
        
        for i, batch in enumerate(train_data):
            if i >= max_samples:
                break
            train_batch.append(batch)
        
        for i, batch in enumerate(val_data):
            if i >= max_samples:
                break
            val_batch.append(batch)
        
        # Split population into worker batches
        batch_size = self.distributed_config.batch_size_per_worker
        topology_batches = [
            population[i:i + batch_size]
            for i in range(0, len(population), batch_size)
        ]
        
        # Submit tasks to workers
        futures = []
        tasks = []
        
        for i, topology_batch in enumerate(topology_batches):
            task_id = f"eval_batch_{self.task_counter}_{i}"
            self.task_counter += 1
            
            task = WorkerTask(
                task_id=task_id,
                topologies=topology_batch,
                model_state=model_state,
                expert_pool_config=expert_pool_config,
                data_batch=train_batch + val_batch,  # Combined for simplicity
                config=self.config
            )
            
            future = self.executor.submit(evaluate_topologies_worker, task)
            futures.append(future)
            tasks.append(task)
            
            self.active_tasks[task_id] = {
                'task': task,
                'future': future,
                'start_time': time.time()
            }
        
        self.logger.info(f"Submitted {len(futures)} evaluation tasks to workers")
        
        # Collect results
        all_fitness_scores = []
        all_detailed_metrics = []
        successful_tasks = 0
        failed_tasks = 0
        
        for future in as_completed(futures, timeout=self.distributed_config.worker_timeout):
            try:
                result = future.result()
                
                if result.success:
                    all_fitness_scores.extend(result.fitness_scores)
                    all_detailed_metrics.extend(result.detailed_metrics)
                    successful_tasks += 1
                    
                    self.logger.debug(
                        f"Task {result.task_id} completed successfully in {result.execution_time:.2f}s"
                    )
                else:
                    # Handle failed task
                    failed_tasks += 1
                    self.logger.warning(f"Task {result.task_id} failed: {result.error_message}")
                    
                    # Add placeholder scores for failed evaluations
                    num_topologies = len(self.active_tasks[result.task_id]['task'].topologies)
                    all_fitness_scores.extend([float('-inf')] * num_topologies)
                    all_detailed_metrics.extend([{'error': result.error_message}] * num_topologies)
                
            except Exception as e:
                failed_tasks += 1
                self.logger.error(f"Future result error: {e}")
                
                # Add placeholder scores
                all_fitness_scores.extend([float('-inf')])
                all_detailed_metrics.extend([{'error': str(e)}])
        
        total_time = time.time() - start_time
        
        # Clean up active tasks
        self.active_tasks.clear()
        
        # Log performance
        self.logger.info(
            f"Parallel evaluation completed: {successful_tasks} successful, "
            f"{failed_tasks} failed, {total_time:.2f}s total"
        )
        
        # Record performance metrics
        self.performance_tracker.record_performance_metrics(
            accuracy=0.0,  # Not measured here
            latency_ms=total_time * 1000,
            throughput_samples_per_sec=len(population) / total_time,
            memory_usage_mb=0.0  # Not measured here
        )
        
        return all_fitness_scores, all_detailed_metrics


class DistributedEvolver:
    """
    Distributed evolution coordinator that manages parallel fitness evaluation
    and distributed population management across multiple workers.
    """
    
    def __init__(
        self,
        expert_pool: ExpertPool,
        evolution_config: EvolutionConfig,
        distributed_config: DistributedConfig
    ):
        self.expert_pool = expert_pool
        self.evolution_config = evolution_config
        self.distributed_config = distributed_config
        
        self.logger = get_evolution_logger(__name__)
        self.performance_tracker = PerformanceTracker()
        
        # Evolution state
        self.population: List[TopologyGenome] = []
        self.fitness_scores: List[float] = []
        self.generation = 0
        self.best_topology: Optional[TopologyGenome] = None
        self.best_fitness = float('-inf')
        
        # Distributed components
        self.parallel_evaluator = None
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize random population."""
        self.population = []
        
        num_tokens = self.expert_pool.expert_dim
        num_experts = self.expert_pool.num_experts
        
        for _ in range(self.evolution_config.population_size):
            sparsity = np.random.uniform(0.5, 0.95)
            
            topology = TopologyGenome(
                num_tokens=num_tokens,
                num_experts=num_experts,
                sparsity=sparsity,
                device='cpu'  # Use CPU for distributed coordination
            )
            
            self.population.append(topology)
        
        self.logger.info(f"Initialized population of {len(self.population)} topologies")
    
    def evolve_distributed(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader,
        generations: Optional[int] = None
    ) -> TopologyGenome:
        """
        Run distributed evolution.
        
        Args:
            model: Model to evolve routing for
            train_data: Training data loader
            val_data: Validation data loader
            generations: Number of generations to run
            
        Returns:
            Best topology found
        """
        if generations is None:
            generations = self.evolution_config.generations
        
        self.logger.info(f"Starting distributed evolution for {generations} generations")
        
        # Start distributed components
        with ParallelFitnessEvaluator(
            self.evolution_config,
            self.distributed_config,
            self.expert_pool
        ) as evaluator:
            
            self.parallel_evaluator = evaluator
            
            for gen in range(generations):
                self.generation = gen
                generation_start_time = time.time()
                
                self.logger.log_generation_start(gen, len(self.population))
                
                # Distributed fitness evaluation
                fitness_scores, detailed_metrics = evaluator.evaluate_population_parallel(
                    self.population, model, train_data, val_data
                )
                
                self.fitness_scores = fitness_scores
                
                # Find best topology
                if fitness_scores:
                    current_best_fitness = max(fitness_scores)
                    current_best_idx = fitness_scores.index(current_best_fitness)
                    
                    if current_best_fitness > self.best_fitness:
                        self.best_fitness = current_best_fitness
                        self.best_topology = self.population[current_best_idx].copy() if hasattr(self.population[current_best_idx], 'copy') else self.population[current_best_idx]
                        
                        self.logger.info(f"New best fitness in generation {gen}: {current_best_fitness:.6f}")
                
                # Create next generation
                self._create_next_generation_distributed()
                
                # Generation timing
                generation_time = time.time() - generation_start_time
                
                # Log generation completion
                avg_fitness = np.mean(fitness_scores) if fitness_scores else 0.0
                diversity = np.std(fitness_scores) if len(fitness_scores) > 1 else 0.0
                
                self.logger.log_generation_end(gen, self.best_fitness, avg_fitness, diversity)
                
                # Record evolution metrics
                self.performance_tracker.record_evolution_metrics(
                    generation=gen,
                    best_fitness=self.best_fitness,
                    avg_fitness=avg_fitness,
                    population_diversity=diversity,
                    mutation_rate=self.evolution_config.mutation_rate,
                    convergence_rate=0.0,  # TODO: Calculate
                    active_experts=self.expert_pool.num_experts,
                    topology_sparsity=self.best_topology.compute_sparsity() if self.best_topology else 0.0
                )
                
                # Checkpoint saving
                if (gen + 1) % self.distributed_config.checkpoint_interval == 0:
                    self._save_checkpoint(gen)
        
        self.logger.info(f"Distributed evolution completed. Best fitness: {self.best_fitness:.6f}")
        return self.best_topology
    
    def _create_next_generation_distributed(self):
        """Create next generation with distributed-aware selection."""
        if not self.fitness_scores:
            return
        
        new_population = []
        
        # Elitism - keep best individuals
        num_elite = max(1, int(self.evolution_config.population_size * self.evolution_config.elitism_ratio))
        elite_indices = np.argsort(self.fitness_scores)[-num_elite:]
        
        for idx in elite_indices:
            elite = self._copy_topology(self.population[idx])
            new_population.append(elite)
        
        # Generate rest through selection, crossover, and mutation
        while len(new_population) < self.evolution_config.population_size:
            # Tournament selection
            parent1_idx = self._tournament_selection()
            parent2_idx = self._tournament_selection()
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            if np.random.random() < self.evolution_config.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = self._copy_topology(parent1)
            
            # Mutation
            if np.random.random() < self.evolution_config.mutation_rate:
                child = child.mutate(self.evolution_config.mutation_rate)
            
            new_population.append(child)
        
        self.population = new_population[:self.evolution_config.population_size]
    
    def _tournament_selection(self) -> int:
        """Tournament selection for parent selection."""
        tournament_size = min(self.evolution_config.tournament_size, len(self.population))
        tournament_indices = np.random.choice(
            len(self.population), 
            size=tournament_size, 
            replace=False
        )
        
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        return winner_idx
    
    def _copy_topology(self, topology: TopologyGenome) -> TopologyGenome:
        """Create a copy of topology for safe manipulation."""
        new_topology = TopologyGenome(
            num_tokens=topology.num_tokens,
            num_experts=topology.num_experts,
            sparsity=topology.sparsity,
            routing_params=topology.routing_params,
            device='cpu'
        )
        
        new_topology.routing_matrix = topology.routing_matrix.clone()
        new_topology.expert_graph = topology.expert_graph.clone()
        new_topology.generation = topology.generation
        
        return new_topology
    
    def _save_checkpoint(self, generation: int):
        """Save evolution checkpoint."""
        checkpoint_dir = Path(f"checkpoints/distributed_evolution_gen_{generation}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best topology
        if self.best_topology:
            self.best_topology.save_topology(str(checkpoint_dir / "best_topology.pt"))
        
        # Save evolution state
        state = {
            'generation': generation,
            'best_fitness': self.best_fitness,
            'population_size': len(self.population),
            'evolution_config': self.evolution_config,
            'distributed_config': self.distributed_config
        }
        
        torch.save(state, checkpoint_dir / "evolution_state.pt")
        
        self.logger.info(f"Saved checkpoint for generation {generation}")
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'population_size': len(self.population),
            'distributed_workers': self.distributed_config.num_workers,
            'performance_summary': self.performance_tracker.get_performance_summary(),
            'evolution_summary': self.performance_tracker.get_evolution_summary()
        }