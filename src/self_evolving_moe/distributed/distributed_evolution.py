"""
Distributed evolution system for Self-Evolving MoE Router.

This module implements distributed and parallel evolution across multiple
devices, processes, and potentially multiple machines for maximum scalability.
"""

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import os
import json
import pickle
import queue
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import socket
import subprocess

from ..utils.logging import get_logger
from ..routing.topology import TopologyGenome
from ..evolution.router import EvolutionConfig

logger = get_logger(__name__)


class DistributionStrategy(Enum):
    """Distribution strategies for parallel evolution."""
    DATA_PARALLEL = "data_parallel"        # Same model, different data
    MODEL_PARALLEL = "model_parallel"      # Different model parts
    POPULATION_PARALLEL = "population_parallel"  # Different population subsets
    ISLAND_MODEL = "island_model"          # Independent populations with migration
    HIERARCHICAL = "hierarchical"          # Multi-level parallelism


@dataclass
class DistributedConfig:
    """Configuration for distributed evolution."""
    strategy: DistributionStrategy = DistributionStrategy.POPULATION_PARALLEL
    
    # Process configuration
    world_size: int = 1
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    master_addr: str = "localhost"
    master_port: str = "12355"
    
    # Population distribution
    population_per_worker: int = 20
    migration_interval: int = 10  # generations
    migration_rate: float = 0.1   # fraction of population to migrate
    
    # Communication
    reduce_frequency: int = 5     # generations between reductions
    async_communication: bool = True
    compression: bool = True
    
    # Load balancing
    enable_dynamic_load_balancing: bool = True
    work_stealing: bool = True
    
    # Fault tolerance
    enable_checkpointing: bool = True
    checkpoint_interval: int = 20
    backup_workers: int = 0


class WorkerProcess:
    """Individual worker process for distributed evolution."""
    
    def __init__(self, rank: int, config: DistributedConfig, evolution_config: EvolutionConfig):
        self.rank = rank
        self.config = config
        self.evolution_config = evolution_config
        
        # Local state
        self.local_population: List[TopologyGenome] = []
        self.local_fitness: List[float] = []
        self.generation = 0
        self.best_topology: Optional[TopologyGenome] = None
        self.best_fitness = float('-inf')
        
        # Communication
        self.message_queue = queue.Queue()
        self.migration_buffer: List[TopologyGenome] = []
        
        logger.info(f"Initialized worker process rank {rank}")
    
    def initialize_distributed(self):
        """Initialize distributed training backend."""
        try:
            # Set environment variables
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port
            os.environ['RANK'] = str(self.rank)
            os.environ['WORLD_SIZE'] = str(self.config.world_size)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                rank=self.rank,
                world_size=self.config.world_size
            )
            
            logger.info(f"Worker {self.rank} initialized distributed backend")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed backend: {e}")
            raise
    
    def initialize_population(self, total_population_size: int):
        """Initialize local population subset."""
        # Calculate population slice for this worker
        pop_per_worker = total_population_size // self.config.world_size
        extra = total_population_size % self.config.world_size
        
        local_pop_size = pop_per_worker + (1 if self.rank < extra else 0)
        
        # Initialize local population
        self.local_population = []
        for _ in range(local_pop_size):
            topology = TopologyGenome(
                num_tokens=64,  # This should come from config
                num_experts=16,  # This should come from config
                sparsity=self.evolution_config.target_sparsity
            )
            self.local_population.append(topology)
        
        logger.info(f"Worker {self.rank} initialized {local_pop_size} individuals")
    
    def evolve_local_population(self, model, data_loader, fitness_evaluator):
        """Evolve local population subset."""
        logger.info(f"Worker {self.rank} evolving generation {self.generation}")
        
        try:
            # Evaluate fitness for local population
            self.local_fitness = []
            
            for i, topology in enumerate(self.local_population):
                try:
                    fitness_scores = fitness_evaluator.evaluate(
                        topology, model, data_loader, self.generation, self.local_population
                    )
                    fitness = fitness_scores['combined_fitness']
                    self.local_fitness.append(fitness)
                    
                    # Track local best
                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.best_topology = topology
                        
                except Exception as e:
                    logger.warning(f"Worker {self.rank} fitness evaluation failed for individual {i}: {e}")
                    self.local_fitness.append(-1000.0)
            
            # Create next generation
            self._create_next_generation()
            
            # Handle migration
            if self.generation % self.config.migration_interval == 0:
                self._prepare_migration()
            
            self.generation += 1
            
            return {
                'local_best_fitness': max(self.local_fitness) if self.local_fitness else -float('inf'),
                'local_avg_fitness': np.mean(self.local_fitness) if self.local_fitness else 0,
                'local_population_size': len(self.local_population)
            }
            
        except Exception as e:
            logger.error(f"Worker {self.rank} evolution failed: {e}")
            return {'error': str(e)}
    
    def _create_next_generation(self):
        """Create next generation from local population."""
        if not self.local_population or not self.local_fitness:
            return
        
        new_population = []
        
        # Elitism - keep best individuals
        elite_count = max(1, int(0.1 * len(self.local_population)))
        elite_indices = np.argsort(self.local_fitness)[-elite_count:]
        for idx in elite_indices:
            new_population.append(self.local_population[idx])
        
        # Generate rest through selection and mutation
        while len(new_population) < len(self.local_population):
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.random() < self.evolution_config.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = parent1
            
            # Mutation
            if np.random.random() < self.evolution_config.mutation_rate:
                child = child.mutate(self.evolution_config.mutation_rate)
            
            new_population.append(child)
        
        # Apply migrations if available
        if self.migration_buffer:
            self._apply_migrations(new_population)
        
        self.local_population = new_population[:len(self.local_population)]
    
    def _tournament_selection(self, tournament_size: int = 3) -> TopologyGenome:
        """Tournament selection from local population."""
        if not self.local_population or not self.local_fitness:
            return self.local_population[0] if self.local_population else None
        
        tournament_indices = np.random.choice(
            len(self.local_population),
            size=min(tournament_size, len(self.local_population)),
            replace=False
        )
        
        best_idx = max(tournament_indices, key=lambda i: self.local_fitness[i])
        return self.local_population[best_idx]
    
    def _prepare_migration(self):
        """Prepare individuals for migration to other workers."""
        if not self.local_population or not self.local_fitness:
            return
        
        # Select best individuals for migration
        migration_count = max(1, int(self.config.migration_rate * len(self.local_population)))
        best_indices = np.argsort(self.local_fitness)[-migration_count:]
        
        migrants = [self.local_population[idx] for idx in best_indices]
        
        # Send migrants to next worker (ring topology)
        next_worker = (self.rank + 1) % self.config.world_size
        self._send_migrants(migrants, next_worker)
    
    def _send_migrants(self, migrants: List[TopologyGenome], target_worker: int):
        """Send migrants to target worker."""
        try:
            # Serialize migrants
            migrant_data = [topology.to_dict() for topology in migrants]
            
            # Use distributed communication
            tensor_data = torch.tensor([len(migrant_data)], dtype=torch.long)
            dist.send(tensor_data, dst=target_worker)
            
            # Send each migrant
            for migrant_dict in migrant_data:
                migrant_bytes = pickle.dumps(migrant_dict)
                migrant_tensor = torch.frombuffer(migrant_bytes, dtype=torch.uint8)
                
                # Send size first
                size_tensor = torch.tensor([len(migrant_tensor)], dtype=torch.long)
                dist.send(size_tensor, dst=target_worker)
                
                # Send data
                dist.send(migrant_tensor, dst=target_worker)
            
            logger.debug(f"Worker {self.rank} sent {len(migrants)} migrants to worker {target_worker}")
            
        except Exception as e:
            logger.error(f"Failed to send migrants: {e}")
    
    def _receive_migrants(self):
        """Receive migrants from other workers."""
        try:
            # Check for incoming migrants (non-blocking)
            prev_worker = (self.rank - 1) % self.config.world_size
            
            # Try to receive count
            count_tensor = torch.tensor([0], dtype=torch.long)
            try:
                dist.recv(count_tensor, src=prev_worker)
                migrant_count = count_tensor.item()
                
                if migrant_count > 0:
                    migrants = []
                    
                    for _ in range(migrant_count):
                        # Receive size
                        size_tensor = torch.tensor([0], dtype=torch.long)
                        dist.recv(size_tensor, src=prev_worker)
                        migrant_size = size_tensor.item()
                        
                        # Receive data
                        migrant_tensor = torch.zeros(migrant_size, dtype=torch.uint8)
                        dist.recv(migrant_tensor, src=prev_worker)
                        
                        # Deserialize
                        migrant_bytes = migrant_tensor.numpy().tobytes()
                        migrant_dict = pickle.loads(migrant_bytes)
                        migrant_topology = TopologyGenome.from_dict(migrant_dict)
                        
                        migrants.append(migrant_topology)
                    
                    self.migration_buffer.extend(migrants)
                    logger.debug(f"Worker {self.rank} received {len(migrants)} migrants")
                    
            except Exception as e:
                # No migrants available (expected in non-blocking mode)
                pass
                
        except Exception as e:
            logger.error(f"Failed to receive migrants: {e}")
    
    def _apply_migrations(self, new_population: List[TopologyGenome]):
        """Apply received migrants to new population."""
        if not self.migration_buffer:
            return
        
        # Replace worst individuals with migrants
        replace_count = min(len(self.migration_buffer), len(new_population) // 4)
        
        if replace_count > 0:
            # Sort new population by fitness (assuming we have fitness values)
            # For now, replace random individuals
            replace_indices = np.random.choice(
                len(new_population), size=replace_count, replace=False
            )
            
            for i, idx in enumerate(replace_indices):
                if i < len(self.migration_buffer):
                    new_population[idx] = self.migration_buffer[i]
        
        # Clear migration buffer
        self.migration_buffer.clear()


class DistributedEvolutionMaster:
    """Master coordinator for distributed evolution."""
    
    def __init__(self, config: DistributedConfig, evolution_config: EvolutionConfig):
        self.config = config
        self.evolution_config = evolution_config
        
        # Global state
        self.global_best_fitness = float('-inf')
        self.global_best_topology: Optional[TopologyGenome] = None
        self.generation = 0
        
        # Worker management
        self.workers: List[WorkerProcess] = []
        self.worker_stats: Dict[int, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_time': 0,
            'communication_time': 0,
            'computation_time': 0,
            'load_balance_efficiency': []
        }
        
        logger.info(f"Initialized distributed evolution master with {config.world_size} workers")
    
    def initialize_workers(self):
        """Initialize all worker processes."""
        self.workers = []
        
        for rank in range(self.config.world_size):
            worker = WorkerProcess(rank, self.config, self.evolution_config)
            self.workers.append(worker)
        
        logger.info(f"Created {len(self.workers)} worker processes")
    
    def run_distributed_evolution(self, model, data_loader, fitness_evaluator, generations: int):
        """Run distributed evolution across all workers."""
        logger.info(f"Starting distributed evolution for {generations} generations")
        
        start_time = time.time()
        
        try:
            # Initialize distributed backend
            self._setup_distributed_environment()
            
            # Initialize populations
            total_population_size = self.evolution_config.population_size
            
            # Run evolution loop
            for gen in range(generations):
                gen_start_time = time.time()
                
                # Evolve each worker's population
                worker_results = self._evolve_generation_parallel(
                    model, data_loader, fitness_evaluator
                )
                
                # Aggregate results
                self._aggregate_results(worker_results)
                
                # Periodic synchronization
                if gen % self.config.reduce_frequency == 0:
                    self._synchronize_workers()
                
                # Log progress
                gen_time = time.time() - gen_start_time
                self.performance_stats['computation_time'] += gen_time
                
                if gen % 10 == 0:
                    logger.info(
                        f"Generation {gen}: "
                        f"Global best fitness = {self.global_best_fitness:.4f}, "
                        f"Time = {gen_time:.2f}s"
                    )
                
                self.generation = gen
            
            total_time = time.time() - start_time
            self.performance_stats['total_time'] = total_time
            
            logger.info(f"Distributed evolution completed in {total_time:.2f}s")
            
            return {
                'best_fitness': self.global_best_fitness,
                'best_topology': self.global_best_topology,
                'generations_completed': generations,
                'performance_stats': self.performance_stats,
                'worker_stats': self.worker_stats
            }
            
        except Exception as e:
            logger.error(f"Distributed evolution failed: {e}")
            raise
        
        finally:
            self._cleanup_distributed()
    
    def _setup_distributed_environment(self):
        """Setup distributed computing environment."""
        try:
            # Set CUDA device for each worker if available
            if torch.cuda.is_available():
                for rank, worker in enumerate(self.workers):
                    device_id = rank % torch.cuda.device_count()
                    torch.cuda.set_device(device_id)
                    logger.info(f"Worker {rank} assigned to CUDA device {device_id}")
            
            # Initialize each worker's distributed backend
            for worker in self.workers:
                worker.initialize_distributed()
                worker.initialize_population(self.evolution_config.population_size)
            
            logger.info("Distributed environment setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup distributed environment: {e}")
            raise
    
    def _evolve_generation_parallel(self, model, data_loader, fitness_evaluator):
        """Evolve one generation across all workers in parallel."""
        # This would typically use multiprocessing or distributed training
        # For demonstration, we'll simulate parallel execution
        
        worker_results = {}
        
        for worker in self.workers:
            try:
                # In real implementation, this would be executed in parallel
                result = worker.evolve_local_population(model, data_loader, fitness_evaluator)
                worker_results[worker.rank] = result
                
            except Exception as e:
                logger.error(f"Worker {worker.rank} evolution failed: {e}")
                worker_results[worker.rank] = {'error': str(e)}
        
        return worker_results
    
    def _aggregate_results(self, worker_results: Dict[int, Dict[str, Any]]):
        """Aggregate results from all workers."""
        valid_results = {k: v for k, v in worker_results.items() if 'error' not in v}
        
        if not valid_results:
            logger.warning("No valid worker results to aggregate")
            return
        
        # Find global best
        for rank, result in valid_results.items():
            local_best = result.get('local_best_fitness', float('-inf'))
            if local_best > self.global_best_fitness:
                self.global_best_fitness = local_best
                # Get the actual topology from the worker
                if rank < len(self.workers) and self.workers[rank].best_topology:
                    self.global_best_topology = self.workers[rank].best_topology
        
        # Update worker statistics
        self.worker_stats = worker_results
        
        # Calculate load balance efficiency
        if len(valid_results) > 1:
            population_sizes = [r.get('local_population_size', 0) for r in valid_results.values()]
            if population_sizes:
                mean_size = np.mean(population_sizes)
                variance = np.var(population_sizes)
                efficiency = 1.0 / (1.0 + variance / max(mean_size, 1))
                self.performance_stats['load_balance_efficiency'].append(efficiency)
    
    def _synchronize_workers(self):
        """Synchronize workers and share best solutions."""
        logger.info(f"Synchronizing workers at generation {self.generation}")
        
        try:
            # Share global best with all workers
            if self.global_best_topology:
                for worker in self.workers:
                    # In real implementation, this would use distributed communication
                    # For now, we'll update the worker's best topology directly
                    if worker.best_fitness < self.global_best_fitness:
                        worker.best_topology = self.global_best_topology
                        worker.best_fitness = self.global_best_fitness
            
            # Handle migration
            for worker in self.workers:
                worker._receive_migrants()
            
            logger.info("Worker synchronization complete")
            
        except Exception as e:
            logger.error(f"Worker synchronization failed: {e}")
    
    def _cleanup_distributed(self):
        """Cleanup distributed resources."""
        try:
            # Cleanup distributed backend
            if dist.is_initialized():
                dist.destroy_process_group()
            
            # Clear worker resources
            for worker in self.workers:
                # In real implementation, would cleanup worker processes
                pass
            
            logger.info("Distributed cleanup complete")
            
        except Exception as e:
            logger.error(f"Distributed cleanup failed: {e}")
    
    def get_distributed_statistics(self) -> Dict[str, Any]:
        """Get comprehensive distributed execution statistics."""
        stats = {
            'configuration': {
                'world_size': self.config.world_size,
                'strategy': self.config.strategy.value,
                'backend': self.config.backend,
                'migration_interval': self.config.migration_interval,
                'migration_rate': self.config.migration_rate
            },
            'performance': self.performance_stats,
            'worker_statistics': self.worker_stats,
            'global_results': {
                'best_fitness': self.global_best_fitness,
                'generations_completed': self.generation
            }
        }
        
        # Calculate efficiency metrics
        if self.performance_stats['load_balance_efficiency']:
            stats['efficiency_metrics'] = {
                'avg_load_balance': np.mean(self.performance_stats['load_balance_efficiency']),
                'communication_overhead': (
                    self.performance_stats['communication_time'] / 
                    max(self.performance_stats['total_time'], 1)
                ),
                'parallel_efficiency': (
                    self.performance_stats['computation_time'] / 
                    max(self.performance_stats['total_time'], 1)
                )
            }
        
        return stats


def spawn_distributed_evolution(
    config: DistributedConfig,
    evolution_config: EvolutionConfig,
    model_factory: Callable,
    data_loader_factory: Callable,
    fitness_evaluator_factory: Callable,
    generations: int
):
    """
    Spawn distributed evolution using multiprocessing.
    
    This function handles the complexity of starting multiple processes
    and coordinating distributed evolution.
    """
    
    def worker_main(rank: int, world_size: int):
        """Main function for worker process."""
        try:
            # Initialize worker
            worker = WorkerProcess(rank, config, evolution_config)
            worker.initialize_distributed()
            
            # Create model, data loader, and fitness evaluator for this worker
            model = model_factory()
            data_loader = data_loader_factory()
            fitness_evaluator = fitness_evaluator_factory()
            
            # Initialize population
            worker.initialize_population(evolution_config.population_size)
            
            # Run evolution loop
            for gen in range(generations):
                result = worker.evolve_local_population(model, data_loader, fitness_evaluator)
                
                # Periodic synchronization
                if gen % config.reduce_frequency == 0:
                    # Synchronization logic would go here
                    pass
            
            logger.info(f"Worker {rank} completed evolution")
            
        except Exception as e:
            logger.error(f"Worker {rank} failed: {e}")
            raise
    
    # Launch worker processes
    try:
        mp.spawn(
            worker_main,
            args=(config.world_size,),
            nprocs=config.world_size,
            join=True
        )
        
        logger.info("Distributed evolution completed successfully")
        
    except Exception as e:
        logger.error(f"Distributed evolution spawn failed: {e}")
        raise


def setup_distributed_environment(config: DistributedConfig) -> bool:
    """
    Setup distributed computing environment.
    
    Returns:
        True if setup successful, False otherwise
    """
    try:
        # Check if distributed training is available
        if not dist.is_available():
            logger.warning("Distributed training not available")
            return False
        
        # Check for CUDA if using NCCL backend
        if config.backend == "nccl" and not torch.cuda.is_available():
            logger.warning("NCCL backend requires CUDA, falling back to Gloo")
            config.backend = "gloo"
        
        # Validate network connectivity
        try:
            socket.create_connection((config.master_addr, int(config.master_port)), timeout=5)
        except Exception:
            logger.warning(f"Cannot connect to {config.master_addr}:{config.master_port}")
            return False
        
        logger.info("Distributed environment validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Distributed environment setup failed: {e}")
        return False