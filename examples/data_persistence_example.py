#!/usr/bin/env python3
"""
Data persistence and caching example.

This example demonstrates the data layer functionality including:
- Topology repository for storing evolved topologies
- Experiment tracking and history
- Caching for performance optimization
- Model storage and versioning
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from self_evolving_moe import (
    EvolvingMoERouter, ExpertPool, TopologyGenome,
    TopologyRepository, ExperimentRepository, EvolutionCache, ModelStorage
)
from self_evolving_moe.evolution.router import EvolutionConfig
from self_evolving_moe.data.repository import SQLiteRepository
from self_evolving_moe.data.storage import LocalStorageBackend


def create_sample_topology(num_tokens=64, num_experts=8):
    """Create a sample topology for demonstration."""
    topology = TopologyGenome(
        num_tokens=num_tokens,
        num_experts=num_experts,
        sparsity=0.8,
        device="cpu"
    )
    
    # Simulate some evolution
    for _ in range(5):
        topology = topology.mutate(0.1)
        topology.fitness_history.append(0.5 + len(topology.fitness_history) * 0.1)
    
    return topology


def demonstrate_topology_repository():
    """Demonstrate topology repository functionality."""
    print("\n" + "="*60)
    print("TOPOLOGY REPOSITORY DEMONSTRATION")
    print("="*60)
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "topologies.db")
        storage_dir = os.path.join(temp_dir, "topology_storage")
        
        # Initialize repository
        backend = SQLiteRepository(db_path)
        repository = TopologyRepository(backend, storage_dir)
        
        print(f"✅ Initialized repository with database: {db_path}")
        
        # Create and save some topologies
        experiment_id = "exp_001"
        topologies = []
        
        for i in range(3):
            topology = create_sample_topology()
            fitness = 0.6 + i * 0.1
            
            topology_id = repository.save_topology(
                topology=topology,
                experiment_id=experiment_id,
                fitness=fitness,
                metadata={"iteration": i, "notes": f"Sample topology {i}"}
            )
            
            topologies.append((topology_id, topology, fitness))
            print(f"💾 Saved topology {i+1}: ID={topology_id}, fitness={fitness:.3f}")
        
        # Retrieve topologies
        print(f"\n📊 Repository Statistics:")
        best_topologies = repository.get_best_topologies(experiment_id, limit=5)
        print(f"   • Total topologies for experiment: {len(best_topologies)}")
        
        for i, record in enumerate(best_topologies):
            print(f"   • #{i+1}: ID={record.topology_id[:8]}..., "
                  f"fitness={record.fitness:.3f}, sparsity={record.sparsity:.3f}")
        
        # Load a topology
        if best_topologies:
            best_record = best_topologies[0]
            loaded_topology = repository.load_topology(best_record.topology_id)
            
            if loaded_topology:
                print(f"✅ Successfully loaded topology: {best_record.topology_id[:8]}...")
                summary = loaded_topology.get_topology_summary()
                print(f"   • Connections: {summary['total_connections']}")
                print(f"   • Sparsity: {summary['sparsity']:.3f}")
                print(f"   • Generation: {summary['generation']}")
        
        # Evolution history
        evolution_history = repository.get_topology_evolution(experiment_id)
        print(f"\n📈 Evolution History: {len(evolution_history)} topologies")
        for record in evolution_history:
            print(f"   • Gen {record.generation}: fitness={record.fitness:.3f}")


def demonstrate_experiment_repository():
    """Demonstrate experiment repository functionality."""
    print("\n" + "="*60)
    print("EXPERIMENT REPOSITORY DEMONSTRATION")
    print("="*60)
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "experiments.db")
        
        # Initialize repository
        backend = SQLiteRepository(db_path)
        exp_repository = ExperimentRepository(backend)
        
        print(f"✅ Initialized experiment repository")
        
        # Create experiments
        configs = [
            EvolutionConfig(population_size=50, generations=100, mutation_rate=0.1),
            EvolutionConfig(population_size=30, generations=200, mutation_rate=0.2),
            EvolutionConfig(population_size=100, generations=50, mutation_rate=0.05)
        ]
        
        experiment_ids = []
        for i, config in enumerate(configs):
            exp_id = exp_repository.create_experiment(
                name=f"Evolution Experiment {i+1}",
                config=config,
                metadata={"researcher": "Demo", "task": "routing_optimization"}
            )
            experiment_ids.append(exp_id)
            print(f"🔬 Created experiment {i+1}: {exp_id}")
        
        # Update experiment progress
        for i, exp_id in enumerate(experiment_ids):
            best_fitness = 0.7 + i * 0.1
            generations_run = 25 + i * 10
            
            exp_repository.update_experiment(
                experiment_id=exp_id,
                status="running",
                best_fitness=best_fitness,
                generations_run=generations_run
            )
            print(f"📊 Updated experiment {i+1}: fitness={best_fitness:.3f}, "
                  f"generations={generations_run}")
        
        # Complete first experiment
        exp_repository.update_experiment(
            experiment_id=experiment_ids[0],
            status="completed",
            end_time=None  # Would use datetime.now() in real usage
        )
        print(f"✅ Completed experiment: {experiment_ids[0]}")
        
        # List experiments
        print(f"\n📋 All Experiments:")
        all_experiments = exp_repository.list_experiments()
        for exp in all_experiments:
            print(f"   • {exp.name}: {exp.status}, best_fitness={exp.best_fitness}")
        
        # Get statistics
        stats = exp_repository.get_experiment_stats()
        print(f"\n📊 Repository Statistics:")
        print(f"   • Total experiments: {stats['total_experiments']}")
        print(f"   • Status counts: {stats['status_counts']}")
        print(f"   • Best fitness overall: {stats['best_fitness']}")


def demonstrate_evolution_cache():
    """Demonstrate evolution caching functionality."""
    print("\n" + "="*60)
    print("EVOLUTION CACHE DEMONSTRATION")
    print("="*60)
    
    # Initialize cache
    cache = EvolutionCache(
        memory_cache_size=100,
        memory_cache_mb=10,
        persistent_cache_dir=None,  # Use default
        enable_persistent=False     # Disable for demo
    )
    
    print(f"✅ Initialized evolution cache")
    
    # Create sample data
    topology1 = create_sample_topology(num_tokens=32, num_experts=4)
    topology2 = create_sample_topology(num_tokens=32, num_experts=4)
    model_hash = "demo_model_abc123"
    
    # Cache fitness evaluations
    print(f"\n🏃 Fitness Caching:")
    
    # Cache miss - first evaluation
    cached_fitness = cache.get_fitness(topology1, model_hash)
    print(f"   • Cache miss for topology 1: {cached_fitness is None}")
    
    # Cache the fitness
    fitness_score = 0.85
    metrics = {"accuracy": 0.9, "latency": 15.2, "sparsity": 0.82}
    cache.cache_fitness(topology1, model_hash, fitness_score, metrics)
    print(f"   • Cached fitness for topology 1: {fitness_score}")
    
    # Cache hit - second access
    cached_fitness = cache.get_fitness(topology1, model_hash)
    print(f"   • Cache hit for topology 1: {cached_fitness is not None}")
    if cached_fitness:
        cached_score, cached_metrics = cached_fitness
        print(f"     - Cached score: {cached_score}")
        print(f"     - Cached metrics: {cached_metrics}")
    
    # Topology similarity caching
    print(f"\n🔗 Topology Similarity Caching:")
    
    # Cache miss
    similarity = cache.get_topology_similarity(topology1, topology2)
    print(f"   • Cache miss for similarity: {similarity is None}")
    
    # Cache similarity
    similarity_score = 0.73
    cache.cache_topology_similarity(topology1, topology2, similarity_score)
    print(f"   • Cached similarity: {similarity_score}")
    
    # Cache hit
    cached_similarity = cache.get_topology_similarity(topology1, topology2)
    print(f"   • Cache hit for similarity: {cached_similarity}")
    
    # Cache statistics
    print(f"\n📊 Cache Statistics:")
    stats = cache.get_stats()
    mem_stats = stats['memory_cache']
    print(f"   • Memory cache size: {mem_stats['size']}/{mem_stats['max_size']}")
    print(f"   • Memory usage: {mem_stats['memory_mb']:.2f}/{mem_stats['max_memory_mb']:.2f} MB")
    print(f"   • Hit rate: {mem_stats['hit_rate']:.2%}")
    print(f"   • Hits/Misses: {mem_stats['hits']}/{mem_stats['misses']}")


def demonstrate_model_storage():
    """Demonstrate model storage functionality."""
    print("\n" + "="*60)
    print("MODEL STORAGE DEMONSTRATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize storage
        backend = LocalStorageBackend(os.path.join(temp_dir, "model_storage"))
        registry_path = os.path.join(temp_dir, "model_registry.json")
        storage = ModelStorage(backend, registry_path)
        
        print(f"✅ Initialized model storage")
        
        # Create and save sample models
        print(f"\n💾 Saving Models:")
        
        # Save topology
        topology = create_sample_topology()
        topology_id = storage.save_topology(
            topology=topology,
            name="Best Evolved Topology",
            version="1.0",
            description="High-performance sparse routing topology",
            tags=["optimized", "sparse", "v1"]
        )
        print(f"   • Saved topology: {topology_id}")
        
        # Save expert pool
        expert_pool = ExpertPool(
            num_experts=8,
            expert_dim=128,
            expert_type="mlp",
            device="cpu"
        )
        pool_id = storage.save_expert_pool(
            expert_pool=expert_pool,
            name="MLP Expert Pool",
            version="1.0",
            description="Collection of MLP experts for routing",
            tags=["mlp", "experts", "base"]
        )
        print(f"   • Saved expert pool: {pool_id}")
        
        # List stored models
        print(f"\n📋 Stored Models:")
        all_models = storage.list_models()
        for model in all_models:
            print(f"   • {model.name} ({model.model_type})")
            print(f"     - ID: {model.model_id}")
            print(f"     - Size: {model.size_bytes / 1024:.1f} KB")
            print(f"     - Parameters: {model.num_parameters:,}")
            print(f"     - Tags: {', '.join(model.tags)}")
        
        # Load a model
        print(f"\n📥 Loading Models:")
        loaded_topology = storage.load_topology(topology_id)
        if loaded_topology:
            print(f"   • Successfully loaded topology: {topology_id}")
            summary = loaded_topology.get_topology_summary()
            print(f"     - Sparsity: {summary['sparsity']:.3f}")
            print(f"     - Connections: {summary['total_connections']}")
        
        # Storage statistics
        print(f"\n📊 Storage Statistics:")
        stats = storage.get_storage_stats()
        registry_stats = stats['registry']
        print(f"   • Total models: {registry_stats['total_models']}")
        print(f"   • Total size: {registry_stats['total_size_mb']:.2f} MB")
        print(f"   • Model types: {registry_stats['model_types']}")


def main():
    """Run all data layer demonstrations."""
    print("🚀 SELF-EVOLVING MOE-ROUTER DATA LAYER DEMONSTRATION")
    print("This example shows the data persistence and caching capabilities.")
    
    try:
        demonstrate_topology_repository()
        demonstrate_experiment_repository()  
        demonstrate_evolution_cache()
        demonstrate_model_storage()
        
        print("\n" + "="*60)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe data layer provides:")
        print("• 🗄️  Topology repository with SQLite backend")
        print("• 🔬 Experiment tracking and history")
        print("• ⚡ Multi-level caching (memory + persistent)")
        print("• 💾 Model storage with versioning")
        print("• 📊 Comprehensive statistics and monitoring")
        print("\nThis enables:")
        print("• Experiment reproducibility")
        print("• Performance optimization through caching")
        print("• Model versioning and deployment")
        print("• Long-term data persistence")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()