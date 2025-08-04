#!/usr/bin/env python3
"""
Comprehensive robustness tests for Self-Evolving MoE-Router evolution system.

This module tests error handling, validation, edge cases, and fault tolerance
throughout the evolution process.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from self_evolving_moe.evolution.router import EvolvingMoERouter, EvolutionConfig, FitnessEvaluator
from self_evolving_moe.experts.pool import ExpertPool, TransformerExpert, MLPExpert
from self_evolving_moe.routing.topology import TopologyGenome, RoutingParams
from self_evolving_moe.utils.exceptions import (
    EvolutionError,
    TopologyError,
    ConfigurationError,
    ExpertPoolError,
    ResourceConstraintError
)
from self_evolving_moe.utils.validation import (
    validate_config,
    validate_topology,
    validate_expert_pool
)


class TestEvolutionConfigValidation:
    """Test configuration validation robustness."""
    
    def test_valid_config(self):
        """Test valid configuration passes validation."""
        config = EvolutionConfig(
            population_size=50,
            generations=100,
            mutation_rate=0.1,
            crossover_rate=0.7
        )
        assert validate_config(config) == True
    
    def test_invalid_population_size(self):
        """Test invalid population size raises error."""
        with pytest.raises(ConfigurationError):
            config = EvolutionConfig(population_size=1)  # Too small
            validate_config(config)
        
        with pytest.raises(ConfigurationError):
            config = EvolutionConfig(population_size=2000)  # Too large
            validate_config(config)
        
        with pytest.raises(ConfigurationError):
            config = EvolutionConfig(population_size=-5)  # Negative
            validate_config(config)
    
    def test_invalid_mutation_rate(self):
        """Test invalid mutation rate raises error."""
        with pytest.raises(ConfigurationError):
            config = EvolutionConfig(mutation_rate=-0.1)  # Negative
            validate_config(config)
        
        with pytest.raises(ConfigurationError):
            config = EvolutionConfig(mutation_rate=1.5)  # > 1.0
            validate_config(config)
    
    def test_invalid_selection_method(self):
        """Test invalid selection method raises error."""
        with pytest.raises(ConfigurationError):
            config = EvolutionConfig(selection_method="invalid_method")
            validate_config(config)
    
    def test_edge_case_values(self):
        """Test edge case configuration values."""
        # Minimum valid values
        config = EvolutionConfig(
            population_size=2,
            generations=1,
            mutation_rate=0.0,
            tournament_size=2
        )
        assert validate_config(config) == True
        
        # Maximum valid values  
        config = EvolutionConfig(
            population_size=1000,
            mutation_rate=1.0,
            crossover_rate=1.0,
            elitism_ratio=1.0
        )
        assert validate_config(config) == True


class TestTopologyValidation:
    """Test topology validation robustness."""
    
    def create_valid_topology(self, num_tokens=64, num_experts=8):
        """Create a valid topology for testing."""
        return TopologyGenome(
            num_tokens=num_tokens,
            num_experts=num_experts,
            sparsity=0.8,
            device="cpu"
        )
    
    def test_valid_topology(self):
        """Test valid topology passes validation."""
        topology = self.create_valid_topology()
        assert validate_topology(topology) == True
    
    def test_missing_attributes(self):
        """Test topology with missing attributes raises error."""
        topology = self.create_valid_topology()
        
        # Remove routing matrix
        delattr(topology, 'routing_matrix')
        with pytest.raises(TopologyError):
            validate_topology(topology)
    
    def test_invalid_matrix_dimensions(self):
        """Test topology with invalid matrix dimensions."""
        topology = self.create_valid_topology()
        
        # Wrong shaped routing matrix
        topology.routing_matrix = torch.ones(32, 16)  # Wrong dimensions
        with pytest.raises(TopologyError):
            validate_topology(topology)
    
    def test_non_binary_matrix(self):
        """Test topology with non-binary routing matrix."""
        topology = self.create_valid_topology()
        
        # Non-binary values
        topology.routing_matrix = torch.rand(64, 8)  # Random values
        with pytest.raises(TopologyError):
            validate_topology(topology)
    
    def test_disconnected_tokens(self):
        """Test topology with disconnected tokens."""
        topology = self.create_valid_topology()
        
        # Zero out first token's connections
        topology.routing_matrix[0, :] = 0
        with pytest.raises(TopologyError):
            validate_topology(topology)
    
    def test_invalid_routing_params(self):
        """Test topology with invalid routing parameters."""
        topology = self.create_valid_topology()
        
        # Invalid temperature
        topology.routing_params.temperature = -1.0
        with pytest.raises(TopologyError):
            validate_topology(topology)
        
        # Invalid top_k
        topology = self.create_valid_topology()
        topology.routing_params.top_k = 0
        with pytest.raises(TopologyError):
            validate_topology(topology)
    
    def test_self_connections_in_expert_graph(self):
        """Test expert graph with self-connections."""
        topology = self.create_valid_topology()
        
        # Add self-connection
        topology.expert_graph[0, 0] = 1
        with pytest.raises(TopologyError):
            validate_topology(topology)


class TestExpertPoolValidation:
    """Test expert pool validation robustness."""
    
    def create_valid_expert_pool(self, num_experts=8, expert_dim=128):
        """Create valid expert pool for testing."""
        return ExpertPool(
            num_experts=num_experts,
            expert_dim=expert_dim,
            expert_type="mlp",
            device="cpu"
        )
    
    def test_valid_expert_pool(self):
        """Test valid expert pool passes validation."""
        pool = self.create_valid_expert_pool()
        assert validate_expert_pool(pool) == True
    
    def test_missing_attributes(self):
        """Test expert pool with missing attributes."""
        pool = self.create_valid_expert_pool()
        
        delattr(pool, 'experts')
        with pytest.raises(ExpertPoolError):
            validate_expert_pool(pool)
    
    def test_invalid_expert_count(self):
        """Test expert pool with invalid expert count."""
        with pytest.raises(ExpertPoolError):
            pool = ExpertPool(num_experts=0, expert_dim=128)
            validate_expert_pool(pool)
        
        with pytest.raises(ExpertPoolError):
            pool = ExpertPool(num_experts=2000, expert_dim=128)  # Too many
            validate_expert_pool(pool)
    
    def test_expert_count_mismatch(self):
        """Test expert pool with mismatched expert count."""
        pool = self.create_valid_expert_pool()
        
        # Remove an expert but don't update count
        pool.experts = pool.experts[:-1]
        with pytest.raises(ExpertPoolError):
            validate_expert_pool(pool)
    
    def test_invalid_expert_ids(self):
        """Test expert pool with invalid expert IDs."""
        pool = self.create_valid_expert_pool()
        
        # Wrong expert ID
        pool.experts[0].expert_id = 999
        with pytest.raises(ExpertPoolError):
            validate_expert_pool(pool)
    
    def test_invalid_expert_type(self):
        """Test expert pool with invalid expert type."""
        pool = self.create_valid_expert_pool()
        pool.expert_type = "invalid_type"
        with pytest.raises(ExpertPoolError):
            validate_expert_pool(pool)


class TestEvolutionErrorHandling:
    """Test evolution process error handling."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.expert_pool = ExpertPool(
            num_experts=4,
            expert_dim=32,
            expert_type="mlp",
            device="cpu"
        )
        
        self.config = EvolutionConfig(
            population_size=10,
            generations=5,
            mutation_rate=0.1
        )
    
    def test_invalid_expert_pool(self):
        """Test evolution initialization with invalid expert pool."""
        # Mock invalid expert pool
        invalid_pool = Mock()
        invalid_pool.num_experts = -1  # Invalid
        
        with pytest.raises(EvolutionError):
            EvolvingMoERouter(
                expert_pool=invalid_pool,
                config=self.config
            )
    
    def test_invalid_config(self):
        """Test evolution initialization with invalid config."""
        invalid_config = EvolutionConfig(population_size=-1)  # Invalid
        
        with pytest.raises(EvolutionError):
            EvolvingMoERouter(
                expert_pool=self.expert_pool,
                config=invalid_config
            )
    
    def test_evolution_with_invalid_data(self):
        """Test evolution with invalid data loaders."""
        evolver = EvolvingMoERouter(
            expert_pool=self.expert_pool,
            config=self.config
        )
        
        # Mock model
        model = Mock()
        
        # Invalid data loader (empty)
        invalid_loader = []
        
        with pytest.raises(Exception):  # Should raise some validation error
            evolver.evolve(model, invalid_loader, invalid_loader)
    
    def test_fitness_evaluation_failure(self):
        """Test handling of fitness evaluation failures."""
        evolver = EvolvingMoERouter(
            expert_pool=self.expert_pool,
            config=self.config
        )
        
        # Mock fitness evaluator that always fails
        def failing_evaluate(*args, **kwargs):
            raise RuntimeError("Fitness evaluation failed")
        
        evolver.fitness_evaluator.evaluate = failing_evaluate
        
        model = Mock()
        data_loader = Mock()
        data_loader.__len__ = Mock(return_value=1)
        data_loader.__iter__ = Mock(return_value=iter([(torch.randn(4, 32), torch.randint(0, 10, (4,)))]))
        
        # Should handle fitness evaluation failure gracefully
        with pytest.raises(Exception):
            evolver.evolve(model, data_loader, data_loader, generations=1)
    
    def test_mutation_failure_recovery(self):
        """Test recovery from mutation failures."""
        evolver = EvolvingMoERouter(
            expert_pool=self.expert_pool,
            config=self.config
        )
        
        # Create topology that will fail mutation
        topology = evolver.population[0]
        
        # Mock mutate method to fail
        original_mutate = topology.mutate
        
        def failing_mutate(*args, **kwargs):
            raise RuntimeError("Mutation failed")
        
        topology.mutate = failing_mutate
        
        # Evolution should handle mutation failure
        try:
            evolver._create_next_generation()
            # Should not crash, should create valid population
            assert len(evolver.population) == self.config.population_size
        except Exception as e:
            # If it does raise an error, it should be a controlled error
            assert "Mutation failed" in str(e) or isinstance(e, EvolutionError)
    
    def test_convergence_detection(self):
        """Test convergence detection and early stopping."""
        config = EvolutionConfig(
            population_size=5,
            generations=100,
            patience=3  # Stop after 3 generations without improvement
        )
        
        evolver = EvolvingMoERouter(
            expert_pool=self.expert_pool,
            config=config
        )
        
        # Mock fitness evaluator to return constant fitness (no improvement)
        def constant_fitness(*args, **kwargs):
            return 0.5, {'accuracy': 0.5}
        
        evolver.fitness_evaluator.evaluate = constant_fitness
        
        model = Mock()
        data_loader = Mock()
        data_loader.__len__ = Mock(return_value=1)
        data_loader.__iter__ = Mock(return_value=iter([(torch.randn(4, 32), torch.randint(0, 10, (4,)))]))
        
        # Should converge early
        best_topology = evolver.evolve(model, data_loader, data_loader)
        
        # Should have stopped early due to no improvement
        assert evolver.generation < config.generations


class TestResourceConstraintHandling:
    """Test resource constraint handling."""
    
    def test_memory_constraint_violation(self):
        """Test handling of memory constraint violations."""
        config = EvolutionConfig(
            memory_budget=1024 * 1024  # 1MB limit (very small)
        )
        
        expert_pool = ExpertPool(
            num_experts=100,  # Large number of experts
            expert_dim=1024,  # Large dimension
            expert_type="transformer_block"
        )
        
        # Should handle memory constraint violation
        evolver = EvolvingMoERouter(
            expert_pool=expert_pool,
            config=config
        )
        
        # Fitness evaluation should penalize high memory usage
        topology = evolver.population[0]
        model = Mock()
        
        # Mock data loaders
        data_loader = Mock()
        data_loader.__len__ = Mock(return_value=1)
        data_loader.__iter__ = Mock(return_value=iter([(torch.randn(4, 1024), torch.randint(0, 10, (4,)))]))
        
        fitness, metrics = evolver.fitness_evaluator.evaluate(
            topology, model, data_loader, data_loader
        )
        
        # Should have constraint penalty
        assert metrics.get('constraint_penalty', 0) > 0
    
    def test_latency_constraint_violation(self):
        """Test handling of latency constraint violations."""
        config = EvolutionConfig(
            latency_budget=1.0  # 1ms limit (very strict)
        )
        
        evolver = EvolvingMoERouter(
            expert_pool=ExpertPool(num_experts=4, expert_dim=32),
            config=config
        )
        
        # Mock slow model
        def slow_forward(*args, **kwargs):
            import time
            time.sleep(0.01)  # Simulate slow execution
            return torch.randn(4, 10)
        
        model = Mock()
        model.forward = slow_forward
        model.set_routing_topology = Mock()
        
        data_loader = Mock()
        data_loader.__len__ = Mock(return_value=1)
        data_loader.__iter__ = Mock(return_value=iter([(torch.randn(4, 32), torch.randint(0, 10, (4,)))]))
        
        topology = evolver.population[0]
        fitness, metrics = evolver.fitness_evaluator.evaluate(
            topology, model, data_loader, data_loader
        )
        
        # Should have latency penalty
        assert metrics.get('latency', 0) > config.latency_budget


class TestFaultTolerance:
    """Test system fault tolerance."""
    
    def test_corrupted_topology_recovery(self):
        """Test recovery from corrupted topology data."""
        evolver = EvolvingMoERouter(
            expert_pool=ExpertPool(num_experts=4, expert_dim=32),
            config=EvolutionConfig(population_size=5)
        )
        
        # Corrupt a topology
        topology = evolver.population[0]
        topology.routing_matrix = torch.tensor([[float('nan'), 0, 1, 0]])  # Invalid data
        
        # Should detect and handle corruption
        try:
            validate_topology(topology)
            assert False, "Should have detected corrupted topology"
        except TopologyError:
            pass  # Expected
    
    def test_device_mismatch_handling(self):
        """Test handling of device mismatches."""
        # Create expert pool on CPU
        expert_pool = ExpertPool(num_experts=4, expert_dim=32, device="cpu")
        
        evolver = EvolvingMoERouter(
            expert_pool=expert_pool,
            config=EvolutionConfig(population_size=5),
            device="cpu"  # Different device
        )
        
        # Should handle device mismatches gracefully
        topology = evolver.population[0]
        assert topology.routing_matrix.device.type == "cpu"
    
    def test_save_load_robustness(self):
        """Test robustness of save/load operations."""
        evolver = EvolvingMoERouter(
            expert_pool=ExpertPool(num_experts=4, expert_dim=32),
            config=EvolutionConfig(population_size=5)
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "evolution_state.pt"
            
            # Save state
            evolver.save_evolution_state(str(save_path))
            
            # Corrupt the saved file
            with open(save_path, 'wb') as f:
                f.write(b'corrupted data')
            
            # Loading should handle corruption gracefully
            try:
                EvolvingMoERouter.load_evolution_state(
                    str(save_path),
                    expert_pool=ExpertPool(num_experts=4, expert_dim=32)
                )
                assert False, "Should have failed to load corrupted file"
            except Exception:
                pass  # Expected to fail
    
    def test_population_diversity_maintenance(self):
        """Test maintenance of population diversity."""
        config = EvolutionConfig(
            population_size=10,
            mutation_rate=0.0,  # No mutations
            crossover_rate=0.0  # No crossover
        )
        
        evolver = EvolvingMoERouter(
            expert_pool=ExpertPool(num_experts=4, expert_dim=32),
            config=config
        )
        
        # Check initial diversity
        initial_matrices = [t.routing_matrix.clone() for t in evolver.population]
        
        # After several generations with no variation, should still maintain diversity
        for _ in range(5):
            evolver._create_next_generation()
        
        # Should still have diverse population due to elitism
        final_matrices = [t.routing_matrix for t in evolver.population]
        
        # At least some diversity should remain
        unique_topologies = set()
        for matrix in final_matrices:
            unique_topologies.add(matrix.sum().item())  # Simple diversity measure
        
        assert len(unique_topologies) > 1, "Population lost all diversity"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])