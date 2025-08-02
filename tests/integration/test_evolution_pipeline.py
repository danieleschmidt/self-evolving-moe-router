"""Integration tests for the complete evolution pipeline."""

import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestEvolutionPipeline:
    """Integration tests for the evolution pipeline."""

    @pytest.mark.integration
    def test_complete_evolution_cycle(self, evolution_config, temp_dir, device):
        """Test a complete evolution cycle from start to finish."""
        # Mock the main evolution components
        population = self._create_mock_population(
            evolution_config["evolution"]["population_size"], device
        )
        
        fitness_evaluator = Mock()
        fitness_evaluator.evaluate = Mock(
            return_value=np.random.uniform(0.1, 0.9, len(population))
        )
        
        selector = Mock()
        selector.select = Mock(return_value=population[:10])  # Select top 10
        
        mutator = Mock()
        mutator.mutate = Mock(side_effect=lambda x: x)  # Identity for simplicity
        
        # Run evolution loop
        generations = evolution_config["evolution"]["generations"]
        fitness_history = []
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = fitness_evaluator.evaluate(population)
            fitness_history.append(fitness_scores)
            
            # Select parents
            parents = selector.select(population, fitness_scores)
            
            # Create new population through mutation/crossover
            new_population = []
            for parent in parents:
                child = mutator.mutate(parent)
                new_population.append(child)
            
            population = new_population
        
        # Verify evolution completed
        assert len(fitness_history) == generations
        assert len(population) > 0
        
        # Check that fitness evaluator was called
        assert fitness_evaluator.evaluate.call_count == generations

    @pytest.mark.integration
    def test_evolution_with_checkpoints(self, evolution_config, checkpoint_dir, device):
        """Test evolution with checkpointing enabled."""
        population = self._create_mock_population(20, device)
        
        # Mock checkpointing
        checkpoint_manager = Mock()
        checkpoint_manager.save_checkpoint = Mock()
        checkpoint_manager.load_checkpoint = Mock(return_value=population)
        
        # Simulate evolution with checkpoints every 3 generations
        checkpoint_interval = 3
        total_generations = 10
        
        for gen in range(total_generations):
            # Simulate evolution step
            fitness_scores = np.random.uniform(0.1, 0.9, len(population))
            
            if gen % checkpoint_interval == 0:
                checkpoint_manager.save_checkpoint(
                    population, gen, checkpoint_dir / f"checkpoint_{gen}.pkl"
                )
        
        # Verify checkpoints were saved
        expected_checkpoints = total_generations // checkpoint_interval + 1
        assert checkpoint_manager.save_checkpoint.call_count >= expected_checkpoints

    @pytest.mark.integration
    def test_multi_objective_optimization(self, evolution_config, device):
        """Test multi-objective optimization integration."""
        population = self._create_mock_population(50, device)
        
        # Mock multi-objective fitness evaluator
        objectives = ["accuracy", "latency", "memory"]
        
        def mock_evaluate_multi_objective(pop):
            results = []
            for individual in pop:
                scores = {
                    "accuracy": np.random.uniform(0.7, 0.95),
                    "latency": np.random.uniform(5, 50),  # Lower is better
                    "memory": np.random.uniform(1, 8),    # Lower is better
                }
                results.append(scores)
            return results
        
        fitness_evaluator = Mock()
        fitness_evaluator.evaluate_multi_objective = Mock(
            side_effect=mock_evaluate_multi_objective
        )
        
        # Run multi-objective evolution
        generations = 10
        pareto_fronts = []
        
        for gen in range(generations):
            fitness_results = fitness_evaluator.evaluate_multi_objective(population)
            
            # Mock Pareto front calculation
            pareto_front = fitness_results[:10]  # Simplified
            pareto_fronts.append(pareto_front)
        
        # Verify multi-objective evaluation
        assert len(pareto_fronts) == generations
        assert all(len(front) > 0 for front in pareto_fronts)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_evolution_convergence(self, evolution_config, device):
        """Test that evolution converges to better solutions."""
        population = self._create_mock_population(30, device)
        
        # Mock fitness function with some structure (not purely random)
        def structured_fitness(individual):
            # Simple fitness based on routing matrix sparsity and connectivity
            matrix = individual["routing_matrix"]
            sparsity = (matrix == 0).float().mean().item()
            connectivity = matrix.sum().item()
            
            # Prefer moderate sparsity and reasonable connectivity
            sparsity_score = 1.0 - abs(sparsity - 0.8)  # Target 80% sparsity
            connectivity_score = min(connectivity / 10.0, 1.0)  # Normalize
            
            return 0.7 * sparsity_score + 0.3 * connectivity_score
        
        # Run evolution
        generations = 50
        fitness_history = []
        best_fitness_history = []
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = [structured_fitness(ind) for ind in population]
            fitness_history.append(fitness_scores)
            best_fitness_history.append(max(fitness_scores))
            
            # Simple selection and mutation
            # Select top 50%
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_size = len(population) // 2
            elite_indices = sorted_indices[:elite_size]
            
            # Create new population
            new_population = []
            for i in range(len(population)):
                parent_idx = np.random.choice(elite_indices)
                parent = population[parent_idx]
                
                # Simple mutation
                child = self._mutate_individual(parent, device)
                new_population.append(child)
            
            population = new_population
        
        # Check for convergence (fitness should generally improve)
        early_fitness = np.mean(best_fitness_history[:10])
        late_fitness = np.mean(best_fitness_history[-10:])
        
        assert late_fitness >= early_fitness, "Evolution should show improvement"

    @pytest.mark.integration
    def test_hardware_aware_evolution(self, evolution_config, device):
        """Test hardware-aware evolution constraints."""
        population = self._create_mock_population(25, device)
        
        # Mock hardware constraints
        hardware_constraints = {
            "max_memory": 4.0,  # GB
            "target_latency": 20.0,  # ms
            "max_experts_per_token": 2
        }
        
        def mock_hardware_fitness(individual, constraints):
            # Mock hardware evaluation
            base_fitness = np.random.uniform(0.5, 0.9)
            
            # Penalty for constraint violations
            penalty = 0.0
            
            # Mock memory usage (based on routing matrix density)
            memory_usage = individual["routing_matrix"].sum().item() * 0.1
            if memory_usage > constraints["max_memory"]:
                penalty += (memory_usage - constraints["max_memory"]) * 0.1
            
            # Mock latency (based on active experts)
            active_experts = (individual["routing_matrix"] > 0).sum(dim=1).float().mean()
            if active_experts > constraints["max_experts_per_token"]:
                penalty += (active_experts - constraints["max_experts_per_token"]) * 0.2
            
            return max(0.0, base_fitness - penalty)
        
        # Run hardware-aware evolution
        generations = 20
        constraint_violations = []
        
        for gen in range(generations):
            # Evaluate with hardware constraints
            fitness_scores = [
                mock_hardware_fitness(ind, hardware_constraints) 
                for ind in population
            ]
            
            # Count constraint violations
            violations = sum(1 for score in fitness_scores if score < 0.3)
            constraint_violations.append(violations)
            
            # Simple evolution step
            population = self._evolve_population_step(population, fitness_scores, device)
        
        # Constraint violations should decrease over time
        early_violations = np.mean(constraint_violations[:5])
        late_violations = np.mean(constraint_violations[-5:])
        
        assert late_violations <= early_violations

    @pytest.mark.integration
    def test_distributed_evolution(self, evolution_config, device):
        """Test distributed evolution across multiple workers."""
        population = self._create_mock_population(40, device)
        
        # Mock distributed evaluation
        def mock_distributed_evaluate(pop, num_workers=4):
            # Simulate parallel evaluation
            worker_populations = np.array_split(pop, num_workers)
            all_results = []
            
            for worker_pop in worker_populations:
                worker_results = [
                    np.random.uniform(0.1, 0.9) for _ in worker_pop
                ]
                all_results.extend(worker_results)
            
            return all_results
        
        # Run distributed evolution
        generations = 10
        num_workers = 4
        
        for gen in range(generations):
            # Distributed fitness evaluation
            fitness_scores = mock_distributed_evaluate(population, num_workers)
            
            # Evolution step
            population = self._evolve_population_step(population, fitness_scores, device)
        
        # Verify we maintained population
        assert len(population) == 40

    @pytest.mark.integration
    def test_online_evolution(self, evolution_config, device):
        """Test online evolution during deployment."""
        # Smaller population for online evolution
        population = self._create_mock_population(10, device)
        
        # Mock deployment scenario
        deployment_steps = 100
        evolution_interval = 10  # Evolve every 10 steps
        
        performance_history = []
        evolution_triggered = 0
        
        for step in range(deployment_steps):
            # Simulate deployment performance
            current_performance = np.random.uniform(0.6, 0.9)
            performance_history.append(current_performance)
            
            # Trigger evolution periodically
            if step % evolution_interval == 0 and step > 0:
                evolution_triggered += 1
                
                # Quick evolution step
                fitness_scores = [np.random.uniform(0.1, 0.9) for _ in population]
                population = self._evolve_population_step(
                    population, fitness_scores, device, quick=True
                )
        
        # Verify evolution was triggered
        expected_evolutions = deployment_steps // evolution_interval
        assert evolution_triggered == expected_evolutions

    def _create_mock_population(self, size: int, device: torch.device):
        """Create a mock population for testing."""
        population = []
        
        for _ in range(size):
            # Random routing topology
            routing_matrix = torch.rand((16, 4), device=device)
            routing_matrix = (routing_matrix > 0.7).float()  # Make sparse
            
            individual = {
                "routing_matrix": routing_matrix,
                "routing_params": {
                    "temperature": np.random.uniform(0.5, 2.0),
                    "top_k": np.random.choice([1, 2, 3]),
                    "load_balancing_weight": np.random.uniform(0.001, 0.1)
                }
            }
            population.append(individual)
        
        return population

    def _mutate_individual(self, individual, device: torch.device):
        """Apply simple mutation to an individual."""
        child = {
            "routing_matrix": individual["routing_matrix"].clone(),
            "routing_params": individual["routing_params"].copy()
        }
        
        # Simple bit-flip mutation
        mutation_mask = torch.rand_like(child["routing_matrix"]) < 0.1
        child["routing_matrix"][mutation_mask] = 1.0 - child["routing_matrix"][mutation_mask]
        
        # Parameter mutation
        if np.random.random() < 0.1:
            child["routing_params"]["temperature"] *= np.random.uniform(0.8, 1.2)
        
        return child

    def _evolve_population_step(self, population, fitness_scores, device: torch.device, quick=False):
        """Perform one evolution step."""
        if quick:
            # Quick evolution: just select best and add some noise
            best_idx = np.argmax(fitness_scores)
            best_individual = population[best_idx]
            
            new_population = [best_individual]  # Elite
            for _ in range(len(population) - 1):
                child = self._mutate_individual(best_individual, device)
                new_population.append(child)
            
            return new_population
        
        # Regular evolution
        # Tournament selection
        new_population = []
        for _ in range(len(population)):
            # Select parents via tournament
            tournament_size = 3
            tournament_indices = np.random.choice(
                len(population), tournament_size, replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            parent = population[winner_idx]
            child = self._mutate_individual(parent, device)
            new_population.append(child)
        
        return new_population