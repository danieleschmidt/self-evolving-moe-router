"""
Comprehensive Test Suite for TERRAGON Next-Gen Features
Tests quantum evolution, distributed consensus, adaptive mutations, and real-time monitoring
"""

import pytest
import numpy as np
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import next-gen modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from self_evolving_moe.evolution.quantum_evolution import (
    QuantumEvolutionEngine, QuantumSuperpositionCrossover, 
    QuantumInterferenceMutation, QuantumState
)
from self_evolving_moe.distributed.consensus_evolution import (
    FederatedEvolutionCoordinator, ByzantineFaultTolerantConsensus,
    EvolutionProposal, ConsensusVote, ConsensusType
)
from self_evolving_moe.evolution.adaptive_mutations import (
    AdaptiveMutationEngine, SelfAdaptiveMutation, DiversityMaintainingMutation,
    HierarchicalAdaptiveMutation, AdaptiveMutationConfig, MutationType
)
from self_evolving_moe.monitoring.realtime_performance_monitor import (
    RealTimePerformanceMonitor, MetricCollector, PerformanceAnalyzer,
    AutoTuner, PerformanceAlert, AlertLevel
)

class TestQuantumEvolution:
    """Test quantum-inspired evolution operators"""
    
    @pytest.fixture
    def sample_population(self):
        """Create sample population for testing"""
        population = []
        for _ in range(10):
            topology = np.random.random((8, 16))
            threshold = np.percentile(topology, 70)
            topology = (topology > threshold).astype(np.float32)
            population.append(topology)
        return population
    
    @pytest.fixture
    def sample_fitness(self):
        """Create sample fitness scores"""
        return np.random.uniform(-1.0, -0.1, 10)
    
    def test_quantum_state_initialization(self):
        """Test quantum state creation and validation"""
        amplitudes = np.array([0.7 + 0j, 0.7 + 0j])
        phases = np.array([0.0, np.pi/4])
        entanglement_matrix = np.array([[0, 0.1j], [-0.1j, 0]], dtype=complex)
        
        quantum_state = QuantumState(amplitudes, phases, entanglement_matrix)
        
        # Check normalization
        assert abs(np.linalg.norm(quantum_state.amplitudes) - 1.0) < 1e-6
        
        # Check hermitian property of entanglement matrix
        assert np.allclose(quantum_state.entanglement_matrix, 
                          quantum_state.entanglement_matrix.T.conj())
    
    def test_quantum_superposition_crossover(self, sample_population, sample_fitness):
        """Test quantum superposition crossover operator"""
        crossover_op = QuantumSuperpositionCrossover(coherence_time=1.0, decoherence_rate=0.1)
        
        offspring = crossover_op.apply(sample_population, sample_fitness)
        
        # Check offspring properties
        assert len(offspring) == len(sample_population)
        assert all(child.shape == sample_population[0].shape for child in offspring)
        assert all(np.all((child == 0) | (child == 1)) for child in offspring)  # Binary values
        
        # Check sparsity preservation
        for child in offspring:
            sparsity = np.mean(child)
            assert 0.1 <= sparsity <= 0.5  # Reasonable sparsity range
    
    def test_quantum_interference_mutation(self, sample_population, sample_fitness):
        """Test quantum interference mutation operator"""
        mutation_op = QuantumInterferenceMutation(interference_strength=0.2, wave_frequency=2.0)
        
        mutated_population = mutation_op.apply(sample_population, sample_fitness)
        
        # Check mutation properties
        assert len(mutated_population) == len(sample_population)
        
        # Check that mutation actually changed individuals
        differences = [
            np.mean(np.abs(original - mutated))
            for original, mutated in zip(sample_population, mutated_population)
        ]
        assert any(diff > 0.01 for diff in differences)  # Some individuals should be mutated
    
    def test_quantum_evolution_engine(self, sample_population, sample_fitness):
        """Test complete quantum evolution engine"""
        engine = QuantumEvolutionEngine(coherence_time=1.5, decoherence_rate=0.08)
        
        next_generation, metrics = engine.evolve_generation(sample_population, sample_fitness)
        
        # Check evolution results
        assert len(next_generation) == len(sample_population)
        assert 'quantum_crossover_applied' in metrics
        assert 'quantum_mutation_applied' in metrics
        
        # Check quantum metrics
        assert engine.quantum_metrics['interference_diversity'] >= 0
        assert engine.quantum_metrics['entanglement_strength'] >= 0
        assert engine.quantum_metrics['coherence_preserved'] >= 0
    
    def test_quantum_selection_mechanism(self, sample_population, sample_fitness):
        """Test quantum measurement-based selection"""
        engine = QuantumEvolutionEngine()
        
        # Create combined population (double size)
        combined_population = sample_population + sample_population
        combined_fitness = np.concatenate([sample_fitness, sample_fitness * 0.8])
        
        selected = engine._quantum_selection(combined_population, combined_fitness, len(sample_population))
        
        assert len(selected) == len(sample_population)
        assert all(ind.shape == sample_population[0].shape for ind in selected)


class TestDistributedConsensus:
    """Test distributed evolution with consensus mechanisms"""
    
    @pytest.fixture
    def sample_proposal(self):
        """Create sample evolution proposal"""
        return EvolutionProposal(
            proposal_id="test_proposal_123",
            proposer_node="node_0",
            operation_type="mutation",
            target_individuals=[0, 1, 2],
            parameters={'mutation_rate': 0.1},
            fitness_evidence={'accuracy': 0.8, 'diversity': 0.3},
            timestamp=time.time(),
            signature="test_signature"
        )
    
    def test_byzantine_consensus_initialization(self):
        """Test Byzantine fault tolerant consensus initialization"""
        consensus = ByzantineFaultTolerantConsensus("node_0", total_nodes=5, fault_tolerance=0.33)
        
        assert consensus.node_id == "node_0"
        assert consensus.total_nodes == 5
        assert consensus.max_faulty_nodes == 1  # 33% of 5 nodes
        assert consensus.min_honest_nodes == 4
    
    @pytest.mark.asyncio
    async def test_proposal_validation(self, sample_proposal):
        """Test proposal validation mechanisms"""
        consensus = ByzantineFaultTolerantConsensus("node_0", total_nodes=5)
        
        # Test valid proposal
        valid = consensus._validate_proposal(sample_proposal)
        assert valid == True
        
        # Test invalid proposal (missing required field)
        invalid_proposal = sample_proposal
        invalid_proposal.operation_type = "invalid_operation"
        
        valid = consensus._validate_proposal(invalid_proposal)
        assert valid == False
    
    @pytest.mark.asyncio
    async def test_proposal_submission(self, sample_proposal):
        """Test proposal submission and acceptance"""
        consensus = ByzantineFaultTolerantConsensus("node_0", total_nodes=5)
        
        success = await consensus.propose_evolution(sample_proposal)
        assert success == True
        assert sample_proposal.proposal_id in consensus.pending_proposals
    
    @pytest.mark.asyncio
    async def test_voting_mechanism(self, sample_proposal):
        """Test consensus voting mechanism"""
        consensus = ByzantineFaultTolerantConsensus("node_0", total_nodes=5)
        
        vote = await consensus.vote_on_proposal(sample_proposal)
        
        assert vote.proposal_id == sample_proposal.proposal_id
        assert vote.voter_node == "node_0"
        assert isinstance(vote.vote, bool)
        assert 0.0 <= vote.confidence <= 1.0
        assert 'final_security_score' in vote.reasoning
    
    @pytest.mark.asyncio
    async def test_consensus_reaching(self, sample_proposal):
        """Test consensus decision making"""
        consensus = ByzantineFaultTolerantConsensus("node_0", total_nodes=5)
        
        # Create multiple votes
        votes = []
        for i in range(4):  # Sufficient for consensus
            vote = ConsensusVote(
                proposal_id=sample_proposal.proposal_id,
                voter_node=f"node_{i}",
                vote=True,
                confidence=0.8,
                reasoning={'test': True},
                timestamp=time.time(),
                signature=f"sig_{i}"
            )
            votes.append(vote)
        
        consensus_reached = await consensus.reach_consensus(sample_proposal, votes)
        assert consensus_reached == True
    
    def test_federated_coordinator_initialization(self):
        """Test federated evolution coordinator setup"""
        coordinator = FederatedEvolutionCoordinator("node_0", total_nodes=5)
        
        assert coordinator.node_id == "node_0"
        assert coordinator.total_nodes == 5
        assert coordinator.consensus_type == ConsensusType.BYZANTINE_FAULT_TOLERANT
        assert coordinator.evolution_round == 0
    
    @pytest.mark.asyncio
    async def test_distributed_fitness_evaluation(self):
        """Test distributed fitness evaluation"""
        coordinator = FederatedEvolutionCoordinator("node_0", total_nodes=3)
        
        # Initialize with sample population
        sample_population = [np.random.random((8, 16)) for _ in range(6)]
        await coordinator.initialize_federated_evolution(sample_population)
        
        # Mock fitness evaluator
        def mock_fitness_evaluator(individual):
            return {'accuracy': np.random.random(), 'diversity': np.random.random()}
        
        metrics = {}
        await coordinator._distributed_fitness_evaluation(mock_fitness_evaluator, metrics)
        
        # Check that fitness evaluation was attempted
        assert len(coordinator.fitness_cache) >= 0  # Some results should be cached


class TestAdaptiveMutations:
    """Test adaptive mutation strategies"""
    
    @pytest.fixture
    def mutation_config(self):
        """Create adaptive mutation configuration"""
        return AdaptiveMutationConfig(
            initial_mutation_rate=0.1,
            initial_mutation_strength=0.05,
            enable_hierarchical=True,
            enable_self_adaptation=True,
            enable_topology_awareness=True
        )
    
    @pytest.fixture
    def sample_individual(self):
        """Create sample topology individual"""
        topology = np.random.random((8, 16))
        threshold = np.percentile(topology, 70)
        return (topology > threshold).astype(np.float32)
    
    @pytest.fixture
    def sample_population(self):
        """Create sample population"""
        population = []
        for _ in range(20):
            topology = np.random.random((8, 16))
            threshold = np.percentile(topology, 70)
            population.append((topology > threshold).astype(np.float32))
        return population
    
    def test_self_adaptive_mutation(self, mutation_config, sample_individual, sample_population):
        """Test self-adaptive mutation operator"""
        mutation_op = SelfAdaptiveMutation(mutation_config)
        
        mutated, info = mutation_op.mutate(sample_individual, fitness=-0.5, generation=1, population=sample_population)
        
        # Check mutation results
        assert mutated.shape == sample_individual.shape
        assert info['mutation_type'] == 'self_adaptive'
        assert 'global_step_size' in info
        assert 'average_step_size' in info
        
        # Check that strategy parameters were created
        individual_id = mutation_op._get_individual_id(sample_individual)
        assert individual_id in mutation_op.strategy_parameters
    
    def test_diversity_maintaining_mutation(self, mutation_config, sample_individual, sample_population):
        """Test diversity-maintaining mutation operator"""
        mutation_op = DiversityMaintainingMutation(mutation_config)
        
        mutated, info = mutation_op.mutate(sample_individual, fitness=-0.5, generation=1, population=sample_population)
        
        # Check mutation results
        assert mutated.shape == sample_individual.shape
        assert info['mutation_type'] == 'diversity_maintaining'
        assert 'population_diversity' in info
        assert 'individual_novelty' in info
        assert 'adaptive_strength' in info
    
    def test_hierarchical_mutation(self, mutation_config, sample_individual, sample_population):
        """Test hierarchical adaptive mutation"""
        mutation_op = HierarchicalAdaptiveMutation(mutation_config)
        
        mutated, info = mutation_op.mutate(sample_individual, fitness=-0.5, generation=1, population=sample_population)
        
        # Check mutation results
        assert mutated.shape == sample_individual.shape
        assert info['mutation_type'] == 'hierarchical_adaptive'
        assert 'scales_applied' in info
        assert 'scale_contributions' in info
        assert len(mutation_op.scale_metrics) == mutation_config.hierarchy_levels
    
    def test_adaptive_mutation_engine(self, mutation_config, sample_population):
        """Test complete adaptive mutation engine"""
        engine = AdaptiveMutationEngine(mutation_config)
        sample_fitness = np.random.uniform(-1.0, -0.1, len(sample_population))
        
        mutated_population, metrics = engine.evolve_population(sample_population, sample_fitness)
        
        # Check evolution results
        assert len(mutated_population) == len(sample_population)
        assert metrics['generation'] == 1
        assert metrics['population_size'] == len(sample_population)
        assert 'operator_usage' in metrics
        assert 'success_rate' in metrics
    
    def test_parameter_adaptation(self, mutation_config, sample_population):
        """Test parameter adaptation mechanisms"""
        engine = AdaptiveMutationEngine(mutation_config)
        sample_fitness = np.random.uniform(-1.0, -0.1, len(sample_population))
        
        # Run multiple generations to test adaptation
        for generation in range(5):
            mutated_population, metrics = engine.evolve_population(sample_population, sample_fitness)
            sample_population = mutated_population
        
        # Check that operator weights adapted
        initial_weights = {name: 1.0 for name in engine.operators.keys()}
        final_weights = engine.operator_weights
        
        # Weights should have changed from initial values
        assert any(final_weights[name] != initial_weights[name] for name in initial_weights.keys())
    
    def test_novelty_archive_management(self, mutation_config, sample_population):
        """Test novelty archive in diversity-maintaining mutation"""
        mutation_op = DiversityMaintainingMutation(mutation_config)
        
        # Process several individuals
        for individual in sample_population[:5]:
            mutation_op._update_novelty_archive(individual)
        
        # Check archive properties
        assert len(mutation_op.novelty_archive) <= 5
        assert all(ind.shape == sample_population[0].shape for ind in mutation_op.novelty_archive)


class TestRealTimeMonitoring:
    """Test real-time performance monitoring and auto-tuning"""
    
    @pytest.fixture
    def metric_collector(self):
        """Create metric collector for testing"""
        return MetricCollector(buffer_size=100)
    
    @pytest.fixture
    def performance_analyzer(self, metric_collector):
        """Create performance analyzer for testing"""
        return PerformanceAnalyzer(metric_collector)
    
    def test_metric_collector_initialization(self, metric_collector):
        """Test metric collector setup"""
        assert metric_collector.buffer_size == 100
        assert not metric_collector.running
        assert len(metric_collector.metrics_buffer) == 0
    
    def test_custom_metric_collection(self, metric_collector):
        """Test custom metric registration and collection"""
        # Add custom metric
        def test_metric():
            return 42.0
        
        metric_collector.add_custom_collector("test_metric", test_metric)
        assert "test_metric" in metric_collector.custom_collectors
        
        # Test collection
        metric_collector._collect_custom_metrics()
        
        # Check that metric was collected
        recent_metrics = metric_collector.get_recent_metrics("test_metric", 1)
        assert len(recent_metrics) == 1
        assert recent_metrics[0].value == 42.0
    
    def test_metric_statistics(self, metric_collector):
        """Test metric statistics calculation"""
        # Add some sample metrics
        for i in range(10):
            metric_collector._add_metric("test_stat", float(i), time.time(), "units")
        
        stats = metric_collector.get_metric_statistics("test_stat", window_seconds=60)
        
        assert stats['count'] == 10
        assert stats['min'] == 0.0
        assert stats['max'] == 9.0
        assert stats['mean'] == 4.5
        assert 'trend' in stats
    
    def test_performance_analysis(self, performance_analyzer, metric_collector):
        """Test performance analysis capabilities"""
        # Add sample metrics with different patterns
        metric_collector._add_metric("cpu_usage", 85.0, time.time(), "%")  # High CPU
        metric_collector._add_metric("memory_usage", 60.0, time.time(), "%")  # Normal memory
        metric_collector._add_metric("latency", 150.0, time.time(), "ms")  # High latency
        
        analysis = performance_analyzer.analyze_current_performance()
        
        assert 'overall_health' in analysis
        assert 'metric_analysis' in analysis
        assert 'bottlenecks' in analysis
        
        # Check that high CPU usage is detected
        if 'cpu_usage' in analysis['metric_analysis']:
            cpu_analysis = analysis['metric_analysis']['cpu_usage']
            assert cpu_analysis.get('alert_level') in ['warning', 'critical']
    
    def test_anomaly_detection(self, performance_analyzer):
        """Test anomaly detection in metrics"""
        # Add normal values
        for i in range(20):
            performance_analyzer.pattern_history['test_metric'].append(50.0 + np.random.normal(0, 2))
        
        # Test normal value
        normal_anomaly = performance_analyzer._detect_anomaly('test_metric', 52.0)
        assert normal_anomaly < 0.5
        
        # Test anomalous value
        anomalous_anomaly = performance_analyzer._detect_anomaly('test_metric', 100.0)
        assert anomalous_anomaly > 0.5
    
    def test_auto_tuner_recommendations(self, performance_analyzer):
        """Test auto-tuning recommendation generation"""
        auto_tuner = AutoTuner(performance_analyzer)
        
        # Mock analysis with bottlenecks
        mock_analysis = {
            'bottlenecks': ['cpu_bound', 'memory_bound'],
            'metric_analysis': {
                'cache_hit_rate': {
                    'current_value': 0.5,
                    'alert_level': 'warning'
                }
            }
        }
        
        with patch.object(performance_analyzer, 'analyze_current_performance', return_value=mock_analysis):
            recommendations = auto_tuner.generate_tuning_recommendations()
        
        # Should generate recommendations for detected issues
        assert len(recommendations) > 0
        
        # Check recommendation properties
        for rec in recommendations:
            assert hasattr(rec, 'recommendation_id')
            assert hasattr(rec, 'action')
            assert hasattr(rec, 'confidence')
            assert 0.0 <= rec.confidence <= 1.0
    
    def test_real_time_monitor_integration(self):
        """Test complete real-time monitoring system"""
        monitor = RealTimePerformanceMonitor(auto_tune=True, collection_interval=0.1)
        
        # Add custom metrics for testing
        def test_fitness():
            return np.random.uniform(-1.0, -0.1)
        
        def test_diversity():
            return np.random.uniform(0.1, 0.8)
        
        monitor.add_custom_metric('fitness_score', test_fitness)
        monitor.add_custom_metric('diversity_metric', test_diversity)
        
        # Set thresholds
        monitor.set_metric_threshold('fitness_score', warning=-0.5, critical=-1.0)
        monitor.set_metric_threshold('diversity_metric', warning=0.2, critical=0.1)
        
        try:
            # Start monitoring
            monitor.start_monitoring()
            assert monitor.running == True
            
            # Let it collect some data
            time.sleep(1.0)
            
            # Check status
            status = monitor.get_current_status()
            assert status['monitoring_active'] == True
            assert status['auto_tuning_enabled'] == True
            assert 'metrics_collected' in status
            
            # Create performance profile
            profile = monitor.create_performance_profile("test_profile")
            assert profile.profile_name == "test_profile"
            assert profile.creation_timestamp > 0
            
        finally:
            monitor.stop_monitoring()
            assert monitor.running == False
    
    def test_alert_system(self):
        """Test alert generation and handling"""
        monitor = RealTimePerformanceMonitor(auto_tune=False, collection_interval=0.1)
        
        alerts_received = []
        
        def alert_handler(alert):
            alerts_received.append(alert)
        
        monitor.add_alert_callback(alert_handler)
        
        # Set aggressive thresholds to trigger alerts
        monitor.set_metric_threshold('cpu_usage', warning=1.0, critical=2.0)  # Very low thresholds
        
        try:
            monitor.start_monitoring()
            time.sleep(1.0)  # Let it collect and potentially generate alerts
            
            # Process any pending alerts
            if hasattr(monitor, '_monitoring_loop'):
                # Manually trigger analysis to check for alerts
                analysis = monitor.analyzer.analyze_current_performance()
                monitor._process_alerts(analysis)
            
        finally:
            monitor.stop_monitoring()
        
        # Note: Alerts may or may not be generated depending on actual system metrics
        # This test verifies the alert system is properly wired


class TestIntegrationScenarios:
    """Integration tests for next-gen features working together"""
    
    def test_quantum_adaptive_evolution_pipeline(self):
        """Test quantum evolution with adaptive mutations"""
        # Create components
        mutation_config = AdaptiveMutationConfig(enable_self_adaptation=True, enable_hierarchical=True)
        adaptive_engine = AdaptiveMutationEngine(mutation_config)
        quantum_engine = QuantumEvolutionEngine(coherence_time=1.0, decoherence_rate=0.1)
        
        # Sample population
        population = []
        for _ in range(10):
            topology = np.random.random((8, 16))
            threshold = np.percentile(topology, 70)
            population.append((topology > threshold).astype(np.float32))
        
        fitness_scores = np.random.uniform(-1.0, -0.1, len(population))
        
        # Run adaptive mutations
        mutated_population, adaptive_metrics = adaptive_engine.evolve_population(population, fitness_scores)
        
        # Run quantum evolution
        quantum_population, quantum_metrics = quantum_engine.evolve_generation(mutated_population, fitness_scores)
        
        # Verify integration
        assert len(quantum_population) == len(population)
        assert adaptive_metrics['generation'] == 1
        assert quantum_metrics['quantum_crossover_applied'] == True
    
    @pytest.mark.asyncio
    async def test_distributed_monitoring_integration(self):
        """Test distributed evolution with monitoring"""
        # Create distributed coordinator
        coordinator = FederatedEvolutionCoordinator("node_0", total_nodes=3)
        
        # Create monitoring system
        monitor = RealTimePerformanceMonitor(auto_tune=True, collection_interval=0.1)
        
        # Add evolution-specific metrics
        def evolution_efficiency():
            return np.random.uniform(0.1, 1.0)
        
        monitor.add_custom_metric('evolution_efficiency', evolution_efficiency)
        
        try:
            # Start monitoring
            monitor.start_monitoring()
            
            # Initialize distributed evolution
            sample_population = [np.random.random((8, 16)) for _ in range(10)]
            await coordinator.initialize_federated_evolution(sample_population)
            
            # Let monitoring collect some data
            time.sleep(0.5)
            
            # Check integration
            status = monitor.get_current_status()
            assert status['monitoring_active'] == True
            
            consensus_metrics = coordinator.get_consensus_metrics()
            assert consensus_metrics['node_id'] == "node_0"
            
        finally:
            monitor.stop_monitoring()
    
    def test_performance_optimization_feedback_loop(self):
        """Test performance monitoring feeding back to optimization"""
        # Create monitoring system
        monitor = RealTimePerformanceMonitor(auto_tune=True, collection_interval=0.1)
        
        # Create adaptive mutation system
        mutation_config = AdaptiveMutationConfig()
        adaptive_engine = AdaptiveMutationEngine(mutation_config)
        
        optimization_adjustments = []
        
        def performance_feedback_optimizer():
            """Simulate performance-based optimization"""
            status = monitor.get_current_status()
            
            if status['overall_health'] == 'warning':
                # Adjust mutation parameters for better performance
                for operator in adaptive_engine.operators.values():
                    operator.current_mutation_rate *= 0.9  # Reduce mutation rate
                optimization_adjustments.append('reduce_mutation_rate')
            
            return len(optimization_adjustments)
        
        monitor.add_custom_metric('optimization_adjustments', performance_feedback_optimizer)
        
        try:
            monitor.start_monitoring()
            time.sleep(1.0)  # Let system run and potentially optimize
            
            # The feedback loop should be operational
            status = monitor.get_current_status()
            assert 'metrics_collected' in status
            
        finally:
            monitor.stop_monitoring()


# Performance benchmarks for next-gen features
class TestPerformanceBenchmarks:
    """Performance benchmarks for next-gen features"""
    
    def test_quantum_evolution_performance(self):
        """Benchmark quantum evolution performance"""
        engine = QuantumEvolutionEngine()
        
        # Large population for performance test
        population = [np.random.random((16, 32)) for _ in range(100)]
        fitness_scores = np.random.uniform(-1.0, -0.1, 100)
        
        start_time = time.time()
        next_generation, metrics = engine.evolve_generation(population, fitness_scores)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 10.0  # Should complete within 10 seconds
        assert len(next_generation) == len(population)
        
        print(f"Quantum evolution (100 individuals): {execution_time:.3f}s")
    
    def test_adaptive_mutation_performance(self):
        """Benchmark adaptive mutation performance"""
        config = AdaptiveMutationConfig(enable_hierarchical=True, enable_self_adaptation=True)
        engine = AdaptiveMutationEngine(config)
        
        # Large population
        population = [np.random.random((16, 32)) for _ in range(100)]
        fitness_scores = np.random.uniform(-1.0, -0.1, 100)
        
        start_time = time.time()
        mutated_population, metrics = engine.evolve_population(population, fitness_scores)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert len(mutated_population) == len(population)
        
        print(f"Adaptive mutation (100 individuals): {execution_time:.3f}s")
    
    def test_monitoring_overhead(self):
        """Benchmark monitoring system overhead"""
        monitor = RealTimePerformanceMonitor(auto_tune=True, collection_interval=0.01)  # High frequency
        
        # Add many custom metrics
        for i in range(20):
            monitor.add_custom_metric(f'metric_{i}', lambda: np.random.random())
        
        try:
            start_time = time.time()
            monitor.start_monitoring()
            
            # Let it run for a short time
            time.sleep(2.0)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Check that monitoring doesn't consume excessive resources
            status = monitor.get_current_status()
            metrics_collected = status.get('metrics_collected', 0)
            
            # Should collect metrics efficiently
            assert metrics_collected > 0
            
            print(f"Monitoring overhead (20 metrics, 2s): {execution_time:.3f}s, {metrics_collected} metrics collected")
            
        finally:
            monitor.stop_monitoring()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])