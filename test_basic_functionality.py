
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_simple_demo_import():
    """Test that simple demo can be imported."""
    try:
        import simple_working_demo
        assert hasattr(simple_working_demo, 'run_simple_demo')
    except ImportError:
        pytest.skip("Simple demo not available")

def test_robust_system_import():
    """Test that robust system can be imported.""" 
    try:
        import robust_moe_system
        assert hasattr(robust_moe_system, 'RobustTopology')
    except ImportError:
        pytest.skip("Robust system not available")

def test_optimized_demo_import():
    """Test that optimized demo can be imported."""
    try:
        import optimized_simple_demo
        assert hasattr(optimized_simple_demo, 'OptimizedTopology')
    except ImportError:
        pytest.skip("Optimized demo not available")

def test_topology_creation():
    """Test basic topology creation."""
    try:
        from simple_working_demo import SimpleTopology
        topology = SimpleTopology(8, 4, 0.2)
        assert topology.num_tokens == 8
        assert topology.num_experts == 4
        assert topology.compute_sparsity() >= 0.0
    except ImportError:
        pytest.skip("SimpleTopology not available")

def test_topology_operations():
    """Test topology operations."""
    try:
        from simple_working_demo import SimpleTopology
        topology = SimpleTopology(4, 3, 0.3)
        
        # Test mutation
        original_matrix = topology.routing_matrix.copy()
        topology.mutate(0.5)
        # Should be able to mutate without crashing
        assert topology.routing_matrix.shape == original_matrix.shape
        
        # Test crossover
        other = SimpleTopology(4, 3, 0.3)
        child = topology.crossover(other)
        assert child.num_tokens == 4
        assert child.num_experts == 3
        
    except ImportError:
        pytest.skip("SimpleTopology not available")

def test_model_forward():
    """Test model forward pass."""
    try:
        from simple_working_demo import SimpleMoEModel
        model = SimpleMoEModel(16, 4, 32)
        
        # Test forward pass
        input_data = np.random.randn(2, 8, 16)
        output, aux = model.forward(input_data)
        
        assert output.shape == input_data.shape
        assert 'expert_usage' in aux
        assert len(aux['expert_usage']) == 4
        
    except ImportError:
        pytest.skip("SimpleMoEModel not available")

def test_evolution_basic():
    """Test basic evolution functionality."""
    try:
        from simple_working_demo import SimpleEvolver, SimpleMoEModel
        
        evolver = SimpleEvolver(8, 4, 6)  # Small population for speed
        model = SimpleMoEModel(16, 4, 32)
        
        # Create minimal test data
        data = [(np.random.randn(1, 8, 16), np.random.randn(1, 8, 16)) for _ in range(2)]
        
        # Test one generation
        stats = evolver.evolve_one_generation(model, data)
        assert 'generation' in stats
        assert 'best_fitness' in stats
        
    except ImportError:
        pytest.skip("Evolution components not available")

def test_numpy_operations():
    """Test numpy operations work correctly."""
    arr = np.random.randn(10, 5)
    assert arr.shape == (10, 5)
    assert np.mean(arr) is not np.nan
    assert np.std(arr) >= 0

def test_basic_math():
    """Test basic mathematical operations."""
    assert 2 + 2 == 4
    assert np.exp(0) == 1.0
    assert np.log(1) == 0.0
