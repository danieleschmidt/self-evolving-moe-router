#!/usr/bin/env python3
"""
Basic import test for Self-Evolving MoE-Router components.
Tests the import structure without requiring full PyTorch functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test basic imports."""
    print("🧪 Testing Self-Evolving MoE-Router imports...")
    
    try:
        # Test basic imports
        print("  ✓ Testing basic package import...")
        import self_evolving_moe
        print(f"    Package version: {self_evolving_moe.__version__}")
        print(f"    Author: {self_evolving_moe.__author__}")
        
        # Test component imports
        print("  ✓ Testing component imports...")
        from self_evolving_moe import (
            EvolvingMoERouter, 
            ExpertPool, 
            SlimmableMoE, 
            TopologyGenome
        )
        print("    Core components imported successfully")
        
        # Test submodule imports
        print("  ✓ Testing submodule imports...")
        from self_evolving_moe.evolution.router import EvolutionConfig, FitnessEvaluator
        from self_evolving_moe.routing.topology import RoutingParams
        from self_evolving_moe.experts.pool import TransformerExpert, MLPExpert
        from self_evolving_moe.experts.slimmable import SlimmableLinear, SlimmableExpert
        print("    Submodule imports successful")
        
        # Test data layer imports
        print("  ✓ Testing data layer imports...")
        from self_evolving_moe.data import (
            TopologyRepository, 
            ExperimentRepository, 
            EvolutionCache, 
            ModelStorage
        )
        print("    Data layer imports successful")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without PyTorch."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        from self_evolving_moe.evolution.router import EvolutionConfig
        from self_evolving_moe.routing.topology import RoutingParams
        
        # Test configuration creation
        config = EvolutionConfig(
            population_size=10,
            generations=5,
            mutation_rate=0.1
        )
        print(f"  ✓ Created EvolutionConfig: {config.population_size} pop, {config.generations} gen")
        
        # Test routing params
        routing_params = RoutingParams(
            temperature=1.0,
            top_k=2,
            load_balancing_weight=0.01
        )
        print(f"  ✓ Created RoutingParams: temp={routing_params.temperature}, k={routing_params.top_k}")
        
        print("✅ Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Self-Evolving MoE-Router - Basic Test Suite")
    print("="*60)
    
    success = True
    
    # Test imports
    success &= test_imports()
    
    # Test basic functionality
    success &= test_basic_functionality()
    
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS PASSED! System is ready for evolution.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    print("="*60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())