"""Unit tests for custom exceptions."""

import pytest

from src.self_evolving_moe.utils.exceptions import (
    MoEValidationError,
    EvolutionError,
    TopologyError,
    ExpertPoolError,
    ResourceConstraintError,
    ConfigurationError,
    ModelLoadError,
    CompatibilityError
)


class TestMoEValidationError:
    """Test MoEValidationError base exception."""

    @pytest.mark.unit
    def test_basic_creation(self):
        """Test basic exception creation."""
        error = MoEValidationError("Test validation error")
        
        assert str(error) == "Test validation error"
        assert isinstance(error, ValueError)

    @pytest.mark.unit
    def test_inheritance(self):
        """Test exception inheritance."""
        error = MoEValidationError("Test error")
        
        assert isinstance(error, ValueError)
        assert isinstance(error, MoEValidationError)


class TestEvolutionError:
    """Test EvolutionError exception."""

    @pytest.mark.unit
    def test_basic_creation(self):
        """Test basic EvolutionError creation."""
        error = EvolutionError("Evolution failed")
        
        assert str(error) == "Evolution failed"
        assert error.generation is None
        assert error.population_size is None

    @pytest.mark.unit
    def test_creation_with_parameters(self):
        """Test EvolutionError creation with parameters."""
        error = EvolutionError(
            "Evolution failed at generation 5",
            generation=5,
            population_size=100
        )
        
        assert str(error) == "Evolution failed at generation 5"
        assert error.generation == 5
        assert error.population_size == 100

    @pytest.mark.unit
    def test_exception_handling(self):
        """Test exception handling in try-catch."""
        with pytest.raises(EvolutionError) as exc_info:
            raise EvolutionError("Test evolution error", generation=10)
        
        assert exc_info.value.generation == 10
        assert "Test evolution error" in str(exc_info.value)


class TestTopologyError:
    """Test TopologyError exception."""

    @pytest.mark.unit
    def test_basic_creation(self):
        """Test basic TopologyError creation."""
        error = TopologyError("Invalid topology")
        
        assert str(error) == "Invalid topology"
        assert error.topology_id is None
        assert error.sparsity is None

    @pytest.mark.unit
    def test_creation_with_parameters(self):
        """Test TopologyError creation with parameters."""
        error = TopologyError(
            "Topology too sparse",
            topology_id="topo_001",
            sparsity=0.95
        )
        
        assert str(error) == "Topology too sparse"
        assert error.topology_id == "topo_001"
        assert error.sparsity == 0.95

    @pytest.mark.unit
    def test_exception_handling(self):
        """Test exception handling in try-catch."""
        with pytest.raises(TopologyError) as exc_info:
            raise TopologyError(
                "Invalid routing matrix",
                topology_id="bad_topology"
            )
        
        assert exc_info.value.topology_id == "bad_topology"
        assert "Invalid routing matrix" in str(exc_info.value)


class TestExpertPoolError:
    """Test ExpertPoolError exception."""

    @pytest.mark.unit
    def test_basic_creation(self):
        """Test basic ExpertPoolError creation."""
        error = ExpertPoolError("Expert pool error")
        
        assert str(error) == "Expert pool error"
        assert error.num_experts is None
        assert error.expert_type is None

    @pytest.mark.unit
    def test_creation_with_parameters(self):
        """Test ExpertPoolError creation with parameters."""
        error = ExpertPoolError(
            "Too many experts",
            num_experts=1000,
            expert_type="transformer"
        )
        
        assert str(error) == "Too many experts"
        assert error.num_experts == 1000
        assert error.expert_type == "transformer"

    @pytest.mark.unit
    def test_exception_handling(self):
        """Test exception handling in try-catch."""
        with pytest.raises(ExpertPoolError) as exc_info:
            raise ExpertPoolError(
                "Expert initialization failed",
                num_experts=16,
                expert_type="mlp"
            )
        
        assert exc_info.value.num_experts == 16
        assert exc_info.value.expert_type == "mlp"
        assert "Expert initialization failed" in str(exc_info.value)


class TestResourceConstraintError:
    """Test ResourceConstraintError exception."""

    @pytest.mark.unit
    def test_basic_creation(self):
        """Test basic ResourceConstraintError creation."""
        error = ResourceConstraintError("Resource constraint violated")
        
        assert str(error) == "Resource constraint violated"
        assert error.resource_type is None
        assert error.limit is None
        assert error.actual is None

    @pytest.mark.unit
    def test_creation_with_parameters(self):
        """Test ResourceConstraintError creation with parameters."""
        error = ResourceConstraintError(
            "Memory limit exceeded",
            resource_type="memory",
            limit=1024.0,
            actual=2048.0
        )
        
        assert str(error) == "Memory limit exceeded"
        assert error.resource_type == "memory"
        assert error.limit == 1024.0
        assert error.actual == 2048.0

    @pytest.mark.unit
    def test_exception_handling(self):
        """Test exception handling in try-catch."""
        with pytest.raises(ResourceConstraintError) as exc_info:
            raise ResourceConstraintError(
                "Latency constraint violated",
                resource_type="latency",
                limit=100.0,
                actual=150.0
            )
        
        assert exc_info.value.resource_type == "latency"
        assert exc_info.value.limit == 100.0
        assert exc_info.value.actual == 150.0
        assert "Latency constraint violated" in str(exc_info.value)


class TestConfigurationError:
    """Test ConfigurationError exception."""

    @pytest.mark.unit
    def test_basic_creation(self):
        """Test basic ConfigurationError creation."""
        error = ConfigurationError("Invalid configuration")
        
        assert str(error) == "Invalid configuration"
        assert error.parameter is None
        assert error.value is None

    @pytest.mark.unit
    def test_creation_with_parameters(self):
        """Test ConfigurationError creation with parameters."""
        error = ConfigurationError(
            "Invalid population size",
            parameter="population_size",
            value=0
        )
        
        assert str(error) == "Invalid population size"
        assert error.parameter == "population_size"
        assert error.value == 0

    @pytest.mark.unit
    def test_inheritance_from_validation_error(self):
        """Test ConfigurationError inherits from MoEValidationError."""
        error = ConfigurationError("Config error")
        
        assert isinstance(error, MoEValidationError)
        assert isinstance(error, ValueError)

    @pytest.mark.unit
    def test_exception_handling(self):
        """Test exception handling in try-catch."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError(
                "Invalid mutation rate",
                parameter="mutation_rate",
                value=1.5
            )
        
        assert exc_info.value.parameter == "mutation_rate"
        assert exc_info.value.value == 1.5
        assert "Invalid mutation rate" in str(exc_info.value)


class TestModelLoadError:
    """Test ModelLoadError exception."""

    @pytest.mark.unit
    def test_basic_creation(self):
        """Test basic ModelLoadError creation."""
        error = ModelLoadError("Failed to load model")
        
        assert str(error) == "Failed to load model"
        assert error.model_path is None
        assert error.expected_format is None

    @pytest.mark.unit
    def test_creation_with_parameters(self):
        """Test ModelLoadError creation with parameters."""
        error = ModelLoadError(
            "Unsupported model format",
            model_path="/path/to/model.bin",
            expected_format="pytorch"
        )
        
        assert str(error) == "Unsupported model format"
        assert error.model_path == "/path/to/model.bin"
        assert error.expected_format == "pytorch"

    @pytest.mark.unit
    def test_exception_handling(self):
        """Test exception handling in try-catch."""
        with pytest.raises(ModelLoadError) as exc_info:
            raise ModelLoadError(
                "Model file not found",
                model_path="/missing/model.pth"
            )
        
        assert exc_info.value.model_path == "/missing/model.pth"
        assert "Model file not found" in str(exc_info.value)


class TestCompatibilityError:
    """Test CompatibilityError exception."""

    @pytest.mark.unit
    def test_basic_creation(self):
        """Test basic CompatibilityError creation."""
        error = CompatibilityError("Components incompatible")
        
        assert str(error) == "Components incompatible"
        assert error.component1 is None
        assert error.component2 is None

    @pytest.mark.unit
    def test_creation_with_parameters(self):
        """Test CompatibilityError creation with parameters."""
        error = CompatibilityError(
            "Version mismatch",
            component1="router_v2.1",
            component2="expert_pool_v1.5"
        )
        
        assert str(error) == "Version mismatch"
        assert error.component1 == "router_v2.1"
        assert error.component2 == "expert_pool_v1.5"

    @pytest.mark.unit
    def test_exception_handling(self):
        """Test exception handling in try-catch."""
        with pytest.raises(CompatibilityError) as exc_info:
            raise CompatibilityError(
                "Incompatible tensor shapes",
                component1="topology",
                component2="expert_pool"
            )
        
        assert exc_info.value.component1 == "topology"
        assert exc_info.value.component2 == "expert_pool"
        assert "Incompatible tensor shapes" in str(exc_info.value)


class TestExceptionChaining:
    """Test exception chaining and multiple exception types."""

    @pytest.mark.unit
    def test_exception_chaining(self):
        """Test exception chaining with from clause."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise TopologyError("Topology validation failed") from e
        except TopologyError as te:
            assert te.__cause__ is not None
            assert isinstance(te.__cause__, ValueError)
            assert str(te.__cause__) == "Original error"

    @pytest.mark.unit
    def test_multiple_exception_types_in_hierarchy(self):
        """Test catching multiple exception types."""
        # Test that specific exceptions can be caught by base class
        with pytest.raises(Exception):
            raise TopologyError("Test error")
        
        with pytest.raises(MoEValidationError):
            raise ConfigurationError("Test config error")
        
        # Test multiple exception types in same try-catch
        for exception_class, message in [
            (EvolutionError, "Evolution failed"),
            (TopologyError, "Topology invalid"),
            (ExpertPoolError, "Expert pool error"),
            (ConfigurationError, "Config error")
        ]:
            with pytest.raises(Exception) as exc_info:
                raise exception_class(message)
            
            assert isinstance(exc_info.value, exception_class)
            assert str(exc_info.value) == message

    @pytest.mark.unit
    def test_exception_attributes_preservation(self):
        """Test that custom exception attributes are preserved during propagation."""
        def inner_function():
            raise EvolutionError(
                "Inner evolution error",
                generation=5,
                population_size=100
            )
        
        def outer_function():
            try:
                inner_function()
            except EvolutionError as e:
                # Re-raise with additional context
                e.generation = 6  # Modify generation
                raise e
        
        with pytest.raises(EvolutionError) as exc_info:
            outer_function()
        
        # Check that modified attributes are preserved
        assert exc_info.value.generation == 6
        assert exc_info.value.population_size == 100

    @pytest.mark.unit
    def test_exception_with_none_values(self):
        """Test exceptions with None values for optional parameters."""
        exceptions_with_params = [
            (EvolutionError("test", None, None), ["generation", "population_size"]),
            (TopologyError("test", None, None), ["topology_id", "sparsity"]),
            (ExpertPoolError("test", None, None), ["num_experts", "expert_type"]),
            (ResourceConstraintError("test", None, None, None), ["resource_type", "limit", "actual"]),
            (ConfigurationError("test", None, None), ["parameter", "value"]),
            (ModelLoadError("test", None, None), ["model_path", "expected_format"]),
            (CompatibilityError("test", None, None), ["component1", "component2"])
        ]
        
        for exception, param_names in exceptions_with_params:
            for param_name in param_names:
                assert hasattr(exception, param_name)
                assert getattr(exception, param_name) is None