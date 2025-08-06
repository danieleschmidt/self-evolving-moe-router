"""Unit tests for validation utilities."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.self_evolving_moe.utils.validation import (
    validate_config,
    validate_topology,
    validate_expert_pool,
    validate_device,
    validate_data_loader,
    validate_resource_constraints,
    validate_file_path
)
from src.self_evolving_moe.utils.exceptions import (
    ConfigurationError,
    TopologyError,
    ExpertPoolError,
    MoEValidationError,
    ResourceConstraintError
)


class TestValidateConfig:
    """Test configuration validation."""

    @pytest.mark.unit
    def test_valid_config(self):
        """Test validation with valid configuration."""
        config = Mock()
        config.population_size = 50
        config.generations = 100
        config.mutation_rate = 0.1
        config.crossover_rate = 0.7
        config.elitism_ratio = 0.1
        config.selection_method = "tournament"
        config.tournament_size = 3
        config.max_active_experts = 16
        config.min_expert_usage = 0.1
        config.target_sparsity = 0.8
        config.patience = 10
        config.min_improvement = 0.001
        
        assert validate_config(config) == True

    @pytest.mark.unit
    def test_invalid_population_size(self):
        """Test validation with invalid population size."""
        config = Mock()
        config.population_size = 1  # Too small
        config.generations = 100
        config.mutation_rate = 0.1
        config.crossover_rate = 0.7
        config.elitism_ratio = 0.1
        config.selection_method = "tournament"
        config.tournament_size = 3
        config.max_active_experts = 16
        config.min_expert_usage = 0.1
        config.target_sparsity = 0.8
        config.patience = 10
        config.min_improvement = 0.001
        
        with pytest.raises(ConfigurationError, match="Population size must be an integer >= 2"):
            validate_config(config)

    @pytest.mark.unit
    def test_population_size_too_large(self):
        """Test validation with population size too large."""
        config = Mock()
        config.population_size = 1001  # Too large
        config.generations = 100
        config.mutation_rate = 0.1
        config.crossover_rate = 0.7
        config.elitism_ratio = 0.1
        config.selection_method = "tournament"
        config.tournament_size = 3
        config.max_active_experts = 16
        config.min_expert_usage = 0.1
        config.target_sparsity = 0.8
        config.patience = 10
        config.min_improvement = 0.001
        
        with pytest.raises(ConfigurationError, match="Population size too large"):
            validate_config(config)

    @pytest.mark.unit
    def test_invalid_generations(self):
        """Test validation with invalid generations."""
        config = Mock()
        config.population_size = 50
        config.generations = 0  # Invalid
        config.mutation_rate = 0.1
        config.crossover_rate = 0.7
        config.elitism_ratio = 0.1
        config.selection_method = "tournament"
        config.tournament_size = 3
        config.max_active_experts = 16
        config.min_expert_usage = 0.1
        config.target_sparsity = 0.8
        config.patience = 10
        config.min_improvement = 0.001
        
        with pytest.raises(ConfigurationError, match="Generations must be an integer >= 1"):
            validate_config(config)

    @pytest.mark.unit
    def test_invalid_mutation_rate(self):
        """Test validation with invalid mutation rate."""
        config = Mock()
        config.population_size = 50
        config.generations = 100
        config.mutation_rate = 1.5  # Too high
        config.crossover_rate = 0.7
        config.elitism_ratio = 0.1
        config.selection_method = "tournament"
        config.tournament_size = 3
        config.max_active_experts = 16
        config.min_expert_usage = 0.1
        config.target_sparsity = 0.8
        config.patience = 10
        config.min_improvement = 0.001
        
        with pytest.raises(ConfigurationError, match="Mutation rate must be between 0.0 and 1.0"):
            validate_config(config)

    @pytest.mark.unit
    def test_invalid_selection_method(self):
        """Test validation with invalid selection method."""
        config = Mock()
        config.population_size = 50
        config.generations = 100
        config.mutation_rate = 0.1
        config.crossover_rate = 0.7
        config.elitism_ratio = 0.1
        config.selection_method = "invalid_method"
        config.tournament_size = 3
        config.max_active_experts = 16
        config.min_expert_usage = 0.1
        config.target_sparsity = 0.8
        config.patience = 10
        config.min_improvement = 0.001
        
        with pytest.raises(ConfigurationError, match="Unknown selection method"):
            validate_config(config)

    @pytest.mark.unit
    def test_invalid_tournament_size(self):
        """Test validation with invalid tournament size."""
        config = Mock()
        config.population_size = 50
        config.generations = 100
        config.mutation_rate = 0.1
        config.crossover_rate = 0.7
        config.elitism_ratio = 0.1
        config.selection_method = "tournament"
        config.tournament_size = 1  # Too small
        config.max_active_experts = 16
        config.min_expert_usage = 0.1
        config.target_sparsity = 0.8
        config.patience = 10
        config.min_improvement = 0.001
        
        with pytest.raises(ConfigurationError, match="Tournament size must be >= 2"):
            validate_config(config)

    @pytest.mark.unit
    def test_missing_parameter(self):
        """Test validation with missing required parameter."""
        config = Mock()
        config.population_size = 50
        # Missing generations parameter
        config.mutation_rate = 0.1
        
        with pytest.raises(ConfigurationError, match="Missing required configuration parameter"):
            validate_config(config)


class TestValidateTopology:
    """Test topology validation."""

    def create_valid_topology(self, device="cpu"):
        """Helper to create valid topology."""
        topology = Mock()
        topology.num_tokens = 10
        topology.num_experts = 4
        topology.routing_matrix = torch.zeros((10, 4), device=device)
        # Add some connections
        topology.routing_matrix[0, [0, 1]] = 1.0
        topology.routing_matrix[1, [1, 2]] = 1.0
        topology.routing_matrix[2, [2, 3]] = 1.0
        topology.routing_matrix[3, [0, 3]] = 1.0
        
        topology.expert_graph = torch.zeros((4, 4), device=device)
        topology.expert_graph[0, 1] = 1.0
        topology.expert_graph[1, 2] = 1.0
        
        routing_params = Mock()
        routing_params.temperature = 1.0
        routing_params.top_k = 2
        routing_params.load_balancing_weight = 0.01
        routing_params.diversity_weight = 0.1
        topology.routing_params = routing_params
        
        topology.compute_sparsity = Mock(return_value=0.8)
        
        return topology

    @pytest.mark.unit
    def test_valid_topology(self, device):
        """Test validation with valid topology."""
        topology = self.create_valid_topology(device)
        assert validate_topology(topology) == True

    @pytest.mark.unit
    def test_missing_routing_matrix(self):
        """Test validation with missing routing matrix."""
        topology = Mock()
        # Missing routing_matrix
        topology.expert_graph = torch.zeros((4, 4))
        topology.routing_params = Mock()
        
        with pytest.raises(TopologyError, match="Topology missing routing_matrix"):
            validate_topology(topology)

    @pytest.mark.unit
    def test_wrong_matrix_dimensions(self, device):
        """Test validation with wrong matrix dimensions."""
        topology = Mock()
        topology.routing_matrix = torch.zeros((10, 4, 2), device=device)  # 3D instead of 2D
        topology.expert_graph = torch.zeros((4, 4), device=device)
        topology.routing_params = Mock()
        
        with pytest.raises(TopologyError, match="Routing matrix must be 2D"):
            validate_topology(topology)

    @pytest.mark.unit
    def test_dimension_mismatch(self, device):
        """Test validation with dimension mismatch."""
        topology = Mock()
        topology.num_tokens = 10
        topology.num_experts = 4
        topology.routing_matrix = torch.zeros((8, 4), device=device)  # Wrong token dimension
        topology.expert_graph = torch.zeros((4, 4), device=device)
        topology.routing_params = Mock()
        
        with pytest.raises(TopologyError, match="Routing matrix token dimension mismatch"):
            validate_topology(topology)

    @pytest.mark.unit
    def test_non_binary_matrix(self, device):
        """Test validation with non-binary routing matrix."""
        topology = Mock()
        topology.num_tokens = 4
        topology.num_experts = 4
        topology.routing_matrix = torch.rand((4, 4), device=device)  # Non-binary values
        topology.expert_graph = torch.zeros((4, 4), device=device)
        topology.routing_params = Mock()
        
        with pytest.raises(TopologyError, match="Routing matrix must be binary"):
            validate_topology(topology)

    @pytest.mark.unit
    def test_disconnected_tokens(self, device):
        """Test validation with disconnected tokens."""
        topology = Mock()
        topology.num_tokens = 4
        topology.num_experts = 4
        topology.routing_matrix = torch.zeros((4, 4), device=device)
        # Token 0 has no connections (all zeros)
        topology.routing_matrix[1, [0, 1]] = 1.0
        topology.routing_matrix[2, [1, 2]] = 1.0
        topology.routing_matrix[3, [2, 3]] = 1.0
        
        topology.expert_graph = torch.zeros((4, 4), device=device)
        topology.routing_params = Mock()
        
        with pytest.raises(TopologyError, match="Tokens .* have no expert connections"):
            validate_topology(topology)

    @pytest.mark.unit
    def test_self_connections_in_expert_graph(self, device):
        """Test validation with self-connections in expert graph."""
        topology = self.create_valid_topology(device)
        # Add self-connection
        topology.expert_graph[0, 0] = 1.0
        
        with pytest.raises(TopologyError, match="Expert graph should not have self-connections"):
            validate_topology(topology)

    @pytest.mark.unit
    def test_invalid_temperature(self, device):
        """Test validation with invalid temperature."""
        topology = self.create_valid_topology(device)
        topology.routing_params.temperature = -1.0  # Invalid
        
        with pytest.raises(TopologyError, match="Temperature must be positive"):
            validate_topology(topology)

    @pytest.mark.unit
    def test_invalid_top_k(self, device):
        """Test validation with invalid top_k."""
        topology = self.create_valid_topology(device)
        topology.routing_params.top_k = 0  # Invalid
        
        with pytest.raises(TopologyError, match="Top-k must be between 1 and"):
            validate_topology(topology)

    @pytest.mark.unit
    def test_invalid_load_balancing_weight(self, device):
        """Test validation with invalid load balancing weight."""
        topology = self.create_valid_topology(device)
        topology.routing_params.load_balancing_weight = 1.5  # Out of range
        
        with pytest.raises(TopologyError, match="Load balancing weight must be between 0 and 1"):
            validate_topology(topology)

    @pytest.mark.unit
    def test_invalid_sparsity(self, device):
        """Test validation with invalid sparsity value."""
        topology = self.create_valid_topology(device)
        topology.compute_sparsity = Mock(return_value=-0.1)  # Invalid sparsity
        
        with pytest.raises(TopologyError, match="Invalid sparsity value"):
            validate_topology(topology)


class TestValidateExpertPool:
    """Test expert pool validation."""

    def create_valid_expert_pool(self):
        """Helper to create valid expert pool."""
        pool = Mock()
        pool.num_experts = 4
        pool.expert_dim = 256
        pool.expert_type = "transformer_block"
        
        experts = []
        for i in range(4):
            expert = Mock()
            expert.forward = Mock()
            expert.expert_id = i
            experts.append(expert)
        
        pool.experts = experts
        pool.active_experts = [0, 1, 2, 3]
        
        return pool

    @pytest.mark.unit
    def test_valid_expert_pool(self):
        """Test validation with valid expert pool."""
        pool = self.create_valid_expert_pool()
        assert validate_expert_pool(pool) == True

    @pytest.mark.unit
    def test_missing_experts_list(self):
        """Test validation with missing experts list."""
        pool = Mock()
        # Missing experts attribute
        pool.num_experts = 4
        pool.expert_dim = 256
        
        with pytest.raises(ExpertPoolError, match="Expert pool missing experts list"):
            validate_expert_pool(pool)

    @pytest.mark.unit
    def test_zero_experts(self):
        """Test validation with zero experts."""
        pool = Mock()
        pool.experts = []
        pool.num_experts = 0
        pool.expert_dim = 256
        pool.expert_type = "transformer_block"
        
        with pytest.raises(ExpertPoolError, match="Must have at least 1 expert"):
            validate_expert_pool(pool)

    @pytest.mark.unit
    def test_too_many_experts(self):
        """Test validation with too many experts."""
        pool = Mock()
        pool.experts = [Mock() for _ in range(1025)]  # Too many
        pool.num_experts = 1025
        pool.expert_dim = 256
        pool.expert_type = "transformer_block"
        
        with pytest.raises(ExpertPoolError, match="Too many experts"):
            validate_expert_pool(pool)

    @pytest.mark.unit
    def test_expert_count_mismatch(self):
        """Test validation with expert count mismatch."""
        pool = Mock()
        pool.experts = [Mock(), Mock()]  # 2 experts
        pool.num_experts = 4  # But says 4
        pool.expert_dim = 256
        pool.expert_type = "transformer_block"
        
        with pytest.raises(ExpertPoolError, match="Expert count mismatch"):
            validate_expert_pool(pool)

    @pytest.mark.unit
    def test_expert_missing_forward(self):
        """Test validation with expert missing forward method."""
        pool = Mock()
        pool.num_experts = 2
        pool.expert_dim = 256
        pool.expert_type = "transformer_block"
        
        expert1 = Mock()
        expert1.forward = Mock()
        expert1.expert_id = 0
        
        expert2 = Mock()
        # Missing forward method
        expert2.expert_id = 1
        
        pool.experts = [expert1, expert2]
        
        with pytest.raises(ExpertPoolError, match="Expert 1 missing forward method"):
            validate_expert_pool(pool)

    @pytest.mark.unit
    def test_expert_wrong_id(self):
        """Test validation with expert having wrong ID."""
        pool = Mock()
        pool.num_experts = 2
        pool.expert_dim = 256
        pool.expert_type = "transformer_block"
        
        expert1 = Mock()
        expert1.forward = Mock()
        expert1.expert_id = 0
        
        expert2 = Mock()
        expert2.forward = Mock()
        expert2.expert_id = 5  # Wrong ID (should be 1)
        
        pool.experts = [expert1, expert2]
        
        with pytest.raises(ExpertPoolError, match="Expert 1 has wrong ID: 5"):
            validate_expert_pool(pool)

    @pytest.mark.unit
    def test_invalid_active_expert_id(self):
        """Test validation with invalid active expert ID."""
        pool = self.create_valid_expert_pool()
        pool.active_experts = [0, 1, 5]  # 5 is invalid (only have 4 experts)
        
        with pytest.raises(ExpertPoolError, match="Invalid active expert ID: 5"):
            validate_expert_pool(pool)

    @pytest.mark.unit
    def test_invalid_expert_type(self):
        """Test validation with invalid expert type."""
        pool = self.create_valid_expert_pool()
        pool.expert_type = "invalid_type"
        
        with pytest.raises(ExpertPoolError, match="Unknown expert type"):
            validate_expert_pool(pool)


class TestValidateDevice:
    """Test device validation."""

    @pytest.mark.unit
    def test_cpu_device(self):
        """Test CPU device validation."""
        assert validate_device("cpu") == "cpu"
        assert validate_device("CPU") == "cpu"
        assert validate_device(" cpu ") == "cpu"

    @pytest.mark.unit
    @patch('src.self_evolving_moe.utils.validation.torch')
    def test_cuda_device_available(self, mock_torch):
        """Test CUDA device validation when available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        
        assert validate_device("cuda") == "cuda"
        assert validate_device("CUDA") == "cuda"
        assert validate_device("cuda:0") == "cuda:0"
        assert validate_device("cuda:1") == "cuda:1"

    @pytest.mark.unit
    @patch('src.self_evolving_moe.utils.validation.torch')
    def test_cuda_device_unavailable(self, mock_torch):
        """Test CUDA device validation when unavailable."""
        mock_torch.cuda.is_available.return_value = False
        
        with pytest.raises(MoEValidationError, match="CUDA requested but not available"):
            validate_device("cuda")

    @pytest.mark.unit
    @patch('src.self_evolving_moe.utils.validation.torch')
    def test_invalid_cuda_device_id(self, mock_torch):
        """Test invalid CUDA device ID."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        
        with pytest.raises(MoEValidationError, match="CUDA device 5 not available"):
            validate_device("cuda:5")

    @pytest.mark.unit
    @patch('src.self_evolving_moe.utils.validation.torch')
    def test_invalid_cuda_device_format(self, mock_torch):
        """Test invalid CUDA device format."""
        mock_torch.cuda.is_available.return_value = True
        
        with pytest.raises(MoEValidationError, match="Invalid CUDA device ID"):
            validate_device("cuda:abc")

    @pytest.mark.unit
    def test_invalid_device_type(self):
        """Test invalid device type."""
        with pytest.raises(MoEValidationError, match="Device must be string"):
            validate_device(123)

    @pytest.mark.unit
    def test_unknown_device(self):
        """Test unknown device."""
        with pytest.raises(MoEValidationError, match="Unknown device"):
            validate_device("unknown_device")


class TestValidateDataLoader:
    """Test data loader validation."""

    @pytest.mark.unit
    def test_valid_data_loader(self):
        """Test validation with valid data loader."""
        # Mock data loader
        data_loader = Mock()
        data_loader.__iter__ = Mock()
        data_loader.__len__ = Mock(return_value=10)
        
        # Mock sample batch
        inputs = torch.randn(4, 128)
        targets = torch.randint(0, 2, (4,))
        data_loader.__iter__.return_value = iter([(inputs, targets)])
        
        assert validate_data_loader(data_loader) == True

    @pytest.mark.unit
    def test_non_iterable_data_loader(self):
        """Test validation with non-iterable data loader."""
        data_loader = Mock()
        # Missing __iter__
        data_loader.__len__ = Mock(return_value=10)
        
        with pytest.raises(MoEValidationError, match="Data loader must be iterable"):
            validate_data_loader(data_loader)

    @pytest.mark.unit
    def test_data_loader_no_length(self):
        """Test validation with data loader without length."""
        data_loader = Mock()
        data_loader.__iter__ = Mock()
        # Missing __len__
        
        with pytest.raises(MoEValidationError, match="Data loader must have length"):
            validate_data_loader(data_loader)

    @pytest.mark.unit
    def test_empty_data_loader(self):
        """Test validation with empty data loader."""
        data_loader = Mock()
        data_loader.__iter__ = Mock()
        data_loader.__len__ = Mock(return_value=0)
        
        with pytest.raises(MoEValidationError, match="Data loader is empty"):
            validate_data_loader(data_loader)

    @pytest.mark.unit
    def test_wrong_batch_format(self):
        """Test validation with wrong batch format."""
        data_loader = Mock()
        data_loader.__iter__ = Mock()
        data_loader.__len__ = Mock(return_value=10)
        
        # Wrong format: should return (inputs, targets) tuple
        wrong_batch = (torch.randn(4, 128), torch.randint(0, 2, (4,)), torch.ones(4))
        data_loader.__iter__.return_value = iter([wrong_batch])
        
        with pytest.raises(MoEValidationError, match="Data loader must return \\(inputs, targets\\) tuples"):
            validate_data_loader(data_loader)

    @pytest.mark.unit
    def test_batch_size_mismatch(self):
        """Test validation with batch size mismatch."""
        data_loader = Mock()
        data_loader.__iter__ = Mock()
        data_loader.__len__ = Mock(return_value=10)
        
        inputs = torch.randn(4, 128)
        targets = torch.randint(0, 2, (8,))  # Different batch size
        data_loader.__iter__.return_value = iter([(inputs, targets)])
        
        with pytest.raises(MoEValidationError, match="Batch size mismatch"):
            validate_data_loader(data_loader)


class TestValidateResourceConstraints:
    """Test resource constraint validation."""

    @pytest.mark.unit
    def test_valid_constraints(self):
        """Test validation with valid constraints."""
        assert validate_resource_constraints(
            memory_limit=1024 * 1024 * 1024,  # 1GB
            latency_limit=100.0,
            compute_budget=1e9
        ) == True

    @pytest.mark.unit
    def test_invalid_memory_limit_type(self):
        """Test validation with invalid memory limit type."""
        with pytest.raises(ResourceConstraintError, match="Memory limit must be positive integer"):
            validate_resource_constraints(memory_limit="invalid")

    @pytest.mark.unit
    def test_negative_memory_limit(self):
        """Test validation with negative memory limit."""
        with pytest.raises(ResourceConstraintError, match="Memory limit must be positive integer"):
            validate_resource_constraints(memory_limit=-1000)

    @pytest.mark.unit
    def test_memory_limit_too_small(self):
        """Test validation with memory limit too small."""
        with pytest.raises(ResourceConstraintError, match="Memory limit too small"):
            validate_resource_constraints(memory_limit=1000)  # Less than 1MB

    @pytest.mark.unit
    def test_memory_limit_too_large(self):
        """Test validation with memory limit too large."""
        huge_limit = 1024 * 1024 * 1024 * 1024 + 1  # > 1TB
        with pytest.raises(ResourceConstraintError, match="Memory limit too large"):
            validate_resource_constraints(memory_limit=huge_limit)

    @pytest.mark.unit
    def test_invalid_latency_limit(self):
        """Test validation with invalid latency limit."""
        with pytest.raises(ResourceConstraintError, match="Latency limit must be positive number"):
            validate_resource_constraints(latency_limit=-5.0)

    @pytest.mark.unit
    def test_latency_limit_too_small(self):
        """Test validation with latency limit too small."""
        with pytest.raises(ResourceConstraintError, match="Latency limit too small"):
            validate_resource_constraints(latency_limit=0.05)  # Less than 0.1ms

    @pytest.mark.unit
    def test_latency_limit_too_large(self):
        """Test validation with latency limit too large."""
        with pytest.raises(ResourceConstraintError, match="Latency limit too large"):
            validate_resource_constraints(latency_limit=4000000)  # > 1 hour

    @pytest.mark.unit
    def test_invalid_compute_budget(self):
        """Test validation with invalid compute budget."""
        with pytest.raises(ResourceConstraintError, match="Compute budget must be positive number"):
            validate_resource_constraints(compute_budget=-1e6)

    @pytest.mark.unit
    def test_none_constraints(self):
        """Test validation with None constraints."""
        assert validate_resource_constraints() == True
        assert validate_resource_constraints(None, None, None) == True


class TestValidateFilePath:
    """Test file path validation."""

    @pytest.mark.unit
    def test_valid_existing_file(self, temp_dir):
        """Test validation with valid existing file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        validated_path = validate_file_path(
            str(test_file),
            must_exist=True
        )
        
        assert validated_path == test_file
        assert isinstance(validated_path, Path)

    @pytest.mark.unit
    def test_non_existing_file_required(self, temp_dir):
        """Test validation when file must exist but doesn't."""
        non_existing = temp_dir / "nonexistent.txt"
        
        with pytest.raises(MoEValidationError, match="File does not exist"):
            validate_file_path(str(non_existing), must_exist=True)

    @pytest.mark.unit
    def test_non_existing_file_optional(self, temp_dir):
        """Test validation when file doesn't need to exist."""
        non_existing = temp_dir / "nonexistent.txt"
        
        validated_path = validate_file_path(
            str(non_existing),
            must_exist=False
        )
        
        assert validated_path == non_existing

    @pytest.mark.unit
    def test_directory_not_file(self, temp_dir):
        """Test validation when path is directory, not file."""
        with pytest.raises(MoEValidationError, match="Path is not a file"):
            validate_file_path(str(temp_dir), must_exist=True)

    @pytest.mark.unit
    def test_wrong_file_extension(self, temp_dir):
        """Test validation with wrong file extension."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        with pytest.raises(MoEValidationError, match="Expected file extension"):
            validate_file_path(
                str(test_file),
                must_exist=True,
                expected_suffix=".json"
            )

    @pytest.mark.unit
    def test_correct_file_extension(self, temp_dir):
        """Test validation with correct file extension."""
        test_file = temp_dir / "test.json"
        test_file.write_text('{"test": true}')
        
        validated_path = validate_file_path(
            str(test_file),
            must_exist=True,
            expected_suffix=".json"
        )
        
        assert validated_path == test_file

    @pytest.mark.unit
    def test_case_insensitive_extension(self, temp_dir):
        """Test validation with case insensitive extension."""
        test_file = temp_dir / "test.JSON"
        test_file.write_text('{"test": true}')
        
        validated_path = validate_file_path(
            str(test_file),
            must_exist=True,
            expected_suffix=".json"
        )
        
        assert validated_path == test_file

    @pytest.mark.unit
    def test_create_parent_directories(self, temp_dir):
        """Test creation of parent directories."""
        nested_file = temp_dir / "nested" / "deep" / "test.txt"
        
        validated_path = validate_file_path(
            str(nested_file),
            must_exist=False,
            create_parent=True
        )
        
        assert validated_path == nested_file
        assert nested_file.parent.exists()
        assert nested_file.parent.is_dir()

    @pytest.mark.unit
    def test_invalid_path_type(self):
        """Test validation with invalid path type."""
        with pytest.raises(MoEValidationError, match="File path must be string or Path"):
            validate_file_path(123)

    @pytest.mark.unit
    def test_path_object_input(self, temp_dir):
        """Test validation with Path object input."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        validated_path = validate_file_path(
            test_file,  # Path object instead of string
            must_exist=True
        )
        
        assert validated_path == test_file
        assert isinstance(validated_path, Path)