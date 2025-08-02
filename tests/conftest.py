"""Shared test configuration and fixtures for Self-Evolving MoE-Router."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, Optional
from unittest.mock import Mock, MagicMock

import pytest
import torch
import numpy as np
import yaml
from omegaconf import OmegaConf


# Configure pytest marks for test categorization
def pytest_configure(config):
    """Configure custom pytest marks."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (slower)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (slowest)")
    config.addinivalue_line("markers", "slow: Tests that take significant time")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "distributed: Tests requiring multiple processes")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if no GPU available."""
    skip_gpu = pytest.mark.skip(reason="No GPU available")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


# ============================================================================
# CORE FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def device():
    """Get appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def random_seed():
    """Set random seeds for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp(prefix="moe_test_"))
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def small_config() -> Dict[str, Any]:
    """Small configuration for fast testing."""
    return {
        "num_experts": 4,
        "expert_dim": 32,
        "hidden_dim": 64,
        "population_size": 10,
        "generations": 5,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
        "elitism_rate": 0.1,
        "sequence_length": 16,
        "batch_size": 4,
        "vocab_size": 100,
        "num_classes": 2,
    }


@pytest.fixture
def large_config() -> Dict[str, Any]:
    """Larger configuration for comprehensive testing."""
    return {
        "num_experts": 32,
        "expert_dim": 768,
        "hidden_dim": 512,
        "population_size": 100,
        "generations": 100,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
        "elitism_rate": 0.1,
        "sequence_length": 128,
        "batch_size": 16,
        "vocab_size": 10000,
        "num_classes": 10,
    }


@pytest.fixture
def evolution_config(temp_dir: Path) -> Dict[str, Any]:
    """Complete evolution configuration."""
    config = {
        "evolution": {
            "algorithm": "genetic_algorithm",
            "population_size": 20,
            "generations": 10,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "elitism_rate": 0.1,
            "selection": {
                "method": "tournament",
                "tournament_size": 3
            }
        },
        "objectives": [
            {"name": "accuracy", "weight": 1.0, "maximize": True},
            {"name": "latency", "weight": -0.2, "maximize": False},
            {"name": "sparsity", "weight": 0.1, "maximize": True}
        ],
        "hardware": {
            "max_memory": "4GB",
            "target_device": "cpu",
            "precision": "fp32"
        },
        "logging": {
            "level": "INFO",
            "save_dir": str(temp_dir / "logs")
        }
    }
    return config


# ============================================================================
# DATA FIXTURES
# ============================================================================

@pytest.fixture
def dummy_input_data(small_config: Dict[str, Any], device: torch.device):
    """Generate dummy input data for testing."""
    return {
        "input_ids": torch.randint(
            0, small_config["vocab_size"],
            (small_config["batch_size"], small_config["sequence_length"]),
            device=device
        ),
        "attention_mask": torch.ones(
            (small_config["batch_size"], small_config["sequence_length"]),
            device=device
        ),
        "labels": torch.randint(
            0, small_config["num_classes"],
            (small_config["batch_size"],),
            device=device
        )
    }


@pytest.fixture
def batch_data(small_config: Dict[str, Any], device: torch.device):
    """Generate a batch of training data."""
    batch_size = small_config["batch_size"]
    seq_len = small_config["sequence_length"]
    vocab_size = small_config["vocab_size"]
    
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones((batch_size, seq_len), device=device),
        "token_type_ids": torch.zeros((batch_size, seq_len), device=device, dtype=torch.long),
        "labels": torch.randint(0, 2, (batch_size,), device=device),
        "loss_mask": torch.ones((batch_size,), device=device),
    }


@pytest.fixture
def validation_data(small_config: Dict[str, Any], device: torch.device):
    """Generate validation dataset."""
    num_samples = 50
    return [
        {
            "input_ids": torch.randint(
                0, small_config["vocab_size"],
                (small_config["sequence_length"],),
                device=device
            ),
            "labels": torch.randint(0, small_config["num_classes"], (1,), device=device)
        }
        for _ in range(num_samples)
    ]


# ============================================================================
# MODEL FIXTURES
# ============================================================================

@pytest.fixture
def mock_expert():
    """Create mock expert for testing."""
    expert = Mock()
    expert.forward = Mock(return_value=torch.randn(4, 32))
    expert.parameters = Mock(return_value=[torch.randn(32, 32, requires_grad=True)])
    expert.eval = Mock()
    expert.train = Mock()
    return expert


@pytest.fixture
def mock_expert_pool(small_config: Dict[str, Any]):
    """Create mock expert pool."""
    pool = Mock()
    pool.num_experts = small_config["num_experts"]
    pool.expert_dim = small_config["expert_dim"]
    pool.experts = [Mock() for _ in range(small_config["num_experts"])]
    
    def mock_forward(x, expert_indices):
        batch_size = x.shape[0]
        return torch.randn(batch_size, small_config["expert_dim"])
    
    pool.forward = Mock(side_effect=mock_forward)
    return pool


@pytest.fixture
def sample_topology(small_config: Dict[str, Any], device: torch.device):
    """Create sample routing topology for testing."""
    num_experts = small_config["num_experts"]
    seq_len = small_config["sequence_length"]
    
    # Create sparse routing matrix (90% sparsity)
    routing_matrix = torch.zeros((seq_len, num_experts), device=device)
    # Each token connects to 1-2 experts
    for i in range(seq_len):
        num_connections = np.random.choice([1, 2], p=[0.7, 0.3])
        expert_indices = np.random.choice(num_experts, num_connections, replace=False)
        routing_matrix[i, expert_indices] = 1.0
    
    return {
        "routing_matrix": routing_matrix,
        "routing_params": {
            "temperature": 1.0,
            "top_k": 2,
            "load_balancing_weight": 0.01,
            "diversity_weight": 0.1
        },
        "sparsity": 1.0 - (routing_matrix.sum() / routing_matrix.numel()).item()
    }


# ============================================================================
# EVOLUTION FIXTURES
# ============================================================================

@pytest.fixture
def sample_population(small_config: Dict[str, Any], device: torch.device):
    """Create sample population for evolution testing."""
    population = []
    num_experts = small_config["num_experts"]
    seq_len = small_config["sequence_length"]
    
    for _ in range(small_config["population_size"]):
        # Random sparse topology
        routing_matrix = torch.zeros((seq_len, num_experts), device=device)
        for i in range(seq_len):
            num_connections = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            expert_indices = np.random.choice(num_experts, num_connections, replace=False)
            routing_matrix[i, expert_indices] = torch.rand(num_connections)
        
        topology = {
            "routing_matrix": routing_matrix,
            "routing_params": {
                "temperature": np.random.uniform(0.5, 2.0),
                "top_k": np.random.choice([1, 2, 3]),
                "load_balancing_weight": np.random.uniform(0.001, 0.1)
            }
        }
        population.append(topology)
    
    return population


@pytest.fixture
def fitness_scores(small_config: Dict[str, Any]):
    """Generate sample fitness scores."""
    return np.random.uniform(0.1, 0.9, small_config["population_size"])


# ============================================================================
# FILE AND CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def config_file(temp_dir: Path, evolution_config: Dict[str, Any]):
    """Create temporary configuration file."""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(evolution_config, f)
    return config_path


@pytest.fixture
def checkpoint_dir(temp_dir: Path):
    """Create checkpoint directory structure."""
    checkpoint_path = temp_dir / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True)
    (checkpoint_path / "models").mkdir(exist_ok=True)
    (checkpoint_path / "topologies").mkdir(exist_ok=True)
    (checkpoint_path / "experiments").mkdir(exist_ok=True)
    return checkpoint_path


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_wandb():
    """Mock Weights & Biases for testing."""
    wandb_mock = Mock()
    wandb_mock.init = Mock()
    wandb_mock.log = Mock()
    wandb_mock.finish = Mock()
    wandb_mock.config = {}
    return wandb_mock


@pytest.fixture
def mock_tensorboard():
    """Mock TensorBoard writer for testing."""
    writer_mock = Mock()
    writer_mock.add_scalar = Mock()
    writer_mock.add_histogram = Mock()
    writer_mock.add_image = Mock()
    writer_mock.close = Mock()
    return writer_mock


# ============================================================================
# PERFORMANCE FIXTURES
# ============================================================================

@pytest.fixture
def performance_metrics():
    """Sample performance metrics for testing."""
    return {
        "accuracy": 0.85,
        "latency": 12.5,  # milliseconds
        "memory": 2.3,    # GB
        "throughput": 1000,  # samples/sec
        "flops": 1e9,     # floating point operations
        "parameters": 1e6,  # model parameters
        "sparsity": 0.9,  # routing sparsity
        "load_balance": 0.8,  # expert load balance
        "diversity": 0.7,  # routing diversity
    }


# ============================================================================
# INTEGRATION TEST FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def integration_model(large_config: Dict[str, Any], device: torch.device):
    """Create model for integration testing."""
    # This would create an actual model instance
    # For now, return a mock that behaves like a real model
    model = Mock()
    model.eval = Mock()
    model.train = Mock()
    model.to = Mock(return_value=model)
    model.device = device
    
    def mock_forward(x):
        batch_size = x.shape[0] if hasattr(x, 'shape') else 1
        return torch.randn(batch_size, large_config["num_classes"], device=device)
    
    model.forward = Mock(side_effect=mock_forward)
    model.__call__ = model.forward
    return model


@pytest.fixture
def integration_dataset():
    """Create dataset for integration testing."""
    # Mock dataset that yields batches
    dataset = Mock()
    dataset.__len__ = Mock(return_value=100)
    
    def mock_getitem(idx):
        return {
            "input_ids": torch.randint(0, 1000, (128,)),
            "labels": torch.randint(0, 2, (1,))
        }
    
    dataset.__getitem__ = Mock(side_effect=mock_getitem)
    return dataset


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def capture_logs():
    """Fixture to capture log output during tests."""
    import logging
    import io
    
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger("self_evolving_moe")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    logger.removeHandler(handler)


@pytest.fixture
def measure_time():
    """Fixture to measure execution time."""
    import time
    times = {}
    
    def timer(name: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                times[name] = end - start
                return result
            return wrapper
        return decorator
    
    timer.times = times
    return timer