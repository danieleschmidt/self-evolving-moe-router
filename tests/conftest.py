"""Shared test configuration and fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed():
    """Set random seeds for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def small_config():
    """Small configuration for fast testing."""
    return {
        "num_experts": 4,
        "expert_dim": 32,
        "population_size": 10,
        "generations": 5,
        "mutation_rate": 0.1,
    }