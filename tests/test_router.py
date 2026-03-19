"""Tests for MoERouter."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from self_evolving_moe import MoERouter


def test_output_shapes():
    router = MoERouter(input_dim=16, num_experts=4, num_active=2)
    x = torch.randn(3, 5, 16)   # batch=3, seq=5, dim=16
    weights, indices = router(x)
    assert weights.shape == (15, 2), f"got {weights.shape}"
    assert indices.shape == (15, 2), f"got {indices.shape}"


def test_weights_sum_to_one():
    router = MoERouter(input_dim=16, num_experts=4, num_active=2)
    x = torch.randn(8, 16)
    weights, _ = router(x)
    assert torch.allclose(weights.sum(-1), torch.ones(8), atol=1e-5)


def test_mask_respected():
    """With mask=[True, True, False, False] only experts 0,1 should be selected."""
    mask = torch.tensor([True, True, False, False])
    router = MoERouter(input_dim=16, num_experts=4, num_active=2, mask=mask)
    x = torch.randn(20, 16)
    _, indices = router(x)
    assert (indices < 2).all(), f"Got expert idx outside {[0,1]}: {indices.unique()}"


def test_set_mask():
    router = MoERouter(input_dim=16, num_experts=4, num_active=2)
    new_mask = torch.tensor([False, True, True, False])
    router.set_mask(new_mask)
    x = torch.randn(20, 16)
    _, indices = router(x)
    assert set(indices.flatten().tolist()).issubset({1, 2})


def test_active_expert_count():
    mask = torch.tensor([True, False, True, False])
    router = MoERouter(input_dim=8, num_experts=4, num_active=2, mask=mask)
    assert router.active_expert_count() == 2


def test_mask_shape_error():
    router = MoERouter(input_dim=8, num_experts=4, num_active=2)
    with pytest.raises(AssertionError):
        router.set_mask(torch.ones(3, dtype=torch.bool))
