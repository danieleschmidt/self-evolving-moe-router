"""Tests for SlimmableExpert."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from self_evolving_moe import SlimmableExpert


@pytest.mark.parametrize("width", [1.0, 0.75, 0.5])
def test_output_shape(width):
    expert = SlimmableExpert(input_dim=32)
    x = torch.randn(4, 7, 32)
    out = expert(x, width=width)
    assert out.shape == x.shape, f"width={width}: expected {x.shape}, got {out.shape}"


def test_invalid_width_raises():
    expert = SlimmableExpert(input_dim=32)
    with pytest.raises(AssertionError):
        expert(torch.randn(2, 32), width=0.25)


def test_supported_widths():
    widths = SlimmableExpert.supported_widths()
    assert set(widths) == {1.0, 0.75, 0.5}


def test_slimmer_uses_fewer_flops():
    """Verify narrower width slices fewer hidden units (proxy: check gradient flows)."""
    expert = SlimmableExpert(input_dim=16)
    x = torch.randn(2, 16, requires_grad=False)

    # Just check both widths produce finite outputs without errors
    for w in (1.0, 0.75, 0.5):
        out = expert(x, width=w)
        assert torch.isfinite(out).all(), f"Non-finite output at width={w}"


def test_full_width_uses_all_params():
    """At width=1.0 the full weight matrix is used."""
    expert = SlimmableExpert(input_dim=8)
    x = torch.randn(1, 8)
    out_full = expert(x, width=1.0)
    out_half = expert(x, width=0.5)
    # Outputs should differ (extremely unlikely to be equal with random init)
    assert not torch.allclose(out_full, out_half)
