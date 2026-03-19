"""
MoERouter: Sparse Mixture-of-Experts router with a binary connectivity mask.

Each token can only route to experts permitted by the mask.
The mask (num_experts,) is a shared binary gate over the router's logits —
tokens still compete via top-k softmax, but only among allowed experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MoERouter(nn.Module):
    """
    Sparse MoE router whose expert connectivity is controlled by a binary mask.

    Args:
        input_dim:    Dimensionality of input tokens.
        num_experts:  Total number of experts available.
        num_active:   Number of experts each token routes to (top-k).
        mask:         Optional initial binary mask (num_experts,). If None,
                      all experts are enabled.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        num_active: int = 2,
        mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        assert 1 <= num_active <= num_experts
        self.num_experts = num_experts
        self.num_active = num_active

        self.gate = nn.Linear(input_dim, num_experts, bias=False)

        # Binary mask over experts — not a parameter (it's evolved, not learned).
        if mask is None:
            mask = torch.ones(num_experts, dtype=torch.bool)
        self.register_buffer("mask", mask.bool())

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-k experts respecting the connectivity mask.

        Args:
            x: (batch, seq_len, input_dim)  or  (N, input_dim)

        Returns:
            weights:          (N, num_active)  — softmax weights for selected experts
            expert_indices:   (N, num_active)  — which experts are active
        """
        flat = x.reshape(-1, x.shape[-1])          # (N, D)
        logits = self.gate(flat)                    # (N, E)

        # Mask out disabled experts with -inf so they never get selected.
        if not self.mask.all():
            logits = logits.masked_fill(~self.mask, float("-inf"))

        # top-k selection
        weights, indices = logits.topk(self.num_active, dim=-1)   # (N, K)
        weights = F.softmax(weights, dim=-1)

        return weights, indices

    # ------------------------------------------------------------------
    def set_mask(self, mask: torch.Tensor) -> None:
        """Replace connectivity mask (called by the evolutionary search)."""
        assert mask.shape == (self.num_experts,), (
            f"Mask shape {mask.shape} != ({self.num_experts},)"
        )
        self.mask.copy_(mask.bool())

    # ------------------------------------------------------------------
    def active_expert_count(self) -> int:
        return int(self.mask.sum().item())
