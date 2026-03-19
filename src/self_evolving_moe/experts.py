"""
SlimmableExpert: an MLP expert that runs at full, 75 %, or 50 % width.

Width is expressed as a ratio.  At each ratio the layer slices its weight
matrices so no extra parameters are ever allocated — it's purely a view.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


_WIDTHS = (1.0, 0.75, 0.5)   # supported ratios, widest first


class SlimmableExpert(nn.Module):
    """
    Feed-forward expert with slimmable hidden dimension.

    Full hidden width is `4 * input_dim` (standard MLP expert ratio).
    At width ratio `r` the hidden size is `int(4 * input_dim * r)`.

    Usage::

        expert = SlimmableExpert(input_dim=64)
        out_full = expert(x, width=1.0)     # full capacity
        out_slim = expert(x, width=0.5)     # half capacity
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.max_hidden = 4 * input_dim

        # Store full-width weights; slimmer widths reuse a prefix slice.
        self.w1 = nn.Linear(input_dim, self.max_hidden, bias=True)
        self.w2 = nn.Linear(self.max_hidden, input_dim, bias=True)

        nn.init.kaiming_uniform_(self.w1.weight)
        nn.init.kaiming_uniform_(self.w2.weight)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, width: float = 1.0) -> torch.Tensor:
        """
        Args:
            x:      (*, input_dim)
            width:  one of 1.0, 0.75, 0.5

        Returns:
            (*, input_dim)
        """
        assert width in _WIDTHS, f"width must be one of {_WIDTHS}, got {width}"
        h = int(self.max_hidden * width)

        # Slice weight matrices to the active hidden prefix.
        hidden = F.linear(x, self.w1.weight[:h], self.w1.bias[:h])
        hidden = F.gelu(hidden)
        out = F.linear(hidden, self.w2.weight[:, :h], self.w2.bias)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def supported_widths() -> Sequence[float]:
        return _WIDTHS
