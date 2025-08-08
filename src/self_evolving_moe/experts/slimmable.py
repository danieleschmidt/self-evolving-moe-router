"""
Slimmable MoE implementation for adaptive deployment.

This module implements width-adjustable MoE models that can dynamically
adapt their capacity based on available computational resources.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING, ForwardRef

if TYPE_CHECKING:
    from .pool import ExpertPool
import math
from pathlib import Path

# from .pool import ExpertPool  # Removed to avoid circular import
from ..routing.topology import TopologyGenome


class SlimmableLinear(nn.Module):
    """Linear layer that supports multiple widths."""
    
    def __init__(self, max_in_features: int, max_out_features: int, min_width_ratio: float = 0.25):
        super().__init__()
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.min_width_ratio = min_width_ratio
        
        # Full-width weight and bias
        self.weight = nn.Parameter(torch.randn(max_out_features, max_in_features))
        self.bias = nn.Parameter(torch.randn(max_out_features))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.size(1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Track current width
        self.current_in_width = max_in_features
        self.current_out_width = max_out_features
    
    def forward(self, x: torch.Tensor, in_width: Optional[int] = None, out_width: Optional[int] = None) -> torch.Tensor:
        """Forward pass with specified widths."""
        if in_width is None:
            in_width = self.current_in_width
        if out_width is None:
            out_width = self.current_out_width
            
        # Validate widths
        min_in_width = int(self.max_in_features * self.min_width_ratio)
        min_out_width = int(self.max_out_features * self.min_width_ratio)
        
        in_width = max(min_in_width, min(in_width, self.max_in_features))
        out_width = max(min_out_width, min(out_width, self.max_out_features))
        
        # Slice input if needed
        if x.shape[-1] > in_width:
            x = x[..., :in_width]
        
        # Use sliced weights and bias
        weight = self.weight[:out_width, :in_width]
        bias = self.bias[:out_width]
        
        return torch.nn.functional.linear(x, weight, bias)
    
    def set_width(self, in_width: int, out_width: int):
        """Set current width for the layer."""
        self.current_in_width = in_width
        self.current_out_width = out_width


class SlimmableMultiHeadAttention(nn.Module):
    """Multi-head attention that supports multiple widths."""
    
    def __init__(self, max_embed_dim: int, max_num_heads: int = 8, min_width_ratio: float = 0.25):
        super().__init__()
        self.max_embed_dim = max_embed_dim
        self.max_num_heads = max_num_heads
        self.min_width_ratio = min_width_ratio
        
        # Ensure embed_dim is divisible by num_heads at all widths
        self.head_dim = max_embed_dim // max_num_heads
        
        # Projection layers
        self.q_proj = SlimmableLinear(max_embed_dim, max_embed_dim, min_width_ratio)
        self.k_proj = SlimmableLinear(max_embed_dim, max_embed_dim, min_width_ratio)
        self.v_proj = SlimmableLinear(max_embed_dim, max_embed_dim, min_width_ratio)
        self.out_proj = SlimmableLinear(max_embed_dim, max_embed_dim, min_width_ratio)
        
        self.current_embed_dim = max_embed_dim
        self.current_num_heads = max_num_heads
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        width: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with specified width."""
        # Handle default arguments
        if key is None:
            key = query
        if value is None:
            value = query
        if width is None:
            width = self.current_embed_dim
            
        # Adjust number of heads based on width
        min_width = int(self.max_embed_dim * self.min_width_ratio)
        width = max(min_width, min(width, self.max_embed_dim))
        
        # Calculate number of heads for this width
        num_heads = max(1, (width * self.max_num_heads) // self.max_embed_dim)
        head_dim = width // num_heads
        
        batch_size, seq_len, _ = query.shape
        
        # Project to Q, K, V
        q = self.q_proj(query, width, width)
        k = self.k_proj(key, width, width)
        v = self.v_proj(value, width, width)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, width)
        output = self.out_proj(attn_output, width, width)
        
        return output, attn_weights
    
    def set_width(self, width: int):
        """Set current width for attention."""
        self.current_embed_dim = width
        self.current_num_heads = max(1, (width * self.max_num_heads) // self.max_embed_dim)


class SlimmableExpert(nn.Module):
    """Expert that can operate at different widths."""
    
    def __init__(
        self,
        max_dim: int,
        max_ffn_dim: Optional[int] = None,
        min_width_ratio: float = 0.25,
        expert_type: str = "transformer"
    ):
        super().__init__()
        self.max_dim = max_dim
        self.max_ffn_dim = max_ffn_dim or (4 * max_dim)
        self.min_width_ratio = min_width_ratio
        self.expert_type = expert_type
        
        if expert_type == "transformer":
            # Transformer-based expert
            self.attention = SlimmableMultiHeadAttention(max_dim, min_width_ratio=min_width_ratio)
            # Use width-flexible normalization  
            self.attn_norm = None
            
            self.ffn = nn.Sequential(
                SlimmableLinear(max_dim, self.max_ffn_dim, min_width_ratio),
                nn.GELU(),
                SlimmableLinear(self.max_ffn_dim, max_dim, min_width_ratio)
            )
            # Use width-flexible normalization
            self.ffn_norm = None
            
        elif expert_type == "mlp":
            # MLP-based expert
            self.network = nn.Sequential(
                SlimmableLinear(max_dim, self.max_ffn_dim, min_width_ratio),
                nn.GELU(),
                SlimmableLinear(self.max_ffn_dim, max_dim, min_width_ratio)
            )
            # Use RMSNorm-like normalization for width flexibility
            self.norm = None  # Will handle normalization manually
        
        self.current_width = max_dim
        self.usage_count = 0
        self.expert_id = None
        
        # Width-specific parameters for better adaptation
        possible_widths = self._get_possible_widths()
        self.width_embeddings = nn.ParameterDict({
            str(w): nn.Parameter(torch.randn(w) * 0.02)
            for w in possible_widths
        })
    
    def _get_possible_widths(self) -> List[int]:
        """Get list of possible widths."""
        min_width = int(self.max_dim * self.min_width_ratio)
        # Create widths in steps that are divisible by common factors
        step = max(1, (self.max_dim - min_width) // 8)
        widths = list(range(min_width, self.max_dim + 1, step))
        if self.max_dim not in widths:
            widths.append(self.max_dim)
        return sorted(widths)
    
    def forward(self, x: torch.Tensor, width: Optional[int] = None, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with specified width."""
        if width is None:
            width = self.current_width
            
        # Ensure width is valid
        min_width = int(self.max_dim * self.min_width_ratio)
        width = max(min_width, min(width, self.max_dim))
        
        self.usage_count += x.shape[0]
        
        # Slice input to target width if needed
        if x.shape[-1] > width:
            x_sliced = x[..., :width]
        else:
            x_sliced = x
            
        # Add width-specific embedding
        width_key = self._find_closest_width(width)
        if width_key in self.width_embeddings:
            width_emb = self.width_embeddings[width_key]
            if width_emb.shape[0] == x_sliced.shape[-1]:
                x_sliced = x_sliced + width_emb
        
        # Process through expert layers
        if self.expert_type == "transformer":
            # Self-attention
            attn_out, _ = self.attention(x_sliced, width, attention_mask)
            x_sliced = self._layer_norm(x_sliced + attn_out[..., :width], width)
            
            # Feed-forward
            ffn_out = self.ffn[0](x_sliced, width, self.max_ffn_dim)  # First linear
            ffn_out = self.ffn[1](ffn_out)  # GELU
            ffn_out = self.ffn[2](ffn_out, self.max_ffn_dim, width)  # Second linear
            x_sliced = self._layer_norm(x_sliced + ffn_out, width)
            
        elif self.expert_type == "mlp":
            net_out = self.network[0](x_sliced, width, self.max_ffn_dim)  # First linear
            net_out = self.network[1](net_out)  # GELU
            net_out = self.network[2](net_out, self.max_ffn_dim, width)  # Second linear
            x_sliced = self._layer_norm(x_sliced + net_out, width)
        
        # Pad back to original width if needed
        if x_sliced.shape[-1] < x.shape[-1]:
            padding = torch.zeros(
                *x.shape[:-1], x.shape[-1] - x_sliced.shape[-1],
                device=x.device, dtype=x.dtype
            )
            x_sliced = torch.cat([x_sliced, padding], dim=-1)
        
        return x_sliced
    
    def _layer_norm(self, x: torch.Tensor, width: int) -> torch.Tensor:
        """Apply layer normalization for given width."""
        # Simple RMS normalization for width flexibility
        x_sliced = x[..., :width]
        variance = x_sliced.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_sliced / torch.sqrt(variance + 1e-6)
        
        # Pad back if needed
        if x_norm.shape[-1] < x.shape[-1]:
            padding = torch.zeros(
                *x.shape[:-1], x.shape[-1] - x_norm.shape[-1],
                device=x.device, dtype=x.dtype
            )
            x_norm = torch.cat([x_norm, padding], dim=-1)
        
        return x_norm
    
    def _find_closest_width(self, target_width: int) -> str:
        """Find closest available width embedding."""
        available_widths = [int(w) for w in self.width_embeddings.keys()]
        closest_width = min(available_widths, key=lambda w: abs(w - target_width))
        return str(closest_width)
    
    def set_width(self, width: int):
        """Set current operating width."""
        self.current_width = width
    
    def get_parameter_count(self, width: Optional[int] = None) -> int:
        """Get effective parameter count at given width."""
        if width is None:
            width = self.current_width
        
        # This is an approximation - actual parameter sharing makes this complex
        ratio = width / self.max_dim
        total_params = sum(p.numel() for p in self.parameters())
        return int(total_params * ratio)


class SlimmableMoE(nn.Module):
    """
    Slimmable Mixture of Experts model that can adapt capacity dynamically.
    
    This model can operate at different expert counts and widths based on
    computational constraints and performance requirements.
    """
    
    def __init__(
        self,
        expert_pool: 'ExpertPool',
        routing_topology: Optional[TopologyGenome] = None,
        width_configs: Optional[List[int]] = None,
        default_width: Optional[int] = None
    ):
        super().__init__()
        self.expert_pool = expert_pool
        self.routing_topology = routing_topology
        
        # Width configurations
        if width_configs is None:
            # Default width configurations (25%, 50%, 75%, 100%)
            max_width = expert_pool.expert_dim
            self.width_configs = [
                max_width // 4,
                max_width // 2,
                3 * max_width // 4,
                max_width
            ]
        else:
            self.width_configs = sorted(width_configs)
        
        self.default_width = default_width or self.width_configs[-1]
        self.current_width = self.default_width
        
        # Routing layer
        self.router = nn.Linear(self.default_width, expert_pool.num_experts)
        
        # Performance tracking
        self.width_performance: Dict[int, float] = {}
        self.width_latency: Dict[int, float] = {}
        
        # Convert regular experts to slimmable if needed
        self._convert_experts_to_slimmable()
    
    def _convert_experts_to_slimmable(self):
        """Convert regular experts to slimmable versions."""
        # This is a simplified conversion - in practice, you'd need to transfer weights
        new_experts = nn.ModuleList()
        
        for i, expert in enumerate(self.expert_pool.experts):
            if isinstance(expert, SlimmableExpert):
                new_experts.append(expert)
            else:
                # Create slimmable version
                slimmable_expert = SlimmableExpert(
                    max_dim=self.expert_pool.expert_dim,
                    max_ffn_dim=self.expert_pool.ffn_dim,
                    expert_type="transformer" if hasattr(expert, 'attention') else "mlp"
                )
                slimmable_expert.expert_id = expert.expert_id
                new_experts.append(slimmable_expert)
        
        self.expert_pool.experts = new_experts
    
    def forward(
        self,
        x: torch.Tensor,
        width: Optional[int] = None,
        num_experts: Optional[int] = None,
        target_latency: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with adaptive capacity.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            width: Target width (if None, uses current_width)
            num_experts: Number of experts to use (if None, uses all active)
            target_latency: Target latency in ms (triggers automatic adaptation)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        if target_latency is not None:
            width, num_experts = self._adapt_to_latency(target_latency)
        
        if width is None:
            width = self.current_width
        if num_experts is None:
            num_experts = len(self.expert_pool.active_experts)
        
        # Ensure width is in available configurations
        width = min(self.width_configs, key=lambda w: abs(w - width))
        
        batch_size, seq_len, hidden_dim = x.shape
        
        # Slice input to target width if needed
        if hidden_dim > width:
            x_sliced = x[..., :width]
        else:
            x_sliced = x
        
        # Compute routing weights
        if self.routing_topology is not None:
            # Use evolved topology
            routing_weights, selected_experts = self.routing_topology.get_routing_weights(x_sliced)
        else:
            # Use learned routing
            router_input = x_sliced.mean(dim=1)  # Average over sequence
            routing_logits = self.router(router_input)
            
            # Select top experts
            top_k = min(num_experts, self.expert_pool.num_experts)
            routing_weights, selected_experts = routing_logits.softmax(dim=-1).topk(top_k, dim=-1)
            
            # Expand for sequence dimension
            routing_weights = routing_weights.unsqueeze(1).expand(-1, seq_len, -1)
            selected_experts = selected_experts.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Route through experts
        output = torch.zeros_like(x_sliced)
        
        for expert_idx in range(self.expert_pool.num_experts):
            if expert_idx not in self.expert_pool.active_experts:
                continue
            
            # Check if this expert is selected
            expert_mask = (selected_experts == expert_idx)
            if not expert_mask.any():
                continue
            
            # Get weights for this expert
            weights = routing_weights * expert_mask.float()
            weights = weights.sum(dim=-1, keepdim=True)
            
            if weights.sum() > 0:
                # Set expert width
                expert = self.expert_pool.experts[expert_idx]
                if isinstance(expert, SlimmableExpert):
                    expert.set_width(width)
                
                # Forward through expert
                expert_output = expert(x_sliced, width, attention_mask)
                
                # Apply weights and accumulate
                weighted_output = expert_output * weights
                output += weighted_output
        
        # Pad output back to original width if needed
        if output.shape[-1] < x.shape[-1]:
            padding = torch.zeros(
                batch_size, seq_len, x.shape[-1] - output.shape[-1],
                device=x.device, dtype=x.dtype
            )
            output = torch.cat([output, padding], dim=-1)
        
        return output
    
    def _adapt_to_latency(self, target_latency: float) -> Tuple[int, int]:
        """
        Automatically adapt width and expert count to meet latency target.
        
        Args:
            target_latency: Target latency in milliseconds
            
        Returns:
            Tuple of (optimal_width, optimal_num_experts)
        """
        # Simple heuristic - in practice, you'd use learned latency models
        if target_latency < 5:  # Very fast
            return self.width_configs[0], 2
        elif target_latency < 10:  # Fast
            return self.width_configs[1], 4
        elif target_latency < 20:  # Medium
            return self.width_configs[2], 8
        else:  # Full capacity
            return self.width_configs[-1], len(self.expert_pool.active_experts)
    
    def set_routing_topology(self, topology: TopologyGenome):
        """Set routing topology for expert selection."""
        self.routing_topology = topology
    
    def get_current_topology(self) -> Optional[TopologyGenome]:
        """Get current routing topology."""
        return self.routing_topology
    
    def benchmark_width(self, x: torch.Tensor, width: int, num_trials: int = 10) -> Dict[str, float]:
        """
        Benchmark performance at specific width.
        
        Args:
            x: Sample input tensor
            width: Width to benchmark
            num_trials: Number of trials for latency measurement
            
        Returns:
            Dictionary with performance metrics
        """
        import time
        
        self.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = self.forward(x, width=width)
            
            # Measure latency
            torch.cuda.synchronize() if x.is_cuda else None
            start_time = time.time()
            
            for _ in range(num_trials):
                output = self.forward(x, width=width)
                
            torch.cuda.synchronize() if x.is_cuda else None
            end_time = time.time()
            
            avg_latency = (end_time - start_time) / num_trials * 1000  # ms
            
            # Estimate memory usage
            param_count = sum(
                expert.get_parameter_count(width) 
                for expert in self.expert_pool.experts 
                if isinstance(expert, SlimmableExpert)
            )
            
            # Store results
            self.width_latency[width] = avg_latency
            
            return {
                'width': width,
                'latency_ms': avg_latency,
                'parameter_count': param_count,
                'throughput_samples_per_sec': (x.shape[0] * 1000) / avg_latency
            }
    
    def auto_select_width(self, target_latency: float, sample_input: torch.Tensor) -> int:
        """
        Automatically select optimal width for target latency.
        
        Args:
            target_latency: Target latency in milliseconds
            sample_input: Sample input for benchmarking
            
        Returns:
            Optimal width
        """
        # Benchmark all width configurations if not done already
        for width in self.width_configs:
            if width not in self.width_latency:
                self.benchmark_width(sample_input, width)
        
        # Find width that meets latency target
        suitable_widths = [
            (width, latency) for width, latency in self.width_latency.items()
            if latency <= target_latency
        ]
        
        if suitable_widths:
            # Choose the largest width that meets the target
            optimal_width = max(suitable_widths, key=lambda x: x[0])[0]
        else:
            # If no width meets target, choose the fastest
            optimal_width = min(self.width_latency.items(), key=lambda x: x[1])[0]
        
        return optimal_width
    
    def get_efficiency_report(self) -> Dict[str, Any]:
        """Get comprehensive efficiency report."""
        return {
            'width_configs': self.width_configs,
            'current_width': self.current_width,
            'width_latency': self.width_latency,
            'width_performance': self.width_performance,
            'active_experts': self.expert_pool.active_experts,
            'total_experts': self.expert_pool.num_experts,
            'expert_utilization': self.expert_pool.get_expert_usage(),
            'memory_usage': self.expert_pool.get_memory_usage()
        }
    
    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cpu") -> 'SlimmableMoE':
        """Load pre-trained slimmable MoE model."""
        checkpoint = torch.load(model_path, map_location=device)
        
        # Reconstruct expert pool
        from .pool import ExpertPool  # Import here to avoid circular import
        expert_pool = ExpertPool.load_experts(
            checkpoint['expert_pool_path'], device=device
        )
        
        # Reconstruct topology if available
        topology = None
        if 'topology_state' in checkpoint:
            topology = TopologyGenome.load_topology(
                checkpoint['topology_path'], device=device
            )
        
        # Create model
        model = cls(
            expert_pool=expert_pool,
            routing_topology=topology,
            width_configs=checkpoint['width_configs'],
            default_width=checkpoint['default_width']
        )
        
        # Load router state
        model.router.load_state_dict(checkpoint['router_state'])
        model.width_performance = checkpoint.get('width_performance', {})
        model.width_latency = checkpoint.get('width_latency', {})
        
        return model
    
    def save_pretrained(self, save_path: str):
        """Save slimmable MoE model."""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save expert pool
        expert_pool_path = save_dir / "expert_pool"
        self.expert_pool.save_experts(str(expert_pool_path))
        
        # Save topology if available
        topology_path = None
        if self.routing_topology is not None:
            topology_path = save_dir / "topology.pt"
            self.routing_topology.save_topology(str(topology_path))
        
        # Save model state
        model_path = save_dir / "model.pt"
        torch.save({
            'router_state': self.router.state_dict(),
            'width_configs': self.width_configs,
            'default_width': self.default_width,
            'width_performance': self.width_performance,
            'width_latency': self.width_latency,
            'expert_pool_path': str(expert_pool_path),
            'topology_path': str(topology_path) if topology_path else None
        }, model_path)