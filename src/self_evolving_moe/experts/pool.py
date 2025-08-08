"""
Expert pool management and routing system for MoE models.

This module implements the ExpertPool class that manages a collection of
expert networks and handles routing decisions based on evolved topologies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from dataclasses import dataclass
import math
from pathlib import Path

from .slimmable import SlimmableLinear, SlimmableMultiHeadAttention
from ..routing.topology import TopologyGenome
from ..utils.exceptions import ExpertPoolError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExpertConfig:
    """Configuration for individual experts."""
    hidden_dim: int = 768
    intermediate_dim: int = 3072
    num_attention_heads: int = 12
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    expert_type: str = "transformer"  # transformer, mlp, conv


class Expert(nn.Module):
    """Base expert module."""
    
    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        self.expert_id = None
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    def get_num_parameters(self) -> int:
        """Get number of parameters in this expert."""
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self) -> float:
        """Get approximate memory usage in MB."""
        return self.get_num_parameters() * 4 / (1024 ** 2)  # Assuming float32


class TransformerExpert(Expert):
    """Transformer-based expert with attention and feed-forward layers."""
    
    def __init__(self, config: ExpertConfig):
        super().__init__(config)
        
        # Multi-head attention
        self.attention = SlimmableMultiHeadAttention(
            max_embed_dim=config.hidden_dim,
            max_num_heads=config.num_attention_heads
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            SlimmableLinear(config.hidden_dim, config.intermediate_dim),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout),
            SlimmableLinear(config.intermediate_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        width: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            width: Target width for slimmable computation
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        attn_out, attn_weights = self.attention(x, x, x, attention_mask=attention_mask, width=width)
        x = residual + attn_out
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        
        return x


class MLPExpert(Expert):
    """Simple MLP-based expert."""
    
    def __init__(self, config: ExpertConfig):
        super().__init__(config)
        
        self.layers = nn.Sequential(
            SlimmableLinear(config.hidden_dim, config.intermediate_dim),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout),
            SlimmableLinear(config.intermediate_dim, config.intermediate_dim // 2),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout),
            SlimmableLinear(config.intermediate_dim // 2, config.hidden_dim)
        )
        
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, width: Optional[int] = None) -> torch.Tensor:
        """Forward pass through MLP expert."""
        return self.layers(x)


class RoutingLayer(nn.Module):
    """Routing layer that decides which experts to use."""
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        temperature: float = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        
        # Learned routing function
        self.router = nn.Linear(input_dim, num_experts)
        self.noise = nn.Parameter(torch.randn(num_experts) * 0.1)
        
        # Load balancing
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        
    def forward(
        self,
        x: torch.Tensor,
        topology: Optional[TopologyGenome] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route tokens to experts.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            topology: Optional topology constraints
            
        Returns:
            routing_weights: Expert weights [batch_size, seq_len, top_k]
            selected_experts: Expert indices [batch_size, seq_len, top_k] 
            aux_loss_info: Auxiliary loss information
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute routing logits
        routing_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        
        # Add noise during training
        if self.training:
            noise = torch.randn_like(routing_logits) * self.noise.unsqueeze(0).unsqueeze(0)
            routing_logits = routing_logits + noise
        
        # Apply temperature
        routing_logits = routing_logits / self.temperature
        
        # Apply topology constraints if provided
        if topology is not None:
            # Create mask from topology
            topology_mask = topology.get_routing_mask(seq_len).to(x.device)
            routing_logits = routing_logits * topology_mask + (1 - topology_mask) * (-1e9)
        
        # Top-k selection
        routing_weights, selected_experts = torch.topk(
            routing_logits, self.top_k, dim=-1
        )
        
        # Normalize weights
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Update expert usage statistics
        if self.training:
            self._update_usage_stats(selected_experts, batch_size * seq_len)
        
        # Compute auxiliary losses
        aux_loss_info = self._compute_aux_losses(routing_logits, selected_experts, routing_weights)
        
        return routing_weights, selected_experts, aux_loss_info
    
    def _update_usage_stats(self, selected_experts: torch.Tensor, num_tokens: int):
        """Update expert usage statistics for load balancing."""
        # Count expert usage
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        expert_usage = expert_mask.sum(dim=[0, 1, 2])  # Sum over batch, seq, top_k
        
        self.expert_counts += expert_usage
        self.total_tokens += num_tokens
    
    def _compute_aux_losses(
        self,
        routing_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for training."""
        aux_losses = {}
        
        # Load balancing loss
        if self.total_tokens > 0:
            # Target usage: uniform distribution
            target_usage = 1.0 / self.num_experts
            current_usage = self.expert_counts / self.total_tokens.clamp(min=1.0)
            
            # L2 loss between current and target usage
            load_balance_loss = F.mse_loss(current_usage, torch.full_like(current_usage, target_usage))
            aux_losses['load_balance'] = load_balance_loss
        
        # Diversity loss - encourage different routing patterns
        routing_probs = F.softmax(routing_logits, dim=-1)
        avg_routing = routing_probs.mean(dim=[0, 1])  # Average over batch and sequence
        
        # Entropy of average routing (higher is better)
        entropy = -(avg_routing * torch.log(avg_routing + 1e-8)).sum()
        max_entropy = math.log(self.num_experts)
        diversity_loss = (max_entropy - entropy) / max_entropy
        aux_losses['diversity'] = diversity_loss
        
        return aux_losses
    
    def get_usage_stats(self) -> Dict[str, torch.Tensor]:
        """Get expert usage statistics."""
        if self.total_tokens > 0:
            usage_rates = self.expert_counts / self.total_tokens
        else:
            usage_rates = torch.zeros_like(self.expert_counts)
        
        return {
            'expert_counts': self.expert_counts,
            'total_tokens': self.total_tokens,
            'usage_rates': usage_rates,
            'load_balance_score': 1.0 - usage_rates.std()
        }
    
    def reset_usage_stats(self):
        """Reset expert usage statistics."""
        self.expert_counts.zero_()
        self.total_tokens.zero_()


class ExpertPool(nn.Module):
    """
    Pool of expert networks with routing capabilities.
    
    This class manages multiple expert networks and handles routing
    decisions based on evolved topologies or learned routing functions.
    """
    
    def __init__(
        self,
        num_experts: int,
        expert_config: ExpertConfig,
        top_k: int = 2,
        routing_temperature: float = 1.0,
        load_balancing_weight: float = 0.01,
        diversity_weight: float = 0.1
    ):
        """
        Initialize expert pool.
        
        Args:
            num_experts: Number of experts in the pool
            expert_config: Configuration for each expert
            top_k: Number of top experts to route to
            routing_temperature: Temperature for routing decisions
            load_balancing_weight: Weight for load balancing loss
            diversity_weight: Weight for diversity loss
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.expert_config = expert_config
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight
        self.diversity_weight = diversity_weight
        
        # Create experts
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            expert = self._create_expert(expert_config)
            expert.expert_id = i
            self.experts.append(expert)
        
        # Routing layer
        self.routing_layer = RoutingLayer(
            input_dim=expert_config.hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            temperature=routing_temperature
        )
        
        # Current topology (if using evolved routing)
        self.current_topology: Optional[TopologyGenome] = None
        
        logger.info(f"Initialized ExpertPool with {num_experts} {expert_config.expert_type} experts")
    
    def _create_expert(self, config: ExpertConfig) -> Expert:
        """Create expert based on configuration."""
        if config.expert_type == "transformer":
            return TransformerExpert(config)
        elif config.expert_type == "mlp":
            return MLPExpert(config)
        else:
            raise ValueError(f"Unknown expert type: {config.expert_type}")
    
    def set_routing_topology(self, topology: Optional[TopologyGenome]):
        """Set routing topology for evolved routing."""
        self.current_topology = topology
        if topology:
            logger.info(f"Applied routing topology with sparsity {topology.sparsity:.3f}")
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expert_capacity_factor: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through expert pool with routing.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            expert_capacity_factor: Capacity factor for load balancing
            **kwargs: Additional arguments passed to experts
            
        Returns:
            output: Combined expert outputs [batch_size, seq_len, hidden_dim]
            aux_losses: Auxiliary loss information
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Get routing decisions
        routing_weights, selected_experts, aux_loss_info = self.routing_layer(
            x, topology=self.current_topology
        )
        
        # Prepare output tensor
        output = torch.zeros_like(x)
        
        # Process tokens through selected experts
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx)  # [batch, seq, top_k]
            
            if not expert_mask.any():
                continue  # Skip unused experts
            
            # Get positions where this expert is used
            token_positions = expert_mask.nonzero()  # [num_tokens, 3] (batch, seq, expert_rank)
            
            if len(token_positions) == 0:
                continue
            
            # Extract tokens for this expert
            batch_indices = token_positions[:, 0]
            seq_indices = token_positions[:, 1]
            expert_rank_indices = token_positions[:, 2]
            
            expert_input = x[batch_indices, seq_indices]  # [num_expert_tokens, hidden_dim]
            
            # Get routing weights for these tokens
            weights = routing_weights[batch_indices, seq_indices, expert_rank_indices].unsqueeze(-1)
            
            # Process through expert
            if expert_input.size(0) > 0:
                # Reshape for expert processing if needed
                if self.expert_config.expert_type == "transformer" and len(expert_input.shape) == 2:
                    expert_input = expert_input.unsqueeze(1)  # Add seq dimension
                    squeeze_output = True
                else:
                    squeeze_output = False
                
                # Forward through expert
                expert_output = self.experts[expert_idx](
                    expert_input,
                    attention_mask=attention_mask[batch_indices, seq_indices].unsqueeze(1) if attention_mask is not None and squeeze_output else None,
                    **kwargs
                )
                
                # Remove seq dimension if added
                if squeeze_output:
                    expert_output = expert_output.squeeze(1)
                
                # Apply routing weights
                weighted_output = expert_output * weights
                
                # Add to output tensor
                output[batch_indices, seq_indices] += weighted_output
        
        # Combine auxiliary losses
        aux_losses = {
            'load_balance_loss': aux_loss_info.get('load_balance', torch.tensor(0.0, device=x.device)),
            'diversity_loss': aux_loss_info.get('diversity', torch.tensor(0.0, device=x.device))
        }
        
        # Weight auxiliary losses
        total_aux_loss = (
            self.load_balancing_weight * aux_losses['load_balance_loss'] +
            self.diversity_weight * aux_losses['diversity_loss']
        )
        aux_losses['total_aux_loss'] = total_aux_loss
        
        return output, aux_losses
    
    def get_expert_utilization(self) -> Dict[str, Any]:
        """Get expert utilization statistics."""
        usage_stats = self.routing_layer.get_usage_stats()
        
        utilization = {
            'expert_usage_rates': usage_stats['usage_rates'].cpu().numpy(),
            'load_balance_score': usage_stats['load_balance_score'].item(),
            'total_tokens_processed': usage_stats['total_tokens'].item(),
            'active_experts': (usage_stats['usage_rates'] > 0.01).sum().item(),
            'most_used_expert': usage_stats['usage_rates'].argmax().item(),
            'least_used_expert': usage_stats['usage_rates'].argmin().item()
        }
        
        return utilization
    
    def get_expert_info(self) -> List[Dict[str, Any]]:
        """Get information about each expert."""
        info = []
        for i, expert in enumerate(self.experts):
            info.append({
                'expert_id': i,
                'expert_type': self.expert_config.expert_type,
                'num_parameters': expert.get_num_parameters(),
                'memory_usage_mb': expert.get_memory_usage(),
                'hidden_dim': self.expert_config.hidden_dim,
                'intermediate_dim': self.expert_config.intermediate_dim
            })
        return info
    
    def reset_expert_stats(self):
        """Reset expert usage statistics."""
        self.routing_layer.reset_usage_stats()
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters in the pool."""
        return sum(expert.get_num_parameters() for expert in self.experts)
    
    def get_active_parameters(self) -> int:
        """Get number of active parameters based on current routing."""
        if self.current_topology is None:
            # Estimate based on top_k
            return int(self.get_total_parameters() * (self.top_k / self.num_experts))
        else:
            # Calculate based on topology sparsity
            active_ratio = 1.0 - self.current_topology.sparsity
            return int(self.get_total_parameters() * active_ratio)
    
    def save_experts(self, save_dir: Union[str, Path]):
        """Save individual expert models."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, expert in enumerate(self.experts):
            expert_path = save_dir / f"expert_{i}.pt"
            torch.save({
                'state_dict': expert.state_dict(),
                'config': self.expert_config.__dict__,
                'expert_id': i
            }, expert_path)
        
        # Save routing layer
        routing_path = save_dir / "routing_layer.pt"
        torch.save({
            'state_dict': self.routing_layer.state_dict(),
            'num_experts': self.num_experts,
            'top_k': self.top_k
        }, routing_path)
        
        logger.info(f"Saved {len(self.experts)} experts to {save_dir}")
    
    def load_experts(self, save_dir: Union[str, Path]):
        """Load individual expert models."""
        save_dir = Path(save_dir)
        
        # Load experts
        for i in range(len(self.experts)):
            expert_path = save_dir / f"expert_{i}.pt"
            if expert_path.exists():
                checkpoint = torch.load(expert_path, map_location='cpu')
                self.experts[i].load_state_dict(checkpoint['state_dict'])
        
        # Load routing layer
        routing_path = save_dir / "routing_layer.pt"
        if routing_path.exists():
            checkpoint = torch.load(routing_path, map_location='cpu')
            self.routing_layer.load_state_dict(checkpoint['state_dict'])
        
        logger.info(f"Loaded experts from {save_dir}")
