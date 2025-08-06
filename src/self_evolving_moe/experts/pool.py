"""
Expert pool management for MoE models.

This module implements the ExpertPool class that manages a collection
of expert networks with different specializations and capabilities.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union, Tuple
import random
import numpy as np
from pathlib import Path


class TransformerExpert(nn.Module):
    """Individual transformer-based expert network."""
    
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # Track expert utilization
        self.usage_count = 0
        self.expert_id = None
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        self.usage_count += x.shape[0]  # Track batch size usage
        
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.attn_norm(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)
        
        return x
    
    def reset_usage(self):
        """Reset usage counter."""
        self.usage_count = 0
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class MLPExpert(nn.Module):
    """Simple MLP-based expert network."""
    
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        
        layers = []
        for i in range(num_layers):
            input_dim = hidden_dim if i == 0 else ffn_dim
            output_dim = hidden_dim if i == num_layers - 1 else ffn_dim
            
            layers.append(nn.Linear(input_dim, output_dim))
            
            if i < num_layers - 1:  # No activation on last layer
                if activation == "gelu":
                    layers.append(nn.GELU())
                else:
                    layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Track expert utilization
        self.usage_count = 0
        self.expert_id = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP expert."""
        self.usage_count += x.shape[0]
        
        # Apply network with residual connection
        output = self.network(x)
        return self.norm(x + output)
    
    def reset_usage(self):
        """Reset usage counter."""
        self.usage_count = 0
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class ExpertPool:
    """
    Manages a pool of expert networks for MoE models.
    
    The pool handles expert creation, specialization, and dynamic loading/unloading
    of experts based on resource constraints and usage patterns.
    """
    
    def __init__(
        self,
        num_experts: int,
        expert_dim: int,
        expert_type: str = "transformer_block",
        ffn_dim: Optional[int] = None,
        device: str = "cpu",
        expert_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize expert pool.
        
        Args:
            num_experts: Number of experts in the pool
            expert_dim: Hidden dimension for experts
            expert_type: Type of expert ("transformer_block" or "mlp")
            ffn_dim: Feed-forward dimension (defaults to 4 * expert_dim)
            device: Device to place experts on
            expert_config: Additional configuration for experts
        """
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.expert_type = expert_type
        self.ffn_dim = ffn_dim or (4 * expert_dim)
        self.device = device
        self.expert_config = expert_config or {}
        
        # Expert management
        self.expert_specializations: Dict[int, str] = {}
        self.expert_performance: Dict[int, float] = {}
        self.active_experts: List[int] = list(range(num_experts))
        
        # Create experts
        self.experts = nn.ModuleList()
        self._create_experts()
        
        # Resource management
        self.max_active_experts = num_experts
        self.memory_budget = None
        
    def _create_experts(self):
        """Create expert networks based on configuration."""
        for i in range(self.num_experts):
            if self.expert_type == "transformer_block":
                expert = TransformerExpert(
                    hidden_dim=self.expert_dim,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.expert_config.get("num_heads", 8),
                    dropout=self.expert_config.get("dropout", 0.1),
                    activation=self.expert_config.get("activation", "gelu")
                )
            elif self.expert_type == "mlp":
                expert = MLPExpert(
                    hidden_dim=self.expert_dim,
                    ffn_dim=self.ffn_dim,
                    num_layers=self.expert_config.get("num_layers", 2),
                    dropout=self.expert_config.get("dropout", 0.1),
                    activation=self.expert_config.get("activation", "gelu")
                )
            else:
                raise ValueError(f"Unknown expert type: {self.expert_type}")
            
            expert.expert_id = i
            expert.to(self.device)
            self.experts.append(expert)
            
            # Initialize performance tracking
            self.expert_performance[i] = 0.0
    
    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Route input through selected experts.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            expert_indices: Selected expert indices [batch_size, seq_len, top_k]
            expert_weights: Expert weights [batch_size, seq_len, top_k]
            attention_mask: Optional attention mask
            
        Returns:
            Combined expert outputs [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        _, _, top_k = expert_indices.shape
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            if expert_idx not in self.active_experts:
                continue
                
            # Find positions where this expert is selected
            expert_mask = (expert_indices == expert_idx)
            if not expert_mask.any():
                continue
            
            # Get expert weights for this expert
            weights = expert_weights * expert_mask.float()
            weights = weights.sum(dim=-1, keepdim=True)  # Sum across top_k dimension
            
            # Only process if expert has non-zero weight
            if weights.sum() > 0:
                # Forward through expert
                if self.expert_type == "transformer_block":
                    expert_output = self.experts[expert_idx](x, attention_mask)
                else:
                    expert_output = self.experts[expert_idx](x)
                
                # Apply weights and accumulate
                weighted_output = expert_output * weights
                output += weighted_output
        
        return output
    
    def get_expert(self, expert_id: int) -> nn.Module:
        """Get expert by ID."""
        if 0 <= expert_id < self.num_experts:
            return self.experts[expert_id]
        raise ValueError(f"Expert ID {expert_id} out of range")
    
    def get_active_experts(self) -> List[int]:
        """Get list of currently active expert IDs."""
        return self.active_experts.copy()
    
    def set_active_experts(self, expert_ids: List[int]):
        """Set which experts are currently active."""
        # Validate expert IDs
        for expert_id in expert_ids:
            if not 0 <= expert_id < self.num_experts:
                raise ValueError(f"Expert ID {expert_id} out of range")
        
        self.active_experts = expert_ids
    
    def get_expert_usage(self) -> Dict[int, int]:
        """Get usage statistics for all experts."""
        usage = {}
        for i, expert in enumerate(self.experts):
            usage[i] = expert.usage_count
        return usage
    
    def reset_expert_usage(self):
        """Reset usage counters for all experts."""
        for expert in self.experts:
            expert.reset_usage()
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters across all experts."""
        return sum(expert.get_parameter_count() for expert in self.experts)
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage of the expert pool."""
        # Rough estimate based on parameter count
        total_params = self.get_total_parameters()
        
        # Assume 4 bytes per parameter (float32)
        param_memory = total_params * 4
        
        # Add overhead for activations and gradients (rough estimate)
        activation_memory = param_memory * 0.5
        gradient_memory = param_memory if any(p.requires_grad for p in self.parameters()) else 0
        
        return {
            'parameters': param_memory,
            'activations': activation_memory,
            'gradients': gradient_memory,
            'total': param_memory + activation_memory + gradient_memory
        }
    
    def prune_experts(self, keep_ratio: float = 0.8):
        """
        Prune least used experts to reduce memory usage.
        
        Args:
            keep_ratio: Fraction of experts to keep (0.0 to 1.0)
        """
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be between 0.0 and 1.0")
        
        # Get usage statistics
        usage = self.get_expert_usage()
        
        # Sort experts by usage (descending)
        sorted_experts = sorted(usage.items(), key=lambda x: x[1], reverse=True)
        
        # Determine how many experts to keep
        num_to_keep = max(1, int(self.num_experts * keep_ratio))
        
        # Update active experts to most used ones
        self.active_experts = [expert_id for expert_id, _ in sorted_experts[:num_to_keep]]
        
        print(f"Pruned expert pool: keeping {num_to_keep}/{self.num_experts} experts")
        print(f"Active experts: {self.active_experts}")
    
    def specialize_expert(self, expert_id: int, specialization: str, performance: float):
        """
        Mark an expert as specialized for a particular task/domain.
        
        Args:
            expert_id: ID of the expert to specialize
            specialization: Description of the specialization
            performance: Performance score for the specialization
        """
        if 0 <= expert_id < self.num_experts:
            self.expert_specializations[expert_id] = specialization
            self.expert_performance[expert_id] = performance
        else:
            raise ValueError(f"Expert ID {expert_id} out of range")
    
    def get_specialized_experts(self, specialization: str) -> List[int]:
        """Get experts specialized for a particular task."""
        return [
            expert_id for expert_id, spec in self.expert_specializations.items()
            if spec == specialization
        ]
    
    def adaptive_expert_selection(self, target_memory: int) -> List[int]:
        """
        Adaptively select experts based on memory constraints.
        
        Args:
            target_memory: Target memory usage in bytes
            
        Returns:
            List of expert IDs that fit within memory budget
        """
        # Estimate memory per expert
        memory_per_expert = self.get_memory_usage()['total'] // self.num_experts
        
        # Calculate how many experts we can afford
        max_experts = min(self.num_experts, target_memory // memory_per_expert)
        
        if max_experts == 0:
            raise ValueError("Target memory too low to fit any experts")
        
        # Select top performing experts
        expert_scores = [(i, self.expert_performance.get(i, 0.0)) for i in range(self.num_experts)]
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_experts = [expert_id for expert_id, _ in expert_scores[:max_experts]]
        
        return selected_experts
    
    def save_experts(self, save_dir: str):
        """Save all experts to directory."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save each expert
        for i, expert in enumerate(self.experts):
            expert_path = save_path / f"expert_{i}.pt"
            torch.save({
                'expert_state': expert.state_dict(),
                'expert_type': self.expert_type,
                'expert_config': {
                    'hidden_dim': self.expert_dim,
                    'ffn_dim': self.ffn_dim,
                    **self.expert_config
                },
                'usage_count': expert.usage_count,
                'specialization': self.expert_specializations.get(i, ""),
                'performance': self.expert_performance.get(i, 0.0)
            }, expert_path)
        
        # Save pool metadata
        metadata_path = save_path / "pool_metadata.pt"
        torch.save({
            'num_experts': self.num_experts,
            'expert_dim': self.expert_dim,
            'expert_type': self.expert_type,
            'ffn_dim': self.ffn_dim,
            'expert_config': self.expert_config,
            'active_experts': self.active_experts,
            'expert_specializations': self.expert_specializations,
            'expert_performance': self.expert_performance
        }, metadata_path)
    
    @classmethod
    def load_experts(cls, save_dir: str, device: str = "cpu") -> 'ExpertPool':
        """Load expert pool from directory."""
        save_path = Path(save_dir)
        
        # Load metadata
        metadata_path = save_path / "pool_metadata.pt"
        metadata = torch.load(metadata_path, map_location=device)
        
        # Create pool
        pool = cls(
            num_experts=metadata['num_experts'],
            expert_dim=metadata['expert_dim'],
            expert_type=metadata['expert_type'],
            ffn_dim=metadata['ffn_dim'],
            device=device,
            expert_config=metadata['expert_config']
        )
        
        # Load expert states
        for i in range(pool.num_experts):
            expert_path = save_path / f"expert_{i}.pt"
            if expert_path.exists():
                checkpoint = torch.load(expert_path, map_location=device)
                pool.experts[i].load_state_dict(checkpoint['expert_state'])
                pool.experts[i].usage_count = checkpoint.get('usage_count', 0)
                
                # Restore specializations and performance
                if checkpoint.get('specialization'):
                    pool.expert_specializations[i] = checkpoint['specialization']
                if checkpoint.get('performance'):
                    pool.expert_performance[i] = checkpoint['performance']
        
        # Restore pool state
        pool.active_experts = metadata['active_experts']
        pool.expert_specializations.update(metadata['expert_specializations'])
        pool.expert_performance.update(metadata['expert_performance'])
        
        return pool
    
    def parameters(self):
        """Get all parameters from active experts."""
        for expert_id in self.active_experts:
            yield from self.experts[expert_id].parameters()
    
    def named_parameters(self):
        """Get all named parameters from active experts."""
        for expert_id in self.active_experts:
            for name, param in self.experts[expert_id].named_parameters():
                yield f"expert_{expert_id}.{name}", param
    
    def train(self, mode: bool = True):
        """Set training mode for all experts."""
        for expert in self.experts:
            expert.train(mode)
    
    def eval(self):
        """Set evaluation mode for all experts."""
        self.train(False)
    
    def to(self, device):
        """Move all experts to device."""
        for expert in self.experts:
            expert.to(device)
        self.device = device