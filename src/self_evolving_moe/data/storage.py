"""
Model storage and checkpointing system.

This module provides comprehensive model storage, versioning,
and checkpoint management for MoE models and evolution states.
"""

import json
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import hashlib
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import torch
import numpy as np

from ..routing.topology import TopologyGenome
from ..experts.pool import ExpertPool
from ..experts.slimmable import SlimmableMoE
from ..evolution.router import EvolvingMoERouter, EvolutionConfig


@dataclass
class ModelMetadata:
    """Metadata for stored models."""
    model_id: str
    name: str
    version: str
    model_type: str  # "slimmable_moe", "expert_pool", "topology"
    created_time: datetime
    size_bytes: int
    num_parameters: int
    config: Dict[str, Any]
    metrics: Dict[str, float] = None
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.tags is None:
            self.tags = []


@dataclass
class CheckpointInfo:
    """Information about model checkpoints."""
    checkpoint_id: str
    model_id: str
    step: int
    epoch: Optional[int]
    created_time: datetime
    metrics: Dict[str, float]
    file_path: str
    size_bytes: int


class StorageBackend(ABC):
    """Abstract storage backend."""
    
    @abstractmethod
    def save(self, key: str, data: bytes, metadata: Optional[Dict[str, Any]] = None):
        """Save data to storage."""
        pass
    
    @abstractmethod
    def load(self, key: str) -> Optional[bytes]:
        """Load data from storage."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data."""
        pass
    
    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List keys with optional prefix."""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save(self, key: str, data: bytes, metadata: Optional[Dict[str, Any]] = None):
        """Save data to local file."""
        file_path = self.base_path / key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(data)
        
        # Save metadata if provided
        if metadata:
            metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        self.logger.debug(f"Saved {len(data)} bytes to {file_path}")
    
    def load(self, key: str) -> Optional[bytes]:
        """Load data from local file."""
        file_path = self.base_path / key
        
        if file_path.exists():
            with open(file_path, 'rb') as f:
                data = f.read()
            self.logger.debug(f"Loaded {len(data)} bytes from {file_path}")
            return data
        
        return None
    
    def exists(self, key: str) -> bool:
        """Check if file exists."""
        return (self.base_path / key).exists()
    
    def delete(self, key: str) -> bool:
        """Delete file."""
        file_path = self.base_path / key
        metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
        
        deleted = False
        
        if file_path.exists():
            file_path.unlink()
            deleted = True
        
        if metadata_path.exists():
            metadata_path.unlink()
        
        return deleted
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """List files with optional prefix."""
        keys = []
        
        if prefix:
            pattern = f"{prefix}*"
        else:
            pattern = "*"
        
        for path in self.base_path.rglob(pattern):
            if path.is_file() and not path.name.endswith('.meta'):
                relative_path = path.relative_to(self.base_path)
                keys.append(str(relative_path))
        
        return sorted(keys)
    
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a key."""
        file_path = self.base_path / key
        metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        return None


class ModelRegistry:
    """Registry for tracking stored models."""
    
    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self.models: Dict[str, ModelMetadata] = {}
        self.checkpoints: Dict[str, List[CheckpointInfo]] = {}
        self._load_registry()
        
        self.logger = logging.getLogger(__name__)
    
    def register_model(self, metadata: ModelMetadata):
        """Register a new model."""
        self.models[metadata.model_id] = metadata
        if metadata.model_id not in self.checkpoints:
            self.checkpoints[metadata.model_id] = []
        
        self._save_registry()
        self.logger.info(f"Registered model: {metadata.model_id} ({metadata.name})")
    
    def add_checkpoint(self, checkpoint_info: CheckpointInfo):
        """Add checkpoint for a model."""
        if checkpoint_info.model_id not in self.checkpoints:
            self.checkpoints[checkpoint_info.model_id] = []
        
        self.checkpoints[checkpoint_info.model_id].append(checkpoint_info)
        
        # Sort by step/creation time
        self.checkpoints[checkpoint_info.model_id].sort(key=lambda x: (x.step, x.created_time))
        
        self._save_registry()
        self.logger.info(f"Added checkpoint: {checkpoint_info.checkpoint_id}")
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        return self.models.get(model_id)
    
    def list_models(self, model_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """List models with optional filters."""
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]
        
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x.created_time, reverse=True)
        
        return models
    
    def get_checkpoints(self, model_id: str) -> List[CheckpointInfo]:
        """Get checkpoints for a model."""
        return self.checkpoints.get(model_id, [])
    
    def get_latest_checkpoint(self, model_id: str) -> Optional[CheckpointInfo]:
        """Get latest checkpoint for a model."""
        checkpoints = self.get_checkpoints(model_id)
        return checkpoints[-1] if checkpoints else None
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model from registry."""
        if model_id in self.models:
            del self.models[model_id]
            if model_id in self.checkpoints:
                del self.checkpoints[model_id]
            
            self._save_registry()
            self.logger.info(f"Deleted model from registry: {model_id}")
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_checkpoints = sum(len(checkpoints) for checkpoints in self.checkpoints.values())
        total_size = sum(model.size_bytes for model in self.models.values())
        
        # Count by model type
        type_counts = {}
        for model in self.models.values():
            type_counts[model.model_type] = type_counts.get(model.model_type, 0) + 1
        
        return {
            'total_models': len(self.models),
            'total_checkpoints': total_checkpoints,
            'total_size_mb': total_size / (1024 * 1024),
            'model_types': type_counts
        }
    
    def _load_registry(self):
        """Load registry from file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                
                # Load models
                for model_data in data.get('models', []):
                    metadata = ModelMetadata(
                        model_id=model_data['model_id'],
                        name=model_data['name'],
                        version=model_data['version'],
                        model_type=model_data['model_type'],
                        created_time=datetime.fromisoformat(model_data['created_time']),
                        size_bytes=model_data['size_bytes'],
                        num_parameters=model_data['num_parameters'],
                        config=model_data['config'],
                        metrics=model_data.get('metrics', {}),
                        tags=model_data.get('tags', []),
                        description=model_data.get('description', "")
                    )
                    self.models[metadata.model_id] = metadata
                
                # Load checkpoints
                for model_id, checkpoint_list in data.get('checkpoints', {}).items():
                    self.checkpoints[model_id] = []
                    for checkpoint_data in checkpoint_list:
                        checkpoint_info = CheckpointInfo(
                            checkpoint_id=checkpoint_data['checkpoint_id'],
                            model_id=checkpoint_data['model_id'],
                            step=checkpoint_data['step'],
                            epoch=checkpoint_data.get('epoch'),
                            created_time=datetime.fromisoformat(checkpoint_data['created_time']),
                            metrics=checkpoint_data['metrics'],
                            file_path=checkpoint_data['file_path'],
                            size_bytes=checkpoint_data['size_bytes']
                        )
                        self.checkpoints[model_id].append(checkpoint_info)
                
            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save registry to file."""
        try:
            data = {
                'models': [asdict(model) for model in self.models.values()],
                'checkpoints': {
                    model_id: [asdict(checkpoint) for checkpoint in checkpoints]
                    for model_id, checkpoints in self.checkpoints.items()
                }
            }
            
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")


class ModelStorage:
    """Main model storage system."""
    
    def __init__(
        self,
        storage_backend: StorageBackend,
        registry_path: Optional[str] = None
    ):
        self.storage_backend = storage_backend
        self.registry = ModelRegistry(registry_path or "./model_registry.json")
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initialized model storage system")
    
    def save_topology(
        self,
        topology: TopologyGenome,
        name: str,
        version: str = "1.0",
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """Save topology to storage."""
        model_id = self._generate_model_id(name, "topology")
        
        # Serialize topology
        state = {
            'routing_matrix': topology.routing_matrix.cpu(),
            'expert_graph': topology.expert_graph.cpu(),
            'routing_params': topology.routing_params,
            'num_tokens': topology.num_tokens,
            'num_experts': topology.num_experts,
            'sparsity': topology.sparsity,
            'generation': topology.generation,
            'fitness_history': topology.fitness_history
        }
        
        data = torch.save(state, buffer=True)
        
        # Save to storage
        key = f"topologies/{model_id}.pt"
        self.storage_backend.save(key, data)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            model_type="topology",
            created_time=datetime.now(),
            size_bytes=len(data),
            num_parameters=int(topology.routing_matrix.numel() + topology.expert_graph.numel()),
            config={
                'num_tokens': topology.num_tokens,
                'num_experts': topology.num_experts,
                'sparsity': topology.sparsity,
                'routing_params': asdict(topology.routing_params)
            },
            metrics={
                'sparsity': topology.compute_sparsity(),
                'total_connections': int(topology.routing_matrix.sum().item()),
                'generation': topology.generation
            },
            tags=tags or [],
            description=description
        )
        
        self.registry.register_model(metadata)
        
        self.logger.info(f"Saved topology: {model_id}")
        return model_id
    
    def load_topology(self, model_id: str, device: str = "cpu") -> Optional[TopologyGenome]:
        """Load topology from storage."""
        key = f"topologies/{model_id}.pt"
        data = self.storage_backend.load(key)
        
        if data is None:
            self.logger.error(f"Topology not found: {model_id}")
            return None
        
        try:
            state = torch.load(data, map_location=device)
            
            topology = TopologyGenome(
                num_tokens=state['num_tokens'],
                num_experts=state['num_experts'],
                sparsity=state['sparsity'],
                routing_params=state['routing_params'],
                device=device
            )
            
            topology.routing_matrix = state['routing_matrix'].to(device)
            topology.expert_graph = state['expert_graph'].to(device)
            topology.generation = state.get('generation', 0)
            topology.fitness_history = state.get('fitness_history', [])
            
            self.logger.info(f"Loaded topology: {model_id}")
            return topology
            
        except Exception as e:
            self.logger.error(f"Failed to load topology {model_id}: {e}")
            return None
    
    def save_expert_pool(
        self,
        expert_pool: ExpertPool,
        name: str,
        version: str = "1.0",
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """Save expert pool to storage."""
        model_id = self._generate_model_id(name, "expert_pool")
        
        # Create temporary directory for expert pool
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save expert pool
            expert_pool.save_experts(str(temp_path / "experts"))
            
            # Create archive
            archive_path = temp_path / f"{model_id}.tar.gz"
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(temp_path / "experts", arcname="experts")
            
            # Read archive data
            with open(archive_path, 'rb') as f:
                data = f.read()
        
        # Save to storage
        key = f"expert_pools/{model_id}.tar.gz"
        self.storage_backend.save(key, data)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            model_type="expert_pool",
            created_time=datetime.now(),
            size_bytes=len(data),
            num_parameters=expert_pool.get_total_parameters(),
            config={
                'num_experts': expert_pool.num_experts,
                'expert_dim': expert_pool.expert_dim,
                'expert_type': expert_pool.expert_type,
                'ffn_dim': expert_pool.ffn_dim,
                'expert_config': expert_pool.expert_config
            },
            metrics={
                'num_experts': expert_pool.num_experts,
                'active_experts': len(expert_pool.active_experts),
                'total_parameters': expert_pool.get_total_parameters()
            },
            tags=tags or [],
            description=description
        )
        
        self.registry.register_model(metadata)
        
        self.logger.info(f"Saved expert pool: {model_id}")
        return model_id
    
    def load_expert_pool(self, model_id: str, device: str = "cpu") -> Optional[ExpertPool]:
        """Load expert pool from storage."""
        key = f"expert_pools/{model_id}.tar.gz"
        data = self.storage_backend.load(key)
        
        if data is None:
            self.logger.error(f"Expert pool not found: {model_id}")
            return None
        
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save archive to temp file
                archive_path = temp_path / f"{model_id}.tar.gz"
                with open(archive_path, 'wb') as f:
                    f.write(data)
                
                # Extract archive
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(temp_path)
                
                # Load expert pool
                expert_pool = ExpertPool.load_experts(str(temp_path / "experts"), device)
                
                self.logger.info(f"Loaded expert pool: {model_id}")
                return expert_pool
                
        except Exception as e:
            self.logger.error(f"Failed to load expert pool {model_id}: {e}")
            return None
    
    def save_slimmable_moe(
        self,
        model: SlimmableMoE,
        name: str,
        version: str = "1.0",
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """Save SlimmableMoE model to storage."""
        model_id = self._generate_model_id(name, "slimmable_moe")
        
        # Create temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save model using its built-in method
            model.save_pretrained(str(temp_path / "model"))
            
            # Create archive
            archive_path = temp_path / f"{model_id}.tar.gz"
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(temp_path / "model", arcname="model")
            
            # Read archive data
            with open(archive_path, 'rb') as f:
                data = f.read()
        
        # Save to storage
        key = f"slimmable_moe/{model_id}.tar.gz"
        self.storage_backend.save(key, data)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            model_type="slimmable_moe",
            created_time=datetime.now(),
            size_bytes=len(data),
            num_parameters=model.expert_pool.get_total_parameters(),
            config={
                'width_configs': model.width_configs,
                'default_width': model.default_width,
                'num_experts': model.expert_pool.num_experts,
                'expert_dim': model.expert_pool.expert_dim
            },
            metrics=model.get_efficiency_report(),
            tags=tags or [],
            description=description
        )
        
        self.registry.register_model(metadata)
        
        self.logger.info(f"Saved SlimmableMoE: {model_id}")
        return model_id
    
    def save_evolution_checkpoint(
        self,
        evolver: EvolvingMoERouter,
        model_id: str,
        step: int,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Save evolution checkpoint."""
        checkpoint_id = f"{model_id}_step_{step}"
        
        # Create temporary file for checkpoint
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt') as temp_file:
            # Save evolution state
            evolver.save_evolution_state(temp_file.name)
            
            # Read checkpoint data
            with open(temp_file.name, 'rb') as f:
                data = f.read()
        
        # Save to storage
        key = f"checkpoints/{checkpoint_id}.pt"
        self.storage_backend.save(key, data)
        
        # Create checkpoint info
        checkpoint_info = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            model_id=model_id,
            step=step,
            epoch=epoch,
            created_time=datetime.now(),
            metrics=metrics or evolver.get_evolution_stats(),
            file_path=key,
            size_bytes=len(data)
        )
        
        self.registry.add_checkpoint(checkpoint_info)
        
        self.logger.info(f"Saved evolution checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    def list_models(self, model_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """List stored models."""
        return self.registry.list_models(model_type, tags)
    
    def get_model_info(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model information."""
        return self.registry.get_model(model_id)
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model from storage."""
        # Get model info to determine type
        model_info = self.registry.get_model(model_id)
        if not model_info:
            return False
        
        # Determine storage key based on model type
        if model_info.model_type == "topology":
            key = f"topologies/{model_id}.pt"
        elif model_info.model_type == "expert_pool":
            key = f"expert_pools/{model_id}.tar.gz"
        elif model_info.model_type == "slimmable_moe":
            key = f"slimmable_moe/{model_id}.tar.gz"
        else:
            self.logger.error(f"Unknown model type: {model_info.model_type}")
            return False
        
        # Delete from storage
        deleted = self.storage_backend.delete(key)
        
        # Delete checkpoints
        checkpoints = self.registry.get_checkpoints(model_id)
        for checkpoint in checkpoints:
            self.storage_backend.delete(checkpoint.file_path)
        
        # Remove from registry
        self.registry.delete_model(model_id)
        
        if deleted:
            self.logger.info(f"Deleted model: {model_id}")
        
        return deleted
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        registry_stats = self.registry.get_stats()
        
        # Get storage backend stats if available
        backend_stats = {}
        if hasattr(self.storage_backend, 'get_stats'):
            backend_stats = self.storage_backend.get_stats()
        
        return {
            'registry': registry_stats,
            'backend': backend_stats
        }
    
    def _generate_model_id(self, name: str, model_type: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().isoformat()
        combined = f"{model_type}_{name}_{timestamp}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def cleanup_old_models(self, days_old: int = 30) -> int:
        """Clean up old models."""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        
        old_models = []
        for model in self.registry.list_models():
            if model.created_time.timestamp() < cutoff_time:
                old_models.append(model.model_id)
        
        deleted_count = 0
        for model_id in old_models:
            if self.delete_model(model_id):
                deleted_count += 1
        
        self.logger.info(f"Cleaned up {deleted_count} old models")
        return deleted_count