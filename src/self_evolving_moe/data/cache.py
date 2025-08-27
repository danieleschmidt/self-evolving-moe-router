"""
Caching system for evolution optimization.

This module provides intelligent caching for fitness evaluations,
topology comparisons, and expensive computations during evolution.
"""

import pickle
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import json
import logging

import torch
import numpy as np

from ..routing.topology import TopologyGenome


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_time: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """
        Internal helper function.
        """
        if self.metadata is None:
            self.metadata = {}


class LRUCache:
    """
    Internal helper function.
    """
    """Thread-safe LRU cache implementation."""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.current_memory = 0
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.last_accessed = datetime.now()
                entry.access_count += 1

                # Move to end of access order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)

                self.hits += 1
                return entry.value
            else:
                self.misses += 1
                return None

    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Put value in cache."""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024  # Fallback estimate

            # Check if we need to evict
            while (len(self.cache) >= self.max_size or
                   self.current_memory + size_bytes > self.max_memory_bytes):
                if not self.access_order:
                    break
                self._evict_lru()

            # Remove existing entry if updating
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory -= old_entry.size_bytes
                if key in self.access_order:
                    self.access_order.remove(key)

            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_time=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                metadata=metadata or {}
            )

            self.cache[key] = entry
            self.access_order.append(key)
            self.current_memory += size_bytes

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.access_order:
            return

        lru_key = self.access_order.pop(0)
        if lru_key in self.cache:
            entry = self.cache[lru_key]
            self.current_memory -= entry.size_bytes
            del self.cache[lru_key]
            self.evictions += 1

    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self.lock:
            return key in self.cache

    def remove(self, key: str) -> bool:
        """Remove key from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.current_memory -= entry.size_bytes
                del self.cache[key]

                if key in self.access_order:
                    self.access_order.remove(key)

                return True
            return False

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.current_memory = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_accesses = self.hits + self.misses
            hit_rate = self.hits / total_accesses if total_accesses > 0 else 0

            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_mb': self.current_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions
            }


class FitnessCache:
    """
    Internal helper function.
    """
    """Specialized cache for fitness evaluations."""

    def __init__(self, cache: LRUCache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)

    def get_fitness(self, topology: TopologyGenome, model_hash: str) -> Optional[Tuple[float, Dict[str, float]]]:
        """Get cached fitness for topology and model combination."""
        cache_key = self._get_fitness_key(topology, model_hash)
        result = self.cache.get(cache_key)

        if result is not None:
            self.logger.debug(f"Cache hit for fitness: {cache_key}")
            return result

        return None

    def cache_fitness(
        self,
        topology: TopologyGenome,
        model_hash: str,
        fitness: float,
        metrics: Dict[str, float]
    ):
        """Cache fitness evaluation result."""
        cache_key = self._get_fitness_key(topology, model_hash)
        value = (fitness, metrics)

        metadata = {
            'topology_generation': topology.generation,
            'topology_sparsity': topology.compute_sparsity(),
            'cached_at': datetime.now().isoformat()
        }

        self.cache.put(cache_key, value, metadata)
        self.logger.debug(f"Cached fitness: {cache_key}")

    def _get_fitness_key(self, topology: TopologyGenome, model_hash: str) -> str:
        """Generate cache key for fitness evaluation."""
        # Create hash from topology structure and parameters
        topology_data = {
            'routing_matrix': topology.routing_matrix.cpu().numpy().tobytes(),
            'expert_graph': topology.expert_graph.cpu().numpy().tobytes(),
            'temperature': topology.routing_params.temperature,
            'top_k': topology.routing_params.top_k,
            'load_balancing_weight': topology.routing_params.load_balancing_weight
        }

        topology_str = str(topology_data)
        combined = f"fitness_{model_hash}_{topology_str}"

        return hashlib.md5(combined.encode()).hexdigest()


class TopologyCache:
    """
    Internal helper function.
    """
    """Cache for topology comparisons and similarities."""

    def __init__(self, cache: LRUCache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)

    def get_similarity(self, topology1: TopologyGenome, topology2: TopologyGenome) -> Optional[float]:
        """Get cached similarity between two topologies."""
        cache_key = self._get_similarity_key(topology1, topology2)
        return self.cache.get(cache_key)

    def cache_similarity(self, topology1: TopologyGenome, topology2: TopologyGenome, similarity: float):
        """Cache topology similarity."""
        cache_key = self._get_similarity_key(topology1, topology2)
        metadata = {
            'topology1_gen': topology1.generation,
            'topology2_gen': topology2.generation,
            'cached_at': datetime.now().isoformat()
        }

        self.cache.put(cache_key, similarity, metadata)

    def get_topology_hash(self, topology: TopologyGenome) -> str:
        """Get consistent hash for topology."""
        cache_key = f"topology_hash_{id(topology)}"
        cached_hash = self.cache.get(cache_key)

        if cached_hash is None:
            # Compute hash
            topology_data = {
                'routing_matrix': topology.routing_matrix.cpu().numpy().tobytes(),
                'expert_graph': topology.expert_graph.cpu().numpy().tobytes(),
                'params': asdict(topology.routing_params)
            }

            topology_str = str(topology_data)
            topology_hash = hashlib.md5(topology_str.encode()).hexdigest()

            # Cache the hash
            self.cache.put(cache_key, topology_hash)
            cached_hash = topology_hash

        return cached_hash

    def _get_similarity_key(self, topology1: TopologyGenome, topology2: TopologyGenome) -> str:
        """Generate cache key for topology similarity."""
        hash1 = self.get_topology_hash(topology1)
        hash2 = self.get_topology_hash(topology2)

        # Ensure consistent ordering
        if hash1 < hash2:
            """
            Internal helper function.
            """
            return f"similarity_{hash1}_{hash2}"
        else:
            return f"similarity_{hash2}_{hash1}"


class PersistentCache:
    """Persistent cache that survives program restarts."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.logger = logging.getLogger(__name__)

        # Load existing metadata
        self.metadata = self._load_metadata()

    def save(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Save value to persistent cache."""
        # Generate filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        filename = f"{key_hash}.pkl"
        filepath = self.cache_dir / filename

        # Save data
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(value, f)

            # Update metadata
            self.metadata[key] = {
                'filename': filename,
                'created_time': datetime.now().isoformat(),
                'size_bytes': filepath.stat().st_size,
                'metadata': metadata or {}
            }

            self._save_metadata()
            self.logger.debug(f"Saved to persistent cache: {key}")

        except Exception as e:
            self.logger.error(f"Failed to save to persistent cache: {e}")

    def load(self, key: str) -> Optional[Any]:
        """Load value from persistent cache."""
        if key not in self.metadata:
            return None

        filename = self.metadata[key]['filename']
        filepath = self.cache_dir / filename

        if not filepath.exists():
            # Clean up stale metadata
            del self.metadata[key]
            self._save_metadata()
            return None

        try:
            with open(filepath, 'rb') as f:
                value = pickle.load(f)

            self.logger.debug(f"Loaded from persistent cache: {key}")
            return value

        except Exception as e:
            self.logger.error(f"Failed to load from persistent cache: {e}")
            return None

    def contains(self, key: str) -> bool:
        """Check if key exists in persistent cache."""
        return key in self.metadata

    def remove(self, key: str) -> bool:
        """Remove key from persistent cache."""
        if key not in self.metadata:
            return False

        filename = self.metadata[key]['filename']
        filepath = self.cache_dir / filename

        # Remove file
        if filepath.exists():
            filepath.unlink()

        # Remove metadata
        del self.metadata[key]
        self._save_metadata()

        self.logger.debug(f"Removed from persistent cache: {key}")
        return True

    def cleanup_old(self, days_old: int = 7) -> int:
        """Clean up old cache entries."""
        cutoff_time = datetime.now() - timedelta(days=days_old)

        to_remove = []
        for key, info in self.metadata.items():
            created_time = datetime.fromisoformat(info['created_time'])
            if created_time < cutoff_time:
                to_remove.append(key)

        removed_count = 0
        for key in to_remove:
            if self.remove(key):
                removed_count += 1

        self.logger.info(f"Cleaned up {removed_count} old cache entries")
        return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """Get persistent cache statistics."""
        total_size = sum(info['size_bytes'] for info in self.metadata.values())

        return {
            'entries': len(self.metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load cache metadata: {e}")

        return {}

    def _save_metadata(self):
        """
        Internal helper function.
        """
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")


class EvolutionCache:
    """Main cache system for evolution optimization."""

    def __init__(
        self,
        memory_cache_size: int = 1000,
        memory_cache_mb: int = 100,
        persistent_cache_dir: Optional[str] = None,
        enable_persistent: bool = True
    ):
        # Memory cache
        self.lru_cache = LRUCache(memory_cache_size, memory_cache_mb)

        # Specialized caches
        self.fitness_cache = FitnessCache(self.lru_cache)
        self.topology_cache = TopologyCache(self.lru_cache)

        # Persistent cache
        self.persistent_cache = None
        if enable_persistent:
            cache_dir = persistent_cache_dir or "./evolution_cache"
            self.persistent_cache = PersistentCache(cache_dir)

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized evolution cache system")

    def get_fitness(self, topology: TopologyGenome, model_hash: str) -> Optional[Tuple[float, Dict[str, float]]]:
        """Get cached fitness evaluation."""
        return self.fitness_cache.get_fitness(topology, model_hash)

    def cache_fitness(
        self,
        topology: TopologyGenome,
        model_hash: str,
        fitness: float,
        metrics: Dict[str, float]
    ):
        """Cache fitness evaluation result."""
        self.fitness_cache.cache_fitness(topology, model_hash, fitness, metrics)

        # Also save to persistent cache if enabled
        if self.persistent_cache:
            cache_key = f"fitness_{self.topology_cache.get_topology_hash(topology)}_{model_hash}"
            self.persistent_cache.save(cache_key, (fitness, metrics))

    def get_topology_similarity(self, topology1: TopologyGenome, topology2: TopologyGenome) -> Optional[float]:
        """Get cached topology similarity."""
        return self.topology_cache.get_similarity(topology1, topology2)

    def cache_topology_similarity(self, topology1: TopologyGenome, topology2: TopologyGenome, similarity: float):
        """Cache topology similarity."""
        self.topology_cache.cache_similarity(topology1, topology2, similarity)

    def save_evolution_state(self, key: str, state: Any):
        """Save evolution state to persistent cache."""
        if self.persistent_cache:
            self.persistent_cache.save(f"evolution_state_{key}", state)

    def load_evolution_state(self, key: str) -> Optional[Any]:
        """Load evolution state from persistent cache."""
        if self.persistent_cache:
            return self.persistent_cache.load(f"evolution_state_{key}")
        return None

    def clear_memory_cache(self):
        """Clear memory cache."""
        self.lru_cache.clear()
        self.logger.info("Cleared memory cache")

    def cleanup_old_entries(self, days_old: int = 7) -> int:
        """Clean up old persistent cache entries."""
        if self.persistent_cache:
            return self.persistent_cache.cleanup_old(days_old)
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'memory_cache': self.lru_cache.get_stats()
        }

        if self.persistent_cache:
            stats['persistent_cache'] = self.persistent_cache.get_stats()

        return stats

    def optimize_cache(self):
        """Optimize cache performance."""
        # Remove least accessed entries if cache is getting full
        memory_stats = self.lru_cache.get_stats()

        if memory_stats['memory_mb'] > memory_stats['max_memory_mb'] * 0.9:
            self.logger.info("Memory cache near capacity, triggering cleanup")
            # The LRU cache will automatically evict entries

        # Clean up old persistent entries
        if self.persistent_cache:
            cleaned = self.cleanup_old_entries(days_old=30)
            if cleaned > 0:
                self.logger.info(f"Cleaned up {cleaned} old persistent cache entries")