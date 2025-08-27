"""
Repository pattern implementations for data persistence.

This module implements repositories for storing and retrieving topologies,
experiments, and evolution results with various backend support.
"""

import json
import sqlite3
import pickle
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
from ..evolution.router import EvolutionConfig


@dataclass
class ExperimentRecord:
    """Record structure for evolution experiments."""
    experiment_id: str
    name: str
    config: EvolutionConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # "running", "completed", "failed"
    best_fitness: Optional[float] = None
    generations_run: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """
        Internal helper function.
        """
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TopologyRecord:
    """Record structure for topology storage."""
    topology_id: str
    experiment_id: str
    generation: int
    fitness: float
    sparsity: float
    total_connections: int
    routing_matrix_hash: str
    created_time: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """
        Internal helper function.
        """
        if self.metadata is None:
            self.metadata = {}


class BaseRepository(ABC):
    """Abstract base class for repositories."""

    @abstractmethod
    def initialize(self):
        """Initialize the repository."""
        pass

    @abstractmethod
    def close(self):
        """Close repository connections."""
        pass


class SQLiteRepository(BaseRepository):
    """
    Internal helper function.
    """
    """SQLite-based repository implementation."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection: Optional[sqlite3.Connection] = None
        self.logger = logging.getLogger(__name__)

    def initialize(self):
        """Initialize SQLite database and create tables."""
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row

        # Create experiments table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                best_fitness REAL,
                generations_run INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}'
            )
        """)

        # Create topologies table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS topologies (
                topology_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                fitness REAL NOT NULL,
                sparsity REAL NOT NULL,
                total_connections INTEGER NOT NULL,
                routing_matrix_hash TEXT NOT NULL,
                created_time TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}',
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        """)

        # Create indices for performance
        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_topologies_experiment
            ON topologies (experiment_id)
        """)

        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_topologies_fitness
            ON topologies (fitness DESC)
        """)

        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_topologies_generation
            ON topologies (experiment_id, generation)
        """)

        self.connection.commit()
        self.logger.info(f"Initialized SQLite repository at {self.db_path}")

    def close(self):
        """Close database connection."""
        if self.connection:
            """
            Internal helper function.
            """
            self.connection.close()
            self.connection = None


class TopologyRepository:
    """Repository for managing topology storage and retrieval."""

    def __init__(self, backend: BaseRepository, storage_dir: Optional[str] = None):
        self.backend = backend
        self.storage_dir = Path(storage_dir) if storage_dir else Path("./topology_storage")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Initialize backend
        self.backend.initialize()

    def save_topology(
        self,
        topology: TopologyGenome,
        experiment_id: str,
        fitness: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a topology to the repository.

        Args:
            topology: Topology to save
            experiment_id: Associated experiment ID
            fitness: Fitness score of the topology
            metadata: Additional metadata

        Returns:
            Topology ID
        """
        # Generate topology ID
        topology_id = self._generate_topology_id(topology, experiment_id)

        # Create topology record
        record = TopologyRecord(
            topology_id=topology_id,
            experiment_id=experiment_id,
            generation=topology.generation,
            fitness=fitness,
            sparsity=topology.compute_sparsity(),
            total_connections=int(topology.routing_matrix.sum().item()),
            routing_matrix_hash=self._hash_routing_matrix(topology.routing_matrix),
            created_time=datetime.now(),
            metadata=metadata or {}
        )

        # Save topology file
        topology_path = self.storage_dir / f"{topology_id}.pt"
        topology.save_topology(str(topology_path))

        # Save to database
        if isinstance(self.backend, SQLiteRepository):
            cursor = self.backend.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO topologies
                (topology_id, experiment_id, generation, fitness, sparsity,
                 total_connections, routing_matrix_hash, created_time, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.topology_id,
                record.experiment_id,
                record.generation,
                record.fitness,
                record.sparsity,
                record.total_connections,
                record.routing_matrix_hash,
                record.created_time.isoformat(),
                json.dumps(record.metadata)
            ))
            self.backend.connection.commit()

        self.logger.info(f"Saved topology {topology_id} for experiment {experiment_id}")
        return topology_id

    def load_topology(self, topology_id: str, device: str = "cpu") -> Optional[TopologyGenome]:
        """Load a topology by ID."""
        topology_path = self.storage_dir / f"{topology_id}.pt"

        if topology_path.exists():
            return TopologyGenome.load_topology(str(topology_path), device)

        self.logger.warning(f"Topology file not found: {topology_path}")
        return None

    def get_topology_record(self, topology_id: str) -> Optional[TopologyRecord]:
        """Get topology metadata record."""
        if isinstance(self.backend, SQLiteRepository):
            cursor = self.backend.connection.cursor()
            cursor.execute("""
                SELECT * FROM topologies WHERE topology_id = ?
            """, (topology_id,))

            row = cursor.fetchone()
            if row:
                return TopologyRecord(
                    topology_id=row['topology_id'],
                    experiment_id=row['experiment_id'],
                    generation=row['generation'],
                    fitness=row['fitness'],
                    sparsity=row['sparsity'],
                    total_connections=row['total_connections'],
                    routing_matrix_hash=row['routing_matrix_hash'],
                    created_time=datetime.fromisoformat(row['created_time']),
                    metadata=json.loads(row['metadata_json'])
                )

        return None

    def get_best_topologies(
        self,
        experiment_id: Optional[str] = None,
        limit: int = 10
    ) -> List[TopologyRecord]:
        """Get best topologies across experiments or for specific experiment."""
        if isinstance(self.backend, SQLiteRepository):
            cursor = self.backend.connection.cursor()

            if experiment_id:
                cursor.execute("""
                    SELECT * FROM topologies
                    WHERE experiment_id = ?
                    ORDER BY fitness DESC
                    LIMIT ?
                """, (experiment_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM topologies
                    ORDER BY fitness DESC
                    LIMIT ?
                """, (limit,))

            records = []
            for row in cursor.fetchall():
                records.append(TopologyRecord(
                    topology_id=row['topology_id'],
                    experiment_id=row['experiment_id'],
                    generation=row['generation'],
                    fitness=row['fitness'],
                    sparsity=row['sparsity'],
                    total_connections=row['total_connections'],
                    routing_matrix_hash=row['routing_matrix_hash'],
                    created_time=datetime.fromisoformat(row['created_time']),
                    metadata=json.loads(row['metadata_json'])
                ))

            return records

        return []

    def get_topology_evolution(self, experiment_id: str) -> List[TopologyRecord]:
        """Get topology evolution history for an experiment."""
        if isinstance(self.backend, SQLiteRepository):
            cursor = self.backend.connection.cursor()
            cursor.execute("""
                SELECT * FROM topologies
                WHERE experiment_id = ?
                ORDER BY generation ASC
            """, (experiment_id,))

            records = []
            for row in cursor.fetchall():
                records.append(TopologyRecord(
                    topology_id=row['topology_id'],
                    experiment_id=row['experiment_id'],
                    generation=row['generation'],
                    fitness=row['fitness'],
                    sparsity=row['sparsity'],
                    total_connections=row['total_connections'],
                    routing_matrix_hash=row['routing_matrix_hash'],
                    created_time=datetime.fromisoformat(row['created_time']),
                    metadata=json.loads(row['metadata_json'])
                ))

            return records

        return []

    def delete_topology(self, topology_id: str) -> bool:
        """Delete a topology and its file."""
        # Delete file
        topology_path = self.storage_dir / f"{topology_id}.pt"
        if topology_path.exists():
            topology_path.unlink()

        # Delete from database
        if isinstance(self.backend, SQLiteRepository):
            cursor = self.backend.connection.cursor()
            cursor.execute("DELETE FROM topologies WHERE topology_id = ?", (topology_id,))
            deleted = cursor.rowcount > 0
            self.backend.connection.commit()
            return deleted

        return False

    def _generate_topology_id(self, topology: TopologyGenome, experiment_id: str) -> str:
        """Generate unique topology ID."""
        # Create hash from routing matrix and parameters
        matrix_hash = self._hash_routing_matrix(topology.routing_matrix)
        params_str = f"{topology.routing_params.temperature}_{topology.routing_params.top_k}"
        combined = f"{experiment_id}_{matrix_hash}_{params_str}_{topology.generation}"

        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _hash_routing_matrix(self, matrix: torch.Tensor) -> str:
        """Create hash of routing matrix."""
        matrix_bytes = matrix.cpu().numpy().tobytes()
        return hashlib.md5(matrix_bytes).hexdigest()[:12]

    def cleanup_old_topologies(self, days_old: int = 30) -> int:
        """Clean up topologies older than specified days."""
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)

        if isinstance(self.backend, SQLiteRepository):
            cursor = self.backend.connection.cursor()

            # Get old topology IDs
            cursor.execute("""
                SELECT topology_id FROM topologies
                WHERE created_time < ?
            """, (datetime.fromtimestamp(cutoff_date).isoformat(),))

            old_topology_ids = [row[0] for row in cursor.fetchall()]

            # Delete files and database records
            deleted_count = 0
            for topology_id in old_topology_ids:
                if self.delete_topology(topology_id):
                    """
                    Internal helper function.
                    """
                    deleted_count += 1

            self.logger.info(f"Cleaned up {deleted_count} old topologies")
            return deleted_count

        return 0


class ExperimentRepository:
    """Repository for managing evolution experiments."""

    def __init__(self, backend: BaseRepository):
        self.backend = backend
        self.logger = logging.getLogger(__name__)

        # Initialize backend
        self.backend.initialize()

    def create_experiment(
        self,
        name: str,
        config: EvolutionConfig,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new experiment record."""
        experiment_id = self._generate_experiment_id(name)

        record = ExperimentRecord(
            experiment_id=experiment_id,
            name=name,
            config=config,
            start_time=datetime.now(),
            metadata=metadata or {}
        )

        # Save to database
        if isinstance(self.backend, SQLiteRepository):
            cursor = self.backend.connection.cursor()
            cursor.execute("""
                INSERT INTO experiments
                (experiment_id, name, config_json, start_time, status, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                record.experiment_id,
                record.name,
                json.dumps(asdict(record.config)),
                record.start_time.isoformat(),
                record.status,
                json.dumps(record.metadata)
            ))
            self.backend.connection.commit()

        self.logger.info(f"Created experiment {experiment_id}: {name}")
        return experiment_id

    def update_experiment(
        self,
        experiment_id: str,
        status: Optional[str] = None,
        best_fitness: Optional[float] = None,
        generations_run: Optional[int] = None,
        end_time: Optional[datetime] = None
    ):
        """Update experiment record."""
        if isinstance(self.backend, SQLiteRepository):
            cursor = self.backend.connection.cursor()

            # Build update query dynamically
            updates = []
            params = []

            if status is not None:
                updates.append("status = ?")
                params.append(status)

            if best_fitness is not None:
                updates.append("best_fitness = ?")
                params.append(best_fitness)

            if generations_run is not None:
                updates.append("generations_run = ?")
                params.append(generations_run)

            if end_time is not None:
                updates.append("end_time = ?")
                params.append(end_time.isoformat())

            if updates:
                query = f"UPDATE experiments SET {', '.join(updates)} WHERE experiment_id = ?"
                params.append(experiment_id)
                cursor.execute(query, params)
                self.backend.connection.commit()

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Get experiment record by ID."""
        if isinstance(self.backend, SQLiteRepository):
            cursor = self.backend.connection.cursor()
            cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))

            row = cursor.fetchone()
            if row:
                return ExperimentRecord(
                    experiment_id=row['experiment_id'],
                    name=row['name'],
                    config=EvolutionConfig(**json.loads(row['config_json'])),
                    start_time=datetime.fromisoformat(row['start_time']),
                    end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                    status=row['status'],
                    best_fitness=row['best_fitness'],
                    generations_run=row['generations_run'],
                    metadata=json.loads(row['metadata_json'])
                )

        return None

    def list_experiments(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[ExperimentRecord]:
        """List experiments with optional status filter."""
        if isinstance(self.backend, SQLiteRepository):
            cursor = self.backend.connection.cursor()

            if status:
                cursor.execute("""
                    SELECT * FROM experiments
                    WHERE status = ?
                    ORDER BY start_time DESC
                    LIMIT ?
                """, (status, limit))
            else:
                cursor.execute("""
                    SELECT * FROM experiments
                    ORDER BY start_time DESC
                    LIMIT ?
                """, (limit,))

            records = []
            for row in cursor.fetchall():
                records.append(ExperimentRecord(
                    experiment_id=row['experiment_id'],
                    name=row['name'],
                    config=EvolutionConfig(**json.loads(row['config_json'])),
                    start_time=datetime.fromisoformat(row['start_time']),
                    end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                    status=row['status'],
                    best_fitness=row['best_fitness'],
                    generations_run=row['generations_run'],
                    metadata=json.loads(row['metadata_json'])
                ))

            return records

        return []

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment record."""
        if isinstance(self.backend, SQLiteRepository):
            cursor = self.backend.connection.cursor()

            # First delete associated topologies
            cursor.execute("DELETE FROM topologies WHERE experiment_id = ?", (experiment_id,))

            # Then delete experiment
            cursor.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))

            deleted = cursor.rowcount > 0
            self.backend.connection.commit()

            if deleted:
                self.logger.info(f"Deleted experiment {experiment_id}")

            return deleted

        return False

    def get_experiment_stats(self) -> Dict[str, Any]:
        """Get overall experiment statistics."""
        if isinstance(self.backend, SQLiteRepository):
            cursor = self.backend.connection.cursor()

            # Count experiments by status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM experiments
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())

            # Get total topology count
            cursor.execute("SELECT COUNT(*) FROM topologies")
            total_topologies = cursor.fetchone()[0]

            # Get best overall fitness
            cursor.execute("SELECT MAX(best_fitness) FROM experiments WHERE best_fitness IS NOT NULL")
            best_fitness = cursor.fetchone()[0]

            return {
                'total_experiments': sum(status_counts.values()),
                'status_counts': status_counts,
                'total_topologies': total_topologies,
                'best_fitness': best_fitness
            }

        return {}

    def _generate_experiment_id(self, name: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().isoformat()
        combined = f"{name}_{timestamp}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def close(self):
        """Close repository."""
        self.backend.close()