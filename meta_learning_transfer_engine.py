#!/usr/bin/env python3
"""
TERRAGON V6.0 - Advanced Meta-Learning & Transfer Evolution Engine
Autonomous SDLC Implementation - Meta-Learning for Few-Shot Evolution

Features:
- Few-Shot Learning for Rapid MoE Topology Adaptation
- Transfer Learning Across Different Task Domains
- Gradient-Based Meta-Learning (MAML-inspired)
- Neural Architecture Search Integration
- Memory-Augmented Evolution Networks
- Continual Learning with Catastrophic Forgetting Prevention
- Cross-Domain Knowledge Transfer
"""

import asyncio
import json
import logging
import time
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
from datetime import datetime
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
import copy
from collections import defaultdict, deque
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetaLearningEngine")

@dataclass
class TaskDefinition:
    """Task definition for meta-learning"""
    task_id: str
    task_type: str
    domain: str
    input_dimension: int
    num_experts: int
    num_tokens: int
    complexity_score: float
    task_specific_constraints: Dict[str, Any]
    performance_requirements: Dict[str, float]

@dataclass
class MetaLearningExperience:
    """Detailed meta-learning experience"""
    experience_id: str
    task: TaskDefinition
    solution_topology: np.ndarray
    adaptation_steps: List[Dict]
    performance_trajectory: List[float]
    convergence_time: float
    final_performance: float
    meta_features: Dict[str, float]
    transfer_success_rate: float
    timestamp: float

@dataclass
class FewShotLearningContext:
    """Context for few-shot learning"""
    support_set: List[MetaLearningExperience]
    query_task: TaskDefinition
    initial_adaptation_rate: float
    num_adaptation_steps: int
    meta_gradients: Optional[Dict[str, np.ndarray]]

class MemoryAugmentedNetwork:
    """Memory-augmented neural network for meta-learning"""
    
    def __init__(self, memory_size: int = 1000, key_dim: int = 64):
        self.memory_size = memory_size
        self.key_dim = key_dim
        
        # External memory
        self.memory_keys = np.random.randn(memory_size, key_dim)
        self.memory_values = np.random.randn(memory_size, key_dim)
        self.memory_usage = np.zeros(memory_size)
        
        # Controller parameters
        self.controller_weights = np.random.randn(key_dim, key_dim) * 0.1
        self.read_weights = np.zeros(memory_size)
        self.write_weights = np.zeros(memory_size)
        
    def read_memory(self, query_key: np.ndarray) -> np.ndarray:
        """Read from external memory using content-based addressing"""
        # Compute similarity scores
        similarities = np.dot(self.memory_keys, query_key) / (
            np.linalg.norm(self.memory_keys, axis=1) * np.linalg.norm(query_key) + 1e-8
        )
        
        # Softmax to get read weights
        self.read_weights = self._softmax(similarities)
        
        # Read weighted memory values
        read_vector = np.dot(self.read_weights, self.memory_values)
        
        # Update memory usage statistics
        self.memory_usage += self.read_weights
        
        return read_vector
    
    def write_memory(self, key: np.ndarray, value: np.ndarray, erase_vector: Optional[np.ndarray] = None):
        """Write to external memory"""
        # Find least used memory location
        min_usage_idx = np.argmin(self.memory_usage)
        
        # Optionally erase old memory
        if erase_vector is not None:
            self.memory_values[min_usage_idx] *= (1 - erase_vector)
        
        # Write new memory
        self.memory_keys[min_usage_idx] = key
        self.memory_values[min_usage_idx] = value
        self.memory_usage[min_usage_idx] = 0  # Reset usage counter
    
    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute softmax with temperature"""
        exp_x = np.exp((x - np.max(x)) / temperature)
        return exp_x / np.sum(exp_x)
    
    def encode_experience(self, experience: MetaLearningExperience) -> np.ndarray:
        """Encode experience into memory representation"""
        # Create feature vector from experience
        features = []
        
        # Task features
        features.extend([
            experience.task.input_dimension / 1000.0,  # Normalize
            experience.task.num_experts / 20.0,
            experience.task.num_tokens / 100.0,
            experience.task.complexity_score,
            experience.final_performance,
            experience.convergence_time / 100.0,
            experience.transfer_success_rate
        ])
        
        # Topology features
        topo_features = [
            np.mean(experience.solution_topology),
            np.std(experience.solution_topology),
            np.sum(experience.solution_topology) / experience.solution_topology.size
        ]
        features.extend(topo_features)
        
        # Pad or truncate to key_dim
        feature_vector = np.array(features)
        if len(feature_vector) < self.key_dim:
            feature_vector = np.pad(feature_vector, (0, self.key_dim - len(feature_vector)))
        else:
            feature_vector = feature_vector[:self.key_dim]
        
        return feature_vector

class GradientBasedMetaLearner:
    """Gradient-based meta-learning (MAML-inspired) for MoE topology optimization"""
    
    def __init__(self, learning_rate: float = 0.01, meta_learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        
        # Meta-parameters (simplified topology generation parameters)
        self.meta_params = {
            'sparsity_bias': np.array([0.5]),
            'connectivity_weight': np.array([1.0]),
            'expert_balance_factor': np.array([1.0]),
            'routing_temperature': np.array([1.0])
        }
        
        # Gradient buffers
        self.meta_gradients = {key: np.zeros_like(value) for key, value in self.meta_params.items()}
        
    def generate_initial_topology(self, task: TaskDefinition) -> np.ndarray:
        """Generate initial topology using meta-parameters"""
        sparsity = float(self.meta_params['sparsity_bias'][0])
        connectivity = float(self.meta_params['connectivity_weight'][0])
        
        # Create topology with learned biases
        topology = np.random.random((task.num_tokens, task.num_experts))
        
        # Apply sparsity bias
        sparsity_threshold = np.percentile(topology, (1 - sparsity) * 100)
        topology = (topology > sparsity_threshold).astype(float)
        
        # Adjust connectivity based on meta-learned weights
        if connectivity > 1.0:
            # Increase connectivity
            additional_connections = np.random.random(topology.shape) > (2.0 - connectivity)
            topology = np.logical_or(topology, additional_connections).astype(float)
        
        return topology
    
    def inner_loop_update(self, topology: np.ndarray, task: TaskDefinition, 
                         num_steps: int = 5) -> Tuple[np.ndarray, List[float]]:
        """Perform inner loop adaptation for specific task"""
        current_topology = topology.copy()
        performance_trajectory = []
        
        for step in range(num_steps):
            # Evaluate current topology
            fitness = self._evaluate_topology(current_topology, task)
            performance_trajectory.append(fitness)
            
            # Compute gradients (simplified finite difference)
            gradients = self._compute_topology_gradients(current_topology, task, fitness)
            
            # Update topology
            current_topology = self._update_topology_with_gradients(current_topology, gradients)
            
        return current_topology, performance_trajectory
    
    def outer_loop_update(self, meta_batch: List[Tuple[TaskDefinition, np.ndarray, List[float]]]):
        """Perform outer loop meta-parameter update"""
        # Reset gradient buffers
        for key in self.meta_gradients:
            self.meta_gradients[key].fill(0.0)
        
        # Accumulate meta-gradients across tasks
        for task, adapted_topology, trajectory in meta_batch:
            meta_grads = self._compute_meta_gradients(task, adapted_topology, trajectory)
            
            for key in self.meta_gradients:
                if key in meta_grads:
                    self.meta_gradients[key] += meta_grads[key] / len(meta_batch)
        
        # Update meta-parameters
        for key in self.meta_params:
            self.meta_params[key] -= self.meta_learning_rate * self.meta_gradients[key]
            
            # Clip parameters to reasonable ranges
            if key == 'sparsity_bias':
                self.meta_params[key] = np.clip(self.meta_params[key], 0.1, 0.9)
            elif key == 'routing_temperature':
                self.meta_params[key] = np.clip(self.meta_params[key], 0.1, 10.0)
    
    def _evaluate_topology(self, topology: np.ndarray, task: TaskDefinition) -> float:
        """Evaluate topology performance on task"""
        # Simulate MoE routing evaluation
        sparsity = np.mean(topology)
        connectivity = np.std(np.sum(topology, axis=1))
        
        # Task-specific evaluation
        if task.domain == "language":
            # Language tasks prefer balanced expert usage
            expert_usage = np.sum(topology, axis=0)
            balance_score = 1.0 / (1.0 + np.std(expert_usage))
            fitness = balance_score * 0.6 - sparsity * 0.4
        elif task.domain == "vision":
            # Vision tasks prefer hierarchical structures
            hierarchical_score = self._compute_hierarchical_score(topology)
            fitness = hierarchical_score * 0.7 - connectivity * 0.3
        else:
            # General tasks
            fitness = -(sparsity * 0.6 + connectivity * 0.4)
        
        # Add noise to simulate real evaluation
        fitness += np.random.normal(0, 0.01)
        
        return fitness
    
    def _compute_hierarchical_score(self, topology: np.ndarray) -> float:
        """Compute hierarchical structure score"""
        num_tokens, num_experts = topology.shape
        
        # Check for hierarchical patterns (early tokens use fewer experts)
        hierarchical_score = 0.0
        for i in range(num_tokens - 1):
            if np.sum(topology[i]) <= np.sum(topology[i + 1]):
                hierarchical_score += 1.0
        
        return hierarchical_score / max(1, num_tokens - 1)
    
    def _compute_topology_gradients(self, topology: np.ndarray, task: TaskDefinition, 
                                  current_fitness: float) -> np.ndarray:
        """Compute gradients for topology using finite differences"""
        gradients = np.zeros_like(topology)
        epsilon = 0.01
        
        # Sample a subset of positions for efficiency
        positions = [(i, j) for i in range(topology.shape[0]) for j in range(topology.shape[1])]
        sampled_positions = np.random.choice(len(positions), 
                                           size=min(20, len(positions)), 
                                           replace=False)
        
        for idx in sampled_positions:
            i, j = positions[idx]
            
            # Forward difference
            topology_plus = topology.copy()
            topology_plus[i, j] += epsilon
            fitness_plus = self._evaluate_topology(topology_plus, task)
            
            gradients[i, j] = (fitness_plus - current_fitness) / epsilon
        
        return gradients
    
    def _update_topology_with_gradients(self, topology: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update topology using gradients"""
        # Gradient ascent (maximizing fitness)
        updated_topology = topology + self.learning_rate * gradients
        
        # Apply constraints (binary topology)
        sigmoid_topology = 1.0 / (1.0 + np.exp(-updated_topology))
        binary_topology = (sigmoid_topology > 0.5).astype(float)
        
        return binary_topology
    
    def _compute_meta_gradients(self, task: TaskDefinition, adapted_topology: np.ndarray, 
                              trajectory: List[float]) -> Dict[str, np.ndarray]:
        """Compute meta-gradients for meta-parameter update"""
        meta_grads = {}
        
        # Final performance as meta-objective
        final_performance = trajectory[-1] if trajectory else 0.0
        
        # Compute gradients w.r.t. meta-parameters (simplified)
        epsilon = 0.01
        
        for key in self.meta_params:
            original_value = self.meta_params[key].copy()
            
            # Perturb meta-parameter
            self.meta_params[key] += epsilon
            
            # Generate new initial topology and evaluate
            new_initial_topology = self.generate_initial_topology(task)
            new_adapted_topology, new_trajectory = self.inner_loop_update(new_initial_topology, task)
            new_performance = new_trajectory[-1] if new_trajectory else 0.0
            
            # Compute gradient
            gradient = (new_performance - final_performance) / epsilon
            meta_grads[key] = np.array([gradient])
            
            # Restore original value
            self.meta_params[key] = original_value
        
        return meta_grads

class TransferLearningEngine:
    """Transfer learning engine for cross-domain knowledge transfer"""
    
    def __init__(self):
        self.domain_embeddings = {}
        self.transfer_matrix = {}
        self.domain_expertise = defaultdict(list)
        
    def learn_domain_embedding(self, domain: str, experiences: List[MetaLearningExperience]):
        """Learn embedding for specific domain"""
        if not experiences:
            return
        
        # Extract features from experiences
        features = []
        for exp in experiences:
            feature_vector = [
                exp.task.complexity_score,
                exp.final_performance,
                exp.convergence_time,
                np.mean(exp.solution_topology),
                np.std(exp.solution_topology),
                exp.transfer_success_rate
            ]
            features.append(feature_vector)
        
        # Compute domain embedding (mean of experience features)
        domain_embedding = np.mean(features, axis=0)
        self.domain_embeddings[domain] = domain_embedding
        
        # Store experiences for domain
        self.domain_expertise[domain].extend(experiences)
        
        logger.info(f"Learned embedding for domain '{domain}' from {len(experiences)} experiences")
    
    def compute_domain_similarity(self, source_domain: str, target_domain: str) -> float:
        """Compute similarity between domains"""
        if source_domain not in self.domain_embeddings or target_domain not in self.domain_embeddings:
            return 0.0
        
        source_emb = self.domain_embeddings[source_domain]
        target_emb = self.domain_embeddings[target_domain]
        
        # Cosine similarity
        similarity = np.dot(source_emb, target_emb) / (
            np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8
        )
        
        return float(similarity)
    
    def suggest_transfer_candidates(self, target_task: TaskDefinition, top_k: int = 5) -> List[Tuple[str, float, MetaLearningExperience]]:
        """Suggest transfer learning candidates for target task"""
        candidates = []
        
        # Compute target task features
        target_features = np.array([
            target_task.complexity_score,
            target_task.input_dimension / 1000.0,
            target_task.num_experts / 20.0,
            target_task.num_tokens / 100.0
        ])
        
        # Find similar experiences across all domains
        for domain, experiences in self.domain_expertise.items():
            for exp in experiences:
                # Compute experience features
                exp_features = np.array([
                    exp.task.complexity_score,
                    exp.task.input_dimension / 1000.0,
                    exp.task.num_experts / 20.0,
                    exp.task.num_tokens / 100.0
                ])
                
                # Compute similarity
                similarity = np.dot(target_features, exp_features) / (
                    np.linalg.norm(target_features) * np.linalg.norm(exp_features) + 1e-8
                )
                
                candidates.append((domain, float(similarity), exp))
        
        # Sort by similarity and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def transfer_knowledge(self, source_experiences: List[MetaLearningExperience], 
                          target_task: TaskDefinition) -> np.ndarray:
        """Transfer knowledge from source experiences to target task"""
        if not source_experiences:
            return np.random.random((target_task.num_tokens, target_task.num_experts)) > 0.5
        
        # Weight experiences by their performance and similarity to target
        weighted_topologies = []
        weights = []
        
        for exp in source_experiences:
            # Compute similarity weight
            task_similarity = self._compute_task_similarity(exp.task, target_task)
            performance_weight = max(0.0, exp.final_performance + 1.0)  # Shift to ensure positive
            
            combined_weight = task_similarity * performance_weight
            
            if combined_weight > 0:
                # Adapt topology size if needed
                adapted_topology = self._adapt_topology_size(
                    exp.solution_topology, 
                    (target_task.num_tokens, target_task.num_experts)
                )
                
                weighted_topologies.append(adapted_topology)
                weights.append(combined_weight)
        
        if not weighted_topologies:
            return np.random.random((target_task.num_tokens, target_task.num_experts)) > 0.5
        
        # Weighted average of topologies
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        transferred_topology = np.zeros((target_task.num_tokens, target_task.num_experts))
        for i, topology in enumerate(weighted_topologies):
            transferred_topology += weights[i] * topology
        
        # Convert to binary topology
        binary_topology = (transferred_topology > 0.5).astype(float)
        
        return binary_topology
    
    def _compute_task_similarity(self, source_task: TaskDefinition, target_task: TaskDefinition) -> float:
        """Compute similarity between tasks"""
        # Structural similarity
        size_diff = abs(source_task.num_experts - target_task.num_experts) / max(source_task.num_experts, target_task.num_experts)
        tokens_diff = abs(source_task.num_tokens - target_task.num_tokens) / max(source_task.num_tokens, target_task.num_tokens)
        
        structural_similarity = 1.0 - 0.5 * (size_diff + tokens_diff)
        
        # Domain similarity
        domain_similarity = 1.0 if source_task.domain == target_task.domain else 0.5
        
        # Complexity similarity
        complexity_diff = abs(source_task.complexity_score - target_task.complexity_score)
        complexity_similarity = 1.0 / (1.0 + complexity_diff)
        
        # Combined similarity
        total_similarity = (structural_similarity * 0.4 + 
                          domain_similarity * 0.3 + 
                          complexity_similarity * 0.3)
        
        return total_similarity
    
    def _adapt_topology_size(self, source_topology: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Adapt topology to target size"""
        source_tokens, source_experts = source_topology.shape
        target_tokens, target_experts = target_shape
        
        # Create target topology
        target_topology = np.zeros(target_shape)
        
        # Copy overlapping region
        copy_tokens = min(source_tokens, target_tokens)
        copy_experts = min(source_experts, target_experts)
        
        target_topology[:copy_tokens, :copy_experts] = source_topology[:copy_tokens, :copy_experts]
        
        # Fill remaining regions
        if target_tokens > source_tokens:
            # Repeat pattern for additional tokens
            for i in range(source_tokens, target_tokens):
                pattern_idx = i % source_tokens
                target_topology[i, :copy_experts] = source_topology[pattern_idx, :copy_experts]
        
        if target_experts > source_experts:
            # Extend expert dimension
            for j in range(source_experts, target_experts):
                pattern_idx = j % source_experts
                target_topology[:copy_tokens, j] = source_topology[:copy_tokens, pattern_idx]
        
        return target_topology

class ContinualLearningEngine:
    """Continual learning with catastrophic forgetting prevention"""
    
    def __init__(self, memory_capacity: int = 500):
        self.memory_capacity = memory_capacity
        self.episodic_memory = deque(maxlen=memory_capacity)
        self.task_boundaries = []
        self.importance_weights = {}
        
    def add_experience(self, experience: MetaLearningExperience):
        """Add experience to episodic memory"""
        self.episodic_memory.append(experience)
        
        # Update importance weights using Fisher Information approximation
        self._update_importance_weights(experience)
    
    def _update_importance_weights(self, experience: MetaLearningExperience):
        """Update importance weights for parameters"""
        task_id = experience.task.task_id
        
        if task_id not in self.importance_weights:
            self.importance_weights[task_id] = {}
        
        # Simplified Fisher Information: use gradient magnitude as importance
        topology_variance = np.var(experience.solution_topology)
        performance_sensitivity = abs(experience.final_performance)
        
        self.importance_weights[task_id]['topology_importance'] = topology_variance
        self.importance_weights[task_id]['performance_sensitivity'] = performance_sensitivity
    
    def prevent_catastrophic_forgetting(self, current_task: TaskDefinition, 
                                      new_experience: MetaLearningExperience) -> float:
        """Compute regularization penalty to prevent catastrophic forgetting"""
        if not self.episodic_memory:
            return 0.0
        
        # Compute forgetting penalty based on previous important tasks
        forgetting_penalty = 0.0
        
        for prev_experience in list(self.episodic_memory)[-50:]:  # Recent memories
            prev_task_id = prev_experience.task.task_id
            
            if prev_task_id in self.importance_weights:
                # Compute parameter change penalty
                topology_diff = np.mean(np.abs(
                    new_experience.solution_topology - prev_experience.solution_topology
                ))
                
                importance = self.importance_weights[prev_task_id]['topology_importance']
                
                forgetting_penalty += importance * topology_diff
        
        return forgetting_penalty / max(1, len(self.episodic_memory))
    
    def replay_buffer_sample(self, batch_size: int = 10) -> List[MetaLearningExperience]:
        """Sample experiences for replay"""
        if len(self.episodic_memory) <= batch_size:
            return list(self.episodic_memory)
        
        # Sample with bias towards high-performance experiences
        experiences = list(self.episodic_memory)
        performances = [exp.final_performance for exp in experiences]
        
        # Convert to probabilities (higher performance = higher probability)
        min_perf = min(performances)
        shifted_perfs = [p - min_perf + 1e-6 for p in performances]
        probabilities = np.array(shifted_perfs) / sum(shifted_perfs)
        
        indices = np.random.choice(len(experiences), size=batch_size, 
                                 replace=False, p=probabilities)
        
        return [experiences[i] for i in indices]

class MetaLearningTransferEngine:
    """Main meta-learning and transfer evolution engine"""
    
    def __init__(self):
        self.memory_network = MemoryAugmentedNetwork()
        self.meta_learner = GradientBasedMetaLearner()
        self.transfer_engine = TransferLearningEngine()
        self.continual_engine = ContinualLearningEngine()
        
        # Experience storage
        self.all_experiences: List[MetaLearningExperience] = []
        self.task_history: List[TaskDefinition] = []
        
        # Meta-learning statistics
        self.meta_learning_stats = {
            'total_tasks': 0,
            'successful_transfers': 0,
            'average_adaptation_time': 0.0,
            'domain_coverage': set()
        }
    
    def create_task_definition(self, task_type: str, domain: str, num_experts: int, 
                             num_tokens: int, complexity_score: float) -> TaskDefinition:
        """Create task definition"""
        task = TaskDefinition(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            domain=domain,
            input_dimension=64 * num_tokens,  # Assume 64-dim embeddings
            num_experts=num_experts,
            num_tokens=num_tokens,
            complexity_score=complexity_score,
            task_specific_constraints={},
            performance_requirements={"min_fitness": -0.5}
        )
        
        self.task_history.append(task)
        self.meta_learning_stats['domain_coverage'].add(domain)
        
        return task
    
    def few_shot_adaptation(self, task: TaskDefinition, support_experiences: List[MetaLearningExperience], 
                          num_adaptation_steps: int = 10) -> MetaLearningExperience:
        """Perform few-shot learning adaptation"""
        adaptation_start = time.time()
        
        # Create few-shot learning context
        context = FewShotLearningContext(
            support_set=support_experiences,
            query_task=task,
            initial_adaptation_rate=0.01,
            num_adaptation_steps=num_adaptation_steps,
            meta_gradients=None
        )
        
        # Query memory network for similar experiences
        task_encoding = self.memory_network.encode_experience(
            MetaLearningExperience(
                experience_id="temp",
                task=task,
                solution_topology=np.zeros((task.num_tokens, task.num_experts)),
                adaptation_steps=[],
                performance_trajectory=[],
                convergence_time=0,
                final_performance=0,
                meta_features={},
                transfer_success_rate=0,
                timestamp=time.time()
            )
        )
        
        memory_response = self.memory_network.read_memory(task_encoding)
        
        # Generate initial topology using meta-learning
        if support_experiences:
            initial_topology = self.transfer_engine.transfer_knowledge(support_experiences, task)
        else:
            initial_topology = self.meta_learner.generate_initial_topology(task)
        
        # Perform inner loop adaptation
        adapted_topology, trajectory = self.meta_learner.inner_loop_update(
            initial_topology, task, num_adaptation_steps
        )
        
        # Create experience record
        adaptation_time = time.time() - adaptation_start
        
        # Compute meta-features
        meta_features = {
            'initial_sparsity': np.mean(initial_topology),
            'final_sparsity': np.mean(adapted_topology),
            'topology_change': np.mean(np.abs(adapted_topology - initial_topology)),
            'performance_improvement': trajectory[-1] - trajectory[0] if len(trajectory) > 1 else 0,
            'adaptation_efficiency': (trajectory[-1] - trajectory[0]) / adaptation_time if adaptation_time > 0 else 0
        }
        
        # Compute transfer success rate
        transfer_success_rate = 1.0 if trajectory[-1] > task.performance_requirements.get("min_fitness", -1.0) else 0.0
        
        experience = MetaLearningExperience(
            experience_id=str(uuid.uuid4()),
            task=task,
            solution_topology=adapted_topology,
            adaptation_steps=[{"step": i, "fitness": f} for i, f in enumerate(trajectory)],
            performance_trajectory=trajectory,
            convergence_time=adaptation_time,
            final_performance=trajectory[-1] if trajectory else -float('inf'),
            meta_features=meta_features,
            transfer_success_rate=transfer_success_rate,
            timestamp=time.time()
        )
        
        # Store experience
        self.all_experiences.append(experience)
        self.continual_engine.add_experience(experience)
        
        # Update memory network
        experience_encoding = self.memory_network.encode_experience(experience)
        self.memory_network.write_memory(task_encoding, experience_encoding)
        
        # Update statistics
        self.meta_learning_stats['total_tasks'] += 1
        if transfer_success_rate > 0.5:
            self.meta_learning_stats['successful_transfers'] += 1
        
        self.meta_learning_stats['average_adaptation_time'] = (
            (self.meta_learning_stats['average_adaptation_time'] * (self.meta_learning_stats['total_tasks'] - 1) + 
             adaptation_time) / self.meta_learning_stats['total_tasks']
        )
        
        logger.info(f"Few-shot adaptation complete: Task {task.task_id[:8]}, "
                   f"Performance {trajectory[-1]:.4f}, Time {adaptation_time:.3f}s")
        
        return experience
    
    def meta_update_cycle(self, batch_size: int = 5):
        """Perform meta-learning update cycle"""
        if len(self.all_experiences) < batch_size:
            return
        
        logger.info("Performing meta-learning update...")
        
        # Sample meta-batch
        recent_experiences = self.all_experiences[-20:]  # Recent experiences
        meta_batch = []
        
        for exp in recent_experiences[-batch_size:]:
            meta_batch.append((exp.task, exp.solution_topology, exp.performance_trajectory))
        
        # Update meta-learner
        self.meta_learner.outer_loop_update(meta_batch)
        
        # Update transfer learning models
        domain_experiences = defaultdict(list)
        for exp in recent_experiences:
            domain_experiences[exp.task.domain].append(exp)
        
        for domain, experiences in domain_experiences.items():
            self.transfer_engine.learn_domain_embedding(domain, experiences)
    
    def cross_domain_transfer_experiment(self, source_domain: str, target_domain: str, 
                                       num_tasks: int = 10) -> Dict:
        """Run cross-domain transfer learning experiment"""
        logger.info(f"Running cross-domain transfer: {source_domain} -> {target_domain}")
        
        experiment_results = {
            'source_domain': source_domain,
            'target_domain': target_domain,
            'num_tasks': num_tasks,
            'task_results': [],
            'transfer_statistics': {}
        }
        
        domains = [source_domain, target_domain]
        complexities = [0.3, 0.5, 0.7, 0.9]
        
        for task_idx in range(num_tasks):
            # Create task (alternate between domains)
            current_domain = domains[task_idx % len(domains)]
            complexity = np.random.choice(complexities)
            
            task = self.create_task_definition(
                task_type="moe_routing",
                domain=current_domain,
                num_experts=np.random.randint(6, 14),
                num_tokens=np.random.randint(8, 20),
                complexity_score=complexity
            )
            
            # Get transfer candidates
            if current_domain == target_domain and len(self.all_experiences) > 0:
                # Use transfer learning for target domain tasks
                candidates = self.transfer_engine.suggest_transfer_candidates(task, top_k=3)
                support_experiences = [candidate[2] for candidate in candidates if candidate[1] > 0.3]
            else:
                # For source domain, use fewer support examples
                support_experiences = []
            
            # Perform few-shot adaptation
            experience = self.few_shot_adaptation(task, support_experiences)
            
            experiment_results['task_results'].append({
                'task_id': task.task_id,
                'domain': current_domain,
                'complexity': complexity,
                'final_performance': experience.final_performance,
                'adaptation_time': experience.convergence_time,
                'transfer_success': experience.transfer_success_rate,
                'num_support_examples': len(support_experiences)
            })
            
            # Periodic meta-updates
            if (task_idx + 1) % 5 == 0:
                self.meta_update_cycle()
        
        # Compute transfer statistics
        source_results = [r for r in experiment_results['task_results'] if r['domain'] == source_domain]
        target_results = [r for r in experiment_results['task_results'] if r['domain'] == target_domain]
        
        if source_results and target_results:
            source_performance = np.mean([r['final_performance'] for r in source_results])
            target_performance = np.mean([r['final_performance'] for r in target_results])
            
            experiment_results['transfer_statistics'] = {
                'source_avg_performance': source_performance,
                'target_avg_performance': target_performance,
                'transfer_improvement': target_performance - source_performance,
                'avg_adaptation_time': np.mean([r['adaptation_time'] for r in experiment_results['task_results']]),
                'transfer_success_rate': np.mean([r['transfer_success'] for r in target_results])
            }
        
        return experiment_results
    
    def run_comprehensive_meta_learning_study(self) -> Dict:
        """Run comprehensive meta-learning study"""
        logger.info("Starting comprehensive meta-learning study...")
        
        study_start = time.time()
        study_results = {
            'study_id': str(uuid.uuid4()),
            'start_time': study_start,
            'experiments': [],
            'meta_learning_progression': [],
            'final_analysis': {}
        }
        
        # Define experimental conditions
        domain_pairs = [
            ("language", "vision"),
            ("vision", "language"),
            ("language", "speech"),
            ("speech", "vision"),
            ("vision", "general"),
            ("general", "language")
        ]
        
        # Run transfer experiments
        for source_domain, target_domain in domain_pairs:
            experiment_results = self.cross_domain_transfer_experiment(
                source_domain, target_domain, num_tasks=15
            )
            study_results['experiments'].append(experiment_results)
            
            # Track meta-learning progression
            study_results['meta_learning_progression'].append({
                'timestamp': time.time(),
                'total_experiences': len(self.all_experiences),
                'meta_learning_stats': copy.deepcopy(self.meta_learning_stats),
                'domain_similarities': {
                    f"{source_domain}->{target_domain}": 
                    self.transfer_engine.compute_domain_similarity(source_domain, target_domain)
                }
            })
        
        # Final analysis
        study_end = time.time()
        total_time = study_end - study_start
        
        all_task_results = []
        for exp in study_results['experiments']:
            all_task_results.extend(exp['task_results'])
        
        study_results['final_analysis'] = {
            'total_study_time': total_time,
            'total_tasks_completed': len(all_task_results),
            'overall_performance_distribution': {
                'mean': np.mean([r['final_performance'] for r in all_task_results]),
                'std': np.std([r['final_performance'] for r in all_task_results]),
                'min': min([r['final_performance'] for r in all_task_results]),
                'max': max([r['final_performance'] for r in all_task_results])
            },
            'adaptation_time_analysis': {
                'mean': np.mean([r['adaptation_time'] for r in all_task_results]),
                'median': np.median([r['adaptation_time'] for r in all_task_results]),
                'percentile_90': np.percentile([r['adaptation_time'] for r in all_task_results], 90)
            },
            'transfer_effectiveness': {
                'overall_success_rate': np.mean([r['transfer_success'] for r in all_task_results]),
                'by_domain': self._analyze_transfer_by_domain(study_results['experiments'])
            },
            'meta_learning_convergence': self._analyze_meta_learning_convergence(),
            'memory_network_utilization': {
                'memory_usage': np.mean(self.memory_network.memory_usage),
                'total_experiences_stored': len(self.all_experiences)
            }
        }
        
        # Save results
        results_file = f"/root/repo/meta_learning_study_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(study_results, f, indent=2, default=str)
        
        logger.info(f"Meta-learning study complete! Results saved to {results_file}")
        logger.info(f"Completed {len(all_task_results)} tasks in {total_time:.2f}s")
        logger.info(f"Overall success rate: {study_results['final_analysis']['transfer_effectiveness']['overall_success_rate']:.2%}")
        
        return study_results
    
    def _analyze_transfer_by_domain(self, experiments: List[Dict]) -> Dict:
        """Analyze transfer effectiveness by domain"""
        domain_analysis = {}
        
        for exp in experiments:
            domain_pair = f"{exp['source_domain']}->{exp['target_domain']}"
            
            target_results = [r for r in exp['task_results'] if r['domain'] == exp['target_domain']]
            
            if target_results:
                domain_analysis[domain_pair] = {
                    'avg_performance': np.mean([r['final_performance'] for r in target_results]),
                    'success_rate': np.mean([r['transfer_success'] for r in target_results]),
                    'avg_adaptation_time': np.mean([r['adaptation_time'] for r in target_results])
                }
        
        return domain_analysis
    
    def _analyze_meta_learning_convergence(self) -> Dict:
        """Analyze meta-learning convergence"""
        if len(self.all_experiences) < 10:
            return {"insufficient_data": True}
        
        # Analyze performance improvement over time
        recent_experiences = self.all_experiences[-20:]
        early_experiences = self.all_experiences[:20] if len(self.all_experiences) > 20 else []
        
        if early_experiences:
            early_performance = np.mean([exp.final_performance for exp in early_experiences])
            recent_performance = np.mean([exp.final_performance for exp in recent_experiences])
            
            convergence_analysis = {
                "performance_improvement": recent_performance - early_performance,
                "early_avg_performance": early_performance,
                "recent_avg_performance": recent_performance,
                "learning_trend": "improving" if recent_performance > early_performance else "declining"
            }
        else:
            convergence_analysis = {"insufficient_comparison_data": True}
        
        # Analyze adaptation time improvement
        adaptation_times = [exp.convergence_time for exp in self.all_experiences]
        if len(adaptation_times) > 5:
            recent_times = adaptation_times[-10:]
            early_times = adaptation_times[:10] if len(adaptation_times) > 10 else adaptation_times
            
            convergence_analysis["adaptation_time_improvement"] = {
                "early_avg_time": np.mean(early_times),
                "recent_avg_time": np.mean(recent_times),
                "time_reduction": np.mean(early_times) - np.mean(recent_times)
            }
        
        return convergence_analysis

async def main():
    """Meta-Learning Transfer Engine Main Execution"""
    print("🧠 TERRAGON V6.0 - Advanced Meta-Learning & Transfer Evolution Engine")
    print("=" * 80)
    
    # Initialize meta-learning system
    meta_engine = MetaLearningTransferEngine()
    
    # Run comprehensive study
    study_results = meta_engine.run_comprehensive_meta_learning_study()
    
    # Display results
    print("\n🎯 META-LEARNING STUDY RESULTS:")
    final_analysis = study_results['final_analysis']
    
    print(f"Total Tasks: {final_analysis['total_tasks_completed']}")
    print(f"Study Time: {final_analysis['total_study_time']:.2f}s")
    print(f"Overall Success Rate: {final_analysis['transfer_effectiveness']['overall_success_rate']:.2%}")
    
    perf_dist = final_analysis['overall_performance_distribution']
    print(f"Performance - Mean: {perf_dist['mean']:.4f}, Std: {perf_dist['std']:.4f}")
    
    adaptation = final_analysis['adaptation_time_analysis']
    print(f"Adaptation Time - Mean: {adaptation['mean']:.3f}s, Median: {adaptation['median']:.3f}s")
    
    # Transfer effectiveness by domain
    print("\n🔄 TRANSFER EFFECTIVENESS BY DOMAIN:")
    transfer_by_domain = final_analysis['transfer_effectiveness']['by_domain']
    for domain_pair, stats in transfer_by_domain.items():
        print(f"{domain_pair}: Success {stats['success_rate']:.2%}, "
              f"Performance {stats['avg_performance']:.4f}")
    
    # Meta-learning convergence
    convergence = final_analysis.get('meta_learning_convergence', {})
    if 'performance_improvement' in convergence:
        print(f"\n📈 META-LEARNING CONVERGENCE:")
        print(f"Performance Improvement: {convergence['performance_improvement']:.4f}")
        print(f"Learning Trend: {convergence['learning_trend']}")
        
        if 'adaptation_time_improvement' in convergence:
            time_improve = convergence['adaptation_time_improvement']
            print(f"Time Reduction: {time_improve['time_reduction']:.3f}s")
    
    print("\n✅ META-LEARNING & TRANSFER EVOLUTION COMPLETE")
    return study_results

if __name__ == "__main__":
    results = asyncio.run(main())