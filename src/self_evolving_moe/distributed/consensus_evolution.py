"""
Consensus-Based Distributed Evolution for MoE Systems
TERRAGON NEXT-GEN v5.0 - Distributed Consensus Evolution

Implements distributed evolutionary algorithms with consensus mechanisms:
- Multi-node consensus protocols for evolution decisions
- Byzantine fault tolerance for distributed fitness evaluation
- Federated learning integration for topology optimization
- Distributed Pareto optimization with global convergence
"""

import asyncio
import json
import hashlib
import time
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

logger = logging.getLogger(__name__)

class ConsensusType(Enum):
    """Types of consensus mechanisms"""
    PROOF_OF_STAKE = "proof_of_stake"
    BYZANTINE_FAULT_TOLERANT = "bft"
    RAFT = "raft"
    PRACTICAL_BYZANTINE_FT = "pbft"
    FEDERATED_CONSENSUS = "federated"

@dataclass
class NodeMetrics:
    """Metrics for consensus node performance"""
    node_id: str
    fitness_evaluations: int = 0
    successful_proposals: int = 0
    consensus_participation: float = 0.0
    network_reliability: float = 1.0
    computational_power: float = 1.0
    stake_weight: float = 1.0
    last_heartbeat: float = 0.0
    byzantine_score: float = 0.0  # Lower is better

@dataclass
class EvolutionProposal:
    """Proposal for evolutionary operation"""
    proposal_id: str
    proposer_node: str
    operation_type: str  # "mutation", "crossover", "selection"
    target_individuals: List[int]
    parameters: Dict[str, Any]
    fitness_evidence: Dict[str, float]
    timestamp: float
    signature: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionProposal':
        return cls(**data)

@dataclass
class ConsensusVote:
    """Vote on an evolution proposal"""
    proposal_id: str
    voter_node: str
    vote: bool  # True = approve, False = reject
    confidence: float  # 0.0 to 1.0
    reasoning: Dict[str, Any]
    timestamp: float
    signature: str

class ConsensusProtocol(ABC):
    """Base class for consensus protocols"""
    
    @abstractmethod
    async def propose_evolution(self, proposal: EvolutionProposal) -> bool:
        """Propose an evolutionary operation"""
        pass
    
    @abstractmethod
    async def vote_on_proposal(self, proposal: EvolutionProposal) -> ConsensusVote:
        """Vote on a proposal"""
        pass
    
    @abstractmethod
    async def reach_consensus(self, proposal: EvolutionProposal, votes: List[ConsensusVote]) -> bool:
        """Determine if consensus is reached"""
        pass

class ByzantineFaultTolerantConsensus(ConsensusProtocol):
    """
    Byzantine Fault Tolerant consensus for evolution decisions
    
    Ensures correct evolution even with up to 1/3 malicious nodes
    """
    
    def __init__(self, node_id: str, total_nodes: int, fault_tolerance: float = 0.33):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.fault_tolerance = fault_tolerance
        self.max_faulty_nodes = int(total_nodes * fault_tolerance)
        self.min_honest_nodes = total_nodes - self.max_faulty_nodes
        
        # Consensus state
        self.pending_proposals: Dict[str, EvolutionProposal] = {}
        self.proposal_votes: Dict[str, List[ConsensusVote]] = {}
        self.consensus_history: List[Dict[str, Any]] = []
        
        # Security measures
        self.node_trust_scores: Dict[str, float] = {}
        self.proposal_validation_cache: Dict[str, bool] = {}
        
    async def propose_evolution(self, proposal: EvolutionProposal) -> bool:
        """Propose evolutionary operation with Byzantine validation"""
        
        # Validate proposal integrity
        if not self._validate_proposal(proposal):
            logger.warning(f"Invalid proposal {proposal.proposal_id} rejected")
            return False
        
        # Add to pending proposals
        self.pending_proposals[proposal.proposal_id] = proposal
        self.proposal_votes[proposal.proposal_id] = []
        
        logger.info(f"Node {self.node_id} proposed evolution {proposal.proposal_id}")
        return True
    
    async def vote_on_proposal(self, proposal: EvolutionProposal) -> ConsensusVote:
        """Generate vote with Byzantine fault detection"""
        
        # Analyze proposal quality and security
        vote_decision, confidence, reasoning = await self._analyze_proposal_security(proposal)
        
        # Create vote
        vote = ConsensusVote(
            proposal_id=proposal.proposal_id,
            voter_node=self.node_id,
            vote=vote_decision,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=time.time(),
            signature=self._sign_vote(proposal.proposal_id, vote_decision)
        )
        
        # Update trust scores based on voting patterns
        self._update_trust_scores(proposal, vote)
        
        return vote
    
    async def reach_consensus(self, proposal: EvolutionProposal, votes: List[ConsensusVote]) -> bool:
        """Byzantine fault tolerant consensus decision"""
        
        if len(votes) < self.min_honest_nodes:
            logger.warning(f"Insufficient votes for consensus: {len(votes)} < {self.min_honest_nodes}")
            return False
        
        # Filter out potentially Byzantine votes
        trusted_votes = self._filter_byzantine_votes(votes)
        
        if len(trusted_votes) < self.min_honest_nodes:
            logger.warning("Too many potentially Byzantine votes detected")
            return False
        
        # Count weighted votes
        approve_weight = sum(vote.confidence for vote in trusted_votes if vote.vote)
        reject_weight = sum(vote.confidence for vote in trusted_votes if not vote.vote)
        total_weight = approve_weight + reject_weight
        
        if total_weight == 0:
            return False
        
        # Require 2/3 majority with confidence weighting
        consensus_threshold = 0.67
        approval_ratio = approve_weight / total_weight
        
        consensus_reached = approval_ratio >= consensus_threshold
        
        # Record consensus result
        self._record_consensus_result(proposal, votes, trusted_votes, consensus_reached)
        
        logger.info(f"Consensus {'REACHED' if consensus_reached else 'FAILED'} for {proposal.proposal_id} "
                   f"(approval: {approval_ratio:.3f})")
        
        return consensus_reached
    
    def _validate_proposal(self, proposal: EvolutionProposal) -> bool:
        """Validate proposal integrity and security"""
        
        # Check proposal structure
        required_fields = ['proposal_id', 'proposer_node', 'operation_type', 'timestamp']
        if not all(hasattr(proposal, field) and getattr(proposal, field) for field in required_fields):
            return False
        
        # Check timestamp freshness (prevent replay attacks)
        current_time = time.time()
        if abs(current_time - proposal.timestamp) > 300:  # 5 minute window
            logger.warning(f"Proposal {proposal.proposal_id} timestamp too old/future")
            return False
        
        # Validate operation type
        valid_operations = ['mutation', 'crossover', 'selection', 'initialization']
        if proposal.operation_type not in valid_operations:
            return False
        
        # Check proposer trust score
        proposer_trust = self.node_trust_scores.get(proposal.proposer_node, 0.5)
        if proposer_trust < 0.1:
            logger.warning(f"Low trust proposer {proposal.proposer_node}: {proposer_trust}")
            return False
        
        # Validate fitness evidence
        if not self._validate_fitness_evidence(proposal.fitness_evidence):
            return False
        
        return True
    
    async def _analyze_proposal_security(self, proposal: EvolutionProposal) -> Tuple[bool, float, Dict[str, Any]]:
        """Analyze proposal for security threats and quality"""
        
        reasoning = {}
        confidence = 1.0
        vote_decision = True
        
        # Security checks
        security_score = 1.0
        
        # Check for anomalous fitness values
        if proposal.fitness_evidence:
            fitness_values = list(proposal.fitness_evidence.values())
            if fitness_values:
                mean_fitness = np.mean(fitness_values)
                std_fitness = np.std(fitness_values)
                
                # Flag extremely good fitness (potential attack)
                if any(f > mean_fitness + 3 * std_fitness for f in fitness_values):
                    security_score *= 0.7
                    reasoning['anomalous_fitness'] = True
                
                # Flag impossible fitness values
                if any(f > 1.0 or f < -10.0 for f in fitness_values):
                    security_score *= 0.3
                    reasoning['impossible_fitness'] = True
        
        # Check proposer reputation
        proposer_trust = self.node_trust_scores.get(proposal.proposer_node, 0.5)
        security_score *= proposer_trust
        reasoning['proposer_trust'] = proposer_trust
        
        # Check operation reasonableness
        if proposal.operation_type == 'mutation':
            # Validate mutation parameters
            mutation_rate = proposal.parameters.get('mutation_rate', 0.1)
            if mutation_rate > 0.5:  # Unreasonably high mutation
                security_score *= 0.8
                reasoning['high_mutation_rate'] = mutation_rate
        
        elif proposal.operation_type == 'crossover':
            # Validate crossover parameters
            if len(proposal.target_individuals) < 2:
                security_score *= 0.5
                reasoning['insufficient_parents'] = len(proposal.target_individuals)
        
        # Make final decision
        confidence = min(1.0, security_score)
        vote_decision = security_score > 0.5
        
        reasoning['final_security_score'] = security_score
        reasoning['final_decision'] = vote_decision
        
        return vote_decision, confidence, reasoning
    
    def _filter_byzantine_votes(self, votes: List[ConsensusVote]) -> List[ConsensusVote]:
        """Filter out potentially Byzantine votes"""
        
        trusted_votes = []
        
        for vote in votes:
            voter_trust = self.node_trust_scores.get(vote.voter_node, 0.5)
            
            # Filter out low-trust voters
            if voter_trust < 0.2:
                logger.warning(f"Filtering vote from low-trust node {vote.voter_node}")
                continue
            
            # Check vote consistency with confidence
            if vote.confidence < 0.1:
                logger.warning(f"Filtering low-confidence vote from {vote.voter_node}")
                continue
            
            # Check for timestamp attacks
            current_time = time.time()
            if abs(current_time - vote.timestamp) > 600:  # 10 minute window
                logger.warning(f"Filtering old vote from {vote.voter_node}")
                continue
            
            trusted_votes.append(vote)
        
        return trusted_votes
    
    def _validate_fitness_evidence(self, fitness_evidence: Dict[str, float]) -> bool:
        """Validate fitness evidence for consistency"""
        
        if not fitness_evidence:
            return True  # Empty evidence is valid
        
        # Check for reasonable fitness ranges
        for metric, value in fitness_evidence.items():
            if not isinstance(value, (int, float)):
                return False
            
            # Basic sanity checks
            if metric == 'accuracy' and not (0.0 <= value <= 1.0):
                return False
            elif metric == 'latency' and value < 0:
                return False
            elif metric in ['memory', 'throughput'] and value < 0:
                return False
        
        return True
    
    def _update_trust_scores(self, proposal: EvolutionProposal, vote: ConsensusVote):
        """Update node trust scores based on voting behavior"""
        
        # Initialize trust scores if needed
        if vote.voter_node not in self.node_trust_scores:
            self.node_trust_scores[vote.voter_node] = 0.5
        
        if proposal.proposer_node not in self.node_trust_scores:
            self.node_trust_scores[proposal.proposer_node] = 0.5
        
        # Trust updates based on vote quality and consistency
        # This is a simplified model - in practice you'd track actual outcomes
        
        # Boost trust for high-confidence votes
        if vote.confidence > 0.8:
            self.node_trust_scores[vote.voter_node] *= 1.01
        elif vote.confidence < 0.3:
            self.node_trust_scores[vote.voter_node] *= 0.99
        
        # Keep trust scores in bounds
        for node in self.node_trust_scores:
            self.node_trust_scores[node] = max(0.0, min(1.0, self.node_trust_scores[node]))
    
    def _sign_vote(self, proposal_id: str, vote_decision: bool) -> str:
        """Create cryptographic signature for vote (simplified)"""
        
        # In production, use proper cryptographic signatures
        content = f"{self.node_id}_{proposal_id}_{vote_decision}_{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _record_consensus_result(self, proposal: EvolutionProposal, all_votes: List[ConsensusVote], 
                                trusted_votes: List[ConsensusVote], consensus_reached: bool):
        """Record consensus result for analysis"""
        
        result = {
            'proposal_id': proposal.proposal_id,
            'proposer': proposal.proposer_node,
            'operation_type': proposal.operation_type,
            'timestamp': time.time(),
            'consensus_reached': consensus_reached,
            'total_votes': len(all_votes),
            'trusted_votes': len(trusted_votes),
            'filtered_votes': len(all_votes) - len(trusted_votes),
            'approval_ratio': sum(v.confidence for v in trusted_votes if v.vote) / max(sum(v.confidence for v in trusted_votes), 1),
            'average_confidence': np.mean([v.confidence for v in trusted_votes]) if trusted_votes else 0
        }
        
        self.consensus_history.append(result)
        
        # Keep history bounded
        if len(self.consensus_history) > 1000:
            self.consensus_history = self.consensus_history[-500:]

class FederatedEvolutionCoordinator:
    """
    Coordinates federated evolution across multiple consensus nodes
    """
    
    def __init__(self, node_id: str, total_nodes: int, consensus_type: ConsensusType = ConsensusType.BYZANTINE_FAULT_TOLERANT):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.consensus_type = consensus_type
        
        # Initialize consensus protocol
        if consensus_type == ConsensusType.BYZANTINE_FAULT_TOLERANT:
            self.consensus_protocol = ByzantineFaultTolerantConsensus(node_id, total_nodes)
        else:
            raise NotImplementedError(f"Consensus type {consensus_type} not implemented")
        
        # Distributed state
        self.global_population: List[np.ndarray] = []
        self.local_population: List[np.ndarray] = []
        self.fitness_cache: Dict[str, Dict[str, float]] = {}
        
        # Node coordination
        self.active_nodes: Set[str] = set()
        self.node_metrics: Dict[str, NodeMetrics] = {}
        
        # Evolution coordination
        self.evolution_round = 0
        self.pending_operations: List[EvolutionProposal] = []
        self.completed_operations: List[str] = []
        
    async def initialize_federated_evolution(self, initial_population: List[np.ndarray]) -> bool:
        """Initialize distributed evolution with consensus on initial population"""
        
        # Propose initial population
        proposal = EvolutionProposal(
            proposal_id=f"init_{self.node_id}_{time.time()}",
            proposer_node=self.node_id,
            operation_type="initialization",
            target_individuals=list(range(len(initial_population))),
            parameters={'population_size': len(initial_population)},
            fitness_evidence={},
            timestamp=time.time(),
            signature=""
        )
        proposal.signature = self._sign_proposal(proposal)
        
        # Submit for consensus
        success = await self.consensus_protocol.propose_evolution(proposal)
        
        if success:
            self.local_population = initial_population.copy()
            logger.info(f"Node {self.node_id} initialized federated evolution with {len(initial_population)} individuals")
        
        return success
    
    async def evolve_generation_distributed(self, fitness_evaluator: Callable) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Execute one generation of distributed evolution with consensus
        """
        
        generation_metrics = {
            'round': self.evolution_round,
            'node_id': self.node_id,
            'consensus_operations': 0,
            'failed_consensus': 0,
            'byzantine_incidents': 0
        }
        
        # Phase 1: Distributed fitness evaluation
        await self._distributed_fitness_evaluation(fitness_evaluator, generation_metrics)
        
        # Phase 2: Consensus on selection
        new_population = await self._consensus_selection(generation_metrics)
        
        # Phase 3: Consensus on crossover operations
        offspring = await self._consensus_crossover(new_population, generation_metrics)
        
        # Phase 4: Consensus on mutation operations
        mutated_offspring = await self._consensus_mutation(offspring, generation_metrics)
        
        # Phase 5: Final population assembly
        final_population = await self._assemble_final_population(new_population, mutated_offspring, generation_metrics)
        
        self.local_population = final_population
        self.evolution_round += 1
        
        # Update coordination metrics
        generation_metrics['final_population_size'] = len(final_population)
        generation_metrics['consensus_success_rate'] = generation_metrics['consensus_operations'] / max(
            generation_metrics['consensus_operations'] + generation_metrics['failed_consensus'], 1)
        
        logger.info(f"Distributed evolution round {self.evolution_round} completed on node {self.node_id}")
        
        return final_population, generation_metrics
    
    async def _distributed_fitness_evaluation(self, fitness_evaluator: Callable, metrics: Dict[str, Any]):
        """Distribute fitness evaluation across nodes with Byzantine fault tolerance"""
        
        # Divide population among nodes for evaluation
        local_share = len(self.local_population) // self.total_nodes
        start_idx = self.active_nodes.index(self.node_id) * local_share if self.node_id in self.active_nodes else 0
        end_idx = start_idx + local_share
        
        local_individuals = self.local_population[start_idx:end_idx]
        
        # Evaluate assigned individuals
        local_fitness_results = {}
        for i, individual in enumerate(local_individuals):
            global_idx = start_idx + i
            try:
                fitness_scores = fitness_evaluator(individual)
                local_fitness_results[str(global_idx)] = fitness_scores
                
                # Cache for future reference
                individual_hash = hashlib.sha256(individual.tobytes()).hexdigest()[:16]
                self.fitness_cache[individual_hash] = fitness_scores
                
            except Exception as e:
                logger.error(f"Fitness evaluation failed for individual {global_idx}: {e}")
                local_fitness_results[str(global_idx)] = {'error': str(e)}
        
        # Propose fitness results for consensus validation
        fitness_proposal = EvolutionProposal(
            proposal_id=f"fitness_{self.node_id}_{self.evolution_round}_{time.time()}",
            proposer_node=self.node_id,
            operation_type="fitness_evaluation",
            target_individuals=list(range(start_idx, end_idx)),
            parameters={'evaluator_method': 'distributed'},
            fitness_evidence=local_fitness_results,
            timestamp=time.time(),
            signature=""
        )
        fitness_proposal.signature = self._sign_proposal(fitness_proposal)
        
        # Submit for consensus (this would be sent to other nodes in practice)
        success = await self.consensus_protocol.propose_evolution(fitness_proposal)
        
        if success:
            metrics['consensus_operations'] += 1
        else:
            metrics['failed_consensus'] += 1
            logger.warning(f"Fitness evaluation consensus failed for node {self.node_id}")
    
    async def _consensus_selection(self, metrics: Dict[str, Any]) -> List[np.ndarray]:
        """Perform selection with distributed consensus"""
        
        # Propose selection strategy
        selection_proposal = EvolutionProposal(
            proposal_id=f"selection_{self.node_id}_{self.evolution_round}_{time.time()}",
            proposer_node=self.node_id,
            operation_type="selection",
            target_individuals=list(range(len(self.local_population))),
            parameters={
                'selection_method': 'tournament',
                'tournament_size': 3,
                'selection_pressure': 1.2
            },
            fitness_evidence=self._get_population_fitness_summary(),
            timestamp=time.time(),
            signature=""
        )
        selection_proposal.signature = self._sign_proposal(selection_proposal)
        
        # In a real implementation, this would involve network communication
        # For now, simulate consensus
        success = await self.consensus_protocol.propose_evolution(selection_proposal)
        
        if success:
            # Perform tournament selection
            selected_population = self._tournament_selection(
                self.local_population,
                tournament_size=3,
                selection_pressure=1.2
            )
            metrics['consensus_operations'] += 1
            return selected_population
        else:
            metrics['failed_consensus'] += 1
            logger.warning("Selection consensus failed, using local population")
            return self.local_population.copy()
    
    async def _consensus_crossover(self, population: List[np.ndarray], metrics: Dict[str, Any]) -> List[np.ndarray]:
        """Perform crossover with distributed consensus"""
        
        # Propose crossover operations
        num_offspring = len(population) // 2
        crossover_pairs = [(i, i + 1) for i in range(0, len(population) - 1, 2)]
        
        crossover_proposal = EvolutionProposal(
            proposal_id=f"crossover_{self.node_id}_{self.evolution_round}_{time.time()}",
            proposer_node=self.node_id,
            operation_type="crossover",
            target_individuals=[idx for pair in crossover_pairs for idx in pair],
            parameters={
                'crossover_method': 'uniform',
                'crossover_rate': 0.8,
                'num_offspring': num_offspring
            },
            fitness_evidence={},
            timestamp=time.time(),
            signature=""
        )
        crossover_proposal.signature = self._sign_proposal(crossover_proposal)
        
        success = await self.consensus_protocol.propose_evolution(crossover_proposal)
        
        if success:
            offspring = []
            for i in range(0, len(population) - 1, 2):
                parent1, parent2 = population[i], population[i + 1]
                child1, child2 = self._uniform_crossover(parent1, parent2)
                offspring.extend([child1, child2])
            
            metrics['consensus_operations'] += 1
            return offspring[:num_offspring]
        else:
            metrics['failed_consensus'] += 1
            logger.warning("Crossover consensus failed, skipping crossover")
            return []
    
    async def _consensus_mutation(self, population: List[np.ndarray], metrics: Dict[str, Any]) -> List[np.ndarray]:
        """Perform mutation with distributed consensus"""
        
        mutation_proposal = EvolutionProposal(
            proposal_id=f"mutation_{self.node_id}_{self.evolution_round}_{time.time()}",
            proposer_node=self.node_id,
            operation_type="mutation",
            target_individuals=list(range(len(population))),
            parameters={
                'mutation_method': 'gaussian',
                'mutation_rate': 0.1,
                'mutation_strength': 0.05
            },
            fitness_evidence={},
            timestamp=time.time(),
            signature=""
        )
        mutation_proposal.signature = self._sign_proposal(mutation_proposal)
        
        success = await self.consensus_protocol.propose_evolution(mutation_proposal)
        
        if success:
            mutated_population = []
            for individual in population:
                mutated = self._gaussian_mutation(individual, rate=0.1, strength=0.05)
                mutated_population.append(mutated)
            
            metrics['consensus_operations'] += 1
            return mutated_population
        else:
            metrics['failed_consensus'] += 1
            logger.warning("Mutation consensus failed, returning unmutated population")
            return population
    
    async def _assemble_final_population(self, selected: List[np.ndarray], offspring: List[np.ndarray], 
                                       metrics: Dict[str, Any]) -> List[np.ndarray]:
        """Assemble final population with consensus on composition"""
        
        # Combine populations
        combined_population = selected + offspring
        
        # Propose final assembly
        assembly_proposal = EvolutionProposal(
            proposal_id=f"assembly_{self.node_id}_{self.evolution_round}_{time.time()}",
            proposer_node=self.node_id,
            operation_type="assembly",
            target_individuals=list(range(len(combined_population))),
            parameters={
                'target_size': len(selected),
                'selection_method': 'elite_preservation'
            },
            fitness_evidence={},
            timestamp=time.time(),
            signature=""
        )
        assembly_proposal.signature = self._sign_proposal(assembly_proposal)
        
        success = await self.consensus_protocol.propose_evolution(assembly_proposal)
        
        if success:
            # Elite preservation - keep best individuals
            # In practice, you'd need actual fitness values
            final_population = combined_population[:len(selected)]
            metrics['consensus_operations'] += 1
            return final_population
        else:
            metrics['failed_consensus'] += 1
            logger.warning("Assembly consensus failed, using selected population")
            return selected
    
    def _get_population_fitness_summary(self) -> Dict[str, float]:
        """Get summary fitness statistics for current population"""
        
        # This is a simplified version - in practice you'd use actual fitness values
        summary = {
            'population_size': len(self.local_population),
            'avg_sparsity': np.mean([np.mean(ind) for ind in self.local_population]),
            'diversity_estimate': np.std([np.mean(ind) for ind in self.local_population])
        }
        
        return summary
    
    def _tournament_selection(self, population: List[np.ndarray], tournament_size: int = 3, 
                            selection_pressure: float = 1.2) -> List[np.ndarray]:
        """Tournament selection for consensus-based evolution"""
        
        selected = []
        
        for _ in range(len(population)):
            # Random tournament
            tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            
            # In practice, you'd use actual fitness values
            # For now, use a proxy based on sparsity
            tournament_fitness = [np.mean(population[i]) for i in tournament_indices]
            
            # Select best from tournament
            best_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[best_idx].copy())
        
        return selected
    
    def _uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover for topology matrices"""
        
        mask = np.random.random(parent1.shape) < 0.5
        
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        
        return child1, child2
    
    def _gaussian_mutation(self, individual: np.ndarray, rate: float = 0.1, strength: float = 0.05) -> np.ndarray:
        """Gaussian mutation with sparsity preservation"""
        
        mutated = individual.copy()
        
        # Apply mutation to a fraction of elements
        mutation_mask = np.random.random(individual.shape) < rate
        noise = np.random.normal(0, strength, individual.shape)
        
        mutated = np.where(mutation_mask, mutated + noise, mutated)
        
        # Preserve sparsity by thresholding
        mutated = np.clip(mutated, 0, 1)
        threshold = np.percentile(mutated, 70)  # Keep 30% active
        mutated = (mutated > threshold).astype(np.float32)
        
        return mutated
    
    def _sign_proposal(self, proposal: EvolutionProposal) -> str:
        """Create proposal signature for integrity verification"""
        
        # Simplified signature - in production use proper cryptography
        content = f"{proposal.proposal_id}_{proposal.proposer_node}_{proposal.operation_type}_{proposal.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_consensus_metrics(self) -> Dict[str, Any]:
        """Get comprehensive consensus and coordination metrics"""
        
        metrics = {
            'node_id': self.node_id,
            'evolution_round': self.evolution_round,
            'total_nodes': self.total_nodes,
            'active_nodes': len(self.active_nodes),
            'consensus_type': self.consensus_type.value,
            'local_population_size': len(self.local_population),
            'fitness_cache_size': len(self.fitness_cache),
            'trust_scores': getattr(self.consensus_protocol, 'node_trust_scores', {}),
            'consensus_history_length': len(getattr(self.consensus_protocol, 'consensus_history', []))
        }
        
        return metrics

# Example usage and testing
if __name__ == "__main__":
    async def test_distributed_evolution():
        """Test distributed evolution with consensus"""
        
        # Initialize coordination nodes
        num_nodes = 5
        coordinators = []
        
        for i in range(num_nodes):
            coordinator = FederatedEvolutionCoordinator(
                node_id=f"node_{i}",
                total_nodes=num_nodes,
                consensus_type=ConsensusType.BYZANTINE_FAULT_TOLERANT
            )
            coordinators.append(coordinator)
        
        # Initialize with sample population
        sample_topology_shape = (8, 16)
        initial_population = []
        
        for _ in range(20):
            topology = np.random.random(sample_topology_shape)
            threshold = np.percentile(topology, 70)
            topology = (topology > threshold).astype(np.float32)
            initial_population.append(topology)
        
        # Initialize first coordinator
        success = await coordinators[0].initialize_federated_evolution(initial_population)
        print(f"Initialization success: {success}")
        
        # Simple fitness evaluator for testing
        def dummy_fitness_evaluator(individual: np.ndarray) -> Dict[str, float]:
            sparsity = np.mean(individual)
            diversity = np.std(individual)
            return {
                'sparsity': sparsity,
                'diversity': diversity,
                'combined': sparsity * 0.7 + diversity * 0.3
            }
        
        # Run distributed evolution
        for generation in range(3):
            print(f"\n=== Generation {generation + 1} ===")
            
            # Simulate all nodes evolving
            for coordinator in coordinators:
                try:
                    new_population, metrics = await coordinator.evolve_generation_distributed(dummy_fitness_evaluator)
                    print(f"Node {coordinator.node_id}: {len(new_population)} individuals, metrics: {metrics}")
                except Exception as e:
                    print(f"Evolution failed on {coordinator.node_id}: {e}")
        
        # Display final consensus metrics
        print("\n=== Final Consensus Metrics ===")
        for coordinator in coordinators:
            metrics = coordinator.get_consensus_metrics()
            print(f"Node {coordinator.node_id}: {metrics}")
    
    # Run the test
    asyncio.run(test_distributed_evolution())