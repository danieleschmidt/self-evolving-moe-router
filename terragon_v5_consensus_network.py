"""
TERRAGON v5.0 - Distributed Consensus Network
Advanced peer-to-peer evolution with Byzantine fault tolerance.
"""

import asyncio
import aiohttp
import json
import time
import hashlib
import numpy as np
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import logging
import websockets
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

@dataclass
class NetworkNode:
    """Distributed network node configuration"""
    node_id: str
    host: str
    port: int
    is_leader: bool = False
    trusted_peers: List[str] = None
    reputation_score: float = 1.0

@dataclass
class ConsensusProposal:
    """Blockchain-like consensus proposal"""
    proposal_id: str
    node_id: str
    timestamp: float
    evolution_data: Dict
    fitness_score: float
    signature: str
    previous_hash: str

class ByzantineFaultTolerantConsensus:
    """Byzantine fault tolerant consensus for evolution"""
    
    def __init__(self, node_id: str, fault_tolerance: int = 1):
        self.node_id = node_id
        self.fault_tolerance = fault_tolerance
        self.min_nodes = 3 * fault_tolerance + 1
        self.active_nodes = set()
        self.proposal_ledger = []
        self.votes = {}
        self.committed_proposals = []
        
    def create_proposal(self, evolution_data: Dict, fitness_score: float) -> ConsensusProposal:
        """Create a new consensus proposal"""
        previous_hash = self.get_latest_hash()
        
        proposal = ConsensusProposal(
            proposal_id=str(uuid.uuid4()),
            node_id=self.node_id,
            timestamp=time.time(),
            evolution_data=evolution_data,
            fitness_score=fitness_score,
            signature=self.sign_data(evolution_data),
            previous_hash=previous_hash
        )
        
        return proposal
    
    def sign_data(self, data: Dict) -> str:
        """Create cryptographic signature for data integrity"""
        data_bytes = json.dumps(data, sort_keys=True).encode()
        signature = hashlib.sha256(data_bytes + self.node_id.encode()).hexdigest()
        return signature
    
    def verify_proposal(self, proposal: ConsensusProposal) -> bool:
        """Verify proposal integrity and authenticity"""
        # Verify signature
        expected_signature = self.sign_data(proposal.evolution_data)
        if proposal.signature != expected_signature:
            return False
        
        # Verify chain continuity
        if self.proposal_ledger and proposal.previous_hash != self.get_latest_hash():
            return False
            
        return True
    
    def get_latest_hash(self) -> str:
        """Get hash of the latest committed proposal"""
        if not self.committed_proposals:
            return "genesis_hash"
        
        latest = self.committed_proposals[-1]
        data = f"{latest.proposal_id}{latest.timestamp}{latest.fitness_score}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def vote_on_proposal(self, proposal: ConsensusProposal) -> bool:
        """Vote on a consensus proposal"""
        if not self.verify_proposal(proposal):
            return False
        
        proposal_key = proposal.proposal_id
        if proposal_key not in self.votes:
            self.votes[proposal_key] = set()
        
        self.votes[proposal_key].add(self.node_id)
        
        # Check if consensus reached (2/3 majority)
        required_votes = (2 * len(self.active_nodes)) // 3 + 1
        if len(self.votes[proposal_key]) >= required_votes:
            self.commit_proposal(proposal)
            return True
        
        return False
    
    def commit_proposal(self, proposal: ConsensusProposal):
        """Commit proposal to the distributed ledger"""
        self.committed_proposals.append(proposal)
        self.proposal_ledger.append(proposal)
        logger.info(f"✅ Proposal {proposal.proposal_id[:8]} committed with fitness {proposal.fitness_score:.6f}")

class DistributedEvolutionNetwork:
    """Peer-to-peer evolution network with consensus"""
    
    def __init__(self, node_config: NetworkNode):
        self.node = node_config
        self.consensus_engine = ByzantineFaultTolerantConsensus(node_config.node_id)
        self.peer_connections = {}
        self.evolution_state = {}
        self.network_topology = []
        self.performance_metrics = {
            'proposals_created': 0,
            'proposals_validated': 0,
            'consensus_rounds': 0,
            'network_latency': []
        }
        
    async def start_node_server(self):
        """Start the distributed node server"""
        async def handle_peer_message(websocket, path):
            try:
                async for message in websocket:
                    await self.process_peer_message(json.loads(message), websocket)
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Peer connection closed: {path}")
        
        server = await websockets.serve(
            handle_peer_message, 
            self.node.host, 
            self.node.port
        )
        
        logger.info(f"🌐 Node {self.node.node_id} started on {self.node.host}:{self.node.port}")
        return server
    
    async def connect_to_peers(self, peer_addresses: List[str]):
        """Connect to trusted peer nodes"""
        for peer_addr in peer_addresses:
            try:
                websocket = await websockets.connect(f"ws://{peer_addr}")
                self.peer_connections[peer_addr] = websocket
                
                # Send introduction message
                intro_message = {
                    'type': 'node_introduction',
                    'node_id': self.node.node_id,
                    'timestamp': time.time(),
                    'capabilities': ['evolution', 'consensus', 'bft']
                }
                await websocket.send(json.dumps(intro_message))
                
                logger.info(f"🤝 Connected to peer: {peer_addr}")
                
            except Exception as e:
                logger.warning(f"Failed to connect to peer {peer_addr}: {e}")
    
    async def process_peer_message(self, message: Dict, websocket):
        """Process incoming peer messages"""
        message_type = message.get('type')
        
        if message_type == 'node_introduction':
            await self.handle_node_introduction(message, websocket)
        elif message_type == 'consensus_proposal':
            await self.handle_consensus_proposal(message)
        elif message_type == 'evolution_sync':
            await self.handle_evolution_sync(message)
        elif message_type == 'network_topology_update':
            await self.handle_topology_update(message)
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    async def handle_node_introduction(self, message: Dict, websocket):
        """Handle new node introduction"""
        peer_id = message['node_id']
        self.consensus_engine.active_nodes.add(peer_id)
        
        # Send network topology update
        topology_message = {
            'type': 'network_topology_update',
            'nodes': list(self.consensus_engine.active_nodes),
            'timestamp': time.time()
        }
        await websocket.send(json.dumps(topology_message))
    
    async def handle_consensus_proposal(self, message: Dict):
        """Handle consensus proposals from peers"""
        proposal_data = message['proposal']
        
        proposal = ConsensusProposal(
            proposal_id=proposal_data['proposal_id'],
            node_id=proposal_data['node_id'],
            timestamp=proposal_data['timestamp'],
            evolution_data=proposal_data['evolution_data'],
            fitness_score=proposal_data['fitness_score'],
            signature=proposal_data['signature'],
            previous_hash=proposal_data['previous_hash']
        )
        
        # Vote on the proposal
        consensus_reached = self.consensus_engine.vote_on_proposal(proposal)
        
        if consensus_reached:
            # Broadcast consensus achievement
            await self.broadcast_consensus_result(proposal)
        
        self.performance_metrics['proposals_validated'] += 1
    
    async def handle_evolution_sync(self, message: Dict):
        """Synchronize evolution state with peers"""
        peer_state = message['evolution_state']
        
        # Merge evolution states using consensus
        merged_state = self.merge_evolution_states(
            self.evolution_state, 
            peer_state
        )
        
        self.evolution_state = merged_state
    
    async def handle_topology_update(self, message: Dict):
        """Update network topology information"""
        self.network_topology = message['nodes']
        for node_id in message['nodes']:
            self.consensus_engine.active_nodes.add(node_id)
    
    def merge_evolution_states(self, local_state: Dict, peer_state: Dict) -> Dict:
        """Merge evolution states using fitness-based consensus"""
        merged_state = local_state.copy()
        
        for key, value in peer_state.items():
            if key not in merged_state:
                merged_state[key] = value
            else:
                # Use higher fitness state
                if isinstance(value, dict) and 'fitness' in value:
                    if value['fitness'] > merged_state[key].get('fitness', float('-inf')):
                        merged_state[key] = value
        
        return merged_state
    
    async def propose_evolution_candidate(self, candidate_data: Dict, fitness_score: float):
        """Propose an evolution candidate for consensus"""
        proposal = self.consensus_engine.create_proposal(candidate_data, fitness_score)
        
        # Broadcast to all peers
        message = {
            'type': 'consensus_proposal',
            'proposal': asdict(proposal),
            'timestamp': time.time()
        }
        
        await self.broadcast_to_peers(message)
        self.performance_metrics['proposals_created'] += 1
        
        return proposal
    
    async def broadcast_to_peers(self, message: Dict):
        """Broadcast message to all connected peers"""
        disconnected_peers = []
        
        for peer_addr, websocket in self.peer_connections.items():
            try:
                await websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_peers.append(peer_addr)
        
        # Clean up disconnected peers
        for peer_addr in disconnected_peers:
            del self.peer_connections[peer_addr]
    
    async def broadcast_consensus_result(self, proposal: ConsensusProposal):
        """Broadcast consensus achievement to network"""
        message = {
            'type': 'consensus_achieved',
            'proposal_id': proposal.proposal_id,
            'fitness_score': proposal.fitness_score,
            'timestamp': time.time(),
            'committed_by': self.node.node_id
        }
        
        await self.broadcast_to_peers(message)
        self.performance_metrics['consensus_rounds'] += 1
    
    async def run_distributed_evolution_round(self, local_candidates: List[Dict]) -> Dict:
        """Run one round of distributed evolution"""
        round_start = time.time()
        
        # Evaluate local candidates
        best_local = max(local_candidates, key=lambda x: x.get('fitness', float('-inf')))
        
        # Propose best candidate for consensus
        proposal = await self.propose_evolution_candidate(
            best_local, 
            best_local.get('fitness', 0.0)
        )
        
        # Wait for consensus (with timeout)
        consensus_timeout = 10.0  # seconds
        start_wait = time.time()
        
        while time.time() - start_wait < consensus_timeout:
            if proposal in self.consensus_engine.committed_proposals:
                break
            await asyncio.sleep(0.1)
        
        # Collect network results
        network_latency = time.time() - round_start
        self.performance_metrics['network_latency'].append(network_latency)
        
        results = {
            'local_best_fitness': best_local.get('fitness', 0.0),
            'consensus_achieved': proposal in self.consensus_engine.committed_proposals,
            'network_latency': network_latency,
            'active_peers': len(self.peer_connections),
            'total_network_nodes': len(self.consensus_engine.active_nodes),
            'committed_proposals': len(self.consensus_engine.committed_proposals)
        }
        
        return results
    
    def get_network_performance_report(self) -> Dict:
        """Generate comprehensive network performance report"""
        return {
            'node_id': self.node.node_id,
            'performance_metrics': self.performance_metrics,
            'network_topology': {
                'total_nodes': len(self.consensus_engine.active_nodes),
                'connected_peers': len(self.peer_connections),
                'consensus_participation': len(self.consensus_engine.committed_proposals)
            },
            'consensus_statistics': {
                'total_proposals': len(self.consensus_engine.proposal_ledger),
                'committed_proposals': len(self.consensus_engine.committed_proposals),
                'average_latency': np.mean(self.performance_metrics['network_latency']) if self.performance_metrics['network_latency'] else 0,
                'consensus_rate': len(self.consensus_engine.committed_proposals) / max(1, self.performance_metrics['proposals_created'])
            }
        }

async def create_test_network():
    """Create a test distributed evolution network"""
    
    # Create test nodes
    nodes = [
        NetworkNode("node_1", "localhost", 8001, is_leader=True),
        NetworkNode("node_2", "localhost", 8002),
        NetworkNode("node_3", "localhost", 8003),
        NetworkNode("node_4", "localhost", 8004)
    ]
    
    # Initialize network nodes
    networks = []
    for node_config in nodes:
        network = DistributedEvolutionNetwork(node_config)
        networks.append(network)
    
    # Start servers
    servers = []
    for network in networks:
        server = await network.start_node_server()
        servers.append(server)
    
    # Allow servers to start
    await asyncio.sleep(1.0)
    
    # Connect nodes to each other
    peer_addresses = [f"{node.host}:{node.port}" for node in nodes[1:]]
    await networks[0].connect_to_peers(peer_addresses)
    
    logger.info("🚀 Test network initialized successfully")
    return networks, servers

async def run_consensus_test():
    """Run distributed consensus test"""
    logger.info("🔬 Starting Distributed Consensus Test")
    
    networks, servers = await create_test_network()
    
    # Generate test evolution candidates
    test_candidates = []
    for i in range(5):
        candidate = {
            'genome': np.random.randint(0, 2, (8, 16)).tolist(),
            'fitness': np.random.uniform(-0.5, -0.3),
            'generation': i,
            'mutations': np.random.randint(1, 5)
        }
        test_candidates.append(candidate)
    
    # Run distributed evolution rounds
    results = []
    for round_num in range(3):
        logger.info(f"Running consensus round {round_num + 1}")
        
        # Each node runs evolution round
        round_results = []
        for network in networks:
            result = await network.run_distributed_evolution_round(test_candidates)
            round_results.append(result)
        
        results.append({
            'round': round_num + 1,
            'node_results': round_results
        })
        
        await asyncio.sleep(2.0)  # Allow consensus to propagate
    
    # Generate performance reports
    performance_reports = []
    for network in networks:
        report = network.get_network_performance_report()
        performance_reports.append(report)
    
    # Save test results
    Path("consensus_test_results").mkdir(exist_ok=True)
    
    test_results = {
        'test_timestamp': time.time(),
        'consensus_rounds': results,
        'performance_reports': performance_reports,
        'network_configuration': {
            'total_nodes': len(networks),
            'test_candidates': len(test_candidates),
            'rounds_completed': len(results)
        }
    }
    
    with open("consensus_test_results/distributed_consensus_test.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Cleanup
    for server in servers:
        server.close()
        await server.wait_closed()
    
    logger.info("✅ Distributed consensus test completed")
    return test_results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_consensus_test())