# Self-Evolving MoE-Router

Implements a neuro-evolution loop (mutate→score→select) that discovers sparse routing topologies for Mixture of Experts models. Seeded by evolutionary attention studies, this framework ships with slimmable expert checkpoints for adaptive deployment.

## Overview

Self-Evolving MoE-Router uses evolutionary algorithms to automatically discover optimal routing patterns for Mixture of Experts (MoE) models. Unlike fixed routing strategies, our system continuously evolves the routing topology based on task performance, creating more efficient and specialized expert networks that can dynamically adapt to different computational budgets.

## Key Features

- **Evolutionary Routing**: Automatically discovers optimal expert routing patterns
- **Slimmable Experts**: Dynamically adjust model capacity based on resources
- **Continuous Evolution**: Routing topology improves during deployment
- **Hardware-Aware**: Evolves different topologies for different devices
- **Checkpoint Zoo**: Pre-evolved topologies for common tasks
- **Visualization**: Real-time routing pattern analysis

## Installation

```bash
# Basic installation
pip install self-evolving-moe-router

# With visualization support
pip install self-evolving-moe-router[viz]

# With distributed evolution
pip install self-evolving-moe-router[distributed]

# Development installation
git clone https://github.com/yourusername/self-evolving-moe-router
cd self-evolving-moe-router
pip install -e ".[dev]"
```

## Quick Start

### Basic Evolution

```python
from self_evolving_moe import EvolvingMoERouter, ExpertPool
import torch

# Create expert pool
experts = ExpertPool(
    num_experts=32,
    expert_dim=768,
    expert_type="transformer_block"
)

# Initialize evolving router
router = EvolvingMoERouter(
    experts=experts,
    population_size=100,
    mutation_rate=0.1,
    evolution_strategy="tournament"
)

# Evolution loop
for generation in range(1000):
    # Evaluate current population
    fitness_scores = []
    for topology in router.population:
        score = evaluate_topology(topology, validation_data)
        fitness_scores.append(score)
    
    # Evolve to next generation
    router.evolve(fitness_scores)
    
    # Get best topology
    best_topology = router.get_best_topology()
    print(f"Gen {generation}: Best fitness = {max(fitness_scores):.4f}")
    
# Deploy best topology
router.deploy_topology(best_topology)
```

### Slimmable Deployment

```python
from self_evolving_moe import SlimmableMoE

# Load pre-evolved model
slimmable_moe = SlimmableMoE.from_pretrained(
    "evolved-moe-gpt2-medium"
)

# Adapt to different resource constraints
# Full capacity (32 experts)
output_full = slimmable_moe(input_ids, num_experts=32)

# Medium capacity (16 experts)
output_medium = slimmable_moe(input_ids, num_experts=16)

# Minimal capacity (4 experts)
output_minimal = slimmable_moe(input_ids, num_experts=4)

# Automatic adaptation based on latency
output_auto = slimmable_moe(
    input_ids,
    target_latency=10  # ms
)
```

## Architecture

```
self-evolving-moe-router/
├── self_evolving_moe/
│   ├── evolution/
│   │   ├── population.py          # Population management
│   │   ├── mutations.py           # Mutation operators
│   │   ├── crossover.py           # Crossover strategies
│   │   ├── selection.py           # Selection algorithms
│   │   └── fitness.py             # Fitness functions
│   ├── routing/
│   │   ├── topology.py            # Routing topology representation
│   │   ├── sparse_router.py       # Sparse routing implementation
│   │   ├── learned_router.py      # Learnable routing
│   │   └── hardware_aware.py      # Device-specific routing
│   ├── experts/
│   │   ├── expert_types.py        # Different expert architectures
│   │   ├── slimmable_expert.py    # Width-adjustable experts
│   │   ├── specialized_expert.py  # Task-specific experts
│   │   └── expert_pool.py         # Expert management
│   ├── training/
│   │   ├── evolution_trainer.py   # Evolution loop
│   │   ├── fine_tuning.py         # Expert fine-tuning
│   │   ├── distillation.py        # Knowledge distillation
│   │   └── online_evolution.py    # Continuous evolution
│   ├── visualization/
│   │   ├── topology_viz.py        # Routing visualization
│   │   ├── evolution_plot.py      # Evolution progress
│   │   └── expert_analysis.py     # Expert specialization
│   └── checkpoints/
│       ├── model_zoo.py           # Pre-evolved models
│       ├── topology_library.py    # Routing patterns
│       └── export.py              # Model export
├── configs/
├── examples/
└── benchmarks/
```

## Evolutionary Algorithms

### Topology Representation

```python
from self_evolving_moe.evolution import TopologyGenome

class RoutingTopology:
    """Represents an evolvable routing pattern"""
    
    def __init__(self, num_tokens, num_experts, sparsity=0.1):
        # Binary routing matrix (tokens × experts)
        self.routing_matrix = self.random_sparse_matrix(
            num_tokens, num_experts, sparsity
        )
        
        # Expert connectivity graph
        self.expert_graph = self.init_expert_connections()
        
        # Routing function parameters
        self.routing_params = {
            'temperature': 1.0,
            'top_k': 2,
            'load_balancing_weight': 0.01,
            'diversity_weight': 0.1
        }
    
    def mutate(self, mutation_rate=0.1):
        """Apply mutations to topology"""
        # Structural mutations
        if random.random() < mutation_rate:
            self.mutate_routing_matrix()
            
        # Parameter mutations
        if random.random() < mutation_rate:
            self.mutate_routing_params()
            
        # Connection mutations
        if random.random() < mutation_rate:
            self.mutate_expert_connections()
    
    def crossover(self, other):
        """Crossover with another topology"""
        child = RoutingTopology(
            self.num_tokens,
            self.num_experts
        )
        
        # Uniform crossover for routing matrix
        mask = torch.rand_like(self.routing_matrix) > 0.5
        child.routing_matrix = torch.where(
            mask,
            self.routing_matrix,
            other.routing_matrix
        )
        
        # Average parameters
        child.routing_params = {
            k: (self.routing_params[k] + other.routing_params[k]) / 2
            for k in self.routing_params
        }
        
        return child
```

### Mutation Operators

```python
from self_evolving_moe.evolution import MutationOperators

mutations = MutationOperators()

# Structural mutations
@mutations.register("add_connection")
def add_expert_connection(topology):
    """Add new token-expert connection"""
    token_idx = random.randint(0, topology.num_tokens - 1)
    expert_idx = random.randint(0, topology.num_experts - 1)
    topology.routing_matrix[token_idx, expert_idx] = 1

@mutations.register("remove_connection")
def remove_expert_connection(topology):
    """Remove existing connection"""
    active_connections = topology.routing_matrix.nonzero()
    if len(active_connections) > 0:
        idx = random.choice(range(len(active_connections)))
        topology.routing_matrix[tuple(active_connections[idx])] = 0

@mutations.register("rewire")
def rewire_expert(topology):
    """Rewire all connections of an expert"""
    expert_idx = random.randint(0, topology.num_experts - 1)
    topology.routing_matrix[:, expert_idx] = 0
    # Add random new connections
    num_connections = int(topology.num_tokens * topology.sparsity)
    token_indices = random.sample(
        range(topology.num_tokens),
        num_connections
    )
    topology.routing_matrix[token_indices, expert_idx] = 1

# Parameter mutations
@mutations.register("perturb_temperature")
def mutate_temperature(topology):
    """Perturb routing temperature"""
    topology.routing_params['temperature'] *= random.uniform(0.8, 1.2)
    topology.routing_params['temperature'] = max(0.1, min(10.0, topology.routing_params['temperature']))
```

### Fitness Evaluation

```python
from self_evolving_moe.evolution import FitnessEvaluator

class MultiObjectiveFitness:
    """Evaluate topology on multiple objectives"""
    
    def __init__(self, objectives):
        self.objectives = objectives
        
    def evaluate(self, topology, model, data):
        scores = {}
        
        # Apply topology to model
        model.set_routing_topology(topology)
        
        # Task performance
        if 'accuracy' in self.objectives:
            scores['accuracy'] = self.evaluate_accuracy(model, data)
            
        # Efficiency metrics
        if 'latency' in self.objectives:
            scores['latency'] = self.measure_latency(model, data)
            
        if 'memory' in self.objectives:
            scores['memory'] = self.measure_memory(model)
            
        # Routing metrics
        if 'load_balance' in self.objectives:
            scores['load_balance'] = self.measure_load_balance(topology)
            
        if 'diversity' in self.objectives:
            scores['diversity'] = self.measure_routing_diversity(topology)
        
        # Combine objectives
        fitness = self.combine_objectives(scores)
        return fitness, scores
    
    def combine_objectives(self, scores):
        """Weighted combination of objectives"""
        weights = {
            'accuracy': 1.0,
            'latency': -0.1,    # Negative weight (minimize)
            'memory': -0.05,
            'load_balance': 0.2,
            'diversity': 0.1
        }
        
        fitness = sum(
            weights.get(obj, 0) * score
            for obj, score in scores.items()
        )
        return fitness
```

## Slimmable Experts

### Width-Adjustable Experts

```python
from self_evolving_moe.experts import SlimmableExpert

class SlimmableTransformerExpert(nn.Module):
    """Expert that can operate at different widths"""
    
    def __init__(self, max_dim=768, min_dim=192, granularity=64):
        super().__init__()
        self.max_dim = max_dim
        self.min_dim = min_dim
        self.granularity = granularity
        
        # Slimmable layers
        self.attention = SlimmableMultiHeadAttention(
            max_dim, num_heads=12
        )
        self.ffn = SlimmableFeedForward(
            max_dim, expansion=4
        )
        
        # Width-specific parameters
        self.width_embeddings = nn.ParameterDict({
            str(w): nn.Parameter(torch.randn(w))
            for w in range(min_dim, max_dim + 1, granularity)
        })
    
    def forward(self, x, width=None):
        if width is None:
            width = self.max_dim
            
        # Slice to target width
        x_sliced = x[..., :width]
        
        # Add width-specific embedding
        width_emb = self.width_embeddings[str(width)]
        x_sliced = x_sliced + width_emb
        
        # Process through slimmable layers
        x_sliced = self.attention(x_sliced, width=width)
        x_sliced = self.ffn(x_sliced, width=width)
        
        # Pad back if needed
        if width < x.shape[-1]:
            x = torch.cat([x_sliced, x[..., width:]], dim=-1)
        else:
            x = x_sliced
            
        return x
```

### Dynamic Expert Selection

```python
from self_evolving_moe.routing import DynamicRouter

class ResourceAwareRouter(nn.Module):
    """Routes to different numbers of experts based on resources"""
    
    def __init__(self, num_experts, hidden_dim):
        super().__init__()
        self.num_experts = num_experts
        
        # Learned routing functions for different capacities
        self.routers = nn.ModuleDict({
            'full': nn.Linear(hidden_dim, num_experts),
            'half': nn.Linear(hidden_dim, num_experts // 2),
            'quarter': nn.Linear(hidden_dim, num_experts // 4),
            'minimal': nn.Linear(hidden_dim, 2)
        })
        
        # Resource predictor
        self.resource_predictor = nn.Linear(hidden_dim, 4)
    
    def forward(self, x, target_latency=None):
        if target_latency is None:
            # Use all experts
            router = self.routers['full']
            active_experts = self.num_experts
        else:
            # Predict resource level needed
            resource_logits = self.resource_predictor(x.mean(dim=1))
            resource_level = self.latency_to_resource_level(target_latency)
            
            # Select appropriate router
            router_key = ['minimal', 'quarter', 'half', 'full'][resource_level]
            router = self.routers[router_key]
            active_experts = [2, self.num_experts // 4, 
                            self.num_experts // 2, self.num_experts][resource_level]
        
        # Route to active experts
        routing_weights = router(x).softmax(dim=-1)
        
        # Apply top-k selection
        top_k = min(2, active_experts)
        routing_weights, selected_experts = routing_weights.topk(top_k, dim=-1)
        
        return routing_weights, selected_experts
```

## Online Evolution

### Continuous Evolution During Deployment

```python
from self_evolving_moe.training import OnlineEvolution

class ContinuousEvolver:
    """Evolve routing topology during deployment"""
    
    def __init__(self, model, evolution_interval=1000):
        self.model = model
        self.evolution_interval = evolution_interval
        self.step_count = 0
        
        # Maintain small population during deployment
        self.population_size = 10
        self.population = self.init_population()
        
        # Performance tracking
        self.performance_history = []
        
    def step(self, batch):
        """Single training/inference step with evolution"""
        # Regular forward pass
        output = self.model(batch)
        
        self.step_count += 1
        
        # Periodic evolution
        if self.step_count % self.evolution_interval == 0:
            self.evolve_population(batch)
            
        return output
    
    def evolve_population(self, validation_batch):
        """Run one evolution step"""
        # Evaluate current population
        fitness_scores = []
        for topology in self.population:
            self.model.set_routing_topology(topology)
            score = self.evaluate_topology(validation_batch)
            fitness_scores.append(score)
        
        # Track best performance
        best_idx = np.argmax(fitness_scores)
        self.performance_history.append(fitness_scores[best_idx])
        
        # Generate new population
        new_population = []
        
        # Elitism - keep best
        new_population.append(self.population[best_idx])
        
        # Generate rest through mutation/crossover
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self.tournament_select(self.population, fitness_scores)
            parent2 = self.tournament_select(self.population, fitness_scores)
            
            # Crossover
            child = parent1.crossover(parent2)
            
            # Mutation
            child.mutate(self.mutation_rate)
            
            new_population.append(child)
        
        self.population = new_population
        
        # Set model to best topology
        self.model.set_routing_topology(self.population[0])
```

### Hardware-Aware Evolution

```python
from self_evolving_moe.routing import HardwareAwareEvolution

class DeviceSpecificEvolver:
    """Evolve different topologies for different devices"""
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.device_topologies = {}
        
    def evolve_for_device(self, device_profile, training_data):
        """Evolve topology for specific device"""
        # Set hardware constraints
        constraints = {
            'max_memory': device_profile['memory'],
            'target_latency': device_profile['target_latency'],
            'compute_units': device_profile['compute_units']
        }
        
        # Create hardware-aware fitness function
        fitness_fn = self.create_hardware_fitness(constraints)
        
        # Evolution with hardware constraints
        evolver = EvolvingMoERouter(
            experts=self.base_model.experts,
            fitness_function=fitness_fn,
            population_size=50
        )
        
        # Evolve
        for generation in range(100):
            evolver.evolve_one_generation(training_data)
            
        # Store device-specific topology
        best_topology = evolver.get_best_topology()
        self.device_topologies[device_profile['name']] = best_topology
        
        return best_topology
    
    def create_hardware_fitness(self, constraints):
        """Fitness function with hardware constraints"""
        def fitness(topology, model, data):
            # Measure hardware metrics
            latency = measure_latency(model, topology, data)
            memory = measure_memory(model, topology)
            
            # Task performance
            accuracy = evaluate_accuracy(model, topology, data)
            
            # Penalize constraint violations
            penalty = 0
            if latency > constraints['target_latency']:
                penalty += (latency - constraints['target_latency']) * 10
            if memory > constraints['max_memory']:
                penalty += (memory - constraints['max_memory']) * 5
                
            return accuracy - penalty
            
        return fitness
```

## Visualization and Analysis

### Routing Pattern Visualization

```python
from self_evolving_moe.visualization import TopologyVisualizer

viz = TopologyVisualizer()

# Visualize routing topology
viz.plot_routing_matrix(
    topology.routing_matrix,
    title="Evolved Routing Pattern",
    save_path="routing_pattern.png"
)

# Animate evolution progress
viz.animate_evolution(
    evolution_history,
    metric='fitness',
    save_path="evolution_animation.gif"
)

# Expert specialization analysis
specialization = viz.analyze_expert_specialization(
    model,
    topology,
    validation_data
)

viz.plot_expert_specialization(
    specialization,
    save_path="expert_specialization.png"
)
```

### Interactive Dashboard

```python
from self_evolving_moe.visualization import EvolutionDashboard

# Launch real-time monitoring dashboard
dashboard = EvolutionDashboard(
    evolver=continuous_evolver,
    port=8080
)

# Add custom metrics
@dashboard.metric
def routing_efficiency(model, topology):
    """Compute routing efficiency metric"""
    total_connections = topology.routing_matrix.sum()
    possible_connections = topology.routing_matrix.numel()
    return total_connections / possible_connections

@dashboard.plot
def expert_load_distribution(model, topology):
    """Plot load distribution across experts"""
    loads = topology.routing_matrix.sum(dim=0)
    return loads.numpy()

dashboard.start()
```

## Pre-Evolved Model Zoo

### Available Checkpoints

```python
from self_evolving_moe import ModelZoo

# List available models
zoo = ModelZoo()
available_models = zoo.list_models()

# Load pre-evolved model
model = zoo.load_model(
    "evolved-moe-bert-base",
    device="cuda",
    num_experts=16  # Can use fewer experts than evolved with
)

# Model information
info = zoo.get_model_info("evolved-moe-bert-base")
print(f"Evolved for: {info['task']}")
print(f"Generations: {info['evolution_generations']}")
print(f"Best fitness: {info['final_fitness']}")
print(f"Topology sparsity: {info['sparsity']}")
```

### Export and Deployment

```python
from self_evolving_moe.checkpoints import ModelExporter

exporter = ModelExporter()

# Export for different frameworks
exporter.export_pytorch(model, "evolved_moe.pt")
exporter.export_onnx(model, "evolved_moe.onnx")
exporter.export_tensorrt(model, "evolved_moe.trt")

# Export just the topology
exporter.export_topology(
    model.get_current_topology(),
    "topology.json"
)

# Create slimmable checkpoint
exporter.export_slimmable(
    model,
    width_configs=[192, 384, 576, 768],
    "slimmable_moe.pt"
)
```

## Benchmarks

### Evolution Efficiency

```python
from self_evolving_moe.benchmarks import EvolutionBenchmark

benchmark = EvolutionBenchmark()

# Compare evolution strategies
strategies = ['genetic', 'evolution_strategies', 'neat', 'cma_es']
results = benchmark.compare_strategies(
    task="language_modeling",
    model_size="base",
    generations=500,
    strategies=strategies
)

benchmark.plot_convergence(results, "convergence_comparison.png")
```

### Routing Efficiency

```python
from self_evolving_moe.benchmarks import RoutingBenchmark

routing_bench = RoutingBenchmark()

# Compare with baseline routing
baselines = {
    'dense': DenseRouting(),
    'top2': TopKRouting(k=2),
    'switch': SwitchRouting(),
    'evolved': model.get_current_topology()
}

metrics = routing_bench.evaluate(
    baselines,
    tasks=['classification', 'generation', 'reasoning'],
    metrics=['flops', 'memory', 'accuracy', 'latency']
)

routing_bench.generate_report(metrics, "routing_comparison.html")
```

## Advanced Features

### Meta-Evolution

```python
from self_evolving_moe.evolution import MetaEvolution

# Evolve the evolution process itself
meta_evolver = MetaEvolution(
    base_evolver=EvolvingMoERouter,
    meta_parameters={
        'mutation_rate': (0.01, 0.5),
        'crossover_rate': (0.1, 0.9),
        'population_size': (10, 200),
        'selection_pressure': (1.0, 5.0)
    }
)

# Find optimal evolution parameters
best_evolution_params = meta_evolver.evolve(
    task=language_modeling_task,
    meta_generations=50,
    trials_per_config=5
)

print(f"Optimal mutation rate: {best_evolution_params['mutation_rate']}")
print(f"Optimal population size: {best_evolution_params['population_size']}")
```

### Compositional Evolution

```python
from self_evolving_moe.evolution import CompositionalEvolution

# Evolve modular routing patterns
comp_evolver = CompositionalEvolution(
    primitive_patterns=['star', 'chain', 'tree', 'mesh'],
    composition_rules=['sequential', 'parallel', 'hierarchical']
)

# Discover complex routing from simple patterns
evolved_composition = comp_evolver.evolve(
    base_model=model,
    target_complexity=1000,  # Max routing connections
    generations=200
)

# Resulting topology is composition of primitives
print(f"Composition: {evolved_composition.describe()}")
# Output: "Hierarchical(Parallel(Star(8), Chain(4)), Tree(16))"
```

## Configuration

### Evolution Configuration

```yaml
# config/evolution_config.yaml
evolution:
  population_size: 100
  generations: 1000
  mutation_rate: 0.1
  crossover_rate: 0.7
  selection:
    method: "tournament"
    tournament_size: 3
  
objectives:
  - name: "accuracy"
    weight: 1.0
    target: 0.95
  - name: "latency"
    weight: -0.2
    target: 10  # ms
  - name: "sparsity"
    weight: 0.1
    target: 0.9
    
constraints:
  max_active_experts: 4
  min_expert_usage: 0.1
  max_memory: "4GB"
  
hardware_targets:
  - name: "edge"
    memory: "1GB"
    compute: "int8"
  - name: "mobile" 
    memory: "500MB"
    compute: "fp16"
  - name: "server"
    memory: "32GB"
    compute: "fp32"
```

## Troubleshooting

### Common Issues

```python
from self_evolving_moe.diagnostics import EvolutionDiagnostics

diagnostics = EvolutionDiagnostics()

# Check evolution health
health = diagnostics.check_evolution_health(evolver)

if health.diversity < 0.1:
    print("Low diversity - increase mutation rate or population size")
    evolver.mutation_rate *= 1.5

if health.convergence_rate < 0.001:
    print("Slow convergence - check fitness function")
    diagnostics.analyze_fitness_landscape(evolver)

if health.expert_utilization < 0.5:
    print("Poor expert utilization - adjust routing sparsity")
    evolver.adjust_sparsity_target(0.2)
```

## Citation

```bibtex
@article{self_evolving_moe_router,
  title={Self-Evolving MoE-Router: Evolutionary Discovery of Sparse Expert Routing},
  author={Your Name},
  journal={NeurIPS},
  year={2025}
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Evolutionary computation community
- Mixture of Experts researchers
- Neural Architecture Search community

## Resources

- [Documentation](https://self-evolving-moe.readthedocs.io)
- [Model Zoo](https://huggingface.co/self-evolving-moe)
- [Evolution Visualizer](https://self-evolving-moe.github.io/visualizer)
- [Paper](https://arxiv.org/abs/self-evolving-moe)
