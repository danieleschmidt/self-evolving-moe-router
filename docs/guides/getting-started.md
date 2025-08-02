# Getting Started with Self-Evolving MoE-Router

Welcome to the Self-Evolving MoE-Router! This guide will help you get up and running with evolutionary optimization of Mixture of Experts routing patterns.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **Memory**: 16GB+ RAM for development
- **Storage**: 10GB free space for models and data

### Knowledge Prerequisites
- Basic understanding of PyTorch and neural networks
- Familiarity with Mixture of Experts concepts
- Basic knowledge of evolutionary algorithms (helpful but not required)

## Installation

### Option 1: pip Install (Recommended for Users)

```bash
# Basic installation
pip install self-evolving-moe-router

# With visualization support
pip install self-evolving-moe-router[viz]

# With distributed evolution
pip install self-evolving-moe-router[distributed]

# Full installation with all features
pip install self-evolving-moe-router[viz,distributed,benchmark]
```

### Option 2: Development Installation

```bash
# Clone the repository
git clone https://github.com/terragon-labs/self-evolving-moe-router
cd self-evolving-moe-router

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,viz,distributed,benchmark]"

# Install pre-commit hooks
pre-commit install
```

### Option 3: Docker Installation

```bash
# Pull the official image
docker pull self-evolving-moe/router:latest

# Or build from source
git clone https://github.com/terragon-labs/self-evolving-moe-router
cd self-evolving-moe-router
docker build -t self-evolving-moe-router .

# Run with GPU support
docker run --gpus all -it self-evolving-moe-router
```

## Quick Start

### 1. Your First Evolution Experiment

Create a simple evolution experiment:

```python
from self_evolving_moe import EvolvingMoERouter, ExpertPool
import torch

# Create expert pool
experts = ExpertPool(
    num_experts=16,
    expert_dim=512,
    expert_type="transformer_block"
)

# Initialize evolving router
router = EvolvingMoERouter(
    experts=experts,
    population_size=50,
    mutation_rate=0.1,
    evolution_strategy="genetic_algorithm"
)

# Generate some dummy data for fitness evaluation
def create_dummy_data():
    return {
        'input_ids': torch.randint(0, 1000, (32, 128)),
        'labels': torch.randint(0, 2, (32,))
    }

# Simple fitness function
def evaluate_topology(topology, data):
    # In practice, this would evaluate model performance
    # Here we use a dummy fitness based on sparsity and random performance
    sparsity = 1.0 - (topology.routing_matrix.sum() / topology.routing_matrix.numel())
    performance = torch.rand(1).item()  # Replace with actual model evaluation
    return performance + 0.1 * sparsity  # Balance performance and efficiency

# Run evolution
print("Starting evolution...")
for generation in range(100):
    # Evaluate current population
    fitness_scores = []
    data = create_dummy_data()
    
    for topology in router.population:
        score = evaluate_topology(topology, data)
        fitness_scores.append(score)
    
    # Evolve to next generation
    router.evolve(fitness_scores)
    
    # Print progress
    best_fitness = max(fitness_scores)
    avg_fitness = sum(fitness_scores) / len(fitness_scores)
    print(f"Generation {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")

# Get the best evolved topology
best_topology = router.get_best_topology()
print(f"Evolution complete! Best topology sparsity: {best_topology.get_sparsity():.3f}")
```

### 2. Using Pre-Evolved Models

Skip the evolution process and use pre-evolved models:

```python
from self_evolving_moe import ModelZoo

# Load model zoo
zoo = ModelZoo()

# List available models
models = zoo.list_models()
print("Available models:", models)

# Load a pre-evolved model
model = zoo.load_model(
    "evolved-moe-gpt2-small",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Use the model for inference
input_text = "The quick brown fox"
output = model.generate(input_text, max_length=50)
print(f"Generated: {output}")
```

### 3. Hardware-Aware Evolution

Evolve topologies optimized for specific hardware:

```python
from self_evolving_moe import HardwareAwareEvolver

# Define target hardware profile
mobile_profile = {
    'name': 'mobile',
    'memory': '2GB',
    'target_latency': 50,  # milliseconds
    'compute_units': 'fp16',
    'max_active_experts': 4
}

# Create hardware-aware evolver
evolver = HardwareAwareEvolver(base_model=model)

# Evolve for mobile deployment
mobile_topology = evolver.evolve_for_device(
    device_profile=mobile_profile,
    training_data=your_training_data,
    generations=200
)

print(f"Mobile-optimized topology: {mobile_topology.describe()}")
```

## Core Concepts

### Routing Topologies

A routing topology defines how input tokens are routed to experts:

```python
from self_evolving_moe.routing import RoutingTopology

# Create a random sparse topology
topology = RoutingTopology(
    num_tokens=512,      # Sequence length
    num_experts=16,      # Number of available experts
    sparsity=0.9        # 90% of connections are zero
)

# Inspect the topology
print(f"Routing matrix shape: {topology.routing_matrix.shape}")
print(f"Active connections: {topology.routing_matrix.sum().item()}")
print(f"Sparsity: {topology.get_sparsity():.3f}")

# Visualize the routing pattern
topology.visualize(save_path="topology.png")
```

### Evolution Operators

The framework includes various mutation and crossover operators:

```python
from self_evolving_moe.evolution import MutationOperators

mutations = MutationOperators()

# Apply different types of mutations
topology_copy = topology.copy()

# Structural mutations
mutations.add_connection(topology_copy)      # Add new routing connection
mutations.remove_connection(topology_copy)   # Remove existing connection
mutations.rewire_expert(topology_copy)       # Rewire all connections of an expert

# Parameter mutations  
mutations.perturb_temperature(topology_copy) # Adjust routing temperature
mutations.mutate_load_balance(topology_copy) # Change load balancing weight
```

### Multi-Objective Optimization

Balance multiple competing objectives:

```python
from self_evolving_moe.evolution import MultiObjectiveFitness

# Define fitness function with multiple objectives
fitness_fn = MultiObjectiveFitness([
    {'name': 'accuracy', 'weight': 1.0, 'maximize': True},
    {'name': 'latency', 'weight': -0.2, 'maximize': False},
    {'name': 'memory', 'weight': -0.1, 'maximize': False},
    {'name': 'diversity', 'weight': 0.15, 'maximize': True}
])

# Use NSGA-II for true multi-objective optimization
from self_evolving_moe.evolution import NSGA2

nsga2 = NSGA2(
    population_size=100,
    objectives=['accuracy', 'latency', 'memory']
)

# Evolution will maintain a Pareto front of solutions
pareto_front = nsga2.evolve(fitness_fn, generations=500)
```

## Configuration

### Evolution Configuration

Create a configuration file `evolution_config.yaml`:

```yaml
# Evolution parameters
evolution:
  algorithm: "genetic_algorithm"  # genetic_algorithm, evolution_strategies, nsga2
  population_size: 100
  generations: 1000
  mutation_rate: 0.1
  crossover_rate: 0.7
  elitism_rate: 0.1
  
  selection:
    method: "tournament"
    tournament_size: 5
  
  # Adaptive parameter control
  adaptive_params: true
  diversity_threshold: 0.1

# Multi-objective configuration
objectives:
  - name: "accuracy"
    weight: 1.0
    target: null
    maximize: true
  - name: "latency"
    weight: 0.2
    target: 10  # milliseconds
    maximize: false
  - name: "memory" 
    weight: 0.1
    target: null
    maximize: false
  - name: "sparsity"
    weight: 0.05
    target: 0.9
    maximize: true

# Hardware constraints
hardware:
  max_memory: "8GB"
  target_device: "cuda"
  precision: "fp16"
  max_experts_per_token: 2

# Logging and monitoring
logging:
  level: "INFO"
  log_file: "evolution.log"
  tensorboard: true
  wandb:
    project: "self-evolving-moe"
    entity: "your-username"
```

Load and use the configuration:

```python
from self_evolving_moe.config import load_config

config = load_config("evolution_config.yaml")
router = EvolvingMoERouter.from_config(config)
```

### Expert Configuration

Configure different types of experts:

```yaml
# Expert pool configuration
experts:
  type: "slimmable_transformer"
  num_experts: 32
  base_config:
    hidden_dim: 768
    num_heads: 12
    intermediate_size: 3072
    num_layers: 2
  
  slimmable:
    width_configs: [192, 384, 576, 768]
    width_multipliers: [0.25, 0.5, 0.75, 1.0]
  
  specialization:
    enable_specialization: true
    specialization_loss_weight: 0.1
    diversity_loss_weight: 0.05
```

## Monitoring and Visualization

### Real-time Dashboard

Launch the evolution dashboard:

```python
from self_evolving_moe.visualization import EvolutionDashboard

dashboard = EvolutionDashboard(
    evolver=router,
    port=8080,
    update_interval=1.0  # seconds
)

# Add custom metrics
@dashboard.metric
def expert_utilization(router):
    topology = router.get_best_topology()
    return topology.get_expert_utilization()

@dashboard.plot  
def fitness_history(router):
    return router.get_fitness_history()

# Start the dashboard
dashboard.start()
# Visit http://localhost:8080 to view
```

### Static Visualizations

Generate static plots and analyses:

```python
from self_evolving_moe.visualization import TopologyVisualizer, EvolutionPlotter

# Visualize routing topology
viz = TopologyVisualizer()
viz.plot_routing_matrix(
    topology.routing_matrix,
    title="Evolved Routing Pattern",
    save_path="routing_heatmap.png"
)

# Plot evolution progress
plotter = EvolutionPlotter()
plotter.plot_fitness_evolution(
    router.get_fitness_history(),
    save_path="evolution_progress.png"
)

# Analyze expert specialization
specialization = viz.analyze_expert_specialization(
    model, topology, validation_data
)
viz.plot_specialization_matrix(
    specialization,
    save_path="expert_specialization.png"
)
```

## Best Practices

### 1. Evolution Configuration

- **Population Size**: Start with 50-100 for simple problems, scale up for complex ones
- **Mutation Rate**: Begin with 0.1, adapt based on diversity
- **Generations**: Budget 200-1000 generations depending on complexity
- **Fitness Function**: Balance task performance with efficiency metrics

### 2. Hardware Optimization

- **Profile Early**: Measure baseline performance on target hardware
- **Set Constraints**: Define memory and latency budgets upfront
- **Test Frequently**: Validate evolved topologies on actual hardware
- **Multi-Device**: Evolve separate topologies for different deployment targets

### 3. Debugging Evolution

- **Monitor Diversity**: Ensure population doesn't converge prematurely
- **Check Fitness Function**: Verify it captures what you want to optimize
- **Visualize Progress**: Use plots to identify convergence issues
- **Save Checkpoints**: Preserve promising intermediate results

### 4. Production Deployment

- **Validate Thoroughly**: Test evolved models extensively before deployment
- **Monitor Performance**: Track real-world performance vs. expected
- **Plan Rollback**: Have fallback to baseline models if needed
- **Version Control**: Track which topology version is deployed where

## Troubleshooting

### Common Issues

#### Evolution Not Converging
```python
# Check population diversity
diversity = router.get_population_diversity()
if diversity < 0.1:
    router.increase_mutation_rate(factor=1.5)
    router.add_random_individuals(count=10)
```

#### Poor Performance on Target Hardware
```python
# Add hardware-specific constraints to fitness
def hardware_aware_fitness(topology, model, data):
    accuracy = evaluate_accuracy(model, data)
    latency = measure_latency(model, target_device)
    memory = measure_memory(model, target_device)
    
    # Heavy penalties for constraint violations
    penalty = 0
    if latency > max_latency:
        penalty += (latency - max_latency) * 10
    if memory > max_memory:
        penalty += (memory - max_memory) * 5
        
    return accuracy - penalty
```

#### Memory Issues During Evolution
```python
# Reduce population size or use gradient checkpointing
router = EvolvingMoERouter(
    population_size=25,  # Reduced from 100
    gradient_checkpointing=True,
    offload_to_cpu=True
)
```

### Getting Help

- **Documentation**: [https://self-evolving-moe.readthedocs.io](https://self-evolving-moe.readthedocs.io)
- **GitHub Issues**: [Report bugs and request features](https://github.com/terragon-labs/self-evolving-moe-router/issues)
- **Discussions**: [Community Q&A](https://github.com/terragon-labs/self-evolving-moe-router/discussions)
- **Examples**: Check the `examples/` directory for more use cases

## Next Steps

Now that you have the basics working:

1. **Explore Examples**: Look at `examples/` for more complex use cases
2. **Read Architecture Guide**: Understand the internals in `ARCHITECTURE.md`
3. **Try Different Algorithms**: Experiment with evolution strategies and NSGA-II
4. **Optimize for Your Hardware**: Use hardware-aware evolution
5. **Contribute**: Help improve the framework by contributing code or documentation

Happy evolving! 🧬🚀