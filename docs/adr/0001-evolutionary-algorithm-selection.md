# ADR-0001: Evolutionary Algorithm Selection for Routing Optimization

## Status

Accepted

## Context

The Self-Evolving MoE-Router requires an optimization algorithm to discover optimal sparse routing topologies for Mixture of Experts models. Traditional gradient-based methods are not suitable for this discrete optimization problem, as routing topologies involve binary connection matrices and categorical expert assignments.

Key requirements:
- Handle discrete/combinatorial search spaces
- Support multi-objective optimization (accuracy, latency, memory)
- Scale to large search spaces (thousands of possible connections)
- Enable hardware-aware optimization
- Allow online/continuous optimization during deployment

## Decision

We will implement a hybrid evolutionary algorithm approach combining:

1. **Primary Algorithm**: Genetic Algorithm (GA) with tournament selection
2. **Secondary Algorithms**: Evolution Strategies (ES) and NEAT for specific use cases
3. **Multi-objective**: NSGA-II for Pareto-optimal solutions
4. **Online Evolution**: (1+λ)-ES for continuous deployment optimization

### Specific Implementation Details:

- **Population Size**: 50-200 individuals (adaptive based on problem complexity)
- **Selection**: Tournament selection with size 3-7
- **Crossover**: Uniform crossover for routing matrices, arithmetic for parameters
- **Mutation Rate**: Adaptive (0.01-0.3) based on population diversity
- **Elitism**: Top 10% preserved across generations

## Consequences

### Benefits:
- **No gradient requirements**: Works with discrete routing topologies
- **Multi-objective optimization**: Can balance competing objectives naturally
- **Robust exploration**: Avoids local optima through population diversity
- **Hardware-aware**: Fitness functions can incorporate device constraints
- **Parallelizable**: Population evaluation can be distributed
- **Online adaptation**: Continuous improvement during deployment

### Drawbacks:
- **Computational cost**: Requires multiple model evaluations per generation
- **Convergence time**: May require hundreds of generations for complex problems
- **Hyperparameter sensitivity**: Evolution parameters need tuning per problem
- **Memory overhead**: Maintaining population of routing topologies

### Risks and Mitigations:
- **Premature convergence**: Mitigated through diversity maintenance and adaptive mutation
- **Expensive fitness evaluation**: Mitigated through surrogate models and incremental evaluation
- **Hyperparameter tuning**: Mitigated through meta-evolution and adaptive parameters

## Alternatives Considered

### 1. Reinforcement Learning (RL)
- **Pros**: Can learn routing policies, handles sequential decisions
- **Cons**: Requires reward shaping, unstable training, complex state representation
- **Verdict**: Rejected due to complexity and instability

### 2. Differentiable Neural Architecture Search (DNAS)
- **Pros**: Gradient-based optimization, faster convergence
- **Cons**: Requires continuous relaxation of discrete choices, memory intensive
- **Verdict**: Rejected due to memory constraints and approximation errors

### 3. Random Search
- **Pros**: Simple implementation, good baseline
- **Cons**: Inefficient exploration, no learning from previous evaluations
- **Verdict**: Rejected due to poor sample efficiency

### 4. Bayesian Optimization
- **Pros**: Sample efficient, uncertainty quantification
- **Cons**: Doesn't scale to high-dimensional discrete spaces well
- **Verdict**: Considered for hyperparameter optimization but not main algorithm

### 5. Simulated Annealing
- **Pros**: Simple, can escape local optima
- **Cons**: Single-point search, no population diversity, parameter sensitive
- **Verdict**: Rejected due to lack of multi-objective support

## Implementation Notes

### Phase 1: Core GA Implementation
- Binary tournament selection
- Uniform crossover for routing matrices
- Bit-flip mutations for connections
- Multi-objective fitness with weighted scalarization

### Phase 2: Advanced Features
- NSGA-II for true multi-objective optimization
- Adaptive parameter control (mutation rate, crossover rate)
- Population diversity maintenance mechanisms
- Surrogate-assisted evaluation for expensive fitness functions

### Phase 3: Online Evolution
- (1+λ)-ES for deployment-time optimization
- Incremental fitness evaluation
- Population size adaptation based on available compute

### Configuration Options:
```yaml
evolution:
  algorithm: "genetic_algorithm"  # genetic_algorithm, evolution_strategies, nsga2
  population_size: 100
  tournament_size: 5
  crossover_rate: 0.7
  mutation_rate: 0.1
  elitism_rate: 0.1
  max_generations: 1000
  
  # Multi-objective settings
  objectives:
    - name: "accuracy"
      weight: 1.0
      maximize: true
    - name: "latency" 
      weight: 0.2
      maximize: false
    - name: "memory"
      weight: 0.1
      maximize: false
```

### Evaluation Metrics:
- **Convergence Speed**: Generations to reach 95% of final fitness
- **Solution Quality**: Final Pareto front coverage
- **Diversity Maintenance**: Population entropy over generations
- **Scalability**: Performance with increasing problem size

---

*Date: 2025-01-15*
*Authors: Self-Evolving MoE Team*
*Reviewers: Architecture Review Board*