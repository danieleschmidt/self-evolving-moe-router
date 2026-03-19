# self-evolving-moe-router

> **Neuroevolution for sparse Mixture-of-Experts routing topology discovery**

Instead of hand-designing which experts a token can route to, let evolution find the best connectivity pattern automatically.

---

## What It Does

Mixture-of-Experts (MoE) models route each token to a small subset of specialist networks.  The *routing topology* — which experts are even reachable — is usually fixed and uniform.  This repo evolves it.

A **routing topology** is a binary mask over the expert pool.  The evolutionary loop:

1. Initialise a population of N random masks (each with exactly `num_active` bits set).
2. Score each mask: plug it into the router, run K gradient steps on the task, measure validation loss.
3. Tournament-select parents → bit-swap mutation → next generation.
4. Repeat until the budget is exhausted.

The result is a sparse connectivity pattern discovered by selection pressure, not gradient descent.

---

## Core Components

### `MoERouter`

Sparse top-k router driven by a binary connectivity mask.

```python
from self_evolving_moe import MoERouter

router = MoERouter(input_dim=256, num_experts=8, num_active=2)
# Only experts 0, 1, 3 are reachable
router.set_mask(torch.tensor([True, True, False, True, False, False, False, False]))

weights, expert_ids = router(x)  # x: (B, T, D)
```

- `mask` is a `bool` tensor of shape `(num_experts,)`.  Disabled experts receive logit `-inf` before softmax — they never win.
- Setting a new mask is O(1) and gradient-free; the gate weights remain trainable.

### `SlimmableExpert`

MLP expert that operates at **100 / 75 / 50 %** of its full hidden width.  Narrower widths slice the weight matrices — no extra parameters.

```python
from self_evolving_moe import SlimmableExpert

expert = SlimmableExpert(input_dim=256)
out = expert(x, width=0.75)   # runs at 75% capacity
```

Useful for adaptive deployment: pick the smallest width that meets a latency budget.

### `EvolutionarySearch`

Population-based evolutionary search over binary routing masks.

```python
from self_evolving_moe import EvolutionarySearch

def fitness(mask):
    """Return validation loss for this routing topology."""
    router.set_mask(mask)
    # ... fine-tune K steps, evaluate, return val_loss
    return val_loss

searcher = EvolutionarySearch(
    num_experts=8,
    num_active=2,
    pop_size=20,
    mutation_rate=0.3,
    tournament_k=5,
    fitness_fn=fitness,
)
best = searcher.run(generations=50, verbose=True)
print(f"Best mask: {best.mask.int().tolist()}, val_loss={best.fitness:.4f}")
```

**Genetic operators:**
- *Mutation*: each active bit has `mutation_rate` chance to swap with a randomly chosen inactive bit.  The number of active experts is always preserved.
- *Selection*: top-1 tournament (size `tournament_k`).
- *Elitism*: the single best individual survives each generation unchanged.

---

## Quick Start

```bash
git clone https://github.com/danieleschmidt/self-evolving-moe-router
cd self-evolving-moe-router
pip install -e .

# Run the demo (evolves routing for a 4-expert toy LM)
python demo.py
```

### Demo output (example)

```
Gen   1/10  best_val_loss=1.7787  active_experts=2/4
Gen   8/10  best_val_loss=1.7087  active_experts=2/4
...
Best topology: mask=[1, 1, 0, 0]  active_experts=[0, 1]  val_loss=1.7087
```

---

## Tests

```bash
pytest tests/ -v
```

21 tests, ~98 % coverage.

---

## Design Choices

| Choice | Rationale |
|--------|-----------|
| Binary mask, not soft gate | Clean separation: *topology* (discrete, evolved) vs *routing weights* (continuous, gradient-trained) |
| Mutation preserves cardinality | Fair comparison across topologies — each uses exactly `num_active` experts |
| Elitism (keep best) | Prevents regression; small population converges faster with it |
| SlimmableExpert slices weights | Zero extra params; deployed width chosen at inference time |
| Fitness = validation loss | Directly optimises generalisation, not a proxy |

---

## Project Structure

```
src/self_evolving_moe/
├── __init__.py        # public API
├── router.py          # MoERouter
├── experts.py         # SlimmableExpert
└── evolution.py       # EvolutionarySearch, Individual

tests/
├── test_router.py
├── test_experts.py
└── test_evolution.py

demo.py                # end-to-end toy demo
```

---

## License

MIT
