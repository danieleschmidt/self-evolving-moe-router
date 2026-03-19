"""Tests for EvolutionarySearch."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
import torch
import pytest
from self_evolving_moe import EvolutionarySearch, Individual


def _dummy_fitness(mask: torch.Tensor) -> float:
    """Fitness = fraction of mask bits that are NOT in positions {0, 1}.
    Optimal mask is [True, True, False, False] → fitness 0.0."""
    active = mask.nonzero(as_tuple=True)[0].tolist()
    bad = sum(1 for a in active if a not in (0, 1))
    return float(bad) / len(active)


def test_population_init():
    es = EvolutionarySearch(num_experts=4, num_active=2, pop_size=6)
    assert len(es.population) == 6
    for ind in es.population:
        assert ind.mask.sum() == 2, "Each mask should have exactly 2 active bits"


def test_mask_constraint_preserved_after_step():
    es = EvolutionarySearch(num_experts=6, num_active=3, pop_size=8, fitness_fn=lambda m: 0.0)
    es.step()
    for ind in es.population:
        assert ind.mask.sum() == 3, f"Mask has {ind.mask.sum()} active bits, expected 3"


def test_fitness_improves_or_stays():
    """Over enough generations a simple fitness should not get worse."""
    torch.manual_seed(0)
    random.seed(0)
    es = EvolutionarySearch(
        num_experts=4, num_active=2, pop_size=10,
        mutation_rate=0.5, tournament_k=3, fitness_fn=_dummy_fitness
    )
    es.run(generations=15, verbose=False)
    assert es.best is not None
    # With C(4,2)=6 possible masks and a clear optimum,
    # the search should almost always find val_loss ≤ initial.
    initial_best = es.history[0]
    final_best = es.history[-1]
    assert final_best <= initial_best + 0.01, (
        f"Fitness got worse: {initial_best:.3f} → {final_best:.3f}"
    )


def test_best_individual_tracked():
    es = EvolutionarySearch(num_experts=4, num_active=2, pop_size=5, fitness_fn=lambda m: 1.0)
    es.run(generations=3, verbose=False)
    assert es.best is not None
    assert isinstance(es.best, Individual)
    assert es.best.mask.sum() == 2


def test_history_length():
    es = EvolutionarySearch(num_experts=4, num_active=2, pop_size=4, fitness_fn=lambda m: 0.5)
    es.run(generations=7, verbose=False)
    assert len(es.history) == 7


def test_elitism_keeps_best():
    """Best individual should never disappear from population."""
    calls = [0]

    def fitness(m):
        calls[0] += 1
        # Expert 0 being active is rewarded
        return 0.0 if m[0] else 1.0

    es = EvolutionarySearch(num_experts=4, num_active=2, pop_size=6, fitness_fn=fitness)
    es.run(generations=5, verbose=False)
    # After elitism, best must have survived
    assert any(ind.mask[0] for ind in es.population), (
        "Elitism failed: best individual (with expert 0 active) lost from population"
    )


def test_no_fitness_fn_raises():
    es = EvolutionarySearch(num_experts=4, num_active=2, pop_size=4)
    with pytest.raises(ValueError):
        es.step()  # no fitness_fn provided


def test_per_step_fitness_fn_override():
    es = EvolutionarySearch(num_experts=4, num_active=2, pop_size=4, fitness_fn=lambda m: 99.0)
    # Override with a different function per step
    override_called = [False]

    def override(m):
        override_called[0] = True
        return 0.0

    es.step(fitness_fn=override)
    assert override_called[0]
