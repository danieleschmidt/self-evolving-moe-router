"""
EvolutionarySearch: neuroevolution loop over sparse MoE routing topologies.

A *routing topology* is a binary mask (num_experts,) that controls which
experts the router is allowed to activate.  The loop:

  1. Initialise a population of N random masks, each with `num_active`
     bits set (exactly the target sparsity).
  2. For each mask, plug it into a shared MoERouter, fine-tune for K steps
     on a small language-modelling task, and record the validation loss.
  3. Tournament-select parents, apply bit-flip mutation, repeat.

Only masks that keep exactly `num_active` experts enabled are allowed
(to maintain a fair comparison across topologies).
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Population individual
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """One routing topology plus its fitness (lower val-loss = better)."""
    mask: torch.Tensor          # bool (num_experts,)
    fitness: float = float("inf")   # validation loss — lower is better

    def clone(self) -> "Individual":
        return Individual(mask=self.mask.clone(), fitness=self.fitness)


# ---------------------------------------------------------------------------
# Evolutionary search engine
# ---------------------------------------------------------------------------

class EvolutionarySearch:
    """
    Evolutionary search over binary routing masks for sparse MoE routers.

    Args:
        num_experts:    Total number of experts.
        num_active:     How many experts each mask keeps enabled.
        pop_size:       Population size (N).
        mutation_rate:  Probability of each bit being flipped during mutation
                        (subject to the constraint that `num_active` bits
                        remain set — bits are swapped in pairs to preserve it).
        tournament_k:   Tournament selection size.
        fitness_fn:     Callable(mask) -> float  — returns validation loss for
                        the given mask.  Lower is better.
    """

    def __init__(
        self,
        num_experts: int,
        num_active: int,
        pop_size: int = 10,
        mutation_rate: float = 0.3,
        tournament_k: int = 3,
        fitness_fn: Callable[[torch.Tensor], float] | None = None,
    ) -> None:
        assert 1 <= num_active <= num_experts
        self.num_experts = num_experts
        self.num_active = num_active
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.fitness_fn = fitness_fn

        self.population: List[Individual] = self._init_population()
        self.best: Individual | None = None
        self.history: List[float] = []    # best fitness per generation

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _random_mask(self) -> torch.Tensor:
        """Random binary mask with exactly `num_active` bits set."""
        idx = torch.randperm(self.num_experts)[: self.num_active]
        mask = torch.zeros(self.num_experts, dtype=torch.bool)
        mask[idx] = True
        return mask

    def _init_population(self) -> List[Individual]:
        return [Individual(mask=self._random_mask()) for _ in range(self.pop_size)]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Bit-swap mutation: each active bit has `mutation_rate` chance to
        swap with a randomly chosen inactive bit.  This preserves the
        exact number of active experts.
        """
        child = mask.clone()
        on_idx  = child.nonzero(as_tuple=True)[0].tolist()
        off_idx = (~child).nonzero(as_tuple=True)[0].tolist()

        for i in on_idx:
            if off_idx and random.random() < self.mutation_rate:
                j = random.choice(off_idx)
                child[i] = False
                child[j] = True
                off_idx.remove(j)
                off_idx.append(i)
        return child

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _tournament(self) -> Individual:
        """Top-1 tournament selection (lower fitness = better)."""
        contestants = random.sample(self.population, min(self.tournament_k, len(self.population)))
        return min(contestants, key=lambda ind: ind.fitness)

    # ------------------------------------------------------------------
    # One generation
    # ------------------------------------------------------------------

    def step(self, fitness_fn: Callable[[torch.Tensor], float] | None = None) -> float:
        """
        Run one generation: evaluate → select → mutate.

        Args:
            fitness_fn: Override the stored fitness function for this step.

        Returns:
            Best fitness value in this generation.
        """
        fn = fitness_fn or self.fitness_fn
        if fn is None:
            raise ValueError("fitness_fn must be provided at construction or per-step")

        # Evaluate any un-scored individuals
        for ind in self.population:
            if ind.fitness == float("inf"):
                ind.fitness = fn(ind.mask)

        # Track best
        gen_best = min(self.population, key=lambda i: i.fitness)
        if self.best is None or gen_best.fitness < self.best.fitness:
            self.best = gen_best.clone()
        self.history.append(gen_best.fitness)

        # Create next generation via tournament + mutation
        new_pop: List[Individual] = [self.best.clone()]   # elitism: keep best
        while len(new_pop) < self.pop_size:
            parent = self._tournament()
            child_mask = self._mutate(parent.mask)
            new_pop.append(Individual(mask=child_mask))

        self.population = new_pop
        return gen_best.fitness

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(
        self,
        generations: int,
        fitness_fn: Callable[[torch.Tensor], float] | None = None,
        verbose: bool = True,
    ) -> Individual:
        """
        Evolve for `generations` generations.

        Returns the best individual found.
        """
        for g in range(generations):
            best_fitness = self.step(fitness_fn)
            if verbose:
                active = self.best.mask.sum().item() if self.best else "?"
                print(
                    f"Gen {g+1:3d}/{generations}  "
                    f"best_val_loss={best_fitness:.4f}  "
                    f"active_experts={active}/{self.num_experts}"
                )
        return self.best  # type: ignore[return-value]
