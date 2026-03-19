"""
demo.py — Evolve sparse routing topology for a 4-expert, 2-active MoE LM task.

Run:
    ~/anaconda3/bin/python3 demo.py

What it does
------------
1. Builds a tiny character-level language model whose routing layer is a
   MoERouter with 4 experts, top-2 active.
2. Uses EvolutionarySearch to search over which 2 of the 4 experts to keep
   enabled (there are C(4,2)=6 possible masks — tiny but illustrative).
3. Evaluates each candidate mask by:
   - setting the mask in the router
   - running K=30 gradient steps on a small text snippet
   - measuring validation cross-entropy loss
4. Reports fitness improvement across generations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# Make reproducible
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ── imports from this package ────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from self_evolving_moe import MoERouter, SlimmableExpert, EvolutionarySearch

# ── Hyperparams ──────────────────────────────────────────────────────────────
NUM_EXPERTS  = 4
NUM_ACTIVE   = 2
INPUT_DIM    = 64     # token embedding size
VOCAB_SIZE   = 65     # printable ASCII chars
SEQ_LEN      = 32
BATCH_SIZE   = 16
TRAIN_STEPS  = 30     # gradient steps per fitness evaluation
LR           = 3e-3
POP_SIZE     = 8
GENERATIONS  = 10

# ── Tiny text corpus (Shakespeare excerpt) ───────────────────────────────────
_TEXT = (
    "To be or not to be that is the question "
    "Whether tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune "
    "Or to take arms against a sea of troubles "
    "And by opposing end them to die to sleep "
    "No more and by a sleep to say we end "
    "The heartache and the thousand natural shocks "
    "That flesh is heir to tis a consummation "
    "Devoutly to be wished to die to sleep "
    "To sleep perchance to dream ay there s the rub "
) * 20

_CHARS = sorted(set(_TEXT))
_C2I = {c: i for i, c in enumerate(_CHARS)}
_VOCAB = len(_CHARS)
_DATA = torch.tensor([_C2I[c] for c in _TEXT], dtype=torch.long)


def _get_batch(split="train"):
    n = int(len(_DATA) * 0.9)
    data = _DATA[:n] if split == "train" else _DATA[n:]
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([data[i : i + SEQ_LEN] for i in ix])
    y = torch.stack([data[i + 1 : i + SEQ_LEN + 1] for i in ix])
    return x, y


# ── Tiny MoE LM ─────────────────────────────────────────────────────────────

class TinyMoELM(nn.Module):
    """Character-level LM: embed → MoE layer → LM head."""

    def __init__(self, vocab: int, dim: int, num_experts: int, num_active: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.router = MoERouter(dim, num_experts, num_active)
        # One slimmable expert per slot
        self.experts = nn.ModuleList([SlimmableExpert(dim) for _ in range(num_experts)])
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) → logits: (B, T, vocab)"""
        B, T = idx.shape
        x = self.embed(idx)                     # (B, T, D)
        weights, expert_ids = self.router(x)    # (B*T, K), (B*T, K)

        x_flat = x.reshape(B * T, -1)           # (N, D)
        out = torch.zeros_like(x_flat)

        K = weights.shape[-1]
        for k in range(K):
            for e in range(len(self.experts)):
                mask = (expert_ids[:, k] == e)
                if not mask.any():
                    continue
                expert_in = x_flat[mask]
                expert_out = self.experts[e](expert_in, width=1.0)
                out[mask] += weights[mask, k : k + 1] * expert_out

        return self.lm_head(out.reshape(B, T, -1))   # (B, T, V)


# ── Fitness function ─────────────────────────────────────────────────────────

def make_fitness_fn(model: TinyMoELM) -> callable:
    """
    Returns a closure that:
      1. Sets the router mask.
      2. Fine-tunes shared model weights for TRAIN_STEPS.
      3. Returns validation cross-entropy loss.
    """
    def fitness(mask: torch.Tensor) -> float:
        # Install mask
        model.router.set_mask(mask)

        # Brief fine-tune (only non-router params to keep it fair across masks)
        opt = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if "router" not in n],
            lr=LR
        )
        model.train()
        for _ in range(TRAIN_STEPS):
            xb, yb = _get_batch("train")
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            xb, yb = _get_batch("val")
            logits = model(xb)
            val_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), yb.reshape(-1)
            ).item()

        return val_loss

    return fitness


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Self-Evolving MoE Router — Topology Search Demo")
    print("=" * 60)
    print(f" Experts: {NUM_EXPERTS}  |  Active per token: {NUM_ACTIVE}")
    print(f" Search space: C({NUM_EXPERTS},{NUM_ACTIVE}) = {_ncr(NUM_EXPERTS, NUM_ACTIVE)} topologies")
    print(f" Population: {POP_SIZE}  |  Generations: {GENERATIONS}")
    print()

    model = TinyMoELM(_VOCAB, INPUT_DIM, NUM_EXPERTS, NUM_ACTIVE)
    fitness_fn = make_fitness_fn(model)

    searcher = EvolutionarySearch(
        num_experts=NUM_EXPERTS,
        num_active=NUM_ACTIVE,
        pop_size=POP_SIZE,
        mutation_rate=0.4,
        tournament_k=3,
        fitness_fn=fitness_fn,
    )

    best = searcher.run(generations=GENERATIONS, verbose=True)

    print()
    print("=" * 60)
    print(f" Best topology found:")
    print(f"   mask            = {best.mask.int().tolist()}")
    print(f"   active experts  = {best.mask.nonzero(as_tuple=True)[0].tolist()}")
    print(f"   val loss        = {best.fitness:.4f}")
    print()
    print(" Fitness history (val loss per generation):")
    for g, f in enumerate(searcher.history):
        bar = "█" * int((4.5 - min(f, 4.5)) * 10)
        print(f"   Gen {g+1:2d}: {f:.4f} {bar}")
    print("=" * 60)

    # SlimmableExpert width demo
    print()
    print(" SlimmableExpert width benchmark:")
    expert = model.experts[0]
    x_demo = torch.randn(4, INPUT_DIM)
    for w in (1.0, 0.75, 0.5):
        out = expert(x_demo, width=w)
        h = int(4 * INPUT_DIM * w)
        print(f"   width={w:.0%}  hidden={h:3d}  out_shape={tuple(out.shape)}")


def _ncr(n, r):
    from math import comb
    return comb(n, r)


if __name__ == "__main__":
    main()
