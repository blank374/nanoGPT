"""Train Cell Graph prefixes with random physical skipping.

This is the robustness stage before halt-threshold/Pareto tuning. It trains
randomly selected prefixes against the full path, so early exits are not just
post-hoc truncations of a model trained only for full depth.
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GPT, GPTConfig


class StopAfter(nn.Module):
    def __init__(self, stop_after):
        super().__init__()
        self.stop_after = stop_after
        self.calls = 0

    def reset(self):
        self.calls = 0

    def forward(self, state):
        self.calls += 1
        value = 10.0 if self.calls >= self.stop_after else -10.0
        return state.new_full((*state.shape[:-1], 1), value)


def make_batches(data, batch_size, block_size, count, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - block_size - 1,
                          size=(count, batch_size))
    return [
        (
            torch.from_numpy(np.stack([data[i:i + block_size] for i in row]).astype(np.int64)),
            torch.from_numpy(np.stack([data[i + 1:i + 1 + block_size] for i in row]).astype(np.int64)),
        )
        for row in starts
    ]


def prefix_logits(model, x, depth, head):
    head.stop_after = depth
    head.reset()
    model.set_cell_graph_halt_head(head)
    return model._forward_cell_graph_logits(
        x, all_logits=True, physical_halt=True, halt_threshold=0.5
    )


def train_model(model, batches, random_skip, steps, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    rng = torch.Generator(device=batches[0][0].device).manual_seed(seed)
    head = StopAfter(1)
    model.train()
    for step, (x, y) in enumerate(batches[:steps]):
        optimizer.zero_grad(set_to_none=True)
        if random_skip:
            with torch.no_grad():
                teacher = model._forward_cell_graph_logits(x, all_logits=True)
            depth = int(torch.randint(1, model.config.n_layer + 1, (), generator=rng))
            logits = prefix_logits(model, x, depth, head)
            temperature = 2.0
            kl = F.kl_div(
                F.log_softmax(logits / temperature, dim=-1),
                F.softmax(teacher.detach() / temperature, dim=-1),
                reduction="none",
            ).sum(dim=-1).mean() * (temperature ** 2)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) + 0.5 * kl
        else:
            logits = model._forward_cell_graph_logits(x, all_logits=True)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
    return head


@torch.no_grad()
def evaluate_prefixes(model, batches):
    model.eval()
    head = StopAfter(1)
    results = {}
    for depth in range(1, model.config.n_layer + 1):
        losses = []
        for x, y in batches:
            logits = prefix_logits(model, x, depth, head)
            losses.append(float(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )))
        results[depth] = sum(losses) / len(losses)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(4)
    root = Path("data/shakespeare_char")
    with open(root / "meta.pkl", "rb") as handle:
        vocab_size = pickle.load(handle)["vocab_size"]
    train_data = np.memmap(root / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(root / "val.bin", dtype=np.uint16, mode="r")
    batch_size, block_size = 32, 64
    train_batches = [(x.to(args.device), y.to(args.device)) for x, y in
                     make_batches(train_data, batch_size, block_size, args.steps, 4007)]
    val_batches = [(x.to(args.device), y.to(args.device)) for x, y in
                   make_batches(val_data, batch_size, block_size, 20, 5007)]
    kwargs = dict(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=4, n_head=4, n_embd=64, dropout=0.0, bias=True,
        cell_graph=True, cell_graph_cells_per_step=2,
        cell_graph_attention_cells=1, cell_graph_atom_size=16,
    )
    torch.manual_seed(11)
    seed = GPT(GPTConfig(**kwargs)).to(args.device)
    initial = {key: value.detach().clone() for key, value in seed.state_dict().items()}
    full = GPT(GPTConfig(**kwargs)).to(args.device)
    random_skip = GPT(GPTConfig(**kwargs)).to(args.device)
    full.load_state_dict(initial)
    random_skip.load_state_dict(initial)
    train_model(full, train_batches, False, args.steps, 7007)
    train_model(random_skip, train_batches, True, args.steps, 8007)
    full_results = evaluate_prefixes(full, val_batches)
    random_results = evaluate_prefixes(random_skip, val_batches)
    print("RANDOM-SKIP TRAINING")
    print("depth,full_training_ce,random_skip_training_ce,delta")
    for depth in full_results:
        print(f"{depth},{full_results[depth]:.6f},{random_results[depth]:.6f},"
              f"{random_results[depth] - full_results[depth]:+.6f}")
    print(f"full_training_ppl={math.exp(full_results[4]):.4f} "
          f"random_skip_ppl={math.exp(random_results[4]):.4f}")


if __name__ == "__main__":
    main()
