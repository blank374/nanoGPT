"""V3-0: test whether Cell states contain a usable halt signal.

This is a probe only. It does not skip computation and makes no latency claim.
The frozen Cell Graph supplies intermediate states; a small halt head is trained
to predict whether that state is already close enough to the final prediction.
"""

import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import GPT, GPTConfig


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


@torch.no_grad()
def collect_probe_data(model, batches, margin):
    features, labels = [], []
    model.eval()
    for x, y in batches:
        final_logits, _ = model(x, y)
        states = model.cell_graph.last_step_states
        step_logits = model.lm_head(model.transformer.ln_f(states))
        final_token_error = F.cross_entropy(
            final_logits.reshape(-1, final_logits.size(-1)),
            y.reshape(-1), reduction="none",
        ).view_as(y)
        step_token_error = F.cross_entropy(
            step_logits.reshape(-1, step_logits.size(-1)),
            y.unsqueeze(-1).expand_as(step_logits[..., 0]).reshape(-1),
            reduction="none",
        ).view(y.size(0), y.size(1), states.size(2))
        # Oracle says STOP when this step is within `margin` nats of the
        # complete path. This is a diagnostic label, not a runtime target.
        stop = step_token_error <= final_token_error.unsqueeze(-1) + margin
        features.append(states.reshape(-1, states.size(-1)).float())
        labels.append(stop.reshape(-1).float())
    return torch.cat(features), torch.cat(labels)


def binary_auc(scores, labels):
    order = torch.argsort(scores)
    ranked = labels[order]
    positives = ranked.sum().item()
    negatives = ranked.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = (torch.arange(1, ranked.numel() + 1)[ranked.bool()]
                .float().sum().item())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


class StopAfter(nn.Module):
    """Deterministic halt head for forced-depth diagnosis."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--probe_steps", type=int, default=150)
    parser.add_argument("--margin", type=float, default=0.05)
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
    train_batches = make_batches(train_data, batch_size, block_size, args.steps, 1007)
    probe_batches = make_batches(train_data, batch_size, block_size, 50, 2007)

    config = GPTConfig(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=4, n_head=4, n_embd=64, dropout=0.0, bias=True,
        cell_graph=True, cell_graph_cells_per_step=2,
        cell_graph_attention_cells=1, cell_graph_atom_size=16,
    )
    model = GPT(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    model.train()
    for x, y in train_batches:
        x, y = x.to(args.device), y.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    features, labels = collect_probe_data(model, [
        (x.to(args.device), y.to(args.device)) for x, y in probe_batches
    ], args.margin)
    split = int(features.size(0) * 0.8)
    probe = nn.Sequential(nn.LayerNorm(features.size(-1)), nn.Linear(features.size(-1), 1))
    probe.to(args.device)
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3)
    train_x, test_x = features[:split].to(args.device), features[split:].to(args.device)
    train_y, test_y = labels[:split].to(args.device), labels[split:].to(args.device)
    for _ in range(args.probe_steps):
        logits = probe(train_x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_y)
        probe_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        probe_optimizer.step()

    with torch.no_grad():
        scores = probe(test_x).squeeze(-1).sigmoid().cpu()
    test_y = test_y.cpu()
    auc = binary_auc(scores, test_y)
    thresholds = torch.linspace(0.05, 0.95, 181)
    accuracies = torch.stack([((scores >= threshold) == test_y.bool()).float().mean()
                              for threshold in thresholds])
    best_index = int(accuracies.argmax())
    positive_rate = test_y.mean().item()
    print("V3-0 HALT PROBE")
    print(f"oracle_margin={args.margin:.3f} samples={test_y.numel()}")
    print(f"stop_positive_rate={positive_rate:.3f}")
    print(f"halt_auc={auc:.4f}")
    print(f"best_accuracy={accuracies[best_index].item():.4f} "
          f"threshold={thresholds[best_index].item():.3f}")
    if math.isfinite(auc) and auc >= 0.75:
        print("decision=promising: proceed to physical skip and latency benchmark")
    else:
        print("decision=weak separation: improve halt features/teacher before latency work")

    # V3-C pilot: the halt head now controls an actual early-exit loop.
    # This is intentionally batch-1: mixed-depth batches need regrouping.
    model.set_cell_graph_halt_head(probe.eval())
    eval_batches = make_batches(val_data, 1, block_size, 64, 3007)
    threshold = thresholds[best_index].item()
    full_times, adaptive_times, depths = [], [], []
    full_nll, adaptive_nll, agreement = [], [], []
    for x, y in eval_batches[:8]:
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            model(x, physical_halt=False)
            model(x, physical_halt=True, halt_threshold=threshold)
    with torch.no_grad():
        for x, y in eval_batches:
            x, y = x.to(args.device), y.to(args.device)
            start = time.perf_counter()
            full_logits, _ = model(x, physical_halt=False)
            full_times.append(time.perf_counter() - start)
            full_steps = model.config.n_layer
            start = time.perf_counter()
            adaptive_logits, _ = model(
                x, physical_halt=True, halt_threshold=threshold
            )
            adaptive_times.append(time.perf_counter() - start)
            depths.append(model.cell_graph.last_step_states.size(2))
            target = y[:, -1]
            full_nll.append(float(F.cross_entropy(full_logits[:, -1], target)))
            adaptive_nll.append(float(F.cross_entropy(adaptive_logits[:, -1], target)))
            agreement.append(float(
                full_logits[:, -1].argmax(dim=-1).eq(
                    adaptive_logits[:, -1].argmax(dim=-1)
                ).float().mean()
            ))
    full_latency = sum(full_times) / len(full_times)
    adaptive_latency = sum(adaptive_times) / len(adaptive_times)
    print("V3-C BATCH-1 PILOT")
    print(f"threshold={threshold:.3f} full_steps={full_steps} "
          f"mean_steps={sum(depths) / len(depths):.2f}")
    print(f"full_latency_ms={1000 * full_latency:.3f} "
          f"adaptive_latency_ms={1000 * adaptive_latency:.3f} "
          f"speedup={full_latency / adaptive_latency:.3f}x")
    print(f"full_last_token_nll={sum(full_nll) / len(full_nll):.4f} "
          f"adaptive_last_token_nll={sum(adaptive_nll) / len(adaptive_nll):.4f} "
          f"agreement={sum(agreement) / len(agreement):.3f}")

    print("FORCED-DEPTH DIAGNOSTIC")
    for forced_depth in range(1, config.n_layer + 1):
        forced_head = StopAfter(forced_depth)
        model.set_cell_graph_halt_head(forced_head)
        nll, agree = [], []
        for x, y in eval_batches:
            x, y = x.to(args.device), y.to(args.device)
            forced_head.reset()
            with torch.no_grad():
                full_logits, _ = model(x, physical_halt=False)
                forced_logits, _ = model(
                    x, physical_halt=True, halt_threshold=0.5
                )
            target = y[:, -1]
            nll.append(float(F.cross_entropy(forced_logits[:, -1], target)))
            agree.append(float(
                forced_logits[:, -1].argmax(dim=-1).eq(
                    full_logits[:, -1].argmax(dim=-1)
                ).float().mean()
            ))
        print(f"depth={forced_depth} nll={sum(nll) / len(nll):.4f} "
              f"agreement={sum(agree) / len(agree):.3f}")


if __name__ == "__main__":
    main()
