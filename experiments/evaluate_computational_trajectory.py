"""Evaluate how compute and route diversity evolve across training checkpoints."""

import argparse
import csv
import glob
import math
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dynamic_resource_eval import load_data, load_model


def fixed_windows(data, count, block_size, seed):
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(data) - block_size - 1, count, replace=False)
    xs, ys = [], []
    for start in starts:
        xs.append(torch.from_numpy(data[start:start + block_size].astype(np.int64)))
        ys.append(torch.from_numpy(data[start + 1:start + block_size + 1].astype(np.int64)))
    return torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def evaluate(path, xs, ys, device, tolerance):
    model, step = load_model(path, device)
    model.eval()
    widths = model.config.dynamic_resource_widths
    full_path = [widths[-1]] * model.config.n_layer
    dynamic_nll = full_nll = 0.0
    dynamic_losses, full_losses = [], []
    costs, routes = [], []
    for x, y in zip(xs, ys):
        x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
        dynamic = model._forward_dynamic_resource_logits(x, all_logits=True)
        full = model._forward_dynamic_resource_logits(
            x, forced_path=full_path, record_stats=False, all_logits=True
        )
        dynamic_item = F.cross_entropy(dynamic.flatten(0, 1), y.flatten()).item()
        full_item = F.cross_entropy(full.flatten(0, 1), y.flatten()).item()
        dynamic_losses.append(dynamic_item)
        full_losses.append(full_item)
        dynamic_nll += dynamic_item * y.numel()
        full_nll += full_item * y.numel()
        modes = model.last_dynamic_resource_modes
        _, hard, _ = model._computational_potential_cost(
            model.last_dynamic_resource_probs, modes
        )
        costs.append(hard.item())
        routes.append(tuple(widths[int(v)] for v in modes[0]))
    tokens = ys.numel()
    dynamic_loss, full_loss = dynamic_nll / tokens, full_nll / tokens
    route_counts = Counter(routes)
    top1 = max(route_counts.values()) / len(routes)
    return {
        "step": step,
        "dynamic_ppl": math.exp(dynamic_loss),
        "full_ppl": math.exp(full_loss),
        "quality_gap": dynamic_loss - full_loss,
        "feasible_fraction": sum(
            a - b <= tolerance for a, b in zip(dynamic_losses, full_losses)
        ) / len(dynamic_losses),
        "average_cost": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "unique_paths": len(route_counts),
        "top1_coverage": top1,
        "route_entropy": -sum(
            (n / len(routes)) * math.log(max(n / len(routes), 1e-12))
            for n in route_counts.values()
        ),
        "quality_feasible_aggregate": int(dynamic_loss - full_loss <= tolerance),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data", default="data/shakespeare_char/val.bin")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--quality-tolerance", type=float, default=0.02)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    data = load_data(args.data)
    xs, ys = fixed_windows(data, args.samples, args.block_size, args.seed)
    paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, "ckpt_iter_*.pt")),
                   key=lambda path: int(os.path.basename(path).split("_")[-1].split(".")[0]))
    if not paths:
        raise SystemExit("no ckpt_iter_*.pt files found")
    rows = [evaluate(path, xs, ys, device, args.quality_tolerance) for path in paths]
    output = args.output or os.path.join(args.checkpoint_dir, "computational_trajectory.csv")
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
    for row in rows:
        print("step={step} ppl={dynamic_ppl:.3f} cost={average_cost:.3f} "
              "cost_std={cost_std:.3f} paths={unique_paths} entropy={route_entropy:.3f} "
              "quality_ok={quality_feasible_aggregate}".format(**row))


if __name__ == "__main__":
    main()
