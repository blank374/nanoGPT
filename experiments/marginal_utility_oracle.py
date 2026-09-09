"""Greedy marginal-utility oracle for quality-constrained compute reduction.

This does not train a router. It answers the upper-bound question for an
existing checkpoint: how much computation can be removed per input while the
candidate CE stays within epsilon of the full path?
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dynamic_resource_eval import load_data, load_model


def windows(data, count, block_size, seed):
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(data) - block_size - 1, count, replace=False)
    xs, ys = [], []
    for start in starts:
        xs.append(torch.from_numpy(data[start:start + block_size].astype(np.int64)))
        ys.append(torch.from_numpy(data[start + 1:start + block_size + 1].astype(np.int64)))
    return torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def score(model, x, y, path):
    logits = model._forward_dynamic_resource_logits(
        x, forced_path=path, record_stats=False, all_logits=True
    )
    return F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()


@torch.no_grad()
def timed(model, x, path, repeats):
    for _ in range(2):
        model._forward_dynamic_resource_logits(x, forced_path=path, record_stats=False)
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        model._forward_dynamic_resource_logits(x, forced_path=path, record_stats=False)
        values.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="out-cpu-potential-validation/ckpt_iter_100.pt")
    parser.add_argument("--data", default="data/shakespeare_char/val.bin")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--quality-tolerance", type=float, default=0.02)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    model, step = load_model(args.checkpoint, device)
    model.eval()
    data = load_data(args.data)
    xs, ys = windows(data, args.samples, min(args.block_size, model.config.block_size), args.seed)
    widths = model.config.dynamic_resource_widths
    full_path = [widths[-1]] * model.config.n_layer
    profile = [0.0, 0.028, 0.046, 0.063, 0.068]
    rows = []
    for index, (x, y) in enumerate(zip(xs, ys)):
        x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
        full_loss = score(model, x, y, full_path)
        path = list(full_path)
        # Coordinate descent: repeatedly try the cheapest action at each
        # layer, accepting only candidates inside the full-model quality gate.
        changed = True
        while changed:
            changed = False
            for layer in range(model.config.n_layer):
                current = path[layer]
                for candidate in widths:
                    if candidate >= current:
                        continue
                    trial = list(path)
                    trial[layer] = candidate
                    trial_loss = score(model, x, y, trial)
                    if trial_loss - full_loss <= args.quality_tolerance:
                        path = trial
                        changed = True
                        break
        loss = score(model, x, y, path)
        cost = sum(profile[widths.index(width)] for width in path)
        full_cost = profile[-1] * model.config.n_layer
        full_ms = timed(model, x, full_path, args.timing_repeats)
        oracle_ms = timed(model, x, path, args.timing_repeats)
        rows.append({
            "input": index,
            "full_loss": full_loss,
            "oracle_loss": loss,
            "quality_gap": loss - full_loss,
            "full_path": ",".join(map(str, full_path)),
            "oracle_path": ",".join(map(str, path)),
            "full_cost": full_cost,
            "oracle_cost": cost,
            "compute_saving": 1.0 - cost / max(full_cost, 1e-9),
            "full_latency_ms": full_ms,
            "oracle_latency_ms": oracle_ms,
            "speedup": full_ms / max(oracle_ms, 1e-9),
            "feasible": int(loss - full_loss <= args.quality_tolerance),
        })
    fields = list(rows[0])
    output = args.output or os.path.join(
        os.path.dirname(args.checkpoint), "marginal_utility_oracle.csv"
    )
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": args.checkpoint,
        "step": step,
        "samples": args.samples,
        "quality_tolerance": args.quality_tolerance,
        "mean_quality_gap": float(np.mean([r["quality_gap"] for r in rows])),
        "feasible_fraction": float(np.mean([r["feasible"] for r in rows])),
        "mean_compute_saving": float(np.mean([r["compute_saving"] for r in rows])),
        "mean_speedup": float(np.mean([r["speedup"] for r in rows])),
        "median_speedup": float(np.median([r["speedup"] for r in rows])),
        "input_cost_std": float(np.std([r["oracle_cost"] for r in rows])),
        "rows": rows,
    }
    json_path = os.path.splitext(output)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("mean_quality_gap={:.4f} feasible={:.1%} compute_saving={:.1%} "
          "mean_speedup={:.3f}x median_speedup={:.3f}x input_cost_std={:.3f}"
          .format(summary["mean_quality_gap"], summary["feasible_fraction"],
                  summary["mean_compute_saving"], summary["mean_speedup"],
                  summary["median_speedup"], summary["input_cost_std"]))
    print(f"wrote {output} and {json_path}")


if __name__ == "__main__":
    main()
