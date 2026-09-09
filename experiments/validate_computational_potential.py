"""Validate lambda as an input-conditioned quality/real-latency control.

This is intentionally CPU-compatible. It calibrates the route actions on the
same machine, then evaluates one fixed set of validation windows at several
lambda values with repeated measurements.
"""

import argparse
import csv
import json
import math
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


def make_windows(data, count, block_size, seed):
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(data) - block_size - 1, size=count, replace=False)
    xs, ys = [], []
    for start in starts:
        xs.append(torch.from_numpy(data[start:start + block_size].astype(np.int64)))
        ys.append(torch.from_numpy(data[start + 1:start + block_size + 1].astype(np.int64)))
    return torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def timed(model, x, path, warmup, repeats):
    for _ in range(warmup):
        model._forward_dynamic_resource_logits(x, forced_path=path, all_logits=False)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        model._forward_dynamic_resource_logits(x, forced_path=path, all_logits=False)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


@torch.no_grad()
def calibrate(model, x, warmup, repeats):
    raw = []
    for width in model.config.dynamic_resource_widths:
        raw.append(timed(model, x[:1], [width] * model.config.n_layer, warmup, repeats))
    baseline = raw[0]
    profile = [max(0.0, (value - baseline) / model.config.n_layer) for value in raw]
    return raw, profile


def pareto_front(rows):
    front = []
    for row in rows:
        dominated = any(
            other["loss"] <= row["loss"]
            and other["latency_ms"] <= row["latency_ms"]
            and (other["loss"] < row["loss"] or other["latency_ms"] < row["latency_ms"])
            for other in rows
        )
        if not dominated:
            front.append(row)
    return front


@torch.no_grad()
def evaluate_lambda(model, xs, ys, lam, profile, repeats, quality_tolerance):
    model.config.computational_potential_enabled = True
    model.config.computational_potential_lambda = lam
    model.config.computational_potential_cost_profile = profile
    losses, reference_losses, latencies, costs, routes = [], [], [], [], []
    for x, y in zip(xs, ys):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        logits = model._forward_dynamic_resource_logits(x, all_logits=True)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        full_path = [model.config.dynamic_resource_widths[-1]] * model.config.n_layer
        full_logits = model._forward_dynamic_resource_logits(
            x, forced_path=full_path, all_logits=True
        )
        reference_loss = F.cross_entropy(
            full_logits.reshape(-1, full_logits.size(-1)), y.reshape(-1)
        ).item()
        modes = model.last_dynamic_resource_modes[0].clone()
        path = [model.config.dynamic_resource_widths[int(mode)] for mode in modes]
        _, hard, _ = model._computational_potential_cost(
            model.last_dynamic_resource_probs, model.last_dynamic_resource_modes
        )
        losses.append(loss)
        reference_losses.append(reference_loss)
        costs.append(hard.item())
        routes.append(tuple(path))
        latencies.append(timed(model, x, path, warmup=1, repeats=repeats))
    return {
        "lambda": lam,
        "loss": statistics.mean(losses),
        "reference_loss": statistics.mean(reference_losses),
        "quality_gap": statistics.mean(a - b for a, b in zip(losses, reference_losses)),
        "feasible_fraction": sum(a - b <= quality_tolerance for a, b in zip(losses, reference_losses)) / len(losses),
        "quality_tolerance": quality_tolerance,
        "latency_ms": statistics.mean(latencies),
        "cost": statistics.mean(costs),
        "loss_std": statistics.pstdev(losses),
        "latency_std": statistics.pstdev(latencies),
        "cost_std": statistics.pstdev(costs),
        "unique_paths": len(set(routes)),
        "routes": routes,
        "per_input_loss": losses,
        "per_input_cost": costs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="out-shakespeare-char-dynamic-resource/ckpt.pt")
    parser.add_argument("--data", default="data/shakespeare_char/val.bin")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--calibration-repeats", type=int, default=8)
    parser.add_argument("--measurement-repeats", type=int, default=5)
    parser.add_argument("--lambdas", default="0,0.1,0.25,0.5,1,2")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--quality-tolerance", type=float, default=0.02)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    device = torch.device(args.device)
    model, step = load_model(args.checkpoint, device)
    assert model.config.dynamic_resource, "checkpoint must enable dynamic_resource"
    data = load_data(args.data)
    block_size = min(args.block_size, model.config.block_size)
    xs, ys = make_windows(data, args.samples, block_size, args.seed)
    raw_latency, profile = calibrate(model, xs, args.calibration_repeats,
                                     args.calibration_repeats)
    rows = []
    for lam in [float(value) for value in args.lambdas.split(",")]:
        result = evaluate_lambda(model, xs, ys, lam, profile, args.measurement_repeats,
                                 args.quality_tolerance)
        rows.append(result)
        print("lambda={:.3f} loss={:.4f} gap={:.4f} feasible={:.1%} latency={:.3f}ms cost={:.3f} paths={}"
              .format(lam, result["loss"], result["quality_gap"],
                      result["feasible_fraction"], result["latency_ms"],
                      result["cost"], result["unique_paths"]))

    costs = [row["cost"] for row in rows]
    monotonic = all(costs[i + 1] <= costs[i] + 1e-9 for i in range(len(costs) - 1))
    route_stability = []
    for row in rows:
        repeat = evaluate_lambda(model, xs, ys, row["lambda"], profile, 1,
                                 args.quality_tolerance)
        route_stability.append(sum(a == b for a, b in zip(row["routes"], repeat["routes"])) / len(xs))
    input_conditioned = max(row["cost_std"] for row in rows) > 1e-9
    feasible_rows = [row for row in rows
                     if row["quality_gap"] <= args.quality_tolerance]
    front = pareto_front(feasible_rows)
    summary = {
        "checkpoint": args.checkpoint,
        "step": step,
        "device": str(device),
        "samples": args.samples,
        "block_size": block_size,
        "seed": args.seed,
        "calibration_latency_ms": raw_latency,
        "cost_profile_ms_per_layer": profile,
        "cost_monotonic_nonincreasing": monotonic,
        "route_repeat_stability_min": min(route_stability),
        "input_conditioned_cost": input_conditioned,
        "feasible_lambda_count": len(feasible_rows),
        "pareto_lambda_count": len(front),
        "rows": [{key: value for key, value in row.items() if key not in ("routes", "per_input_loss", "per_input_cost")} for row in rows],
    }
    output = args.output or os.path.join(os.path.dirname(args.checkpoint),
                                         "computational_potential_validation.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    csv_path = os.path.splitext(output)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fields = [key for key in rows[0] if key not in ("routes", "per_input_loss", "per_input_cost")]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fields} for row in rows)
    print("calibration_profile={}".format(profile))
    print("cost_monotonic_nonincreasing={}".format(monotonic))
    print("route_repeat_stability_min={:.3f}".format(min(route_stability)))
    print("input_conditioned_cost={}".format(input_conditioned))
    print("pareto_points={}".format([row["lambda"] for row in front]))
    print("wrote {} and {}".format(output, csv_path))


if __name__ == "__main__":
    main()
