"""Calibrate the shared route-cost currency from measured GPU latency.

The resulting profile is ordered like dynamic_resource_widths and can be
passed to train.py as computational_potential_cost_profile=[...].  Costs are
incremental milliseconds per layer relative to the zero-width action; this
keeps fixed attention/router overhead out of the action price.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dynamic_resource_eval import load_data, load_model, batch


@torch.no_grad()
def timed(model, x, path, warmup, repeats):
    for _ in range(warmup):
        model._forward_dynamic_resource_logits(x, forced_path=path)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        model._forward_dynamic_resource_logits(x, forced_path=path)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/shakespeare_char/val.bin")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("calibration requires CUDA because CPU costs are not the target currency")
    model, _ = load_model(args.checkpoint, device)
    data = load_data(args.data)
    x, _ = batch(data, args.batch_size, min(args.block_size, model.config.block_size), device)
    widths = model.config.dynamic_resource_widths
    raw = []
    for width in widths:
        latency = timed(model, x, [width] * model.config.n_layer,
                        args.warmup, args.repeats)
        raw.append(latency)
        print(f"width={width}: {latency:.4f} ms")
    baseline = raw[0]
    profile = [max(0.0, (latency - baseline) / model.config.n_layer)
               for latency in raw]
    output = args.output or os.path.join(
        os.path.dirname(args.checkpoint), "computational_potential_profile.json"
    )
    payload = {
        "widths": widths,
        "batch_size": args.batch_size,
        "block_size": min(args.block_size, model.config.block_size),
        "latency_ms_all_layers": raw,
        "cost_profile_ms_per_layer": profile,
        "currency": "milliseconds_per_layer_incremental",
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    csv_path = os.path.splitext(output)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["width", "latency_ms", "cost_ms_per_layer"])
        writer.writeheader()
        writer.writerows({"width": width, "latency_ms": latency,
                          "cost_ms_per_layer": cost}
                         for width, latency, cost in zip(widths, raw, profile))
    print(f"profile={profile}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
