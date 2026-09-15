"""Benchmark dense, legacy packed, and event-driven Cell execution."""

import argparse
import contextlib
import json
import os
import statistics
import sys
import time

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.eval_full_free_attention_v2 import load_model, sample_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--parallel_streams", action="store_true")
    parser.add_argument("--plan", default="")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json", default="event_driven_cell_benchmark.json")
    return parser.parse_args()


def amp_context(device, dtype):
    if device.type != "cuda" or dtype == "float32":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=getattr(torch, dtype))


def set_mode(graph, mode, parallel_streams=False):
    graph.physical_cell_execution = mode == "legacy_queue"
    graph.event_driven_execution = mode == "event_queue"
    graph.event_parallel_streams = parallel_streams and mode == "event_queue"


@torch.no_grad()
def run_once(model, x, dtype):
    with amp_context(x.device, dtype):
        return model(x)[0]


@torch.no_grad()
def timed(model, x, dtype, warmup, iterations):
    for _ in range(warmup):
        run_once(model, x, dtype)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run_once(model, x, dtype)
        end.record()
        torch.cuda.synchronize(x.device)
        return start.elapsed_time(end) / iterations
    start = time.perf_counter()
    for _ in range(iterations):
        run_once(model, x, dtype)
    return (time.perf_counter() - start) * 1000 / iterations


def resolve_plan(args, checkpoint, graph, device):
    text = args.plan or checkpoint.get("config", {}).get("plan", "")
    saved = checkpoint.get("attention_plan")
    if text:
        plan = torch.zeros(graph.num_steps, dtype=torch.bool, device=device)
        plan[[int(value) for value in text.split(",")]] = True
        return plan
    if saved is not None:
        return saved.to(device=device, dtype=torch.bool)
    return None


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_model(
        os.path.join(ROOT, args.out_dir, args.checkpoint), device
    )
    graph = model.cell_graph
    graph.attention_plan_override = resolve_plan(args, checkpoint, graph, device)
    # Research-only graph diversity statistics synchronize the GPU and are not
    # part of the event executor's serving path.
    model._set_cell_graph_stats = lambda valid_mask=None: None
    data = np.memmap(
        os.path.join(ROOT, "data", "shakespeare_char", "val.bin"),
        dtype=np.uint16, mode="r",
    )
    generator = torch.Generator().manual_seed(args.seed)
    x, _ = sample_batch(
        data, args.batch_size, model.config.block_size, device, generator
    )
    modes = ("dense_cells", "legacy_queue", "event_queue")
    logits = {}
    for mode in modes:
        set_mode(graph, mode, args.parallel_streams)
        logits[mode] = run_once(model, x, args.dtype).float()
    event_stats = graph.last_event_executor_stats
    differences = {
        mode: (logits["dense_cells"] - logits[mode]).abs().max().item()
        for mode in modes[1:]
    }

    samples = {mode: [] for mode in modes}
    per_sample_iterations = max(2, args.iterations // 4)
    for repeat in range(args.repeats):
        order = modes if repeat % 2 == 0 else tuple(reversed(modes))
        for mode in order:
            set_mode(graph, mode, args.parallel_streams)
            samples[mode].append(timed(
                model, x, args.dtype, min(args.warmup, 3), per_sample_iterations
            ))
    medians = {mode: statistics.median(values) for mode, values in samples.items()}
    result = {
        "checkpoint": os.path.join(ROOT, args.out_dir, args.checkpoint),
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "block_size": model.config.block_size,
        "parallel_streams": args.parallel_streams,
        "attention_plan": (
            graph.attention_plan_override.nonzero().flatten().tolist()
            if graph.attention_plan_override is not None else None
        ),
        "active_cells": graph.last_node_mask.float().sum(-1).mean().item(),
        "event_stats": event_stats,
        "logits_max_abs_diff": differences,
        "samples_ms": samples,
        "median_ms": medians,
        "legacy_speedup_x": medians["dense_cells"] / medians["legacy_queue"],
        "event_speedup_x": medians["dense_cells"] / medians["event_queue"],
        "event_latency_reduction_fraction": 1.0 - medians["event_queue"] / medians["dense_cells"],
    }
    output = os.path.join(ROOT, args.json)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
