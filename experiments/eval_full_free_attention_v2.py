"""Evaluate Attention depth, quality and whole-step physical skipping in v2."""

import argparse
import contextlib
import json
import os
import sys
import time
import statistics

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import model as frozen_model
from experiments.full_free_attention_v2 import install_into_frozen_model_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-full-free-attention-v2-budget8-seed1337")
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--condense_threshold", type=float, default=0.01)
    parser.add_argument("--interleaved_repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json", default="full_free_attention_v2_eval.json")
    return parser.parse_args()


def amp_context(device, dtype):
    if device.type != "cuda" or dtype == "float32":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=getattr(torch, dtype))


def load_model(path, device):
    install_into_frozen_model_module(frozen_model)
    checkpoint = torch.load(path, map_location=device)
    config = frozen_model.GPTConfig(**checkpoint["model_args"])
    model = frozen_model.GPT(config)
    state = {
        (key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key): value
        for key, value in checkpoint["model"].items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "cell_graph.attention_anchor_mask",
        "cell_graph.v2_dual_updates",
    }
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval(), checkpoint


def sample_batch(data, batch_size, block_size, device, generator):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = np.stack([
        np.asarray(data[int(index):int(index) + block_size], dtype=np.int64)
        for index in starts
    ])
    y = np.stack([
        np.asarray(data[int(index) + 1:int(index) + block_size + 1], dtype=np.int64)
        for index in starts
    ])
    return torch.tensor(x, device=device), torch.tensor(y, device=device)


@torch.no_grad()
def latency_ms(model, x, amp_dtype, warmup, iterations):
    # Graph diversity/entropy diagnostics are research instrumentation, not part
    # of a production inference request. They contain unique(), CPU transfers
    # and synchronizing item() calls and would otherwise dominate this tiny model.
    had_override = "_set_cell_graph_stats" in model.__dict__
    previous_override = model.__dict__.get("_set_cell_graph_stats")
    model._set_cell_graph_stats = lambda valid_mask=None: None
    try:
        for _ in range(warmup):
            with amp_context(x.device, amp_dtype):
                model(x)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                with amp_context(x.device, amp_dtype):
                    model(x)
            end.record()
            torch.cuda.synchronize(x.device)
            return start.elapsed_time(end) / iterations
        start = time.perf_counter()
        for _ in range(iterations):
            model(x)
        return (time.perf_counter() - start) * 1000 / iterations
    finally:
        if had_override:
            model._set_cell_graph_stats = previous_override
        else:
            del model._set_cell_graph_stats


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = os.path.join(ROOT, args.out_dir, args.checkpoint)
    model, checkpoint = load_model(checkpoint_path, device)
    graph = model.cell_graph
    data = np.memmap(
        os.path.join(ROOT, "data", "shakespeare_char", "val.bin"),
        dtype=np.uint16, mode="r",
    )
    generator = torch.Generator().manual_seed(args.seed)
    losses = []
    active_attention = []
    active_cells = []
    executed_steps = []
    per_step = []
    evaluation_batches = []
    timing_x = None
    for _ in range(args.eval_batches):
        x, y = sample_batch(
            data, args.batch_size, model.config.block_size, device, generator
        )
        timing_x = x
        evaluation_batches.append((x, y))
        with amp_context(device, args.dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
        active_attention.append(graph.last_attention_mask.float().sum(dim=-1).mean().item())
        active_cells.append(graph.last_node_mask.float().sum(dim=-1).mean().item())
        executed_steps.append(graph.last_attention_executed_steps.sum().item())
        per_step.append(graph.last_attention_mask.float().mean(dim=(0, 1)).cpu())

    physical_latency = latency_ms(
        model, timing_x, args.dtype, args.warmup, args.iterations
    )
    with amp_context(device, args.dtype):
        physical_logits, _ = model(timing_x)
    physical_mask = graph.last_attention_mask.clone()
    graph.force_dense_attention_execution = True
    with amp_context(device, args.dtype):
        dense_logits, _ = model(timing_x)
    dense_masked_latency = latency_ms(
        model, timing_x, args.dtype, args.warmup, args.iterations
    )
    graph.force_dense_attention_execution = False
    max_abs_diff = (physical_logits.float() - dense_logits.float()).abs().max().item()
    masks_equal = torch.equal(physical_mask, graph.last_attention_mask)

    # Extract a common plan from empirical route coverage. Mandatory anchors are
    # always retained. This is the Familiar Fast Path distilled from the free
    # graph, not a hand-authored depth choice.
    step_usage = torch.stack(per_step).mean(dim=0)
    condensed_plan = (
        (step_usage >= args.condense_threshold)
        | graph.attention_anchor_mask.cpu()
    )
    graph.attention_plan_override = condensed_plan.to(device)
    condensed_losses = []
    for x, y in evaluation_batches:
        with amp_context(device, args.dtype):
            _, condensed_loss = model(x, y)
        condensed_losses.append(condensed_loss.item())
    condensed_latency = latency_ms(
        model, timing_x, args.dtype, args.warmup, args.iterations
    )
    with amp_context(device, args.dtype):
        condensed_dense_logits, _ = model(timing_x)
    graph.physical_cell_execution = True
    with amp_context(device, args.dtype):
        condensed_queue_logits, _ = model(timing_x)
    condensed_queue_latency = latency_ms(
        model, timing_x, args.dtype, args.warmup, args.iterations
    )
    graph.physical_cell_execution = False
    graph.attention_plan_override = None
    condensed_nll = float(np.mean(condensed_losses))
    queue_max_abs_diff = (
        condensed_dense_logits.float() - condensed_queue_logits.float()
    ).abs().max().item()

    interleaved = {"dense_masked": [], "condensed_cell_queue": []}
    interleaved_iterations = max(2, args.iterations // 4)
    for repeat in range(args.interleaved_repeats):
        order = (
            ("dense_masked", "condensed_cell_queue")
            if repeat % 2 == 0 else
            ("condensed_cell_queue", "dense_masked")
        )
        for mode in order:
            if mode == "dense_masked":
                graph.force_dense_attention_execution = True
                graph.attention_plan_override = None
                graph.physical_cell_execution = False
            else:
                graph.force_dense_attention_execution = False
                graph.attention_plan_override = condensed_plan.to(device)
                graph.physical_cell_execution = True
            interleaved[mode].append(latency_ms(
                model, timing_x, args.dtype, min(args.warmup, 3), interleaved_iterations
            ))
    graph.force_dense_attention_execution = False
    graph.attention_plan_override = None
    graph.physical_cell_execution = False
    dense_interleaved_median = statistics.median(interleaved["dense_masked"])
    queue_interleaved_median = statistics.median(interleaved["condensed_cell_queue"])
    result = {
        "architecture": graph.architecture_name,
        "checkpoint": checkpoint_path,
        "checkpoint_iter": int(checkpoint.get("iter_num", -1)),
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "block_size": model.config.block_size,
        "mean_nll": float(np.mean(losses)),
        "ppl": float(np.exp(np.mean(losses))),
        "mean_active_cells": float(np.mean(active_cells)),
        "mean_active_attention_steps": float(np.mean(active_attention)),
        "mean_active_attention_ratio": float(np.mean(active_attention) / graph.num_steps),
        "mean_physically_executed_attention_steps_per_batch": float(np.mean(executed_steps)),
        "attention_step_usage": torch.stack(per_step).mean(dim=0).tolist(),
        "mandatory_attention_steps": graph.attention_anchor_mask.nonzero().flatten().tolist(),
        "dense_masked_latency_ms": dense_masked_latency,
        "physical_skip_latency_ms": physical_latency,
        "physical_speedup_x": dense_masked_latency / physical_latency,
        "physical_latency_reduction_fraction": 1.0 - physical_latency / dense_masked_latency,
        "physical_vs_dense_logits_max_abs_diff": max_abs_diff,
        "physical_vs_dense_attention_masks_equal": masks_equal,
        "condense_threshold": args.condense_threshold,
        "condensed_attention_plan": condensed_plan.nonzero().flatten().tolist(),
        "condensed_attention_steps": int(condensed_plan.sum().item()),
        "condensed_nll": condensed_nll,
        "condensed_ppl": float(np.exp(condensed_nll)),
        "condensed_nll_delta": condensed_nll - float(np.mean(losses)),
        "condensed_latency_ms": condensed_latency,
        "condensed_vs_dense_speedup_x": dense_masked_latency / condensed_latency,
        "condensed_vs_dense_latency_reduction_fraction": 1.0 - condensed_latency / dense_masked_latency,
        "condensed_cell_queue_latency_ms": condensed_queue_latency,
        "condensed_cell_queue_vs_dense_speedup_x": dense_masked_latency / condensed_queue_latency,
        "condensed_cell_queue_vs_dense_latency_reduction_fraction": 1.0 - condensed_queue_latency / dense_masked_latency,
        "condensed_cell_queue_logits_max_abs_diff": queue_max_abs_diff,
        "interleaved_iterations_per_sample": interleaved_iterations,
        "interleaved_dense_samples_ms": interleaved["dense_masked"],
        "interleaved_condensed_cell_queue_samples_ms": interleaved["condensed_cell_queue"],
        "interleaved_dense_median_ms": dense_interleaved_median,
        "interleaved_condensed_cell_queue_median_ms": queue_interleaved_median,
        "interleaved_speedup_x": dense_interleaved_median / queue_interleaved_median,
        "interleaved_latency_reduction_fraction": 1.0 - queue_interleaved_median / dense_interleaved_median,
    }
    output_path = os.path.join(ROOT, args.json)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
