"""
Convert a trained FreeChannelMLP checkpoint with channel-usage reordering, then
benchmark dense-mask free-channel eval against prefix-sliced eval.

The channel permutation preserves the original dense-mask function exactly when
gates are permuted with the hidden channels. Prefix-sliced eval is the hardware-
friendly approximation: for each token, use the original active count K but
compute the first K reordered channels only.
"""

import argparse
import csv
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import FreeChannelMLP, GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-free-channel-budget-0p0")
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--converted_out_dir", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--usage_batches", type=int, default=32)
    parser.add_argument("--eval_batches", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--prefix_granularity", type=int, default=64)
    parser.add_argument("--prefix_impl", choices=["active_count", "cover_active"], default="cover_active")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


def load_model(out_dir, checkpoint_name, device):
    checkpoint = torch.load(os.path.join(out_dir, checkpoint_name), map_location=device)
    model_args = dict(checkpoint["model_args"])
    model_args.setdefault("free_channel_eval_impl", "dense_mask")
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key, _ in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    assert model.config.free_channel_mlp, "checkpoint must have free_channel_mlp=True"
    return model, checkpoint, model_args


def free_channel_modules(model):
    return [block.mlp for block in model.transformer.h if isinstance(block.mlp, FreeChannelMLP)]


def set_eval_impl(model, eval_impl):
    model.config.free_channel_eval_impl = eval_impl
    for mlp in free_channel_modules(model):
        mlp.eval_impl = eval_impl


def get_dataset_name(args, checkpoint):
    if args.dataset is not None:
        return args.dataset
    if "config" in checkpoint and "dataset" in checkpoint["config"]:
        return checkpoint["config"]["dataset"]
    return "shakespeare_char"


def sample_batch(dataset, split, batch_size, block_size, device, targets=False):
    data = np.memmap(os.path.join("data", dataset, f"{split}.bin"), dtype=np.uint16, mode="r")
    max_start = len(data) - block_size - 1
    starts = torch.randint(max_start, (batch_size,))
    x_np = np.stack([np.asarray(data[i:i + block_size], dtype=np.int64) for i in starts])
    x = torch.tensor(x_np, dtype=torch.long, device=device)
    if not targets:
        return x
    y_np = np.stack([np.asarray(data[i + 1:i + 1 + block_size], dtype=np.int64) for i in starts])
    y = torch.tensor(y_np, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def estimate_usage(model, dataset, split, batches, batch_size, block_size, device):
    set_eval_impl(model, "dense_mask")
    modules = free_channel_modules(model)
    usage = [torch.zeros(mlp.max_hidden, device=device) for mlp in modules]
    total = 0
    for _ in range(batches):
        x = sample_batch(dataset, split, batch_size, block_size, device)
        model(x)
        tokens = x.numel()
        total += tokens
        for i, mlp in enumerate(modules):
            usage[i] += mlp.last_gate.detach().float().reshape(-1, mlp.max_hidden).sum(dim=0)
    return [u / max(total, 1) for u in usage]


def apply_permutation(model, usage):
    permutations = []
    for mlp, layer_usage in zip(free_channel_modules(model), usage):
        perm = torch.argsort(layer_usage, descending=True)
        permutations.append(perm.detach().cpu())
        with torch.no_grad():
            mlp.c_fc.weight.data = mlp.c_fc.weight.data.index_select(0, perm)
            if mlp.c_fc.bias is not None:
                mlp.c_fc.bias.data = mlp.c_fc.bias.data.index_select(0, perm)
            mlp.c_proj.weight.data = mlp.c_proj.weight.data.index_select(1, perm)
            mlp.gate_network.weight.data = mlp.gate_network.weight.data.index_select(0, perm)
            if mlp.gate_network.bias is not None:
                mlp.gate_network.bias.data = mlp.gate_network.bias.data.index_select(0, perm)
    return permutations


def save_converted(model, checkpoint, model_args, converted_out_dir, checkpoint_name, permutations, prefix_granularity):
    os.makedirs(converted_out_dir, exist_ok=True)
    converted = dict(checkpoint)
    converted_args = dict(model_args)
    converted_args["free_channel_eval_impl"] = "dense_mask"
    converted_args["free_channel_prefix_granularity"] = prefix_granularity
    converted["model_args"] = converted_args
    converted["model"] = model.state_dict()
    converted["channel_reorder"] = {
        "method": "descending_gate_usage",
        "prefix_granularity": prefix_granularity,
        "permutations": [perm.tolist() for perm in permutations],
    }
    torch.save(converted, os.path.join(converted_out_dir, checkpoint_name))


@torch.no_grad()
def eval_ce(model, dataset, split, batches, batch_size, block_size, device, eval_impl):
    set_eval_impl(model, eval_impl)
    losses = []
    for _ in range(batches):
        x, y = sample_batch(dataset, split, batch_size, block_size, device, targets=True)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    ce = statistics.mean(losses)
    ppl = float(np.exp(ce))
    stats = model.last_free_channel_stats or {}
    return ce, ppl, stats


@torch.no_grad()
def eval_ce_pair(model, dataset, split, batches, batch_size, block_size, device, prefix_eval_impl):
    losses = {"dense_mask": [], prefix_eval_impl: []}
    stats = {}
    for _ in range(batches):
        x, y = sample_batch(dataset, split, batch_size, block_size, device, targets=True)
        for eval_impl in ("dense_mask", prefix_eval_impl):
            set_eval_impl(model, eval_impl)
            _, loss = model(x, y)
            losses[eval_impl].append(float(loss.item()))
            stats[eval_impl] = model.last_free_channel_stats or {}
    dense_ce = statistics.mean(losses["dense_mask"])
    prefix_ce = statistics.mean(losses[prefix_eval_impl])
    return (
        dense_ce,
        float(np.exp(dense_ce)),
        stats["dense_mask"],
        prefix_ce,
        float(np.exp(prefix_ce)),
        stats[prefix_eval_impl],
    )


@torch.no_grad()
def run_once(model, x, eval_impl):
    set_eval_impl(model, eval_impl)
    logits, _ = model(x)
    stats = model.last_free_channel_stats or {}
    return logits, stats


@torch.no_grad()
def time_path(model, x, eval_impl, warmup, iters):
    for _ in range(warmup):
        run_once(model, x, eval_impl)
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _, stats = run_once(model, x, eval_impl)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    tokens = x.numel()
    mean_s = statistics.mean(times)
    q = stats.get("active_width_quantiles", {})
    return {
        "path": eval_impl,
        "mean_ms": mean_s * 1000.0,
        "median_ms": statistics.median(times) * 1000.0,
        "tokens_per_second": tokens / mean_s,
        "mean_active_width": stats.get("mean_active_channels", 0.0),
        "p10_width": q.get("p10", 0.0),
        "p50_width": q.get("p50", 0.0),
        "p90_width": q.get("p90", 0.0),
    }


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    converted_out_dir = args.converted_out_dir or args.out_dir + "-reordered-prefix"
    model, checkpoint, model_args = load_model(args.out_dir, args.checkpoint, args.device)
    model.config.free_channel_prefix_granularity = args.prefix_granularity
    for mlp in free_channel_modules(model):
        mlp.prefix_granularity = args.prefix_granularity
    dataset = get_dataset_name(args, checkpoint)

    usage = estimate_usage(
        model, dataset, args.split, args.usage_batches,
        args.batch_size, args.block_size, args.device,
    )
    permutations = apply_permutation(model, usage)
    save_converted(model, checkpoint, model_args, converted_out_dir, args.checkpoint, permutations, args.prefix_granularity)

    prefix_eval_impl = "prefix_cover_sliced" if args.prefix_impl == "cover_active" else "prefix_sliced"
    ce_dense, ppl_dense, dense_stats, ce_prefix, ppl_prefix, prefix_stats = eval_ce_pair(
        model, dataset, args.split, args.eval_batches,
        args.batch_size, args.block_size, args.device, prefix_eval_impl,
    )

    x = sample_batch(dataset, args.split, args.batch_size, args.block_size, args.device)
    dense_logits, _ = run_once(model, x, "dense_mask")
    prefix_logits, _ = run_once(model, x, prefix_eval_impl)
    max_abs_diff = (dense_logits - prefix_logits).abs().max().item()
    rows = [
        time_path(model, x, "dense_mask", args.warmup, args.iters),
        time_path(model, x, prefix_eval_impl, args.warmup, args.iters),
    ]
    for row in rows:
        row["checkpoint"] = args.checkpoint
        row["iter"] = int(checkpoint.get("iter_num", 0))
        row["batch_size"] = args.batch_size
        row["block_size"] = args.block_size
        row["tokens"] = int(x.numel())
        row["ce"] = ce_dense if row["path"] == "dense_mask" else ce_prefix
        row["ppl"] = ppl_dense if row["path"] == "dense_mask" else ppl_prefix
        row["max_abs_diff_vs_dense_mask"] = 0.0 if row["path"] == "dense_mask" else max_abs_diff

    speedup = rows[0]["mean_ms"] / rows[1]["mean_ms"] if rows[1]["mean_ms"] > 0 else 0.0
    output_csv = args.output_csv or os.path.join(converted_out_dir, "free_channel_reorder_prefix_benchmark.csv")
    write_csv(output_csv, rows)

    usage_means = [float(u.mean().item()) for u in usage]
    usage_stds = [float(u.std(unbiased=False).item()) for u in usage]
    print("===== Free-Channel Reorder Prefix Benchmark =====")
    print(f"source checkpoint: {args.out_dir}/{args.checkpoint} (iter {checkpoint.get('iter_num', 0)})")
    print(f"converted checkpoint: {converted_out_dir}/{args.checkpoint}")
    print(f"prefix granularity: {args.prefix_granularity}")
    print(f"prefix impl: {prefix_eval_impl}")
    print(f"usage mean/std by layer: " + ", ".join(f"{m:.4f}/{s:.4f}" for m, s in zip(usage_means, usage_stds)))
    print(f"dense_mask CE/PPL: {ce_dense:.4f}/{ppl_dense:.2f}")
    print(f"prefix_sliced CE/PPL: {ce_prefix:.4f}/{ppl_prefix:.2f}")
    print(f"max abs diff dense_mask vs prefix_sliced: {max_abs_diff:.6e}")
    for row in rows:
        print(
            f"{row['path']}: mean {row['mean_ms']:.3f} ms, "
            f"median {row['median_ms']:.3f} ms, "
            f"{row['tokens_per_second']:.1f} tok/s, "
            f"mean_width {row['mean_active_width']:.2f}, "
            f"P10/P50/P90 {row['p10_width']:.0f}/{row['p50_width']:.0f}/{row['p90_width']:.0f}"
        )
    print(f"speedup prefix_sliced vs dense_mask: {speedup:.3f}x")
    print(f"wrote: {output_csv}")


if __name__ == "__main__":
    main()
