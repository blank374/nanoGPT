"""
Benchmark AdaptiveWidthMLP dense-mask vs true prefix-sliced eval paths.

Example:
$ python experiments/dynamic_width_benchmark.py --out_dir=out-dynamic-width-8l-ste-anneal-hardaware-1k --device=cpu
"""

import argparse
import csv
import os
import pickle
import statistics
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import AdaptiveWidthMLP, GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-dynamic-width-8l-ste-anneal-hardaware-1k")
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


def load_model(out_dir, checkpoint_name, device):
    checkpoint = torch.load(os.path.join(out_dir, checkpoint_name), map_location=device)
    model_args = dict(checkpoint["model_args"])
    model_args.setdefault("dynamic_width_sliced_eval", True)
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key, _ in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    assert model.config.dynamic_width, "checkpoint must have dynamic_width=True"
    return model, checkpoint


def dynamic_width_modules(model):
    return [block.mlp for block in model.transformer.h if isinstance(block.mlp, AdaptiveWidthMLP)]


def set_sliced_eval(model, enabled):
    for mlp in dynamic_width_modules(model):
        mlp.sliced_eval = enabled


def get_dataset_name(args, checkpoint):
    if args.dataset is not None:
        return args.dataset
    if "config" in checkpoint and "dataset" in checkpoint["config"]:
        return checkpoint["config"]["dataset"]
    return "shakespeare_char"


def sample_batch(dataset, split, batch_size, block_size, device):
    data = np.memmap(os.path.join("data", dataset, f"{split}.bin"), dtype=np.uint16, mode="r")
    max_start = len(data) - block_size - 1
    starts = torch.randint(max_start, (batch_size,))
    x_np = np.stack([np.asarray(data[i:i + block_size], dtype=np.int64) for i in starts])
    return torch.tensor(x_np, dtype=torch.long, device=device)


@torch.no_grad()
def run_once(model, x, sliced):
    set_sliced_eval(model, sliced)
    logits, _ = model(x)
    stats = model.last_dynamic_width_stats or {}
    return logits, stats


@torch.no_grad()
def time_path(model, x, sliced, warmup, iters):
    for _ in range(warmup):
        run_once(model, x, sliced)
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _, stats = run_once(model, x, sliced)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    tokens = x.numel()
    mean_s = statistics.mean(times)
    row = {
        "path": "sliced_prefix" if sliced else "dense_mask",
        "mean_ms": mean_s * 1000.0,
        "median_ms": statistics.median(times) * 1000.0,
        "tokens_per_second": tokens / mean_s,
        "mean_width": stats.get("mean_effective_width", 0.0),
        "mean_width_ratio": stats.get("mean_width_ratio", 0.0),
        "router_entropy": stats.get("router_entropy", 0.0),
    }
    for width in stats.get("width_choices", []):
        row[f"width{width}_fraction"] = stats["width_fractions"][str(width)]
    return row


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, checkpoint = load_model(args.out_dir, args.checkpoint, args.device)
    dataset = get_dataset_name(args, checkpoint)
    x = sample_batch(dataset, args.split, args.batch_size, args.block_size, args.device)

    dense_logits, _ = run_once(model, x, sliced=False)
    sliced_logits, _ = run_once(model, x, sliced=True)
    max_abs_diff = (dense_logits - sliced_logits).abs().max().item()

    rows = [
        time_path(model, x, sliced=False, warmup=args.warmup, iters=args.iters),
        time_path(model, x, sliced=True, warmup=args.warmup, iters=args.iters),
    ]
    for row in rows:
        row["checkpoint"] = args.checkpoint
        row["iter"] = int(checkpoint.get("iter_num", 0))
        row["batch_size"] = args.batch_size
        row["block_size"] = args.block_size
        row["tokens"] = int(x.numel())
        row["max_abs_diff_vs_dense_mask"] = 0.0 if row["path"] == "dense_mask" else max_abs_diff

    speedup = rows[0]["mean_ms"] / rows[1]["mean_ms"] if rows[1]["mean_ms"] > 0 else 0.0
    output_csv = args.output_csv or os.path.join(args.out_dir, "dynamic_width_benchmark.csv")
    write_csv(output_csv, rows)

    print("===== Dynamic-Width Benchmark =====")
    print(f"checkpoint: {args.checkpoint} (iter {checkpoint.get('iter_num', 0)})")
    print(f"batch/block/tokens: {args.batch_size}/{args.block_size}/{x.numel()}")
    print(f"max abs diff dense_mask vs sliced_prefix: {max_abs_diff:.6e}")
    for row in rows:
        dist = ", ".join(
            f"{key.replace('_fraction', '')}={value:.3f}"
            for key, value in row.items()
            if key.startswith("width") and key.endswith("_fraction")
        )
        print(
            f"{row['path']}: mean {row['mean_ms']:.3f} ms, "
            f"median {row['median_ms']:.3f} ms, "
            f"{row['tokens_per_second']:.1f} tok/s, "
            f"mean_width {row['mean_width']:.2f}, {dist}"
        )
    print(f"speedup sliced_prefix vs dense_mask: {speedup:.3f}x")
    print(f"wrote: {output_csv}")


if __name__ == "__main__":
    main()
