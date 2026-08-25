"""
Benchmark BlockPrecisionMLP dense-mask vs grouped sliced eval paths.

This measures real wall-clock execution of the current fake-quant prototype.
It validates compute skipping from hardware-friendly contiguous blocks, but it
does not validate packed int2/int4 memory bandwidth yet.

Example:
$ python experiments/block_precision_benchmark.py --out_dir=out-shakespeare-char-block-precision --device=cpu
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
from model import BlockPrecisionMLP, GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-shakespeare-char-block-precision")
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
    parser.add_argument("--force_blocks", type=int, default=None)
    parser.add_argument("--force_bit", type=int, default=None)
    return parser.parse_args()


def load_model(out_dir, checkpoint_name, device):
    checkpoint = torch.load(os.path.join(out_dir, checkpoint_name), map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key, _ in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    assert model.config.block_precision_mlp, "checkpoint must have block_precision_mlp=True"
    return model, checkpoint


def block_precision_modules(model):
    return [block.mlp for block in model.transformer.h if isinstance(block.mlp, BlockPrecisionMLP)]


def force_route(model, force_blocks, force_bit):
    if force_blocks is None and force_bit is None:
        return
    for mlp in block_precision_modules(model):
        if force_blocks is not None:
            mlp.force_blocks = int(force_blocks)
        if force_bit is not None:
            assert force_bit in mlp.bit_choices, f"force_bit must be one of {mlp.bit_choices}"
            mlp.force_bit = int(force_bit)


def set_eval_path(model, path):
    for mlp in block_precision_modules(model):
        if path == "dense_mask":
            mlp.sliced_eval = False
        else:
            mlp.sliced_eval = True
            mlp.eval_impl = path


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
def run_once(model, x, path):
    set_eval_path(model, path)
    logits, _ = model(x)
    stats = model.last_block_precision_stats or {}
    return logits, stats


@torch.no_grad()
def time_path(model, x, path, warmup, iters):
    for _ in range(warmup):
        run_once(model, x, path)
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    stats = {}
    for _ in range(iters):
        t0 = time.perf_counter()
        _, stats = run_once(model, x, path)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    tokens = x.numel()
    mean_s = statistics.mean(times)
    return {
        "path": path,
        "mean_ms": mean_s * 1000.0,
        "median_ms": statistics.median(times) * 1000.0,
        "tokens_per_second": tokens / mean_s,
        "mean_active_blocks": stats.get("mean_active_blocks", 0.0),
        "mean_active_channels": stats.get("mean_active_channels", 0.0),
        "mean_active_bit": stats.get("mean_active_bit", 0.0),
        "mean_weight_bits_per_token": stats.get("mean_weight_bits_per_token", 0.0),
        "mean_weight_bit_fraction": stats.get("mean_weight_bit_fraction", 0.0),
    }


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, checkpoint = load_model(args.out_dir, args.checkpoint, args.device)
    force_route(model, args.force_blocks, args.force_bit)
    dataset = get_dataset_name(args, checkpoint)
    x = sample_batch(dataset, args.split, args.batch_size, args.block_size, args.device)

    dense_logits, dense_stats = run_once(model, x, path="dense_mask")
    grouped_logits, grouped_stats = run_once(model, x, path="grouped")
    max_abs_diff_grouped = (dense_logits - grouped_logits).abs().max().item()

    rows = [
        time_path(model, x, path="dense_mask", warmup=args.warmup, iters=args.iters),
        time_path(model, x, path="grouped", warmup=args.warmup, iters=args.iters),
    ]
    for row in rows:
        row["checkpoint"] = args.checkpoint
        row["iter"] = int(checkpoint.get("iter_num", 0))
        row["batch_size"] = args.batch_size
        row["block_size"] = args.block_size
        row["tokens"] = int(x.numel())
        row["max_abs_diff_vs_dense_mask"] = 0.0 if row["path"] == "dense_mask" else max_abs_diff_grouped

    speedup = rows[0]["mean_ms"] / rows[1]["mean_ms"] if rows[1]["mean_ms"] > 0 else 0.0
    output_csv = args.output_csv or os.path.join(args.out_dir, "block_precision_benchmark.csv")
    write_csv(output_csv, rows)

    print("===== Block-Precision Benchmark =====")
    print(f"checkpoint: {args.checkpoint} (iter {checkpoint.get('iter_num', 0)})")
    print(f"batch/block/tokens: {args.batch_size}/{args.block_size}/{x.numel()}")
    print(f"max abs diff dense_mask vs grouped: {max_abs_diff_grouped:.6e}")
    print(
        "route stats: "
        f"blocks {dense_stats.get('mean_active_blocks', 0.0):.2f}, "
        f"channels {dense_stats.get('mean_active_channels', 0.0):.2f}, "
        f"bit {dense_stats.get('mean_active_bit', 0.0):.2f}, "
        f"weight_bit_fraction {100 * dense_stats.get('mean_weight_bit_fraction', 0.0):.2f}%"
    )
    for row in rows:
        print(
            f"{row['path']}: mean {row['mean_ms']:.3f} ms, "
            f"median {row['median_ms']:.3f} ms, "
            f"{row['tokens_per_second']:.1f} tok/s, "
            f"mean_blocks {row['mean_active_blocks']:.2f}, "
            f"mean_channels {row['mean_active_channels']:.2f}, "
            f"mean_bit {row['mean_active_bit']:.2f}, "
            f"weight_bit_fraction {100 * row['mean_weight_bit_fraction']:.2f}%"
        )
    print(f"speedup grouped vs dense_mask: {speedup:.3f}x")
    print(f"wrote: {output_csv}")


if __name__ == "__main__":
    main()
