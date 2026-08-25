"""
Benchmark ResourceModeMLP dense-mask vs grouped-by-mode eval paths.

Example:
$ python experiments/resource_mode_benchmark.py --out_dir=out-shakespeare-char-resource-mode --device=cpu
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
from model import GPT, GPTConfig, ResourceModeMLP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-shakespeare-char-resource-mode")
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
    parser.add_argument("--force_mode", default=None)
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
    assert model.config.resource_mode_mlp, "checkpoint must have resource_mode_mlp=True"
    return model, checkpoint


def resource_mode_modules(model):
    return [block.mlp for block in model.transformer.h if isinstance(block.mlp, ResourceModeMLP)]


def set_eval_path(model, path):
    for mlp in resource_mode_modules(model):
        if path == "dense_mask":
            mlp.sliced_eval = False
        else:
            mlp.sliced_eval = True
            mlp.eval_impl = path


def force_mode(model, mode):
    if mode is None:
        return
    for mlp in resource_mode_modules(model):
        labels = [f"{blocks}x{bits}" for blocks, bits in mlp.modes]
        assert mode in labels, f"force_mode must be one of {labels}"
        mlp.force_mode_index = labels.index(mode)


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
    stats = model.last_resource_mode_stats or {}
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
    row = {
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
    for mode, frac in stats.get("mode_fractions", {}).items():
        row[f"mode_{mode}_fraction"] = frac
    return row


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, checkpoint = load_model(args.out_dir, args.checkpoint, args.device)
    force_mode(model, args.force_mode)
    dataset = get_dataset_name(args, checkpoint)
    x = sample_batch(dataset, args.split, args.batch_size, args.block_size, args.device)

    dense_logits, dense_stats = run_once(model, x, path="dense_mask")
    grouped_logits, _ = run_once(model, x, path="grouped")
    max_abs_diff = (dense_logits - grouped_logits).abs().max().item()

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
        row["max_abs_diff_vs_dense_mask"] = 0.0 if row["path"] == "dense_mask" else max_abs_diff

    speedup = rows[0]["mean_ms"] / rows[1]["mean_ms"] if rows[1]["mean_ms"] > 0 else 0.0
    output_csv = args.output_csv or os.path.join(args.out_dir, "resource_mode_benchmark.csv")
    write_csv(output_csv, rows)

    mode_dist = ", ".join(
        f"{mode}={frac:.3f}" for mode, frac in dense_stats.get("mode_fractions", {}).items()
    )
    print("===== Resource-Mode Benchmark =====")
    print(f"checkpoint: {args.checkpoint} (iter {checkpoint.get('iter_num', 0)})")
    print(f"batch/block/tokens: {args.batch_size}/{args.block_size}/{x.numel()}")
    print(f"max abs diff dense_mask vs grouped: {max_abs_diff:.6e}")
    print(
        "route stats: "
        f"blocks {dense_stats.get('mean_active_blocks', 0.0):.2f}, "
        f"channels {dense_stats.get('mean_active_channels', 0.0):.2f}, "
        f"bit {dense_stats.get('mean_active_bit', 0.0):.2f}, "
        f"weight_bit_fraction {100 * dense_stats.get('mean_weight_bit_fraction', 0.0):.2f}%, "
        f"{mode_dist}"
    )
    for row in rows:
        print(
            f"{row['path']}: mean {row['mean_ms']:.3f} ms, "
            f"median {row['median_ms']:.3f} ms, "
            f"{row['tokens_per_second']:.1f} tok/s"
        )
    print(f"speedup grouped vs dense_mask: {speedup:.3f}x")
    print(f"wrote: {output_csv}")


if __name__ == "__main__":
    main()
