"""
Analysis for Free Channel MLP checkpoints.

Measures:
- Final CE and perplexity
- Active channel count distribution and quantiles
- Gate collapse indicators
- Per-channel usage rates for specialization analysis

Example:
$ python experiments/free_channel_analysis.py --out_dir=out-shakespeare-char-free-channel --device=cpu
$ python experiments/free_channel_analysis.py --out_dir=out-shakespeare-char-free-channel --compare_checkpoints --device=cpu
"""

import argparse
import csv
import math
import os
import re
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import BlockSparseMLP, FreeChannelMLP, GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-shakespeare-char-free-channel")
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--compare_checkpoints", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--num_batches", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output_prefix", default=None)
    return parser.parse_args()


def load_model(out_dir, checkpoint_name, device):
    ckpt_path = os.path.join(out_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model, checkpoint


def free_channel_modules(model):
    return [block.mlp for block in model.transformer.h if isinstance(block.mlp, (FreeChannelMLP, BlockSparseMLP))]


def get_dataset_name(args, checkpoint):
    if args.dataset is not None:
        return args.dataset
    if "config" in checkpoint and "dataset" in checkpoint["config"]:
        return checkpoint["config"]["dataset"]
    return "shakespeare_char"


def load_data(dataset, split):
    data_dir = os.path.join("data", dataset)
    return np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")


def sample_batch(data, batch_size, block_size, device):
    max_start = len(data) - block_size - 1
    starts = torch.randint(max_start, (batch_size,))
    x_np = np.stack([np.asarray(data[i:i + block_size], dtype=np.int64) for i in starts])
    y_np = np.stack([np.asarray(data[i + 1:i + 1 + block_size], dtype=np.int64) for i in starts])
    x = torch.tensor(x_np, dtype=torch.long, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)
    return x, y


def checkpoint_names(out_dir, single_checkpoint, compare):
    if not compare:
        return [single_checkpoint]
    names = [name for name in os.listdir(out_dir) if re.match(r"ckpt(_iter_\d+)?\.pt$", name)]
    return sorted(names, key=lambda name: int(re.search(r"\d+", name).group(0)) if re.search(r"\d+", name) else 10**18)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_summary(checkpoint_name, checkpoint, stats, mean_ce):
    q = stats["active_width_quantiles"]
    row = {
        "checkpoint": checkpoint_name,
        "iter": int(checkpoint.get("iter_num", 0)),
        "final_ce": mean_ce,
        "perplexity": math.exp(mean_ce),
        "mean_active_channels": stats["mean_active_channels"],
        "mean_active_ratio": stats["mean_active_ratio"],
        "median_active_channels": stats["median_active_channels"],
        "std_active_channels": stats["std_active_channels"],
        "min_active_channels": stats["min_active_channels"],
        "max_active_channels": stats["max_active_channels"],
        "gate_entropy": stats["gate_entropy"],
        "fraction_gate_gt_0_5": stats["fraction_gate_gt_0_5"],
        "fraction_gate_gt_0_9": stats["fraction_gate_gt_0_9"],
        "fraction_gate_lt_0_1": stats["fraction_gate_lt_0_1"],
        "p10_width": q["p10"],
        "p25_width": q["p25"],
        "p50_width": q["p50"],
        "p75_width": q["p75"],
        "p90_width": q["p90"],
        "mean_channel_usage_rate": stats["mean_channel_usage_rate"],
        "std_channel_usage_rate": stats["std_channel_usage_rate"],
        "min_channel_usage_rate": stats["min_channel_usage_rate"],
        "max_channel_usage_rate": stats["max_channel_usage_rate"],
    }
    for bucket, frac in stats["active_width_histogram"].items():
        row[f"hist_{bucket}"] = frac
    return row


def flatten_layer_rows(checkpoint_name, checkpoint, stats):
    rows = []
    for layer in stats["layers"]:
        q = layer["active_width_quantiles"]
        row = {
            "checkpoint": checkpoint_name,
            "iter": int(checkpoint.get("iter_num", 0)),
            "layer": layer["layer"],
            "mean_active_channels": layer["mean_active_channels"],
            "median_active_channels": layer["median_active_channels"],
            "std_active_channels": layer["std_active_channels"],
            "min_active_channels": layer["min_active_channels"],
            "max_active_channels": layer["max_active_channels"],
            "gate_entropy": layer["gate_entropy"],
            "fraction_gate_gt_0_5": layer["fraction_gate_gt_0_5"],
            "fraction_gate_gt_0_9": layer["fraction_gate_gt_0_9"],
            "fraction_gate_lt_0_1": layer["fraction_gate_lt_0_1"],
            "p10_width": q["p10"],
            "p25_width": q["p25"],
            "p50_width": q["p50"],
            "p75_width": q["p75"],
            "p90_width": q["p90"],
            "mean_channel_usage_rate": layer["mean_channel_usage_rate"],
            "std_channel_usage_rate": layer["std_channel_usage_rate"],
            "min_channel_usage_rate": layer["min_channel_usage_rate"],
            "max_channel_usage_rate": layer["max_channel_usage_rate"],
        }
        for bucket, frac in layer["active_width_histogram"].items():
            row[f"hist_{bucket}"] = frac
        rows.append(row)
    return rows


def flatten_channel_rows(checkpoint_name, checkpoint, stats, layer_channel_usage):
    rows = []
    usage = stats["channel_usage_rate"].numpy()
    for channel_idx, usage_rate in enumerate(usage):
        rows.append({
            "checkpoint": checkpoint_name,
            "iter": int(checkpoint.get("iter_num", 0)),
            "layer": "all",
            "channel": channel_idx,
            "usage_rate": float(usage_rate),
        })
    for layer_idx, usage_tensor in sorted(layer_channel_usage.items()):
        for channel_idx, usage_rate in enumerate(usage_tensor.numpy()):
            rows.append({
                "checkpoint": checkpoint_name,
                "iter": int(checkpoint.get("iter_num", 0)),
                "layer": layer_idx,
                "channel": channel_idx,
                "usage_rate": float(usage_rate),
            })
    return rows


@torch.no_grad()
def analyze_checkpoint(args, checkpoint_name):
    model, checkpoint = load_model(args.out_dir, checkpoint_name, args.device)
    assert model.config.free_channel_mlp or model.config.block_sparse_mlp, \
        "checkpoint must have free_channel_mlp=True or block_sparse_mlp=True"
    assert free_channel_modules(model), "checkpoint must contain FreeChannelMLP or BlockSparseMLP modules"
    dataset = get_dataset_name(args, checkpoint)
    data = load_data(dataset, args.split)

    ce_sum = 0.0
    token_count = 0
    active_values = []
    entropy_values = []
    gate_prob_values = []
    gate_values = []
    layer_gate_values = {}
    layer_accum = {}

    for _ in range(args.num_batches):
        x, y = sample_batch(data, args.batch_size, args.block_size, args.device)
        logits, _ = model(x, y)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        ce_sum += float(ce.item())
        token_count += int(y.numel())

        stats = model.last_free_channel_stats
        active_values.append(torch.cat([
            mlp.last_active_channels.detach().float().reshape(-1).cpu()
            for mlp in free_channel_modules(model)
        ]))
        entropy_values.append(torch.cat([
            mlp.last_gate_entropy.detach().float().reshape(-1).cpu()
            for mlp in free_channel_modules(model)
        ]))
        gate_prob_values.append(torch.cat([
            mlp.last_gate_prob.detach().float().reshape(-1, mlp.max_hidden).cpu()
            for mlp in free_channel_modules(model)
        ]))
        gate_values.append(torch.cat([
            mlp.last_gate.detach().float().reshape(-1, mlp.max_hidden).cpu()
            for mlp in free_channel_modules(model)
        ]))
        for layer_idx, block in enumerate(model.transformer.h, start=1):
            if isinstance(block.mlp, (FreeChannelMLP, BlockSparseMLP)):
                layer_gate_values.setdefault(layer_idx, []).append(
                    block.mlp.last_gate.detach().float().reshape(-1, block.mlp.max_hidden).cpu()
                )
        for layer in stats["layers"]:
            acc = layer_accum.setdefault(layer["layer"], [])
            acc.append(layer)

    active_cat = torch.cat(active_values)
    entropy_cat = torch.cat(entropy_values)
    gate_prob_cat = torch.cat(gate_prob_values)
    gate_cat = torch.cat(gate_values)
    max_hidden = gate_cat.size(-1)
    bins = [(0, 64), (65, 128), (129, 192), (193, 256),
            (257, 320), (321, 384), (385, 448), (449, max_hidden)]
    quantiles = torch.quantile(active_cat, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    channel_usage = gate_cat.mean(dim=0)
    stats = {
        "max_width": float(max_hidden),
        "mean_active_channels": active_cat.mean().item(),
        "mean_active_ratio": (active_cat.mean() / max_hidden).item(),
        "median_active_channels": active_cat.median().item(),
        "std_active_channels": active_cat.std(unbiased=False).item(),
        "min_active_channels": active_cat.min().item(),
        "max_active_channels": active_cat.max().item(),
        "gate_entropy": entropy_cat.mean().item(),
        "fraction_gate_gt_0_5": (gate_prob_cat > 0.5).float().mean().item(),
        "fraction_gate_gt_0_9": (gate_prob_cat > 0.9).float().mean().item(),
        "fraction_gate_lt_0_1": (gate_prob_cat < 0.1).float().mean().item(),
        "active_width_histogram": {
            f"{lo}-{hi}": ((active_cat >= lo) & (active_cat <= hi)).float().mean().item()
            for lo, hi in bins
        },
        "active_width_quantiles": {
            "p10": quantiles[0].item(),
            "p25": quantiles[1].item(),
            "p50": quantiles[2].item(),
            "p75": quantiles[3].item(),
            "p90": quantiles[4].item(),
        },
        "channel_usage_rate": channel_usage,
        "mean_channel_usage_rate": channel_usage.mean().item(),
        "std_channel_usage_rate": channel_usage.std(unbiased=False).item(),
        "min_channel_usage_rate": channel_usage.min().item(),
        "max_channel_usage_rate": channel_usage.max().item(),
        "layers": [],
    }

    for layer_idx, rows in sorted(layer_accum.items()):
        averaged = {"layer": layer_idx}
        for key in rows[0].keys():
            if key == "layer":
                continue
            if isinstance(rows[0][key], dict):
                averaged[key] = {
                    subkey: sum(row[key][subkey] for row in rows) / len(rows)
                    for subkey in rows[0][key]
                }
            else:
                averaged[key] = sum(row[key] for row in rows) / len(rows)
        stats["layers"].append(averaged)

    layer_channel_usage = {
        layer_idx: torch.cat(values).mean(dim=0)
        for layer_idx, values in layer_gate_values.items()
    }

    mean_ce = ce_sum / max(token_count, 1)
    return (
        flatten_summary(checkpoint_name, checkpoint, stats, mean_ce),
        flatten_layer_rows(checkpoint_name, checkpoint, stats),
        flatten_channel_rows(checkpoint_name, checkpoint, stats, layer_channel_usage),
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    summaries = []
    layer_rows = []
    channel_rows = []

    for checkpoint_name in checkpoint_names(args.out_dir, args.checkpoint, args.compare_checkpoints):
        summary, layers, channels = analyze_checkpoint(args, checkpoint_name)
        summaries.append(summary)
        layer_rows.extend(layers)
        channel_rows.extend(channels)

    output_prefix = args.output_prefix or os.path.join(args.out_dir, "free_channel")
    write_csv(f"{output_prefix}_summary.csv", summaries)
    write_csv(f"{output_prefix}_layers.csv", layer_rows)
    write_csv(f"{output_prefix}_channels.csv", channel_rows)

    final = summaries[-1]
    print("===== Free Channel Analysis =====")
    print(f"checkpoint: {final['checkpoint']} (iter {final['iter']})")
    print(f"Final CE: {final['final_ce']:.4f}")
    print(f"Perplexity: {final['perplexity']:.2f}")
    print(f"Mean active channels: {final['mean_active_channels']:.2f}")
    print(f"Active ratio: {100 * final['mean_active_ratio']:.2f}%")
    print(
        "Width quantiles: "
        f"P10={final['p10_width']:.1f}, P25={final['p25_width']:.1f}, "
        f"P50={final['p50_width']:.1f}, P75={final['p75_width']:.1f}, "
        f"P90={final['p90_width']:.1f}"
    )
    print("\nActive width histogram:")
    for key, value in final.items():
        if key.startswith("hist_"):
            print(f"{key.removeprefix('hist_')}: {100 * value:.2f}%")
    print(
        "\nChannel usage: "
        f"mean={final['mean_channel_usage_rate']:.4f}, "
        f"std={final['std_channel_usage_rate']:.4f}, "
        f"min={final['min_channel_usage_rate']:.4f}, "
        f"max={final['max_channel_usage_rate']:.4f}"
    )
    print(f"\nWrote CSV files with prefix: {output_prefix}")


if __name__ == "__main__":
    main()
