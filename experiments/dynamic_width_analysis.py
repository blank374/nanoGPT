"""
Analysis for nested Adaptive Width MLP checkpoints.

Measures:
- Final CE and perplexity
- Learned hard-routing width distribution
- Layer-wise mean width and distribution
- Familiarity buckets using train-set unigram and bigram frequency
- Stable Required Width:
  W_stable(t) = smallest width where this and all larger forced widths agree
                with the full-width top-1 prediction

Example:
$ python experiments/dynamic_width_analysis.py --out_dir=out-shakespeare-char-dynamic-width --device=cpu
$ python experiments/dynamic_width_analysis.py --out_dir=out-shakespeare-char-dynamic-width --compare_checkpoints --device=cpu
"""

import argparse
import csv
import math
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import AdaptiveWidthMLP, GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-shakespeare-char-dynamic-width")
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


def dynamic_width_modules(model):
    return [block.mlp for block in model.transformer.h if isinstance(block.mlp, AdaptiveWidthMLP)]


def set_forced_width(model, width_index):
    for mlp in dynamic_width_modules(model):
        mlp.force_width_index = width_index


def clear_forced_width(model):
    set_forced_width(model, None)


def get_dataset_name(args, checkpoint):
    if args.dataset is not None:
        return args.dataset
    if "config" in checkpoint and "dataset" in checkpoint["config"]:
        return checkpoint["config"]["dataset"]
    return "shakespeare_char"


def load_data(dataset, split):
    data_dir = os.path.join("data", dataset)
    return np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")


def frequency_stats(dataset):
    train = np.asarray(load_data(dataset, "train"), dtype=np.int64)
    unigram = Counter(train.tolist())
    bigram = Counter(zip(train[:-1].tolist(), train[1:].tolist()))
    return unigram, bigram


def frequency_bucket(value):
    if value >= 10000:
        return "very_high"
    if value >= 1000:
        return "high"
    if value >= 100:
        return "medium"
    if value >= 10:
        return "low"
    return "rare"


def sample_batch(data, batch_size, block_size, device):
    max_start = len(data) - block_size - 1
    starts = torch.randint(max_start, (batch_size,))
    x_np = np.stack([np.asarray(data[i:i + block_size], dtype=np.int64) for i in starts])
    y_np = np.stack([np.asarray(data[i + 1:i + 1 + block_size], dtype=np.int64) for i in starts])
    x = torch.tensor(x_np, dtype=torch.long, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)
    return x, y, x_np, y_np


def add_mean_bucket(summary, key, width):
    row = summary.setdefault(key, {"count": 0, "width_sum": 0.0})
    row["count"] += 1
    row["width_sum"] += float(width)


def add_count_bucket(summary, key, width):
    row = summary.setdefault(key, {"count": 0})
    row["count"] += 1
    row[str(width)] = row.get(str(width), 0) + 1


def flatten_width_stats(model, valid_mask=None):
    stats = model.last_dynamic_width_stats
    rows = []
    if stats is None:
        return rows
    for layer in stats["layers"]:
        row = {
            "layer": layer["layer"],
            "mean_width": layer["mean_effective_width"],
            "width_ratio": layer["mean_width_ratio"],
            "router_entropy": layer["router_entropy"],
        }
        for width in stats["width_choices"]:
            row[f"width{width}_fraction"] = layer["width_fractions"][str(width)]
            row[f"width{width}_prob"] = layer["width_prob_means"][str(width)]
        rows.append(row)
    return rows


@torch.no_grad()
def stable_required_width_counts(model, x, y):
    modules = dynamic_width_modules(model)
    assert modules, "checkpoint must have dynamic_width=True"
    width_choices = modules[0].width_choices
    preds = []
    for width_index in range(len(width_choices)):
        set_forced_width(model, width_index)
        logits, _ = model(x, y)
        preds.append(logits.argmax(dim=-1).cpu())
    clear_forced_width(model)

    full_pred = preds[-1]
    stable_counts = {width: 0 for width in width_choices}
    stable_widths = torch.zeros_like(full_pred)
    for i, width in enumerate(width_choices):
        later_agree = torch.ones_like(full_pred, dtype=torch.bool)
        for later in range(i, len(width_choices)):
            later_agree &= preds[later] == full_pred
        first_here = stable_widths == 0
        stable_widths[first_here & later_agree] = width
    stable_widths[stable_widths == 0] = width_choices[-1]
    for width in width_choices:
        stable_counts[width] += int((stable_widths == width).sum().item())
    return stable_counts, stable_widths


@torch.no_grad()
def analyze_checkpoint(args, checkpoint_name):
    model, checkpoint = load_model(args.out_dir, checkpoint_name, args.device)
    assert model.config.dynamic_width, "checkpoint must have dynamic_width=True"
    dataset = get_dataset_name(args, checkpoint)
    data = load_data(dataset, args.split)
    unigram, bigram = frequency_stats(dataset)
    width_choices = dynamic_width_modules(model)[0].width_choices

    ce_sum = 0.0
    token_count = 0
    width_counts = {width: 0 for width in width_choices}
    stable_counts = {width: 0 for width in width_choices}
    layer_rows_accum = {}
    unigram_summary = {}
    bigram_summary = {}
    stable_unigram_summary = {}
    stable_bigram_summary = {}

    for _ in range(args.num_batches):
        x, y, x_np, y_np = sample_batch(data, args.batch_size, args.block_size, args.device)
        clear_forced_width(model)
        logits, _ = model(x, y)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        ce_sum += float(ce.item())
        token_count += int(y.numel())

        stats = model.last_dynamic_width_stats
        for width in width_choices:
            width_counts[width] += int(round(stats["width_fractions"][str(width)] * y.numel() * len(stats["layers"])))
        for row in flatten_width_stats(model):
            acc = layer_rows_accum.setdefault(row["layer"], {"count": 0})
            acc["count"] += 1
            for key, value in row.items():
                if key == "layer":
                    continue
                acc[key] = acc.get(key, 0.0) + float(value)

        selected_by_layer = []
        for mlp in dynamic_width_modules(model):
            selected_by_layer.append(mlp.width_values[mlp.last_selected_width_idx].detach().cpu())
        selected_mean = torch.stack(selected_by_layer).float().mean(dim=0)

        batch_stable_counts, stable_widths = stable_required_width_counts(model, x, y)
        for width, count in batch_stable_counts.items():
            stable_counts[width] += count

        for b in range(args.batch_size):
            for t in range(args.block_size):
                token_id = int(y_np[b, t])
                prev_id = int(x_np[b, t])
                learned_width = float(selected_mean[b, t].item())
                stable_width = int(stable_widths[b, t].item())
                add_mean_bucket(unigram_summary, frequency_bucket(unigram[token_id]), learned_width)
                add_mean_bucket(bigram_summary, frequency_bucket(bigram[(prev_id, token_id)]), learned_width)
                add_mean_bucket(stable_unigram_summary, frequency_bucket(unigram[token_id]), stable_width)
                add_mean_bucket(stable_bigram_summary, frequency_bucket(bigram[(prev_id, token_id)]), stable_width)

    mean_ce = ce_sum / max(token_count, 1)
    total_width_decisions = max(sum(width_counts.values()), 1)
    total_stable = max(sum(stable_counts.values()), 1)
    summary = {
        "checkpoint": checkpoint_name,
        "iter": int(checkpoint.get("iter_num", 0)),
        "final_ce": mean_ce,
        "perplexity": math.exp(mean_ce),
        "mean_effective_width": sum(width * count for width, count in width_counts.items()) / total_width_decisions,
        "effective_width_ratio": sum(width * count for width, count in width_counts.items()) / total_width_decisions / max(width_choices),
    }
    for width in width_choices:
        summary[f"width{width}"] = width_counts[width] / total_width_decisions
        summary[f"stable_width{width}"] = stable_counts[width] / total_stable

    layer_rows = []
    for layer, row in sorted(layer_rows_accum.items()):
        count = max(row.pop("count"), 1)
        averaged = {"checkpoint": checkpoint_name, "iter": summary["iter"], "layer": layer}
        for key, value in row.items():
            averaged[key] = value / count
        layer_rows.append(averaged)

    familiarity_rows = []
    for name, bucket_stats in [
        ("unigram", unigram_summary),
        ("bigram", bigram_summary),
        ("stable_unigram", stable_unigram_summary),
        ("stable_bigram", stable_bigram_summary),
    ]:
        for bucket, row in sorted(bucket_stats.items()):
            familiarity_rows.append({
                "checkpoint": checkpoint_name,
                "iter": summary["iter"],
                "kind": name,
                "bucket": bucket,
                "count": row["count"],
                "mean_width": row["width_sum"] / max(row["count"], 1),
            })

    return summary, layer_rows, familiarity_rows


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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    summaries = []
    layer_rows = []
    familiarity_rows = []

    for checkpoint_name in checkpoint_names(args.out_dir, args.checkpoint, args.compare_checkpoints):
        summary, layers, familiarity = analyze_checkpoint(args, checkpoint_name)
        summaries.append(summary)
        layer_rows.extend(layers)
        familiarity_rows.extend(familiarity)

    output_prefix = args.output_prefix or os.path.join(args.out_dir, "dynamic_width")
    write_csv(f"{output_prefix}_summary.csv", summaries)
    write_csv(f"{output_prefix}_layers.csv", layer_rows)
    write_csv(f"{output_prefix}_familiarity.csv", familiarity_rows)

    final = summaries[-1]
    print("===== Dynamic Width Analysis =====")
    print(f"checkpoint: {final['checkpoint']} (iter {final['iter']})")
    print(f"Final CE: {final['final_ce']:.4f}")
    print(f"Perplexity: {final['perplexity']:.2f}")
    print(f"Mean effective width: {final['mean_effective_width']:.2f}")
    print(f"Effective width ratio: {100 * final['effective_width_ratio']:.2f}%")
    print("\nWidth distribution:")
    for key, value in final.items():
        if key.startswith("width"):
            print(f"{key}: {100 * value:.2f}%")
    print("\nStable required width:")
    for key, value in final.items():
        if key.startswith("stable_width"):
            print(f"{key}: {100 * value:.2f}%")
    print(f"\nWrote CSV files with prefix: {output_prefix}")


if __name__ == "__main__":
    main()
