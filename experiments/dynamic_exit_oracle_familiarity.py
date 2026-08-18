"""
Phase 1.5 / Phase 2 analysis for Dynamic Fast/Slow Path models.

Measures:
- Oracle Required Depth:
  D*(t) = min layer where intermediate top-1 equals final top-1
- Stable Oracle Required Depth:
  D_stable(t) = min layer where this and all later measured layers equal final top-1
- Top-1 agreement per exit layer
- Confidence-bin agreement
- Familiarity buckets using train-set unigram and bigram frequency

Example:
$ python experiments/dynamic_exit_oracle_familiarity.py --out_dir=out-dynamic-exit-8l-long --device=cpu
"""

import argparse
import csv
import os
import pickle
import sys
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import GPTConfig, GPT


CONFIDENCE_BINS = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.01),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--num_batches", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output_prefix", default=None)
    return parser.parse_args()


def load_model(args):
    ckpt_path = os.path.join(args.out_dir, args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(args.device)
    return model, checkpoint


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


def confidence_bucket(value):
    for lo, hi in CONFIDENCE_BINS:
        if lo <= value < hi:
            return f"{lo:.1f}-{min(hi, 1.0):.1f}"
    return "other"


@torch.no_grad()
def collect_layer_logits(model, idx):
    device = idx.device
    _, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    x = model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(pos))
    exit_logits = {}
    for layer_idx, block in enumerate(model.transformer.h, start=1):
        x = block(x)
        if layer_idx in model.exit_layers:
            exit_logits[layer_idx] = model.exit_heads[str(layer_idx)](x)
    final_logits = model.lm_head(model.transformer.ln_f(x))
    return exit_logits, final_logits


def add_bucket(summary, key, oracle_depth, dynamic_exit_layer, agreed, confidence):
    row = summary.setdefault(key, {
        "count": 0,
        "oracle_depth_sum": 0,
        "dynamic_exit_layer_sum": 0,
        "early_oracle_count": 0,
        "agreement_sum": 0,
        "confidence_sum": 0.0,
    })
    row["count"] += 1
    row["oracle_depth_sum"] += oracle_depth
    row["dynamic_exit_layer_sum"] += dynamic_exit_layer
    row["early_oracle_count"] += int(oracle_depth < dynamic_exit_layer)
    row["agreement_sum"] += int(agreed)
    row["confidence_sum"] += confidence


def add_stable_bucket(summary, key, stable_depth, bigram_freq):
    row = summary.setdefault(key, {
        "count": 0,
        "stable_depth_sum": 0,
        "bigram_freq_sum": 0,
    })
    row["count"] += 1
    row["stable_depth_sum"] += stable_depth
    row["bigram_freq_sum"] += bigram_freq


def finalize_bucket_rows(summary, key_name):
    rows = []
    for key, row in sorted(summary.items()):
        count = max(row["count"], 1)
        rows.append({
            key_name: key,
            "count": row["count"],
            "avg_oracle_depth": row["oracle_depth_sum"] / count,
            "avg_dynamic_exit_layer": row["dynamic_exit_layer_sum"] / count,
            "oracle_early_rate": row["early_oracle_count"] / count,
            "agreement_rate": row["agreement_sum"] / count,
            "avg_confidence": row["confidence_sum"] / count,
        })
    return rows


def finalize_stable_bucket_rows(summary, key_name):
    rows = []
    for key, row in sorted(summary.items()):
        count = max(row["count"], 1)
        rows.append({
            key_name: key,
            "count": row["count"],
            "avg_stable_oracle_depth": row["stable_depth_sum"] / count,
            "avg_bigram_frequency": row["bigram_freq_sum"] / count,
        })
    return rows


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def spearman_corr(xs, ys):
    if len(xs) < 2:
        return 0.0
    rx = rankdata(xs)
    ry = rankdata(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(rx, ry))
    den_x = sum((x - mean_x) ** 2 for x in rx) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


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
    model, checkpoint = load_model(args)
    assert model.config.dynamic_exit, "checkpoint must have dynamic_exit=True"

    dataset = get_dataset_name(args, checkpoint)
    data = load_data(dataset, args.split)
    unigram, bigram = frequency_stats(dataset)

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    use_autocast = "cuda" in args.device
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) if use_autocast else torch.no_grad()

    oracle_counts = {layer: 0 for layer in model.exit_layers + [model.config.n_layer]}
    stable_oracle_counts = {layer: 0 for layer in model.exit_layers + [model.config.n_layer]}
    layer_agreement = {layer: [0, 0] for layer in model.exit_layers}
    confidence_summary = {}
    unigram_summary = {}
    bigram_summary = {}
    stable_unigram_summary = {}
    stable_bigram_summary = {}
    log_bigram_freqs = []
    stable_depths = []
    total = 0

    max_start = len(data) - args.block_size - 1
    for _ in range(args.num_batches):
        starts = torch.randint(max_start, (args.batch_size,))
        x_np = np.stack([np.asarray(data[i:i + args.block_size], dtype=np.int64) for i in starts])
        y_np = np.stack([np.asarray(data[i + 1:i + 1 + args.block_size], dtype=np.int64) for i in starts])
        idx = torch.tensor(x_np, dtype=torch.long, device=args.device)

        with ctx:
            exit_logits, final_logits = collect_layer_logits(model, idx)

        final_top1 = final_logits.argmax(dim=-1).cpu()
        exit_top1 = {layer: logits.argmax(dim=-1).cpu() for layer, logits in exit_logits.items()}
        exit_probs = {
            layer: F.softmax(logits.float(), dim=-1).max(dim=-1).values.cpu()
            for layer, logits in exit_logits.items()
        }

        for b in range(args.batch_size):
            for t in range(args.block_size):
                final_pred = int(final_top1[b, t].item())
                oracle_depth = model.config.n_layer
                stable_oracle_depth = model.config.n_layer
                dynamic_exit_layer = model.config.n_layer
                confidence = 0.0
                agreed_at_dynamic_layer = True
                agreements = {}

                for layer in model.exit_layers:
                    pred = int(exit_top1[layer][b, t].item())
                    agreed = pred == final_pred
                    agreements[layer] = agreed
                    layer_agreement[layer][0] += int(agreed)
                    layer_agreement[layer][1] += 1
                    if agreed and oracle_depth == model.config.n_layer:
                        oracle_depth = layer
                    layer_confidence = float(exit_probs[layer][b, t].item())
                    if layer_confidence >= model.config.confidence_threshold and dynamic_exit_layer == model.config.n_layer:
                        dynamic_exit_layer = layer
                        confidence = layer_confidence
                        agreed_at_dynamic_layer = agreed

                for layer in model.exit_layers:
                    later_layers = [later for later in model.exit_layers if later >= layer]
                    if all(agreements[later] for later in later_layers):
                        stable_oracle_depth = layer
                        break

                if dynamic_exit_layer == model.config.n_layer:
                    confidence = 1.0
                    agreed_at_dynamic_layer = True

                oracle_counts[oracle_depth] += 1
                stable_oracle_counts[stable_oracle_depth] += 1
                total += 1

                token_id = int(y_np[b, t])
                prev_id = int(x_np[b, t])
                bigram_freq = bigram[(prev_id, token_id)]
                unigram_bucket = frequency_bucket(unigram[token_id])
                bigram_bucket = frequency_bucket(bigram_freq)
                conf_bucket = confidence_bucket(confidence)

                add_bucket(confidence_summary, conf_bucket, oracle_depth, dynamic_exit_layer, agreed_at_dynamic_layer, confidence)
                add_bucket(unigram_summary, unigram_bucket, oracle_depth, dynamic_exit_layer, agreed_at_dynamic_layer, confidence)
                add_bucket(bigram_summary, bigram_bucket, oracle_depth, dynamic_exit_layer, agreed_at_dynamic_layer, confidence)
                add_stable_bucket(stable_unigram_summary, unigram_bucket, stable_oracle_depth, bigram_freq)
                add_stable_bucket(stable_bigram_summary, bigram_bucket, stable_oracle_depth, bigram_freq)
                log_bigram_freqs.append(float(np.log(bigram_freq + 1)))
                stable_depths.append(float(stable_oracle_depth))

    output_prefix = args.output_prefix or os.path.join(args.out_dir, os.path.splitext(args.checkpoint)[0])
    oracle_rows = [{
        "layer": layer,
        "count": count,
        "rate": count / max(total, 1),
    } for layer, count in oracle_counts.items()]
    agreement_rows = [{
        "layer": layer,
        "agreement_vs_final": agree / count if count else 0.0,
        "count": count,
    } for layer, (agree, count) in layer_agreement.items()]
    stable_oracle_rows = [{
        "layer": layer,
        "count": count,
        "rate": count / max(total, 1),
    } for layer, count in stable_oracle_counts.items()]
    correlation_rows = [{
        "metric": "spearman_corr_log_bigram_freq_stable_depth",
        "value": spearman_corr(log_bigram_freqs, stable_depths),
        "count": len(stable_depths),
    }]

    write_csv(f"{output_prefix}_oracle_depth.csv", oracle_rows)
    write_csv(f"{output_prefix}_stable_oracle_depth.csv", stable_oracle_rows)
    write_csv(f"{output_prefix}_layer_agreement.csv", agreement_rows)
    write_csv(f"{output_prefix}_spearman.csv", correlation_rows)
    write_csv(f"{output_prefix}_confidence_buckets.csv", finalize_bucket_rows(confidence_summary, "confidence_bucket"))
    write_csv(f"{output_prefix}_unigram_familiarity.csv", finalize_bucket_rows(unigram_summary, "unigram_bucket"))
    write_csv(f"{output_prefix}_bigram_familiarity.csv", finalize_bucket_rows(bigram_summary, "bigram_bucket"))
    write_csv(f"{output_prefix}_stable_unigram_familiarity.csv", finalize_stable_bucket_rows(stable_unigram_summary, "unigram_bucket"))
    write_csv(f"{output_prefix}_stable_bigram_familiarity.csv", finalize_stable_bucket_rows(stable_bigram_summary, "bigram_bucket"))

    print("===== Oracle Required Depth =====")
    for row in oracle_rows:
        print(f"Layer {row['layer']}: {100 * row['rate']:.2f}% ({row['count']})")
    print("\n===== Stable Oracle Required Depth =====")
    for row in stable_oracle_rows:
        print(f"Layer {row['layer']}: {100 * row['rate']:.2f}% ({row['count']})")
    print("\n===== Layer Agreement =====")
    for row in agreement_rows:
        print(f"Layer {row['layer']} vs Final: {100 * row['agreement_vs_final']:.2f}%")
    print("\n===== Unigram Familiarity =====")
    for row in finalize_bucket_rows(unigram_summary, "unigram_bucket"):
        print(f"{row['unigram_bucket']}: avg_oracle_depth={row['avg_oracle_depth']:.2f}, oracle_early_rate={100 * row['oracle_early_rate']:.2f}%, count={row['count']}")
    print("\n===== Bigram Familiarity =====")
    for row in finalize_bucket_rows(bigram_summary, "bigram_bucket"):
        print(f"{row['bigram_bucket']}: avg_oracle_depth={row['avg_oracle_depth']:.2f}, oracle_early_rate={100 * row['oracle_early_rate']:.2f}%, count={row['count']}")
    print("\n===== Stable Bigram Familiarity =====")
    for row in finalize_stable_bucket_rows(stable_bigram_summary, "bigram_bucket"):
        print(f"{row['bigram_bucket']}: avg_stable_oracle_depth={row['avg_stable_oracle_depth']:.2f}, count={row['count']}")
    print("\n===== Spearman =====")
    print(f"corr(log(bigram_freq+1), stable_depth) = {correlation_rows[0]['value']:.4f}")
    print(f"\nwrote CSV files with prefix {output_prefix}_*.csv")


if __name__ == "__main__":
    main()
