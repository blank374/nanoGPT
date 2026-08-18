"""
Phase 1: sweep confidence thresholds for Dynamic Fast/Slow Path models.

This script asks the research question directly:

Does confidence predict how much Transformer depth a token needs?

Outputs:
- threshold_sweep.csv
- threshold_sweep.svg

Example:
$ python experiments/dynamic_exit_threshold_sweep.py --out_dir=out-dynamic-exit-8l --device=cpu --max_new_tokens=32
"""

import argparse
import csv
import math
import os
import pickle
import sys
import time
from contextlib import nullcontext

import torch
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import GPTConfig, GPT


DEFAULT_PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "The opposite of hot is",
    "Once upon a time, in a small village",
    "In quantum chromodynamics, confinement refers to",
    "ROMEO:",
    "To be, or not to be",
]

CONFIDENCE_BINS = [
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.01),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float32", "bfloat16", "float16"])
    return parser.parse_args()


def cuda_sync(device):
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, device):
    cuda_sync(device)
    start = time.perf_counter()
    result = fn()
    cuda_sync(device)
    return result, time.perf_counter() - start


def load_model(args):
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
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


def build_codec(checkpoint):
    load_meta = "config" in checkpoint and "dataset" in checkpoint["config"]
    if load_meta:
        meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
        load_meta = os.path.exists(meta_path)
    if load_meta:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        return lambda s: [stoi[c] for c in s if c in stoi], lambda ids: "".join([itos[i] for i in ids])

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}), lambda ids: enc.decode(ids)


def bucket_name(lo, hi):
    if hi > 1.0:
        return f"{lo:.1f}-1.0"
    return f"{lo:.1f}-{hi:.1f}"


@torch.no_grad()
def evaluate_threshold(model, prompts, encode, args, ctx, threshold):
    model.config.confidence_method = "max_prob"
    model.config.confidence_threshold = threshold
    model.config.dynamic_exit = True

    nll_sum = 0.0
    nll_count = 0
    agreement_sum = 0
    token_count = 0
    layer_sum = 0
    early_count = 0
    false_confident = 0
    exit_counts = {layer: 0 for layer in model.exit_layers + [model.config.n_layer]}
    layer_agreement = {layer: [0, 0] for layer in model.exit_layers}
    bin_agreement = {bucket_name(lo, hi): [0, 0] for lo, hi in CONFIDENCE_BINS}

    for prompt in prompts:
        ids = encode(prompt)
        if len(ids) < 2:
            continue
        idx = torch.tensor(ids[:-1], dtype=torch.long, device=args.device)[None, :]
        targets = torch.tensor(ids[1:], dtype=torch.long, device=args.device)[None, :]

        with ctx:
            dynamic_logits, _ = model(idx)
        details = model.last_exit_details

        model.config.dynamic_exit = False
        try:
            with ctx:
                full_logits, _ = model(idx, targets)
        finally:
            model.config.dynamic_exit = True

        per_token_nll = F.cross_entropy(
            dynamic_logits.view(-1, dynamic_logits.size(-1)),
            targets.view(-1),
            reduction="sum",
        )
        nll_sum += float(per_token_nll.item())
        nll_count += targets.numel()

        dynamic_top1 = dynamic_logits.argmax(dim=-1)
        full_top1 = full_logits.argmax(dim=-1)
        agreed = dynamic_top1 == full_top1
        exit_layer = details["exit_layer"]
        max_prob = details["max_prob"]

        agreement_sum += int(agreed.sum().item())
        token_count += targets.numel()
        layer_sum += int(exit_layer.sum().item())

        for layer in exit_counts:
            count = int((exit_layer == layer).sum().item())
            exit_counts[layer] += count

        early_mask = exit_layer < model.config.n_layer
        early_count += int(early_mask.sum().item())
        false_confident += int((early_mask & ~agreed).sum().item())

        for layer in model.exit_layers:
            mask = exit_layer == layer
            count = int(mask.sum().item())
            if count:
                layer_agreement[layer][0] += int(agreed[mask].sum().item())
                layer_agreement[layer][1] += count

        for lo, hi in CONFIDENCE_BINS:
            mask = early_mask & (max_prob >= lo) & (max_prob < hi)
            count = int(mask.sum().item())
            if count:
                name = bucket_name(lo, hi)
                bin_agreement[name][0] += int(agreed[mask].sum().item())
                bin_agreement[name][1] += count

    generation_tokens = max(args.max_new_tokens * len(prompts), 1)
    generation_layers = []
    generation_early_rates = []

    def generate_all():
        outputs = []
        for prompt in prompts:
            ids = encode(prompt)
            if not ids:
                continue
            idx = torch.tensor(ids, dtype=torch.long, device=args.device)[None, :]
            outputs.append(model.generate_dynamic(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k))
        return outputs

    generated, elapsed = timed(generate_all, args.device)
    for _, _, summary in generated:
        generation_layers.append(summary["avg_layers_per_token"])
        generation_early_rates.append(summary["early_exit_rate"])

    loss = nll_sum / max(nll_count, 1)
    avg_layers = layer_sum / max(token_count, 1)
    early_exit_rate = early_count / max(token_count, 1)
    full_path_rate = 1.0 - early_exit_rate
    agreement = agreement_sum / max(token_count, 1)
    false_confident_rate = false_confident / max(early_count, 1)

    row = {
        "threshold": threshold,
        "avg_layers_per_token": avg_layers,
        "generation_avg_layers_per_token": sum(generation_layers) / max(len(generation_layers), 1),
        "tokens_sec": generation_tokens / max(elapsed, 1e-9),
        "ms_token": 1000 * elapsed / generation_tokens,
        "loss": loss,
        "perplexity": math.exp(loss),
        "agreement_with_full_model": agreement,
        "false_confident_exit_rate": false_confident_rate,
        "layer_saving_ratio": 1.0 - avg_layers / model.config.n_layer,
        "early_exit_rate": early_exit_rate,
        "full_path_rate": full_path_rate,
    }

    for layer in model.exit_layers + [model.config.n_layer]:
        row[f"exit_rate_layer_{layer}"] = exit_counts[layer] / max(token_count, 1)
    for layer in model.exit_layers:
        agree, count = layer_agreement[layer]
        row[f"agreement_layer_{layer}_vs_final"] = agree / count if count else 0.0
    for name, (agree, count) in bin_agreement.items():
        row[f"confidence_bin_{name}_agreement"] = agree / count if count else 0.0
        row[f"confidence_bin_{name}_count"] = count
    return row


def svg_polyline(points, x_min, x_max, y_min, y_max, left, top, width, height, color):
    coords = []
    for x, y in points:
        px = left + (x - x_min) / max(x_max - x_min, 1e-9) * width
        py = top + height - (y - y_min) / max(y_max - y_min, 1e-9) * height
        coords.append(f"{px:.1f},{py:.1f}")
    return f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2"/>'


def write_svg(rows, path):
    metrics = [
        ("avg_layers_per_token", "avg layers/token", "#1f77b4"),
        ("tokens_sec", "tokens/sec", "#2ca02c"),
        ("agreement_with_full_model", "agreement", "#9467bd"),
        ("perplexity", "perplexity", "#d62728"),
    ]
    width, height = 920, 680
    panel_w, panel_h = 360, 220
    panels = [(60, 70), (500, 70), (60, 390), (500, 390)]
    thresholds = [r["threshold"] for r in rows]
    x_min, x_max = min(thresholds), max(thresholds)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="60" y="35" font-family="Menlo, monospace" font-size="22">Dynamic Exit Threshold Sweep</text>',
    ]
    for (metric, title, color), (left, top) in zip(metrics, panels):
        values = [r[metric] for r in rows]
        y_min, y_max = min(values), max(values)
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad
        points = [(r["threshold"], r[metric]) for r in rows]
        parts.extend([
            f'<text x="{left}" y="{top - 18}" font-family="Menlo, monospace" font-size="15">{title}</text>',
            f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#cccccc"/>',
            svg_polyline(points, x_min, x_max, y_min, y_max, left, top, panel_w, panel_h, color),
            f'<text x="{left}" y="{top + panel_h + 28}" font-family="Menlo, monospace" font-size="12">{x_min:.2f}</text>',
            f'<text x="{left + panel_w - 34}" y="{top + panel_h + 28}" font-family="Menlo, monospace" font-size="12">{x_max:.2f}</text>',
            f'<text x="{left - 48}" y="{top + 12}" font-family="Menlo, monospace" font-size="12">{y_max:.2f}</text>',
            f'<text x="{left - 48}" y="{top + panel_h}" font-family="Menlo, monospace" font-size="12">{y_min:.2f}</text>',
        ])
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    model, checkpoint = load_model(args)
    assert model.config.dynamic_exit, "checkpoint must have dynamic_exit=True"
    encode, _ = build_codec(checkpoint)

    output_dir = args.output_dir or args.out_dir
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for threshold in args.thresholds:
        row = evaluate_threshold(model, args.prompts, encode, args, ctx, threshold)
        rows.append(row)
        print(
            f"threshold={threshold:.2f} "
            f"avg_layers={row['avg_layers_per_token']:.2f} "
            f"tokens/sec={row['tokens_sec']:.2f} "
            f"agreement={100 * row['agreement_with_full_model']:.2f}% "
            f"ppl={row['perplexity']:.2f} "
            f"early={100 * row['early_exit_rate']:.2f}%"
        )

    csv_path = os.path.join(output_dir, "threshold_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    svg_path = os.path.join(output_dir, "threshold_sweep.svg")
    write_svg(rows, svg_path)
    print(f"\nwrote {csv_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
