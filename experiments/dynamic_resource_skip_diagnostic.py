"""Measure the isolated quality price of every width choice at every layer."""

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.dynamic_resource_eval import batch, load_data, load_model


@torch.no_grad()
def path_nll(model, data, path, batches, batch_size, block_size, device):
    total_nll = 0.0
    tokens = 0
    for _ in range(batches):
        x, y = batch(data, batch_size, block_size, device)
        logits = model._forward_dynamic_resource_logits(
            x, forced_path=path, record_stats=False, all_logits=True
        )
        total_nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        tokens += y.numel()
    return total_nll / tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="out-shakespeare-char-dynamic-resource/ckpt.pt")
    parser.add_argument("--data", default="data/shakespeare_char/val.bin")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    # Reuse identical batches for every path so small quality differences are meaningful.
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device(args.device)
    model, step = load_model(args.checkpoint, device)
    data = load_data(args.data)
    widths = model.config.dynamic_resource_widths
    full = [widths[-1]] * model.config.n_layer
    torch.manual_seed(1234)
    np.random.seed(1234)
    base_nll = path_nll(model, data, full, args.batches, args.batch_size,
                        min(args.block_size, model.config.block_size), device)
    rows = []
    prices = {0: 0.0, 64: 0.125, 128: 0.25, 256: 0.5, 512: 1.0}
    print(f"full path NLL={base_nll:.6f}, PPL={math.exp(base_nll):.4f}")
    for layer in range(model.config.n_layer):
        for width in widths:
            torch.manual_seed(1234)
            np.random.seed(1234)
            path = list(full)
            path[layer] = width
            nll = path_nll(model, data, path, args.batches, args.batch_size,
                           min(args.block_size, model.config.block_size), device)
            row = {
                "step": step, "layer": layer + 1, "width": width,
                "path": str(path), "nll": nll, "ppl": math.exp(nll),
                "delta_nll": nll - base_nll,
                "delta_ppl": math.exp(nll) - math.exp(base_nll),
                "compute_saved": 1.0 - prices[width],
            }
            rows.append(row)
            print(f"L{layer + 1} width={width:3d}: delta_nll={row['delta_nll']:+.6f}, "
                  f"delta_ppl={row['delta_ppl']:+.4f}")
    output = args.output_csv or os.path.join(
        os.path.dirname(args.checkpoint), "dynamic_resource_skip_diagnostic.csv"
    )
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
