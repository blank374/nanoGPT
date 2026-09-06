"""Per-token counterfactual test of marginal compute utility.

For every sampled target position, replay the learned graph everywhere and prune
only that position.  This keeps the prefix unchanged, so the measured NLL delta
is attributable to compute removed from the target token rather than earlier
tokens in its context.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from full_free_cell_graph_analysis import batches, concat, evaluate, load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-full-free-natural-seed1337")
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--dataset", default="shakespeare_char")
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--positions_per_sequence", type=int, default=8)
    parser.add_argument("--fixed_remove", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def rankdata(values):
    """Average ranks, sufficient for a dependency-free Spearman coefficient."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x, y, ranked=False):
    x = rankdata(x) if ranked else np.asarray(x, dtype=np.float64)
    y = rankdata(y) if ranked else np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def prune_one(weights, keep_count):
    result = weights.clone()
    active = torch.nonzero(result > 0, as_tuple=False).flatten()
    if keep_count >= len(active):
        return result
    scores = result[active]
    remove = active[torch.argsort(scores)[:len(active) - keep_count]]
    result[remove] = 0
    return result


@torch.no_grad()
def target_losses(model, x, y, node_weights, edge_weights, positions, mode, value):
    override = node_weights.clone()
    kept = []
    for sequence, position in enumerate(positions):
        original = override[sequence, position]
        count = int((original > 0).sum().item())
        if mode == "fraction":
            keep = max(1, int(round(count * value)))
        else:
            keep = max(1, count - int(value))
        override[sequence, position] = prune_one(original, keep)
        kept.append(keep)
    model.set_cell_graph_overrides(override, edge_weights)
    logits, _ = model(x, y)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
    ).view_as(y)
    selected = torch.stack([losses[i, position] for i, position in enumerate(positions)])
    return selected.cpu().numpy(), kept


def main():
    args = parse_args()
    model = load_model(args.out_dir, args.checkpoint, args.device)
    data = batches(
        args.dataset, args.num_batches, args.batch_size, args.block_size,
        args.device, args.seed,
    )
    _, base_losses, records = evaluate(model, data, collect=True)
    rng = np.random.default_rng(args.seed + 911)
    rows = []
    fractions = (0.8, 0.6, 0.4)

    for batch_index, ((x, y), record) in enumerate(zip(data, records)):
        nodes = record["node_probs"].to(args.device)
        edges = record["edge_probs"].to(args.device)
        for round_index in range(args.positions_per_sequence):
            # One independently selected target per sequence; positions exclude 0.
            positions = rng.integers(1, x.size(1), size=x.size(0)).tolist()
            condition_losses = {}
            condition_kept = {}
            for fraction in fractions:
                loss, kept = target_losses(
                    model, x, y, nodes, edges, positions, "fraction", fraction
                )
                condition_losses[fraction] = loss
                condition_kept[fraction] = kept
            loss_remove4, kept_remove4 = target_losses(
                model, x, y, nodes, edges, positions, "fixed", args.fixed_remove
            )
            for sequence, position in enumerate(positions):
                full = float(base_losses[batch_index * x.size(0) + sequence, position])
                active = int((nodes[sequence, position] > 0).sum().item())
                row = {
                    "seed": args.seed,
                    "batch": batch_index,
                    "round": round_index,
                    "sequence": sequence,
                    "position": position,
                    "target_token": int(y[sequence, position].item()),
                    "active_cells": active,
                    "nll_full": full,
                    "nll_keep80": float(condition_losses[0.8][sequence]),
                    "nll_keep60": float(condition_losses[0.6][sequence]),
                    "nll_keep40": float(condition_losses[0.4][sequence]),
                    "nll_remove4": float(loss_remove4[sequence]),
                    "kept80": condition_kept[0.8][sequence],
                    "kept60": condition_kept[0.6][sequence],
                    "kept40": condition_kept[0.4][sequence],
                    "kept_remove4": kept_remove4[sequence],
                    "marginal_last20": float(condition_losses[0.8][sequence]) - full,
                    "marginal_last4": float(loss_remove4[sequence]) - full,
                    "utility_full_to40": float(condition_losses[0.4][sequence]) - full,
                }
                rows.append(row)

    model.set_cell_graph_overrides(None, None)
    output_dir = os.path.join(ROOT, args.out_dir)
    csv_path = os.path.join(output_dir, "marginal_compute_utility.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    active = np.asarray([row["active_cells"] for row in rows])
    summary = {
        "seed": args.seed,
        "samples": len(rows),
        "mean_active_cells": float(active.mean()),
        "mean_nll_full": float(np.mean([row["nll_full"] for row in rows])),
        "mean_nll_keep80": float(np.mean([row["nll_keep80"] for row in rows])),
        "mean_nll_keep60": float(np.mean([row["nll_keep60"] for row in rows])),
        "mean_nll_keep40": float(np.mean([row["nll_keep40"] for row in rows])),
        "mean_marginal_last20": float(np.mean([row["marginal_last20"] for row in rows])),
        "mean_marginal_last4": float(np.mean([row["marginal_last4"] for row in rows])),
    }
    for metric in ("marginal_last20", "marginal_last4", "utility_full_to40"):
        utility = np.asarray([row[metric] for row in rows])
        summary[f"pearson_active_vs_{metric}"] = correlation(active, utility)
        summary[f"spearman_active_vs_{metric}"] = correlation(active, utility, ranked=True)
    json_path = os.path.join(output_dir, "marginal_compute_utility.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
