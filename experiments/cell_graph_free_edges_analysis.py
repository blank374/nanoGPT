"""Evaluate whether Cell Graph connectivity is input-dependent or merely static.

For one learned-edge checkpoint this compares learned routing with its most-common
fixed graph, a request-shuffled replay, and the canonical current-only graph. It
also reports graph coverage and edge-mask entropy globally, per step, and per Cell.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned_out_dir", default="out-cell-graph-edges-learned-seed1337")
    parser.add_argument("--fixed_out_dir", default="out-cell-graph-edges-fixed-seed1337")
    parser.add_argument("--budget_out_dir", default=None)
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--dataset", default="shakespeare_char")
    parser.add_argument("--num_batches", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_model(out_dir, checkpoint_name, device):
    path = os.path.join(ROOT, out_dir, checkpoint_name)
    checkpoint = torch.load(path, map_location=device)
    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)
    prefix = "_orig_mod."
    state = {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(state)
    if not config.cell_graph:
        raise ValueError(f"{path} is not a Cell Graph checkpoint")
    return model.to(device).eval()


def make_batches(dataset, count, batch_size, block_size, device, seed):
    data = np.memmap(os.path.join(ROOT, "data", dataset, "val.bin"),
                     dtype=np.uint16, mode="r")
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(count):
        starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
        x = np.stack([
            np.asarray(data[int(i):int(i) + block_size], dtype=np.int64) for i in starts
        ])
        y = np.stack([
            np.asarray(data[int(i) + 1:int(i) + block_size + 1], dtype=np.int64)
            for i in starts
        ])
        batches.append((torch.tensor(x, device=device), torch.tensor(y, device=device)))
    return batches


@torch.no_grad()
def evaluate(model, batches, overrides=None, collect_routes=False):
    total_nll = 0.0
    total_tokens = 0
    routes = []
    for batch_index, (x, y) in enumerate(batches):
        model.set_cell_graph_edge_override(
            None if overrides is None else overrides[batch_index]
        )
        logits, _ = model(x, y)
        total_nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        total_tokens += y.numel()
        if collect_routes:
            record = model.cell_graph_route_records()
            routes.append({
                key: value.detach().cpu().clone()
                for key, value in record.items()
            })
    model.set_cell_graph_edge_override(None)
    return total_nll / total_tokens, routes


def signatures(values):
    packed = np.packbits(values.astype(np.uint8), axis=-1)
    return [row.tobytes() for row in packed]


def distribution_metrics(values):
    ids = signatures(values.reshape(values.shape[0], -1))
    counts = Counter(ids)
    ordered = sorted(counts.values(), reverse=True)
    total = max(len(ids), 1)
    probabilities = np.asarray(ordered, dtype=np.float64) / total
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    return {
        "unique": len(counts),
        "entropy": entropy,
        "top1_coverage": sum(ordered[:1]) / total,
        "top4_coverage": sum(ordered[:4]) / total,
        "top8_coverage": sum(ordered[:8]) / total,
    }


def conditional_entropy(mask_values, token_ids):
    result = 0.0
    total = len(token_ids)
    for token in np.unique(token_ids):
        chosen = mask_values[token_ids == token]
        result += len(chosen) / total * distribution_metrics(chosen)["entropy"]
    return result


def active_nodes(model):
    count = model.config.cell_graph_fixed_active_cells
    return [
        step * model.config.cell_graph_cells_per_step + offset
        for step in range(model.config.n_layer)
        for offset in range(count)
    ]


def concatenate_routes(routes, batches):
    edges = torch.cat([record["edge_mask"] for record in routes], dim=0).bool().numpy()
    tokens = torch.cat([x.detach().cpu() for x, _ in batches], dim=0).numpy()
    return edges, tokens


def most_common_graph(model, edges):
    result = torch.zeros(
        model.cell_graph.num_cells, 1 + model.cell_graph.num_cells, dtype=torch.bool
    )
    valid = model.cell_graph._valid_source_mask(torch.device("cpu")).numpy()
    for node in active_nodes(model):
        candidates = edges[:, :, node][:, :, valid[node]].reshape(-1, valid[node].sum())
        ids = signatures(candidates)
        chosen = Counter(ids).most_common(1)[0][0]
        first = ids.index(chosen)
        result[node, torch.from_numpy(valid[node])] = torch.from_numpy(candidates[first]).bool()
    return result


def fixed_overrides(mask, batches):
    return [mask for _ in batches]


def shuffled_overrides(edges, batches, seed):
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(edges.shape[0], generator=generator)
    # Avoid the identity permutation for tiny diagnostic runs.
    if torch.equal(permutation, torch.arange(edges.shape[0])) and edges.shape[0] > 1:
        permutation = permutation.roll(1)
    shuffled = torch.from_numpy(edges).index_select(0, permutation)
    sizes = [x.size(0) for x, _ in batches]
    return list(shuffled.split(sizes, dim=0))


def summarize_graphs(model, edges, tokens):
    selected = active_nodes(model)
    flattened_tokens = tokens.reshape(-1)
    token_edges = edges.reshape(-1, edges.shape[2], edges.shape[3])
    active_edges = token_edges[:, selected]
    global_metrics = distribution_metrics(active_edges)
    global_conditional = conditional_entropy(active_edges, flattened_tokens)
    global_metrics.update({
        "conditional_entropy_token": global_conditional,
        "mask_token_mutual_information": global_metrics["entropy"] - global_conditional,
    })

    valid = model.cell_graph._valid_source_mask(torch.device("cpu")).numpy()
    cell_rows = []
    for node in selected:
        step = node // model.config.cell_graph_cells_per_step
        offset = node % model.config.cell_graph_cells_per_step
        values = token_edges[:, node, valid[node]]
        metrics = distribution_metrics(values)
        cond = conditional_entropy(values, flattened_tokens)
        source_usage = token_edges[:, node].mean(axis=0)
        cell_rows.append({
            "step": step,
            "cell": offset,
            "node": node,
            "available_sources": int(valid[node].sum()),
            "mean_fanin": float(values.sum(axis=-1).mean()),
            "edge_mask_entropy": metrics["entropy"],
            "conditional_entropy_token": cond,
            "mask_token_mutual_information": metrics["entropy"] - cond,
            "unique_masks": metrics["unique"],
            "top1_input_mask_coverage": metrics["top1_coverage"],
            "top4_input_mask_coverage": metrics["top4_coverage"],
            "top8_input_mask_coverage": metrics["top8_coverage"],
            "source_usage_json": json.dumps(source_usage.tolist()),
        })

    step_rows = []
    for step in range(model.config.n_layer):
        nodes = [node for node in selected
                 if node // model.config.cell_graph_cells_per_step == step]
        metrics = distribution_metrics(token_edges[:, nodes])
        step_rows.append({
            "step": step,
            "graph_entropy": metrics["entropy"],
            "unique_graphs": metrics["unique"],
            "top1_graph_coverage": metrics["top1_coverage"],
            "top4_graph_coverage": metrics["top4_coverage"],
            "top8_graph_coverage": metrics["top8_coverage"],
        })
    return global_metrics, cell_rows, step_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    learned = load_model(args.learned_out_dir, args.checkpoint, device)
    if learned.config.cell_graph_fixed_active_cells <= 0:
        raise ValueError("free-edge analysis requires fixed active Cells")
    block_size = min(args.block_size, learned.config.block_size)
    batches = make_batches(
        args.dataset, args.num_batches, args.batch_size, block_size, device, args.seed
    )

    learned_nll, routes = evaluate(learned, batches, collect_routes=True)
    edges, tokens = concatenate_routes(routes, batches)
    common = most_common_graph(learned, edges)
    common_nll, _ = evaluate(learned, batches, fixed_overrides(common, batches))
    shuffled_nll, _ = evaluate(
        learned, batches, shuffled_overrides(edges, batches, args.seed + 1)
    )
    canonical = torch.zeros_like(common)
    canonical[:, 0] = True
    canonical_nll, _ = evaluate(learned, batches, fixed_overrides(canonical, batches))

    rows = [
        {"model": "learned", "intervention": "learned", "nll": learned_nll,
         "ppl": math.exp(learned_nll)},
        {"model": "learned", "intervention": "fixed_most_common", "nll": common_nll,
         "ppl": math.exp(common_nll)},
        {"model": "learned", "intervention": "shuffled_request_routes", "nll": shuffled_nll,
         "ppl": math.exp(shuffled_nll)},
        {"model": "learned", "intervention": "canonical_current_only", "nll": canonical_nll,
         "ppl": math.exp(canonical_nll)},
    ]
    if args.fixed_out_dir:
        fixed = load_model(args.fixed_out_dir, args.checkpoint, device)
        fixed_nll, _ = evaluate(fixed, batches)
        rows.append({"model": "fixed_trained", "intervention": "native", "nll": fixed_nll,
                     "ppl": math.exp(fixed_nll)})
    if args.budget_out_dir:
        budget = load_model(args.budget_out_dir, args.checkpoint, device)
        budget_nll, _ = evaluate(budget, batches)
        rows.append({"model": "edge_budget_trained", "intervention": "native",
                     "nll": budget_nll, "ppl": math.exp(budget_nll)})

    global_metrics, cell_rows, step_rows = summarize_graphs(learned, edges, tokens)
    output_dir = os.path.join(ROOT, args.output_dir or args.learned_out_dir)
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, "cell_graph_edge_interventions.csv"), rows)
    write_csv(os.path.join(output_dir, "cell_graph_edge_diversity_by_cell.csv"), cell_rows)
    write_csv(os.path.join(output_dir, "cell_graph_edge_diversity_by_step.csv"), step_rows)
    np.savez_compressed(
        os.path.join(output_dir, "cell_graph_edge_routes.npz"),
        tokens=tokens,
        edge_masks=edges,
        fixed_most_common=common.numpy(),
    )
    summary = {
        **global_metrics,
        "learned_ppl": math.exp(learned_nll),
        "fixed_most_common_ppl": math.exp(common_nll),
        "shuffled_ppl": math.exp(shuffled_nll),
        "learned_beats_fixed_most_common": learned_nll < common_nll,
        "learned_beats_shuffle": learned_nll < shuffled_nll,
    }
    with open(os.path.join(output_dir, "cell_graph_edge_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    for row in rows:
        print(f"{row['model']:20s} {row['intervention']:24s} PPL {row['ppl']:.4f}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
