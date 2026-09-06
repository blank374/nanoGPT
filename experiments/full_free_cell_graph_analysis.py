"""Causal interventions and structure analysis for Full-Free Cell Graph v1."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import FullFreeDynamicCellGraph, GPT, GPTConfig


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="out-full-free-natural-seed1337")
    p.add_argument("--checkpoint", default="ckpt.pt")
    p.add_argument("--reference_out_dir", default=None)
    p.add_argument("--static_out_dir", default=None)
    p.add_argument("--dataset", default="shakespeare_char")
    p.add_argument("--num_batches", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--position_bucket", type=int, default=16)
    p.add_argument("--marginal_batches", type=int, default=1)
    p.add_argument("--latency_iters", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def load_model(directory, checkpoint, device):
    state = torch.load(os.path.join(ROOT, directory, checkpoint), map_location=device)
    model = GPT(GPTConfig(**state["model_args"]))
    prefix = "_orig_mod."
    weights = {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state["model"].items()
    }
    model.load_state_dict(weights)
    model = model.to(device).eval()
    if getattr(model.config, "cell_graph_static_graph_path", ""):
        graph_path = model.config.cell_graph_static_graph_path
        if not os.path.isabs(graph_path):
            graph_path = os.path.join(ROOT, graph_path)
        graph = np.load(graph_path)
        model.set_cell_graph_overrides(
            torch.from_numpy(graph["node_mask"]).to(device=device, dtype=torch.float32),
            torch.from_numpy(graph["edge_mask"]).to(device=device, dtype=torch.float32),
        )
    return model


def batches(dataset, count, batch_size, block_size, device, seed):
    data = np.memmap(os.path.join(ROOT, "data", dataset, "val.bin"), dtype=np.uint16, mode="r")
    rng = torch.Generator().manual_seed(seed)
    result = []
    for _ in range(count):
        starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=rng)
        x = np.stack([np.asarray(data[int(i):int(i) + block_size], dtype=np.int64) for i in starts])
        y = np.stack([np.asarray(data[int(i) + 1:int(i) + block_size + 1], dtype=np.int64) for i in starts])
        result.append((torch.tensor(x, device=device), torch.tensor(y, device=device)))
    return result


@torch.no_grad()
def evaluate(model, data, overrides=None, collect=False):
    losses, records = [], []
    for i, (x, y) in enumerate(data):
        if overrides is None:
            if (isinstance(model.cell_graph, FullFreeDynamicCellGraph)
                    and not model.config.cell_graph_static_graph_path):
                model.set_cell_graph_overrides(None, None)
        else:
            model.set_cell_graph_overrides(*overrides[i])
        logits, _ = model(x, y)
        losses.append(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
        ).view_as(y).cpu())
        if collect:
            records.append({
                key: value.detach().cpu().clone()
                for key, value in model.cell_graph_route_records().items()
            })
    if (isinstance(model.cell_graph, FullFreeDynamicCellGraph)
            and not model.config.cell_graph_static_graph_path):
        model.set_cell_graph_overrides(None, None)
    per_token = torch.cat(losses, dim=0)
    return per_token.mean().item(), per_token, records


@torch.no_grad()
def reference_losses(model, data):
    losses = []
    for x, y in data:
        logits, _ = model(x, y)
        losses.append(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
        ).view_as(y).cpu())
    return torch.cat(losses, dim=0).numpy()


def concat(records, key):
    return torch.cat([record[key] for record in records], dim=0)


def signature_rows(nodes, edges):
    bits = np.concatenate([nodes.reshape(len(nodes), -1), edges.reshape(len(edges), -1)], axis=-1)
    return [row.tobytes() for row in np.packbits(bits.astype(np.uint8), axis=-1)]


def diversity(nodes, edges):
    ids = signature_rows(nodes, edges)
    counts = Counter(ids)
    ordered = np.asarray(sorted(counts.values(), reverse=True), dtype=np.float64)
    probs = ordered / max(ordered.sum(), 1)
    return {
        "unique_graphs": len(counts),
        "graph_entropy": float(-(probs * np.log(np.maximum(probs, 1e-12))).sum()),
        **{f"top{k}_coverage": float(ordered[:k].sum() / max(ordered.sum(), 1))
           for k in (1, 4, 8, 16)},
    }


def common_mask(rows):
    ids = [row.tobytes() for row in np.packbits(rows.astype(np.uint8), axis=-1)]
    winner = Counter(ids).most_common(1)[0][0]
    return rows[ids.index(winner)]


def fixed_graph(model, node_weights, node_masks, edge_masks, matched=False):
    flat_weights = node_weights.reshape(-1, node_weights.shape[-1])
    flat_nodes = node_masks.reshape(-1, node_masks.shape[-1])
    flat_edges = edge_masks.reshape(-1, edge_masks.shape[-2], edge_masks.shape[-1])
    usage = flat_nodes.mean(axis=0)
    if matched:
        active_count = int(round(flat_nodes.sum(axis=-1).mean()))
        selected = np.zeros_like(usage, dtype=bool)
        selected[np.argsort(-usage)[:active_count]] = True
    else:
        selected = usage >= 0.5
    weights = np.zeros_like(usage, dtype=np.float32)
    positive_mean = np.divide(
        (flat_weights * flat_nodes).sum(axis=0), flat_nodes.sum(axis=0),
        out=np.ones_like(usage, dtype=np.float32), where=flat_nodes.sum(axis=0) > 0,
    )
    weights[selected] = positive_mean[selected]
    edges = np.zeros_like(flat_edges[0], dtype=np.float32)
    valid = model.cell_graph._valid_source_mask(torch.device("cpu")).numpy()
    for node in np.flatnonzero(selected):
        rows = flat_edges[flat_nodes[:, node], node][:, valid[node]]
        chosen = common_mask(rows) if len(rows) else np.eye(1, int(valid[node].sum()), dtype=bool)[0]
        edges[node, valid[node]] = chosen.astype(np.float32)
        if edges[node].sum() == 0:
            edges[node, 0] = 1.0
        edges[node] /= edges[node].sum()
    return torch.from_numpy(weights), torch.from_numpy(edges)


def split_override(nodes, edges, sizes):
    return list(zip(torch.split(nodes, sizes), torch.split(edges, sizes)))


def request_shuffle(nodes, edges, sizes, seed):
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(nodes.size(0), generator=rng)
    if nodes.size(0) > 1 and torch.equal(order, torch.arange(nodes.size(0))):
        order = order.roll(1)
    return split_override(nodes[order], edges[order], sizes)


def grouped_shuffle(nodes, edges, tokens, positions, sizes, seed, with_position=False, bucket=16):
    rng = np.random.default_rng(seed)
    flat_nodes = nodes.reshape(-1, nodes.size(-1)).clone()
    flat_edges = edges.reshape(-1, edges.size(-2), edges.size(-1)).clone()
    token_flat = tokens.reshape(-1).numpy()
    pos_flat = positions.reshape(-1).numpy()
    source_nodes = flat_nodes.clone()
    source_edges = flat_edges.clone()
    groups = {}
    for index, token in enumerate(token_flat):
        key = (int(token), int(pos_flat[index] // bucket)) if with_position else int(token)
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        shuffled = np.asarray(indices)[rng.permutation(len(indices))]
        flat_nodes[indices] = source_nodes[torch.from_numpy(shuffled)]
        flat_edges[indices] = source_edges[torch.from_numpy(shuffled)]
    return split_override(flat_nodes.view_as(nodes), flat_edges.view_as(edges), sizes)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def structural_rows(model, node_masks, edge_masks):
    B, T, N = node_masks.shape
    nodes = node_masks.reshape(-1, N)
    edges = edge_masks.reshape(-1, N, N + 1)
    rows = []
    for index in range(len(nodes)):
        active = nodes[index]
        depths = np.zeros(N, dtype=np.int64)
        outdegree = np.zeros(N, dtype=np.int64)
        indegree = np.zeros(N, dtype=np.int64)
        for node in range(N):
            if not active[node]:
                continue
            parents = np.flatnonzero(edges[index, node, 1:node + 1])
            indegree[node] = len(parents)
            outdegree[parents] += 1
            depths[node] = 1 + (depths[parents].max() if len(parents) else 0)
        active_count = int(active.sum())
        rows.append({
            "active_cells": active_count,
            "active_edges": int(edges[index].sum()),
            "average_path_length": float(depths[active].mean()) if active_count else 0.0,
            "longest_path": int(depths.max()) if active_count else 0,
            "average_branching_factor": float(outdegree[outdegree > 0].mean())
            if (outdegree > 0).any() else 0.0,
            "average_merge_factor": float(indegree[indegree > 0].mean())
            if (indegree > 0).any() else 0.0,
        })
    return rows


def difficulty_rows(node_masks, edge_masks, depths, difficulty):
    cuts = np.quantile(difficulty.reshape(-1), [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(difficulty.reshape(-1), cuts)
    nodes = node_masks.reshape(-1, node_masks.shape[-1])
    edges = edge_masks.reshape(-1, edge_masks.shape[-2], edge_masks.shape[-1])
    depth = depths.reshape(-1)
    result = []
    for q in range(5):
        chosen = bins == q
        active = nodes[chosen]
        active_edges = edges[chosen].sum(axis=(-1, -2))
        fanin = edges[chosen].sum(axis=-1)[active]
        graph = diversity(active, edges[chosen])
        result.append({
            "difficulty_quintile": q + 1,
            "tokens": int(chosen.sum()),
            "avg_active_cells": float(active.sum(axis=-1).mean()),
            "avg_active_edges": float(active_edges.mean()),
            "avg_fanin": float(fanin.mean()) if len(fanin) else 0.0,
            "avg_depth": float(depth[chosen].mean()),
            "graph_entropy": graph["graph_entropy"],
        })
    return result


@torch.no_grad()
def marginal_values(model, data, records, count):
    output = []
    for batch_index in range(min(count, len(data))):
        x, y = data[batch_index]
        record = records[batch_index]
        node_weights = record["node_probs"]
        node_masks = record["node_mask"]
        edge_weights = record["edge_probs"]
        base_logits, _ = model(x, y)
        base = F.cross_entropy(
            base_logits.reshape(-1, base_logits.size(-1)), y.reshape(-1), reduction="none"
        ).view_as(y).cpu()
        for node in range(model.cell_graph.num_cells):
            removed = node_weights.clone()
            removed[..., node] = 0
            model.set_cell_graph_overrides(removed, edge_weights)
            logits, _ = model(x, y)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            ).view_as(y).cpu()
            active = node_masks[..., node]
            remove_delta = (loss - base)[active].mean().item() if active.any() else 0.0

            added = node_weights.clone()
            added[..., node] = torch.where(
                node_masks[..., node], added[..., node], torch.ones_like(added[..., node])
            )
            model.set_cell_graph_overrides(added, edge_weights)
            logits, _ = model(x, y)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            ).view_as(y).cpu()
            inactive = ~node_masks[..., node]
            add_gain = (base - loss)[inactive].mean().item() if inactive.any() else 0.0
            output.append((node, remove_delta, add_gain))
    model.set_cell_graph_overrides(None, None)
    rows = []
    for node in range(model.cell_graph.num_cells):
        values = [item for item in output if item[0] == node]
        rows.append({
            "node": node,
            "step": node // model.cell_graph.cells_per_step,
            "cell": node % model.cell_graph.cells_per_step,
            "remove_delta_nll": float(np.mean([x[1] for x in values])),
            "add_gain_nll": float(np.mean([x[2] for x in values])),
        })
    return rows


def specialization_rows(model, nodes, edges, tokens, difficulty, marginals, position_bucket):
    flat_nodes = nodes.reshape(-1, nodes.shape[-1])
    flat_edges = edges.reshape(-1, edges.shape[-2], edges.shape[-1])
    flat_tokens = tokens.reshape(-1)
    positions = np.tile(np.arange(tokens.shape[1]), tokens.shape[0])
    diff_bins = np.digitize(difficulty.reshape(-1), np.quantile(difficulty, [0.2, 0.4, 0.6, 0.8]))
    marginal = {row["node"]: row for row in marginals}
    rows = []
    for node in range(model.cell_graph.num_cells):
        active = flat_nodes[:, node]
        fanin = flat_edges[active, node].sum(axis=-1) if active.any() else np.asarray([])
        token_counts = Counter(flat_tokens[active].tolist()).most_common(8)
        pos_counts = Counter((positions[active] // position_bucket).tolist())
        rows.append({
            "node": node, "step": node // model.cell_graph.cells_per_step,
            "cell": node % model.cell_graph.cells_per_step,
            "usage_frequency": float(active.mean()),
            "mean_fanin": float(fanin.mean()) if len(fanin) else 0.0,
            "fanin_distribution": json.dumps({int(k): float((fanin == k).mean())
                                               for k in np.unique(fanin)}),
            "source_distribution": json.dumps(flat_edges[active, node].mean(axis=0).tolist()
                                                if active.any() else []),
            "top_token_distribution": json.dumps(token_counts),
            "position_bucket_distribution": json.dumps(pos_counts),
            "difficulty_quintile_usage": json.dumps([
                float(active[diff_bins == q].mean()) for q in range(5)
            ]),
            "remove_delta_nll": marginal[node]["remove_delta_nll"],
            "add_gain_nll": marginal[node]["add_gain_nll"],
        })
    return rows


def export_top_graphs(output, model, nodes, edges):
    flat_nodes = nodes.reshape(-1, nodes.shape[-1])
    flat_edges = edges.reshape(-1, edges.shape[-2], edges.shape[-1])
    ids = signature_rows(flat_nodes, flat_edges)
    counts = Counter(ids)
    top = []
    edge_rows = []
    for rank, (graph_id, count) in enumerate(counts.most_common(8), 1):
        index = ids.index(graph_id)
        active = flat_nodes[index]
        graph_edges = flat_edges[index]
        active_nodes = np.flatnonzero(active).tolist()
        listed_edges = []
        dot = ["digraph G {", "  current [shape=box];"]
        for node in active_nodes:
            dot.append(f"  C{node} [label=\"S{node // model.cell_graph.cells_per_step}C{node % model.cell_graph.cells_per_step}\"];")
            for source in np.flatnonzero(graph_edges[node]):
                source_name = "current" if source == 0 else f"C{source - 1}"
                listed_edges.append([source_name, f"C{node}"])
                dot.append(f"  {source_name} -> C{node};")
                edge_rows.append({"rank": rank, "source": source_name, "target": f"C{node}"})
        dot.append("}")
        with open(os.path.join(output, f"top_graph_{rank}.dot"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(dot))
        top.append({"rank": rank, "count": count, "coverage": count / len(ids),
                    "graph_id": hashlib.sha1(graph_id).hexdigest()[:16],
                    "active_nodes": active_nodes, "edges": listed_edges})
    with open(os.path.join(output, "top8_graphs.json"), "w", encoding="utf-8") as handle:
        json.dump(top, handle, indent=2)
    write_csv(os.path.join(output, "top8_graph_edges.csv"), edge_rows)


@torch.no_grad()
def latency(model, sample, iterations):
    x, _ = sample
    pos = torch.arange(x.size(1), device=x.device)
    anchor = model.transformer.drop(model.transformer.wte(x) + model.transformer.wpe(pos))
    for _ in range(10):
        model.cell_graph.route_only(anchor)
        model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
        def timed(call):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(iterations):
                call()
            end.record(); torch.cuda.synchronize()
            return start.elapsed_time(end) / iterations
    else:
        def timed(call):
            start = time.perf_counter()
            for _ in range(iterations):
                call()
            return (time.perf_counter() - start) * 1000 / iterations
    router_ms = timed(lambda: model.cell_graph.route_only(anchor))
    model_ms = timed(lambda: model(x))
    return {"router_cuda_latency_ms": router_ms, "model_cuda_latency_ms": model_ms,
            "router_model_latency_ratio": router_ms / max(model_ms, 1e-12)}


def main():
    a = args()
    torch.manual_seed(a.seed)
    model = load_model(a.out_dir, a.checkpoint, a.device)
    if not isinstance(model.cell_graph, FullFreeDynamicCellGraph):
        raise ValueError("analysis requires cell_graph_mode='full_free'")
    data = batches(a.dataset, a.num_batches, a.batch_size,
                   min(a.block_size, model.config.block_size), a.device, a.seed)
    learned_nll, learned_token_nll, records = evaluate(model, data, collect=True)
    node_weights = concat(records, "node_probs")
    node_masks = concat(records, "node_mask").bool()
    edge_weights = concat(records, "edge_probs")
    edge_masks = concat(records, "edge_mask").bool()
    depths = concat(records, "depth")
    tokens = torch.cat([x.cpu() for x, _ in data], dim=0)
    sizes = [x.size(0) for x, _ in data]
    positions = torch.arange(tokens.size(1)).view(1, -1).expand_as(tokens)

    common_nodes, common_edges = fixed_graph(
        model, node_weights.numpy(), node_masks.numpy(), edge_masks.numpy(), matched=False
    )
    fixed = [(common_nodes, common_edges) for _ in data]
    global_shuffle = request_shuffle(node_weights, edge_weights, sizes, a.seed + 1)
    token_shuffle = grouped_shuffle(node_weights, edge_weights, tokens, positions, sizes,
                                    a.seed + 2, False, a.position_bucket)
    token_position_shuffle = grouped_shuffle(node_weights, edge_weights, tokens, positions, sizes,
                                             a.seed + 3, True, a.position_bucket)
    conditions = [("learned", None), ("fixed_most_common", fixed),
                  ("global_request_shuffle", global_shuffle),
                  ("same_token_shuffle", token_shuffle),
                  ("same_token_position_shuffle", token_position_shuffle)]
    intervention_rows = []
    for name, override in conditions:
        nll = learned_nll if override is None else evaluate(model, data, override)[0]
        intervention_rows.append({"model": "dynamic", "intervention": name,
                                  "nll": nll, "ppl": math.exp(nll)})
    if a.static_out_dir:
        static_model = load_model(a.static_out_dir, a.checkpoint, a.device)
        nll = evaluate(static_model, data)[0]
        intervention_rows.append({"model": "static_trained", "intervention": "native",
                                  "nll": nll, "ppl": math.exp(nll)})

    if a.reference_out_dir:
        reference = load_model(a.reference_out_dir, a.checkpoint, a.device)
        difficulty = reference_losses(reference, data)
        difficulty_source = a.reference_out_dir
    else:
        difficulty = learned_token_nll.numpy()
        difficulty_source = "learned_self_proxy; pass --reference_out_dir for fixed-reference analysis"

    output = os.path.join(ROOT, a.out_dir)
    os.makedirs(output, exist_ok=True)
    graph_metrics = diversity(node_masks.numpy().reshape(-1, model.cell_graph.num_cells),
                              edge_masks.numpy().reshape(-1, model.cell_graph.num_cells,
                                                         1 + model.cell_graph.num_cells))
    structures = structural_rows(model, node_masks.numpy(), edge_masks.numpy())
    structure_summary = {key: float(np.mean([row[key] for row in structures]))
                         for key in structures[0]}
    marginal = marginal_values(model, data, records, a.marginal_batches)
    specialists = specialization_rows(
        model, node_masks.numpy(), edge_masks.numpy(), tokens.numpy(), difficulty,
        marginal, a.position_bucket,
    )
    difficulty_table = difficulty_rows(
        node_masks.numpy(), edge_masks.numpy(), depths.numpy(), difficulty
    )
    matched_nodes, matched_edges = fixed_graph(
        model, node_weights.numpy(), node_masks.numpy(), edge_masks.numpy(), matched=True
    )
    np.savez_compressed(os.path.join(output, "matched_static_graph.npz"),
                        node_mask=matched_nodes.numpy(), edge_mask=matched_edges.numpy())
    np.savez_compressed(os.path.join(output, "full_free_routes.npz"),
                        tokens=tokens.numpy(), node_weights=node_weights.numpy(),
                        node_masks=node_masks.numpy(), edge_weights=edge_weights.numpy(),
                        edge_masks=edge_masks.numpy(), depths=depths.numpy())
    write_csv(os.path.join(output, "full_free_interventions.csv"), intervention_rows)
    write_csv(os.path.join(output, "full_free_structure_tokens.csv"), structures)
    write_csv(os.path.join(output, "full_free_difficulty_quintiles.csv"), difficulty_table)
    write_csv(os.path.join(output, "full_free_marginal_cells.csv"), marginal)
    write_csv(os.path.join(output, "full_free_specialization.csv"), specialists)
    export_top_graphs(output, model, node_masks.numpy(), edge_masks.numpy())
    latency_stats = latency(model, data[0], a.latency_iters)
    active_counts = node_masks.sum(dim=-1).float()
    width_values = node_masks.view(
        node_masks.size(0), node_masks.size(1), model.cell_graph.num_steps,
        model.cell_graph.cells_per_step,
    ).sum(dim=-1).float()
    runtime_stats = model.last_cell_graph_stats or {}
    edge_usage = edge_masks.float().mean((0, 1))
    common_edges = [
        ["current" if source == 0 else f"C{source - 1}", f"C{node}", float(edge_usage[node, source])]
        for node in range(model.cell_graph.num_cells)
        for source in range(1 + model.cell_graph.num_cells)
        if edge_usage[node, source] >= 0.5
    ]
    summary = {
        **graph_metrics, **structure_summary, **latency_stats,
        "learned_ppl": math.exp(learned_nll),
        "active_cells_std": active_counts.std(unbiased=False).item(),
        "active_cells_min": active_counts.min().item(),
        "active_cells_max": active_counts.max().item(),
        "depth_std": depths.float().std(unbiased=False).item(),
        "depth_min": depths.min().item(),
        "depth_max": depths.max().item(),
        "empty_step_fraction": (width_values == 0).float().mean().item(),
        "step_empty_fractions": (width_values == 0).float().mean((0, 1)).tolist(),
        "router_parameters": runtime_stats.get("router_parameters"),
        "router_theoretical_macs_per_token": runtime_stats.get("router_theoretical_macs_per_token"),
        "theoretical_active_cell_macs": runtime_stats.get("theoretical_active_cell_macs"),
        "theoretical_skipped_cell_macs": runtime_stats.get("theoretical_skipped_cell_macs"),
        "difficulty_source": difficulty_source,
        "mean_step_widths": node_masks.float().view(
            node_masks.size(0), node_masks.size(1), model.cell_graph.num_steps,
            model.cell_graph.cells_per_step).sum(-1).mean((0, 1)).tolist(),
        "node_usage": node_masks.float().mean((0, 1)).tolist(),
        "shared_trunk_cells": [i for i, value in enumerate(node_masks.float().mean((0, 1)))
                               if value >= 0.8],
        "rare_cells": [i for i, value in enumerate(node_masks.float().mean((0, 1)))
                       if 0 < value <= 0.05],
        "dead_cells": [i for i, value in enumerate(node_masks.float().mean((0, 1)))
                       if value == 0],
        "common_edges_usage_ge_0_5": common_edges,
    }
    with open(os.path.join(output, "full_free_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    for row in intervention_rows:
        print(f"{row['model']:16s} {row['intervention']:28s} PPL {row['ppl']:.4f}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
