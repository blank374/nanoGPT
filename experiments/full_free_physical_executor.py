"""Minimal token-queued physical executor for frozen Full-Free Cell Graph v1.

The router and fixed attention remain dense.  For each Cell, token positions with
non-zero node weight are gathered into one queue; LayerNorm and the Cell MLP run
only for that queue, and their residual deltas are scattered back to [B,T,C].

This is deliberately a research prototype: it uses PyTorch indexing rather than
a custom CUDA/Triton kernel and supports only the frozen v1 evaluation path.
"""

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import FullFreeDynamicCellGraph, GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dirs", nargs="+", default=[
            "out-full-free-natural-seed1337",
            "out-full-free-budget8-seed1337",
        ],
    )
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--dataset", default="shakespeare_char")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--amp_dtype", choices=("float32", "float16", "bfloat16"),
                        default="float32")
    parser.add_argument("--executor_mode", choices=("auto", "token_queue", "whole_cell"),
                        default="auto",
                        help="token_queue compacts tokens; whole_cell only skips empty Cells")
    parser.add_argument("--auto_min_token_positions", type=int, default=8192,
                        help="auto mode uses queues only at or above this B*T")
    parser.add_argument("--json", default="full_free_physical_executor_benchmark.json")
    parser.add_argument("--csv", default="full_free_physical_executor_benchmark.csv")
    return parser.parse_args()


def load_model(out_dir, checkpoint, device):
    path = os.path.join(ROOT, out_dir, checkpoint)
    state = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**state["model_args"]))
    prefix = "_orig_mod."
    weights = {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state["model"].items()
    }
    model.load_state_dict(weights)
    return model.to(device).eval(), path


def validate_frozen_v1(model):
    config = model.config
    graph = model.cell_graph
    failures = []
    checks = {
        "cell_graph_mode='full_free'": (
            config.cell_graph and config.cell_graph_mode == "full_free"
            and isinstance(graph, FullFreeDynamicCellGraph)
        ),
        "evaluation mode": not model.training,
        "fixed dense attention": config.cell_graph_fixed_attention,
        "halt disabled": not config.cell_graph_halt,
        "identity input projection": config.cell_graph_input_projection == "identity",
        "no static graph": not bool(config.cell_graph_static_graph_path),
        "no node override": graph.node_override is None,
        "no edge override": graph.edge_override is None,
        "no exploration": float(graph.exploration) == 0.0,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    if failures:
        raise ValueError("physical executor supports only frozen Full-Free v1 eval; failed: "
                         + ", ".join(failures))


def _sparse_forward(self, anchor):
    """Semantics-preserving replacement for FullFreeDynamicCellGraph.forward."""
    validate_frozen_v1(types.SimpleNamespace(config=self.config, cell_graph=self,
                                              training=self.training))
    B, T, C = anchor.shape
    requested_mode = getattr(self, "physical_executor_mode", "token_queue")
    if (requested_mode == "auto"
            and B * T < getattr(self, "physical_executor_min_token_positions", 8192)):
        result = FullFreeDynamicCellGraph.forward(self, anchor)
        active = int(self.last_node_mask.sum().item())
        possible = B * T * self.num_cells
        self.last_physical_executor_stats = {
            "executor_selected": "dense_fallback",
            "active_token_cell_pairs": active,
            "possible_token_cell_pairs": possible,
            "active_pair_ratio": active / possible,
            "nonempty_cell_queues": 0,
            "empty_cell_queues": 0,
            "min_nonempty_queue": 0,
            "mean_nonempty_queue": 0.0,
            "max_nonempty_queue": 0,
        }
        return result
    temperature = max(float(self.temperature), 1e-4)
    context = self.router.graph_context(anchor)
    current = anchor
    cell_outputs = [None] * self.num_cells
    cell_source_features = [None] * self.num_cells
    cell_available = [None] * self.num_cells
    node_weights_records, node_mask_records = [], []
    edge_weights_records, edge_mask_records = [], []
    node_score_records, edge_score_records = [], []
    halt_records = []
    alive = torch.ones(B, T, device=anchor.device, dtype=anchor.dtype)
    queue_sizes = []

    for step in range(self.num_steps):
        current = current + self.fixed_attentions[step](self.fixed_attention_norms[step](current))
        local_context = context + self.router.state_down(current)
        routed, node_scores = self.router.node_weights(
            local_context, step, temperature, self.config.cell_graph_node_selector
        )
        cell_weights = routed[..., :self.cells_per_step] * alive.unsqueeze(-1)
        node_masks = cell_weights > 0
        step_deltas = []

        candidate_indices = self._candidate_indices(step)
        source_tensors = [current]
        source_features = [self.router.source_down(current)]
        source_available = [torch.ones(B, T, dtype=torch.bool, device=anchor.device)]
        for source_index in candidate_indices[1:]:
            prior_node = source_index - 1
            source_tensors.append(cell_outputs[prior_node])
            source_features.append(cell_source_features[prior_node])
            source_available.append(cell_available[prior_node])
        sources = torch.stack(source_tensors, dim=2)
        routed_sources = torch.stack(source_features, dim=2)
        available = torch.stack(source_available, dim=-1)
        step_edge_weights, step_edge_scores = self.router.edge_weights_step(
            local_context, routed_sources, available,
            step * self.cells_per_step, temperature, self.config.cell_graph_edge_selector,
        )

        for offset in range(self.cells_per_step):
            node = step * self.cells_per_step + offset
            edge_weights = step_edge_weights[..., offset, :]
            edge_scores = step_edge_scores[..., offset, :]
            edge_weights = edge_weights * available.to(edge_weights.dtype)
            empty = edge_weights.sum(dim=-1, keepdim=True) == 0
            fallback = torch.zeros_like(edge_weights)
            fallback[..., 0] = 1.0
            edge_weights = torch.where(empty, fallback, edge_weights)
            edge_weights = edge_weights / edge_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            fused = (sources * edge_weights.unsqueeze(-1)).sum(dim=2)

            active_flat = node_masks[..., offset].reshape(-1)
            delta_flat = fused.new_zeros(B * T, C)
            # Boolean gather materializes the queue and exposes its length.  Do not
            # additionally call mask.sum().item(): that would add a second forced
            # CPU/GPU synchronization for every Cell in this Python prototype.
            execution_mode = requested_mode
            if execution_mode == "auto":
                execution_mode = "token_queue"
            if execution_mode == "whole_cell":
                # One scalar synchronization decides whether this Cell launches.
                # No [tokens, channels] gather is materialized in this mode.
                queue_size = int(active_flat.sum().item())
                queued = None
            else:
                queued = fused.reshape(B * T, C)[active_flat]
                queue_size = queued.shape[0]
            queue_sizes.append(queue_size)
            if queue_size and execution_mode == "whole_cell":
                # Hybrid fallback: retain one large, efficient GEMM for a partially
                # active Cell, but launch no MLP kernels when its queue is empty.
                # This avoids fragmented tiny GEMMs and is still an exact physical
                # skip for graph-level empty Cells.
                dense_delta = self.cells[node](self.input_norms[node](fused))
                delta_flat = dense_delta.to(delta_flat.dtype).reshape(B * T, C)
            elif queue_size:
                # One gather, one Cell-sized batched MLP, one scatter per non-empty queue.
                queued = self.input_norms[node](queued)
                # Autocast may produce fp16/bf16 Cell outputs while the residual
                # stream (and therefore the scatter buffer) remains fp32.
                # Dense execution promotes at the later weighted residual add;
                # make that promotion explicit before index_put_.
                cell_delta = self.cells[node](queued).to(delta_flat.dtype)
                delta_flat[active_flat] = cell_delta
            delta = delta_flat.view(B, T, C)
            step_deltas.append(delta)
            cell_outputs[node] = delta
            cell_source_features[node] = self.router.source_down(delta)
            cell_available[node] = node_masks[..., offset]

            padded_weights = torch.zeros(
                B, T, 1 + self.num_cells, device=anchor.device, dtype=edge_weights.dtype
            )
            padded_scores = torch.zeros(
                B, T, 1 + self.num_cells, device=anchor.device, dtype=edge_scores.dtype
            )
            padded_weights[..., candidate_indices] = edge_weights
            padded_scores[..., candidate_indices] = edge_scores
            edge_weights_records.append(padded_weights)
            edge_mask_records.append((padded_weights > 0) & node_masks[..., offset, None])
            edge_score_records.append(padded_scores)

        deltas = torch.stack(step_deltas, dim=2)
        current = current + (deltas * cell_weights.unsqueeze(-1)).sum(dim=2)
        node_weights_records.append(cell_weights)
        node_mask_records.append(node_masks)
        node_score_records.append(node_scores[..., :self.cells_per_step])
        halt_records.append(torch.zeros_like(alive))

    node_weights = torch.cat(node_weights_records, dim=-1)
    node_mask = torch.cat(node_mask_records, dim=-1)
    edge_weights = torch.stack(edge_weights_records, dim=2)
    edge_mask = torch.stack(edge_mask_records, dim=2)
    self.last_node_probs = node_weights
    self.last_node_mask = node_mask.detach()
    self.last_edge_probs = edge_weights
    self.last_edge_mask = edge_mask.detach()
    self.last_node_scores = torch.cat(node_score_records, dim=-1)
    self.last_edge_scores = torch.stack(edge_score_records, dim=2)
    self.last_halt_weights = torch.stack(halt_records, dim=-1)
    self.last_depth = self._hard_depth(node_mask.detach(), edge_mask.detach())
    nonempty = [size for size in queue_sizes if size]
    self.last_physical_executor_stats = {
        "executor_selected": execution_mode,
        "active_token_cell_pairs": int(sum(queue_sizes)),
        "possible_token_cell_pairs": int(B * T * self.num_cells),
        "active_pair_ratio": float(sum(queue_sizes) / (B * T * self.num_cells)),
        "nonempty_cell_queues": len(nonempty),
        "empty_cell_queues": self.num_cells - len(nonempty),
        "min_nonempty_queue": min(nonempty) if nonempty else 0,
        "mean_nonempty_queue": float(np.mean(nonempty)) if nonempty else 0.0,
        "max_nonempty_queue": max(nonempty) if nonempty else 0,
    }
    return current


@contextlib.contextmanager
def physical_executor(model, mode="token_queue", min_token_positions=8192):
    """Temporarily install the queued executor without changing frozen model.py."""
    validate_frozen_v1(model)
    graph = model.cell_graph
    graph.physical_executor_mode = mode
    graph.physical_executor_min_token_positions = int(min_token_positions)
    graph.forward = types.MethodType(_sparse_forward, graph)
    try:
        yield model
    finally:
        del graph.__dict__["forward"]
        del graph.physical_executor_mode
        del graph.physical_executor_min_token_positions


def sample_batch(dataset, batch_size, block_size, device, seed):
    data = np.memmap(os.path.join(ROOT, "data", dataset, "val.bin"),
                     dtype=np.uint16, mode="r")
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = np.stack([np.asarray(data[int(i):int(i) + block_size], dtype=np.int64) for i in starts])
    y = np.stack([np.asarray(data[int(i) + 1:int(i) + block_size + 1], dtype=np.int64)
                  for i in starts])
    return torch.tensor(x, device=device), torch.tensor(y, device=device)


def amp_context(device, dtype_name):
    if device.type != "cuda" or dtype_name == "float32":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=getattr(torch, dtype_name))


@torch.no_grad()
def semantic_check(model, x, y, amp_dtype, executor_mode="token_queue",
                   min_token_positions=8192):
    with amp_context(x.device, amp_dtype):
        dense_logits, _ = model(x, y)
    dense_nodes = model.cell_graph.last_node_mask.clone()
    dense_edges = model.cell_graph.last_edge_mask.clone()
    dense_nll = torch.nn.functional.cross_entropy(
        dense_logits.float().reshape(-1, dense_logits.size(-1)), y.reshape(-1)
    )
    with physical_executor(model, executor_mode, min_token_positions), amp_context(x.device, amp_dtype):
        sparse_logits, _ = model(x, y)
        executor_stats = dict(model.cell_graph.last_physical_executor_stats)
    sparse_nll = torch.nn.functional.cross_entropy(
        sparse_logits.float().reshape(-1, sparse_logits.size(-1)), y.reshape(-1)
    )
    max_abs = (dense_logits.float() - sparse_logits.float()).abs().max().item()
    mean_abs = (dense_logits.float() - sparse_logits.float()).abs().mean().item()
    nll_abs = abs(dense_nll.item() - sparse_nll.item())
    return {
        "dense_nll": dense_nll.item(),
        "sparse_nll": sparse_nll.item(),
        "nll_abs_diff": nll_abs,
        "logits_max_abs_diff": max_abs,
        "logits_mean_abs_diff": mean_abs,
        "node_masks_equal": bool(torch.equal(dense_nodes, model.cell_graph.last_node_mask)),
        "edge_masks_equal": bool(torch.equal(dense_edges, model.cell_graph.last_edge_mask)),
        "semantic_check_passed": bool(
            nll_abs <= (2e-4 if amp_dtype != "float32" else 2e-5)
            and max_abs <= (2e-2 if amp_dtype != "float32" else 2e-4)
            and torch.equal(dense_nodes, model.cell_graph.last_node_mask)
            and torch.equal(dense_edges, model.cell_graph.last_edge_mask)
        ),
        **executor_stats,
    }


@torch.no_grad()
def benchmark(model, x, mode, warmup, iterations, repeats, amp_dtype,
              executor_mode="token_queue", min_token_positions=8192):
    manager = (physical_executor(model, executor_mode, min_token_positions)
               if mode == "sparse" else contextlib.nullcontext(model))
    with manager:
        for _ in range(warmup):
            with amp_context(x.device, amp_dtype):
                model(x)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        samples = []
        for _ in range(repeats):
            if x.is_cuda:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    with amp_context(x.device, amp_dtype):
                        model(x)
                end.record()
                torch.cuda.synchronize(x.device)
                samples.append(start.elapsed_time(end) / iterations)
            else:
                start = time.perf_counter()
                for _ in range(iterations):
                    with amp_context(x.device, amp_dtype):
                        model(x)
                samples.append((time.perf_counter() - start) * 1000 / iterations)
    return samples


@torch.no_grad()
def benchmark_interleaved(model, x, warmup, iterations, repeats, amp_dtype,
                          executor_mode="token_queue", min_token_positions=8192):
    """AB/BA timing to reduce thermal and run-order bias."""
    for mode in ("dense", "sparse"):
        benchmark(model, x, mode, warmup, 1, 1, amp_dtype, executor_mode,
                  min_token_positions)

    def one(mode):
        manager = (physical_executor(model, executor_mode, min_token_positions)
                   if mode == "sparse" else contextlib.nullcontext(model))
        with manager:
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    with amp_context(x.device, amp_dtype):
                        model(x)
                end.record()
                torch.cuda.synchronize(x.device)
                return start.elapsed_time(end) / iterations
            start = time.perf_counter()
            for _ in range(iterations):
                with amp_context(x.device, amp_dtype):
                    model(x)
            return (time.perf_counter() - start) * 1000 / iterations

    samples = {"dense": [], "sparse": []}
    for repeat in range(repeats):
        order = ("dense", "sparse") if repeat % 2 == 0 else ("sparse", "dense")
        for mode in order:
            samples[mode].append(one(mode))
    return samples["dense"], samples["sparse"]


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rows = []
    for out_dir in args.out_dirs:
        model, checkpoint = load_model(out_dir, args.checkpoint, device)
        validate_frozen_v1(model)
        block_size = min(args.block_size, model.config.block_size)
        x, y = sample_batch(args.dataset, args.batch_size, block_size, device, args.seed)
        check = semantic_check(model, x, y, args.amp_dtype, args.executor_mode,
                               args.auto_min_token_positions)
        if not check["semantic_check_passed"]:
            raise RuntimeError(f"dense/sparse semantic check failed for {out_dir}: {check}")
        dense, sparse = benchmark_interleaved(
            model, x, args.warmup, args.iterations, args.repeats,
            args.amp_dtype, args.executor_mode, args.auto_min_token_positions,
        )
        dense_median = float(np.median(dense))
        sparse_median = float(np.median(sparse))
        row = {
            "out_dir": out_dir,
            "checkpoint": checkpoint,
            "device": str(device),
            "amp_dtype": args.amp_dtype,
            "executor_mode": args.executor_mode,
            "batch_size": args.batch_size,
            "block_size": block_size,
            "dense_latency_ms": dense_median,
            "sparse_latency_ms": sparse_median,
            "speedup_x": dense_median / sparse_median,
            "latency_reduction_fraction": 1.0 - sparse_median / dense_median,
            "dense_samples_ms": dense,
            "sparse_samples_ms": sparse,
            **check,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
    json_path = os.path.join(ROOT, args.json)
    csv_path = os.path.join(ROOT, args.csv)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    csv_rows = [{key: value for key, value in row.items()
                 if not isinstance(value, (list, dict))} for row in rows]
    write_csv(csv_path, csv_rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
