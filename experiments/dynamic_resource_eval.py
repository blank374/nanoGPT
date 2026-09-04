"""Quality, path concentration, route stability, and CUDA latency evaluation."""

import argparse
import csv
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import GPT, GPTConfig, ResourceRoutedMLP


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    prefix = "_orig_mod."
    state = {(key[len(prefix):] if key.startswith(prefix) else key): value
             for key, value in checkpoint["model"].items()}
    model.load_state_dict(state)
    return model.to(device).eval(), checkpoint.get("iter", checkpoint.get("iter_num", -1))


def load_data(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


def batch(data, batch_size, block_size, device):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in starts])
    return x.to(device), y.to(device)


def entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(max(count / total, 1e-12))
                for count in counter.values())


@torch.no_grad()
def evaluate(model, data, batches, batch_size, block_size, device):
    widths = model.config.dynamic_resource_widths
    full_path = [widths[-1]] * model.config.n_layer
    path_counts = Counter()
    pattern_paths = defaultdict(Counter)
    dynamic_nll = full_nll = 0.0
    tokens = requests = 0
    expected_compute = entropy_sum = confidence_sum = 0.0
    costs = torch.tensor([0.0, 0.125, 0.25, 0.50, 1.0], device=device)
    hard_cost = active_layers = width_sum = 0.0

    for _ in range(batches):
        x, y = batch(data, batch_size, block_size, device)
        dynamic = model._forward_dynamic_resource_logits(x, all_logits=True)
        probs = model.last_dynamic_resource_probs
        modes = model.last_dynamic_resource_modes
        full = model._forward_dynamic_resource_logits(
            x, forced_path=full_path, record_stats=False, all_logits=True
        )
        dynamic_nll += F.cross_entropy(dynamic.reshape(-1, dynamic.size(-1)), y.reshape(-1),
                                       reduction="sum").item()
        full_nll += F.cross_entropy(full.reshape(-1, full.size(-1)), y.reshape(-1),
                                    reduction="sum").item()
        selected_widths = torch.tensor(widths, device=device)[modes]
        for row in range(x.size(0)):
            path = tuple(int(value) for value in selected_widths[row].tolist())
            path_counts[path] += 1
            pattern_paths[int(x[row, 0].item())][path] += 1
        requests += x.size(0)
        tokens += y.numel()
        expected_compute += (probs * costs).sum(dim=-1).sum().item()
        hard_cost += costs[modes].sum().item()
        active_layers += (selected_widths != 0).sum().item()
        width_sum += selected_widths.sum().item()
        entropy_sum += (-(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log())
                        .sum(dim=-1)).sum().item()
        confidence_sum += probs.max(dim=-1).values.sum().item()

    counts = sorted(path_counts.values(), reverse=True)
    coverage = lambda k: sum(counts[:k]) / requests
    pattern_items = [(sum(routes.values()), entropy(routes)) for routes in pattern_paths.values()]
    split = float(np.median([count for count, _ in pattern_items])) if pattern_items else 0.0
    high = [value for count, value in pattern_items if count >= split]
    low = [value for count, value in pattern_items if count < split]
    dynamic_loss = dynamic_nll / tokens
    full_loss = full_nll / tokens
    average_compute = hard_cost / requests
    return {
        "baseline_ppl": math.exp(full_loss),
        "dynamic_ppl": math.exp(dynamic_loss),
        "delta_ppl": math.exp(dynamic_loss) - math.exp(full_loss),
        "average_compute": average_compute,
        "expected_compute": expected_compute / requests,
        "compute_saving": 1.0 - average_compute / model.config.n_layer,
        "average_active_layers": active_layers / requests,
        "skip_fraction": 1.0 - active_layers / (requests * model.config.n_layer),
        "average_width": width_sum / (requests * model.config.n_layer),
        "theoretical_paths": len(widths) ** model.config.n_layer,
        "observed_paths": len(path_counts),
        "top1_coverage": coverage(1), "top4_coverage": coverage(4),
        "top8_coverage": coverage(8), "top16_coverage": coverage(16),
        "router_entropy": entropy_sum / (requests * model.config.n_layer),
        "router_confidence": confidence_sum / (requests * model.config.n_layer),
        "average_route_entropy": sum(value for _, value in pattern_items) / max(len(pattern_items), 1),
        "high_frequency_route_entropy": sum(high) / max(len(high), 1),
        "low_frequency_route_entropy": sum(low) / max(len(low), 1),
        "most_common_paths": path_counts.most_common(16),
    }


def timed_cuda(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


@torch.no_grad()
def benchmark(model, data, batch_size, block_size, decode_tokens, device, warmup, repeats):
    if device.type != "cuda":
        return {}
    x, _ = batch(data, batch_size, block_size, device)
    full_path = [model.config.dynamic_resource_widths[-1]] * model.config.n_layer
    router_ms = timed_cuda(lambda: model.resource_router(
        model.transformer.drop(model.transformer.wte(x) + model.transformer.wpe(
            torch.arange(x.size(1), device=device)))), warmup, repeats)
    full_ms = timed_cuda(lambda: model._forward_dynamic_resource_logits(
        x, forced_path=full_path, record_stats=False), warmup, repeats)
    dynamic_ms = timed_cuda(lambda: model._forward_dynamic_resource_logits(x), warmup, repeats)

    # Event hooks report component kernel time without synchronizing between layers.
    component_events = {"attention": [], "mlp": []}
    handles = []
    def attach(module, name):
        def before(_module, _inputs):
            event = torch.cuda.Event(enable_timing=True); event.record()
            component_events[name].append([event, None])
        def after(_module, _inputs, _output):
            event = torch.cuda.Event(enable_timing=True); event.record()
            component_events[name][-1][1] = event
        handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    for block in model.transformer.h:
        attach(block.attn, "attention")
        if isinstance(block.mlp, ResourceRoutedMLP):
            attach(block.mlp, "mlp")
    model._forward_dynamic_resource_logits(x)
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    attention_ms = sum(start.elapsed_time(end) for start, end in component_events["attention"])
    mlp_ms = sum(start.elapsed_time(end) for start, end in component_events["mlp"])

    prompt = x[:, :min(32, block_size)]
    model._forward_dynamic_resource_logits(prompt)
    route_widths = torch.tensor(model.config.dynamic_resource_widths, device=device)[
        model.last_dynamic_resource_modes]
    def decode(forced_path):
        seq = prompt
        for _ in range(decode_tokens):
            logits = model._forward_dynamic_resource_logits(
                seq[:, -model.config.block_size:], forced_path=forced_path, record_stats=False
            )
            seq = torch.cat([seq, logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
    full_decode_ms = timed_cuda(lambda: decode(full_path), 1, max(3, repeats // 5))
    dynamic_decode_ms = timed_cuda(lambda: decode(route_widths), 1, max(3, repeats // 5))
    return {
        "baseline_latency_ms": full_ms, "dynamic_latency_ms": dynamic_ms,
        "router_overhead_ms": router_ms, "attention_latency_ms": attention_ms,
        "mlp_latency_ms": mlp_ms, "prefill_latency_ms": dynamic_ms,
        "decode_latency_ms": dynamic_decode_ms,
        "baseline_decode_latency_ms": full_decode_ms,
        "speedup": full_ms / dynamic_ms,
        "tokens_per_second": batch_size * block_size / (dynamic_ms / 1000.0),
    }


def print_report(metrics, latency):
    print("=" * 60)
    print("SELF-ORGANIZED COMPUTE PATH EVALUATION")
    print("=" * 60)
    print(f"Baseline PPL: {metrics['baseline_ppl']:.4f}")
    print(f"Dynamic PPL:  {metrics['dynamic_ppl']:.4f}")
    print(f"Average normalized compute: {metrics['average_compute']:.4f}")
    print(f"Compute saving: {metrics['compute_saving']:.2%}")
    print(f"Average active layers: {metrics['average_active_layers']:.3f}")
    print(f"Skip fraction: {metrics['skip_fraction']:.2%}")
    print(f"Average MLP width: {metrics['average_width']:.2f}")
    print(f"Observed paths: {metrics['observed_paths']}/{metrics['theoretical_paths']}")
    print(f"Top1/4/8/16 coverage: {metrics['top1_coverage']:.2%} / "
          f"{metrics['top4_coverage']:.2%} / {metrics['top8_coverage']:.2%} / "
          f"{metrics['top16_coverage']:.2%}")
    for rank, (path, count) in enumerate(metrics["most_common_paths"], 1):
        print(f"  {rank:2d}. {list(path)}  {count}")
    print(f"Average route entropy: {metrics['average_route_entropy']:.4f}")
    print(f"High-frequency route entropy: {metrics['high_frequency_route_entropy']:.4f}")
    print(f"Low-frequency route entropy: {metrics['low_frequency_route_entropy']:.4f}")
    if latency:
        print(f"Baseline/Dynamic latency: {latency['baseline_latency_ms']:.3f} / "
              f"{latency['dynamic_latency_ms']:.3f} ms")
        print(f"Router/Attention/MLP: {latency['router_overhead_ms']:.3f} / "
              f"{latency['attention_latency_ms']:.3f} / {latency['mlp_latency_ms']:.3f} ms")
        print(f"Decode latency: {latency['decode_latency_ms']:.3f} ms")
        print(f"Speedup: {latency['speedup']:.3f}x; tokens/s: {latency['tokens_per_second']:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="out-shakespeare-char-dynamic-resource/ckpt.pt")
    parser.add_argument("--data", default="data/shakespeare_char/val.bin")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    model, step = load_model(args.checkpoint, device)
    assert model.config.dynamic_resource
    data = load_data(args.data)
    block_size = min(args.block_size, model.config.block_size)
    metrics = evaluate(model, data, args.batches, args.batch_size, block_size, device)
    latency = benchmark(model, data, args.batch_size, block_size, args.decode_tokens,
                        device, args.warmup, args.repeats)
    print_report(metrics, latency)
    output = args.output_csv or os.path.join(os.path.dirname(args.checkpoint),
                                             "dynamic_resource_evaluation.csv")
    row = {"step": step, **{key: value for key, value in metrics.items()
                             if key != "most_common_paths"}, **latency}
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)


if __name__ == "__main__":
    main()
