"""Evaluate quality, routing coverage, and real CUDA latency for dynamic depth."""

import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import GPT, GPTConfig


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)
    state = checkpoint["model"]
    prefix = "_orig_mod."
    state = {(key[len(prefix):] if key.startswith(prefix) else key): value
             for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_validation(data_dir):
    data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as handle:
        meta = pickle.load(handle)
    return data, meta


def make_batch(data, batch_size, block_size, device):
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def quality_and_routes(model, data, batches, batch_size, block_size, device):
    choices = model.config.dynamic_depth_choices
    nll = {depth: 0.0 for depth in choices}
    tokens = 0
    dynamic_nll = 0.0
    counts = {depth: 0 for depth in choices}
    confidence_sum = {depth: 0.0 for depth in choices}
    entropy_sum = 0.0
    requests = 0
    for _ in range(batches):
        x, y = make_batch(data, batch_size, block_size, device)
        model(x, y)  # training/eval-with-targets deliberately exposes all depths
        logits_by_depth = model.last_dynamic_depth_logits
        router_probs = model.last_dynamic_depth_router_logits.softmax(dim=-1)
        selected_mode = model.dynamic_depth_route_indices(model.last_dynamic_depth_router_logits)
        for mode, depth in enumerate(choices):
            logits = logits_by_depth[depth]
            nll[depth] += F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            ).item()
            mask = selected_mode == mode
            counts[depth] += int(mask.sum().item())
            confidence_sum[depth] += router_probs.max(dim=-1).values[mask].sum().item()
            if mask.any():
                dynamic_nll += F.cross_entropy(
                    logits[mask].reshape(-1, logits.size(-1)), y[mask].reshape(-1), reduction="sum"
                ).item()
        entropy_sum += (-(router_probs.clamp_min(1e-9) * router_probs.clamp_min(1e-9).log())
                        .sum(dim=-1).sum().item())
        tokens += y.numel()
        requests += x.size(0)

    print("\nQuality (same checkpoint, shared ln_f/lm_head)")
    for depth in choices:
        print(f"  fixed D{depth}: NLL {nll[depth] / tokens:.4f}, PPL {math.exp(nll[depth] / tokens):.3f}")
    print(f"  dynamic : NLL {dynamic_nll / tokens:.4f}, PPL {math.exp(dynamic_nll / tokens):.3f}")
    mean_depth = sum(depth * counts[depth] for depth in choices) / requests
    print("\nDepth distribution")
    for depth in choices:
        fraction = counts[depth] / requests
        confidence = confidence_sum[depth] / max(counts[depth], 1)
        print(f"  D{depth}: {fraction:.3%}, selected confidence {confidence:.4f}")
    print(f"  average depth: {mean_depth:.3f}/{choices[-1]}")
    print(f"  theoretical skipped-layer fraction: {1.0 - mean_depth / choices[-1]:.3%}")
    print(f"  router entropy: {entropy_sum / requests:.4f}")
    sorted_fractions = sorted((counts[d] / requests for d in choices), reverse=True)
    for k in (1, 3, 4, 8):
        print(f"  top-{k} path coverage: {sum(sorted_fractions[:k]):.3%}")


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
def latency(model, data, batch_sizes, block_size, decode_tokens, device, warmup, repeats):
    if device.type != "cuda":
        print("\nLatency skipped: CUDA is required for synchronized GPU timings.")
        return
    print("\nReal CUDA latency (grouped dynamic path vs ordinary full four-layer path)")
    for batch_size in batch_sizes:
        x, _ = make_batch(data, batch_size, block_size, device)
        baseline_ms = timed_cuda(lambda: model._forward_logits(x)[0], warmup, repeats)
        dynamic_ms = timed_cuda(
            lambda: model.forward_dynamic_depth(x, record_stats=False)[0], warmup, repeats
        )
        print(
            f"  prefill B={batch_size}: full {baseline_ms:.3f} ms, dynamic {dynamic_ms:.3f} ms, "
            f"speedup {baseline_ms / dynamic_ms:.3f}x, "
            f"tokens/s {batch_size * block_size / (dynamic_ms / 1000):.1f}"
        )

        prompt = x[:, :min(32, block_size)]
        request_depth, _ = model.predict_dynamic_depth(prompt)
        route_plan = None
        if batch_size == 1:
            request_depth = int(request_depth.item())
        else:
            route_plan = model.build_dynamic_depth_plan(request_depth)

        def decode_full():
            seq = prompt
            for _ in range(decode_tokens):
                logits, _ = model._forward_logits(seq[:, -model.config.block_size:])
                seq = torch.cat((seq, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)

        def decode_dynamic():
            seq = prompt
            for _ in range(decode_tokens):
                logits, _ = model.forward_dynamic_depth(
                    seq[:, -model.config.block_size:],
                    forced_depth=(None if route_plan is not None else request_depth),
                    record_stats=False, route_plan=route_plan
                )
                seq = torch.cat((seq, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)

        full_decode_ms = timed_cuda(decode_full, 1, max(3, repeats // 5))
        dynamic_decode_ms = timed_cuda(decode_dynamic, 1, max(3, repeats // 5))
        output_tokens = batch_size * decode_tokens
        print(
            f"  decode  B={batch_size}: full {full_decode_ms:.3f} ms, dynamic {dynamic_decode_ms:.3f} ms, "
            f"speedup {full_decode_ms / dynamic_decode_ms:.3f}x, "
            f"tokens/s {output_tokens / (dynamic_decode_ms / 1000):.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="out-shakespeare-char-dynamic-depth-oracle/ckpt.pt")
    parser.add_argument("--data-dir", default="data/shakespeare_char")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--latency-batches", default="1,8,32")
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--cost-bias", type=float, default=0.0,
                        help="subtract this times normalized depth from router scores")
    args = parser.parse_args()
    torch.manual_seed(1337)
    np.random.seed(1337)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    assert model.config.enable_dynamic_depth, "checkpoint does not enable dynamic depth"
    model.config.dynamic_depth_inference_cost_bias = args.cost_bias
    data, _ = load_validation(args.data_dir)
    block_size = min(args.block_size, model.config.block_size)
    quality_and_routes(model, data, args.batches, args.batch_size, block_size, device)
    latency(model, data, [int(x) for x in args.latency_batches.split(",")],
            block_size, args.decode_tokens, device, args.warmup, args.repeats)


if __name__ == "__main__":
    main()
