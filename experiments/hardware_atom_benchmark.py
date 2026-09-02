"""Benchmark fair dense, correctness-reference, and routed atom GPU paths."""

import argparse
import csv
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import GPT, GPTConfig, HardwareAtomMLP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='out-shakespeare-char-hardware-atom')
    parser.add_argument('--checkpoint', default='ckpt.pt')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='float16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--force_width', type=int, default=None)
    parser.add_argument('--cache_route', action='store_true',
                        help='route the prefill once, then cache one majority mode per layer')
    parser.add_argument('--cache_exact_routes', action='store_true',
                        help='cache exact per-request route buckets from one sequence prefill')
    parser.add_argument('--output_csv', default=None)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--synthetic', action='store_true', help='benchmark execution only; do not report quality')
    return parser.parse_args()


def load_model(args):
    path = os.path.join(args.out_dir, args.checkpoint)
    # Checkpoints are produced locally by this repository and contain optimizer
    # metadata in addition to tensor weights.
    checkpoint = torch.load(path, map_location=args.device, weights_only=False)
    model = GPT(GPTConfig(**checkpoint['model_args']))
    state = checkpoint['model']
    for key in list(state):
        if key.startswith('_orig_mod.'):
            state[key[len('_orig_mod.'):]] = state.pop(key)
    model.load_state_dict(state, strict=False)
    model.eval().to(args.device)
    assert model.config.hardware_atom_mlp
    return model, checkpoint


def atom_modules(model):
    return [block.mlp for block in model.transformer.h if isinstance(block.mlp, HardwareAtomMLP)]


def set_path(model, path):
    for mlp in atom_modules(model):
        mlp.eval_impl = path


def force_width(model, width):
    if width is None:
        return
    for mlp in atom_modules(model):
        assert width in mlp.width_choices, f'force_width must be one of {mlp.width_choices}'
        mlp.force_mode_index = mlp.width_choices.index(width)


def clear_cached_routes(model):
    for mlp in atom_modules(model):
        mlp.force_mode_index = None
        mlp.clear_cached_sequence_routes()


def cache_routes_from_prefill(model, x, autocast):
    # One dynamic pass acts as request prefill. Subsequent decoding/batched work
    # uses a host-known width per layer and avoids per-token GPU dispatch.
    set_path(model, 'dense_mask')
    with torch.no_grad(), autocast:
        model(x)
    cached_widths = []
    for mlp in atom_modules(model):
        counts = torch.bincount(mlp.last_selected_mode.reshape(-1), minlength=len(mlp.width_choices))
        mode_idx = int(counts.argmax().item())
        mlp.force_mode_index = mode_idx
        cached_widths.append(mlp.width_choices[mode_idx])
    return cached_widths


def cache_exact_routes_from_prefill(model, x, autocast):
    """Cache each request's route as reusable GPU batch-index buckets."""
    clear_cached_routes(model)
    set_path(model, 'dense_mask')
    with torch.no_grad(), autocast:
        model(x)
    bucket_sizes = []
    for mlp in atom_modules(model):
        mlp.cache_sequence_routes()
        bucket_sizes.append([int(indices.numel()) for indices in mlp.cached_sequence_indices])
    return bucket_sizes


def sample_batch(args, checkpoint):
    if args.synthetic:
        vocab_size = int(checkpoint['model_args']['vocab_size'])
        return torch.randint(vocab_size, (args.batch_size, args.block_size), device=args.device)
    dataset = args.dataset or checkpoint.get('config', {}).get('dataset', 'shakespeare_char')
    data = np.memmap(os.path.join('data', dataset, 'val.bin'), dtype=np.uint16, mode='r')
    starts = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
    array = np.stack([np.asarray(data[i:i + args.block_size], dtype=np.int64) for i in starts])
    return torch.tensor(array, dtype=torch.long, device=args.device)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def run_once(model, x, path, autocast):
    set_path(model, path)
    with autocast:
        logits, _ = model(x)
    return logits, model.last_hardware_atom_stats or {}


@torch.no_grad()
def time_path(model, x, path, autocast, warmup, iters):
    for _ in range(warmup):
        run_once(model, x, path, autocast)
    sync(x.device)
    samples = []
    stats = {}
    for _ in range(iters):
        start = time.perf_counter()
        _, stats = run_once(model, x, path, autocast)
        sync(x.device)
        samples.append(time.perf_counter() - start)
    mean_s = statistics.mean(samples)
    return {
        'path': path,
        'mean_ms': mean_s * 1000.0,
        'median_ms': statistics.median(samples) * 1000.0,
        'tokens_per_second': x.numel() / mean_s,
        'mean_active_atoms': stats.get('mean_active_atoms', 0.0),
        'mean_active_channels': stats.get('mean_active_channels', 0.0),
        'mean_active_ratio': stats.get('mean_active_ratio', 0.0),
        'router_entropy': stats.get('router_entropy', 0.0),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, checkpoint = load_model(args)
    route_options = sum((args.force_width is not None, args.cache_route, args.cache_exact_routes))
    if route_options > 1:
        raise ValueError('--force_width, --cache_route, and --cache_exact_routes are mutually exclusive')
    force_width(model, args.force_width)
    x = sample_batch(args, checkpoint)
    dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
    autocast = torch.amp.autocast(device_type='cuda', dtype=dtype) if x.device.type == 'cuda' else torch.no_grad()
    cached_widths = cache_routes_from_prefill(model, x, autocast) if args.cache_route else []
    exact_bucket_sizes = cache_exact_routes_from_prefill(model, x, autocast) if args.cache_exact_routes else []

    dense_mask_logits, route_stats = run_once(model, x, 'dense_mask', autocast)
    grouped_logits, _ = run_once(model, x, 'grouped', autocast)
    route_diff = (dense_mask_logits - grouped_logits).abs().max().item()
    rows = [time_path(model, x, path, autocast, args.warmup, args.iters)
            for path in ('dense_full', 'dense_mask', 'grouped')]
    dense_ms = rows[0]['mean_ms']
    grouped_ms = rows[2]['mean_ms']
    for row in rows:
        row.update({
            'checkpoint': args.checkpoint,
            'iter': int(checkpoint.get('iter_num', 0)),
            'batch_size': args.batch_size,
            'block_size': args.block_size,
            'tokens': int(x.numel()),
            'dtype': args.dtype,
            'max_abs_diff_grouped_vs_dense_mask': route_diff,
        })
        for width, fraction in route_stats.get('mode_fractions', {}).items():
            row[f'width_{width}_fraction'] = fraction

    output = args.output_csv or os.path.join(args.out_dir, 'hardware_atom_benchmark.csv')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print('===== Hardware Atom Benchmark =====')
    print(f'device: {args.device}, dtype: {args.dtype}, tokens: {x.numel()}')
    print(f'grouped correctness max abs diff: {route_diff:.6e}')
    if cached_widths:
        print(f'cached per-layer widths: {cached_widths}')
    if exact_bucket_sizes:
        print(f'exact per-layer bucket sizes: {exact_bucket_sizes}')
    for row in rows:
        print(f"{row['path']}: {row['mean_ms']:.3f} ms, {row['tokens_per_second']:.1f} tok/s")
    print(f'true speedup grouped vs ordinary dense MLP: {dense_ms / grouped_ms:.3f}x')
    print(f'wrote: {output}')


if __name__ == '__main__':
    main()
