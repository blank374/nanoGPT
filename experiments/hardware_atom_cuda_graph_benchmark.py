"""Benchmark exact-route atom execution with and without CUDA Graph replay."""

import argparse
import csv
import os
import statistics
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hardware_atom_benchmark import (cache_exact_routes_from_prefill, load_model,
                                     run_once, sample_batch, set_path, sync, time_path)
from hardware_atom_runtime import HardwareAtomCUDAGraphRunner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='out-shakespeare-char-hardware-atom-sequence')
    parser.add_argument('--checkpoint', default='ckpt.pt')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='float16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--output_csv', default=None)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--synthetic', action='store_true')
    return parser.parse_args()


@torch.no_grad()
def time_graph(runner, x, path, warmup, iters):
    for _ in range(warmup):
        runner(x)
    sync(x.device)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        runner(x)
        sync(x.device)
        samples.append(time.perf_counter() - start)
    mean_s = statistics.mean(samples)
    return {
        'path': path,
        'mean_ms': mean_s * 1000.0,
        'median_ms': statistics.median(samples) * 1000.0,
        'tokens_per_second': x.numel() / mean_s,
    }


def main():
    args = parse_args()
    if not args.device.startswith('cuda'):
        raise ValueError('CUDA Graph benchmark requires --device=cuda')
    torch.manual_seed(args.seed)
    model, checkpoint = load_model(args)
    x = sample_batch(args, checkpoint)
    dtype = {'float32': torch.float32, 'float16': torch.float16,
             'bfloat16': torch.bfloat16}[args.dtype]
    autocast = (torch.amp.autocast(device_type='cuda', dtype=dtype)
                if dtype != torch.float32 else torch.no_grad())
    bucket_sizes = cache_exact_routes_from_prefill(model, x, autocast)

    rows = []
    reference = {}
    runners = {}
    for path in ('dense_full', 'grouped'):
        logits, _ = run_once(model, x, path, autocast)
        reference[path] = logits.detach().clone()
        eager_row = time_path(model, x, path, autocast, args.warmup, args.iters)
        eager_row['path'] = f'eager_{path}'
        rows.append(eager_row)
        runners[path] = HardwareAtomCUDAGraphRunner(model, x, dtype=dtype)
        graph_logits = runners[path](x).detach().clone()
        sync(x.device)
        diff = (reference[path] - graph_logits).abs().max().item()
        row = time_graph(runners[path], x, f'graph_{path}', args.warmup, args.iters)
        row['max_abs_diff_vs_eager'] = diff
        rows.append(row)

    for row in rows:
        row.update({'batch_size': args.batch_size, 'block_size': args.block_size,
                    'tokens': int(x.numel()), 'dtype': args.dtype})
    output = args.output_csv or os.path.join(args.out_dir, 'hardware_atom_cuda_graph.csv')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_path = {row['path']: row for row in rows}
    dense_ms = by_path['graph_dense_full']['mean_ms']
    atom_ms = by_path['graph_grouped']['mean_ms']
    print('===== Hardware Atom CUDA Graph Benchmark =====')
    print(f'device: {args.device}, dtype: {args.dtype}, tokens: {x.numel()}')
    print(f'exact per-layer bucket sizes: {bucket_sizes}')
    for row in rows:
        diff = row.get('max_abs_diff_vs_eager')
        suffix = f', max diff {diff:.6e}' if diff is not None else ''
        print(f"{row['path']}: {row['mean_ms']:.3f} ms, "
              f"{row['tokens_per_second']:.1f} tok/s{suffix}")
    print(f'graph atom speedup vs graph dense: {dense_ms / atom_ms:.3f}x')
    print(f'wrote: {output}')


if __name__ == '__main__':
    main()
