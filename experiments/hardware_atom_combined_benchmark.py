"""End-to-end benchmark for direct address -> hot CUDA Graph -> cold fallback."""

import argparse
import csv
import json
import os
import statistics
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hardware_atom_benchmark import (clear_cached_routes, load_model, sample_batch,
                                     set_path, sync)
from hardware_atom_runtime import DirectAddressCUDAGraphDispatcher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='out-shakespeare-char-hardware-atom-joint-address')
    parser.add_argument('--checkpoint', default='ckpt.pt')
    parser.add_argument('--address_eval', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--confidence', type=float, default=0.60)
    parser.add_argument('--hot_k', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='float16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--output_csv', default=None)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--synthetic', action='store_true')
    return parser.parse_args()


def measure(operation, x, warmup, iters):
    for _ in range(warmup):
        operation()
    sync(x.device)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        operation()
        sync(x.device)
        samples.append(time.perf_counter() - start)
    mean_s = statistics.mean(samples)
    return mean_s * 1000.0, x.numel() / mean_s


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, checkpoint = load_model(args)
    x = sample_batch(args, checkpoint)
    dtype = {'float32': torch.float32, 'float16': torch.float16,
             'bfloat16': torch.bfloat16}[args.dtype]
    autocast = (torch.amp.autocast(device_type='cuda', dtype=dtype)
                if dtype != torch.float32 else torch.no_grad())
    eval_path = args.address_eval or os.path.join(args.out_dir, 'joint_address_threshold_060.json')
    with open(eval_path, encoding='utf-8') as handle:
        evaluation = json.load(handle)
    hot_plan_ids = [item['id'] for item in evaluation['hot_plans'][:args.hot_k]]

    clear_cached_routes(model)
    set_path(model, 'grouped')

    @torch.no_grad()
    def dynamic():
        clear_cached_routes(model)
        set_path(model, 'grouped')
        with autocast:
            return model.forward_inference_fast(x, compute_address=False)

    dynamic_logits = dynamic().detach().clone()
    dynamic_ms, dynamic_tps = measure(dynamic, x, args.warmup, args.iters)

    dispatcher = DirectAddressCUDAGraphDispatcher(
        model, hot_plan_ids, confidence=args.confidence, dtype=dtype
    )
    # First call creates the required capacity-bucket graphs. Compilation is a
    # one-time cache-fill cost and is excluded from steady-state measurements.
    combined_logits = dispatcher(x).detach().clone()
    sync(x.device)
    combined_ms, combined_tps = measure(
        lambda: dispatcher(x), x, args.warmup, args.iters
    )
    diff = (dynamic_logits - combined_logits).abs().max().item()
    argmax_agreement = float(
        (dynamic_logits.argmax(dim=-1) == combined_logits.argmax(dim=-1)).float().mean()
    )

    rows = [
        {'path': 'dynamic_layered', 'mean_ms': dynamic_ms,
         'tokens_per_second': dynamic_tps},
        {'path': 'direct_address_graph_dispatch', 'mean_ms': combined_ms,
         'tokens_per_second': combined_tps},
    ]
    for row in rows:
        row.update({'batch_size': args.batch_size, 'block_size': args.block_size,
                    'tokens': int(x.numel()), 'dtype': args.dtype,
                    'accepted_fraction': dispatcher.last_stats['accepted_fraction'],
                    'graph_cache_entries': dispatcher.last_stats['graph_cache_entries'],
                    'max_abs_diff_vs_dynamic': diff,
                    'next_token_argmax_agreement': argmax_agreement})
    output = args.output_csv or os.path.join(args.out_dir, 'hardware_atom_combined_benchmark.csv')
    with open(output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print('===== Combined Direct Address + CUDA Graph =====')
    print(f"accepted: {dispatcher.last_stats['accepted_fraction']:.1%}, "
          f"hot groups: {dispatcher.last_stats['hot_groups']}, "
          f"cached graphs: {dispatcher.last_stats['graph_cache_entries']}")
    print(f'dynamic layered: {dynamic_ms:.3f} ms, {dynamic_tps:.1f} tok/s')
    print(f'combined dispatch: {combined_ms:.3f} ms, {combined_tps:.1f} tok/s')
    print(f'end-to-end speedup: {dynamic_ms / combined_ms:.3f}x')
    print(f'next-token argmax agreement: {argmax_agreement:.1%}, max diff: {diff:.6e}')
    print(f'wrote: {output}')


if __name__ == '__main__':
    main()
