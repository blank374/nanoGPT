"""Compare language loss for dynamic, request-cached, and full-width paths."""

import argparse
import csv
import math
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hardware_atom_benchmark import (atom_modules, cache_exact_routes_from_prefill,
                                     cache_routes_from_prefill, clear_cached_routes,
                                     load_model, set_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='out-shakespeare-char-hardware-atom')
    parser.add_argument('--checkpoint', default='ckpt.pt')
    parser.add_argument('--dataset', default='shakespeare_char')
    parser.add_argument('--batches', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='float16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--output_csv', default=None)
    parser.add_argument('--seed', type=int, default=1337)
    return parser.parse_args()


def get_batch(args):
    data = np.memmap(os.path.join('data', args.dataset, 'val.bin'), dtype=np.uint16, mode='r')
    starts = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
    x = np.stack([np.asarray(data[i:i + args.block_size], dtype=np.int64) for i in starts])
    y = np.stack([np.asarray(data[i + 1:i + 1 + args.block_size], dtype=np.int64) for i in starts])
    return (torch.tensor(x, dtype=torch.long, device=args.device),
            torch.tensor(y, dtype=torch.long, device=args.device))


def autocast_context(args):
    if not args.device.startswith('cuda'):
        return torch.no_grad()
    dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
    return torch.amp.autocast(device_type='cuda', dtype=dtype)


@torch.no_grad()
def task_loss(model, x, y, args):
    with autocast_context(args):
        model(x, y)
    return float(model.last_loss_stats['task_loss'])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, _ = load_model(args)
    losses = {'dynamic': [], 'cached_exact': [], 'cached_shared': [], 'dense_full': []}
    cached_patterns = Counter()

    for _ in range(args.batches):
        x, y = get_batch(args)

        clear_cached_routes(model)
        set_path(model, 'grouped')
        losses['dynamic'].append(task_loss(model, x, y, args))

        clear_cached_routes(model)
        cache_exact_routes_from_prefill(model, x, autocast_context(args))
        set_path(model, 'grouped')
        losses['cached_exact'].append(task_loss(model, x, y, args))

        clear_cached_routes(model)
        widths = cache_routes_from_prefill(model, x, autocast_context(args))
        cached_patterns[tuple(widths)] += 1
        set_path(model, 'grouped')
        losses['cached_shared'].append(task_loss(model, x, y, args))

        clear_cached_routes(model)
        set_path(model, 'dense_full')
        losses['dense_full'].append(task_loss(model, x, y, args))

    rows = []
    for path, values in losses.items():
        mean_loss = sum(values) / len(values)
        rows.append({'path': path, 'cross_entropy': mean_loss, 'perplexity': math.exp(mean_loss),
                     'batches': args.batches, 'batch_size': args.batch_size, 'block_size': args.block_size})
    output = args.output_csv or os.path.join(args.out_dir, 'hardware_atom_quality.csv')
    with open(output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print('===== Hardware Atom Quality =====')
    for row in rows:
        print(f"{row['path']}: CE {row['cross_entropy']:.4f}, PPL {row['perplexity']:.3f}")
    print('cached route patterns:')
    for pattern, count in cached_patterns.most_common():
        print(f'  {list(pattern)}: {count}/{args.batches}')
    print(f'wrote: {output}')


if __name__ == '__main__':
    main()
