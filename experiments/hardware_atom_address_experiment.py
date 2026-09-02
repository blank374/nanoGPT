"""Test direct associative addresses and hot/cold fallback for atom routing."""

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hardware_atom_benchmark import (atom_modules, clear_cached_routes, load_model,
                                     set_path, sync)


class DirectAddressRouter(nn.Module):
    """One small projection from an early high-D state to a whole-network address."""

    def __init__(self, hidden_size, num_layers, num_modes):
        super().__init__()
        self.num_layers = num_layers
        self.num_modes = num_modes
        self.address = nn.Linear(hidden_size, num_layers * num_modes)

    def forward(self, features):
        logits = self.address(F.normalize(features.float(), dim=-1))
        return logits.view(-1, self.num_layers, self.num_modes)


class BuiltinAddressAdapter(nn.Module):
    """Expose a jointly trained model address head on pre-aggregated features."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, features):
        logits = self.head.proj(F.normalize(features.float(), dim=-1))
        return logits.view(-1, self.head.num_layers, self.head.num_modes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='out-shakespeare-char-hardware-atom-sequence')
    parser.add_argument('--checkpoint', default='ckpt.pt')
    parser.add_argument('--dataset', default='shakespeare_char')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='float16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_examples', type=int, default=4096)
    parser.add_argument('--val_examples', type=int, default=1024)
    parser.add_argument('--router_steps', type=int, default=800)
    parser.add_argument('--router_batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-3)
    parser.add_argument('--quality_batches', type=int, default=20)
    parser.add_argument('--hot_k', type=int, default=4)
    parser.add_argument('--confidence', type=float, default=0.70)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--router_output', default=None)
    parser.add_argument('--use_model_address', action='store_true',
                        help='evaluate the address head stored inside a jointly trained checkpoint')
    parser.add_argument('--seed', type=int, default=1337)
    return parser.parse_args()


def autocast_context(args):
    if not args.device.startswith('cuda') or args.dtype == 'float32':
        return torch.no_grad()
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
    return torch.amp.autocast(device_type='cuda', dtype=dtype)


def early_features(model, x):
    positions = torch.arange(x.size(1), device=x.device)
    hidden = model.transformer.wte(x) + model.transformer.wpe(positions)
    return hidden.mean(dim=1)


def get_batch(data, count, block_size, device, targets=False):
    starts = torch.randint(len(data) - block_size - 1, (count,))
    x = np.stack([np.asarray(data[i:i + block_size], dtype=np.int64) for i in starts])
    x = torch.tensor(x, dtype=torch.long, device=device)
    if not targets:
        return x
    y = np.stack([np.asarray(data[i + 1:i + 1 + block_size], dtype=np.int64) for i in starts])
    return x, torch.tensor(y, dtype=torch.long, device=device)


def encode_plans(routes, num_modes):
    powers = num_modes ** torch.arange(routes.size(1), device=routes.device)
    return (routes * powers).sum(dim=1)


@torch.no_grad()
def collect_examples(model, data, count, args):
    features, routes = [], []
    clear_cached_routes(model)
    set_path(model, 'dense_mask')
    remaining = count
    while remaining:
        size = min(args.batch_size, remaining)
        x = get_batch(data, size, args.block_size, args.device)
        with autocast_context(args):
            model(x)
        features.append(early_features(model, x).detach())
        routes.append(torch.stack([mlp.last_selected_mode[:, 0] for mlp in atom_modules(model)], dim=1))
        remaining -= size
    return torch.cat(features), torch.cat(routes)


def train_router(router, features, routes, args):
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate)
    losses = []
    for _ in range(args.router_steps):
        indices = torch.randint(features.size(0), (args.router_batch_size,), device=features.device)
        logits = router(features.index_select(0, indices))
        labels = routes.index_select(0, indices)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return statistics.mean(losses[-min(100, len(losses)):])


@torch.no_grad()
def address_metrics(router, features, routes, hot_ids, num_modes, confidence):
    logits = router(features)
    probabilities = logits.softmax(dim=-1)
    predicted = logits.argmax(dim=-1)
    per_layer = (predicted == routes).float().mean(dim=0)
    exact = (predicted == routes).all(dim=1)
    plan_ids = encode_plans(predicted, num_modes)
    hot = torch.isin(plan_ids, hot_ids)
    min_confidence = probabilities.max(dim=-1).values.min(dim=1).values
    accepted = hot & (min_confidence >= confidence)
    return {
        'per_layer_accuracy': [float(value) for value in per_layer],
        'exact_plan_accuracy': float(exact.float().mean()),
        'hot_prediction_fraction': float(hot.float().mean()),
        'accepted_fraction': float(accepted.float().mean()),
        'accepted_exact_accuracy': float(exact[accepted].float().mean()) if accepted.any() else 0.0,
        'mean_min_confidence': float(min_confidence.mean()),
    }


def install_routes(model, routes):
    clear_cached_routes(model)
    set_path(model, 'grouped')
    for layer, mlp in enumerate(atom_modules(model)):
        mlp.cache_sequence_routes(routes[:, layer:layer + 1])


@torch.no_grad()
def task_loss(model, x, y, args):
    with autocast_context(args):
        model(x, y)
    return float(model.last_loss_stats['task_loss'])


@torch.no_grad()
def quality_experiment(model, router, data, hot_ids, args, num_modes):
    losses = {'dynamic': [], 'predicted_all': [], 'hot_confident_dynamic_fallback': [],
              'dense_full': []}
    ratios = {'dynamic': [], 'predicted_all': [], 'hot_confident_dynamic_fallback': []}
    accepted_total = 0
    request_total = 0

    for _ in range(args.quality_batches):
        x, y = get_batch(data, args.batch_size, args.block_size, args.device, targets=True)

        clear_cached_routes(model)
        set_path(model, 'grouped')
        losses['dynamic'].append(task_loss(model, x, y, args))
        dynamic_routes = torch.stack(
            [mlp.last_selected_mode[:, 0] for mlp in atom_modules(model)], dim=1
        )
        ratios['dynamic'].append(float(torch.stack([
            mlp.mode_costs[dynamic_routes[:, layer]].mean()
            for layer, mlp in enumerate(atom_modules(model))
        ]).mean()))

        logits = router(early_features(model, x))
        probabilities = logits.softmax(dim=-1)
        predicted = logits.argmax(dim=-1)
        min_confidence = probabilities.max(dim=-1).values.min(dim=1).values
        predicted_ids = encode_plans(predicted, num_modes)
        accepted = torch.isin(predicted_ids, hot_ids) & (min_confidence >= args.confidence)

        install_routes(model, predicted)
        losses['predicted_all'].append(task_loss(model, x, y, args))
        ratios['predicted_all'].append(float(torch.stack([
            mlp.mode_costs[predicted[:, layer]].mean()
            for layer, mlp in enumerate(atom_modules(model))
        ]).mean()))

        # Familiar/high-confidence requests take the learned shortcut. The
        # remaining requests keep the original layer-wise router: a slower
        # deliberative path rather than an untrained full-width substitution.
        hybrid = dynamic_routes.clone()
        hybrid[accepted] = predicted[accepted]
        install_routes(model, hybrid)
        losses['hot_confident_dynamic_fallback'].append(task_loss(model, x, y, args))
        ratios['hot_confident_dynamic_fallback'].append(float(torch.stack([
            mlp.mode_costs[hybrid[:, layer]].mean()
            for layer, mlp in enumerate(atom_modules(model))
        ]).mean()))
        accepted_total += int(accepted.sum())
        request_total += accepted.numel()

        clear_cached_routes(model)
        set_path(model, 'dense_full')
        losses['dense_full'].append(task_loss(model, x, y, args))

    result = {}
    for name, values in losses.items():
        mean_loss = statistics.mean(values)
        result[name] = {'cross_entropy': mean_loss, 'perplexity': math.exp(mean_loss)}
        if name in ratios:
            result[name]['mean_compute_ratio'] = statistics.mean(ratios[name])
    result['hot_confident_dynamic_fallback']['accepted_fraction'] = accepted_total / request_total
    return result


@torch.no_grad()
def benchmark_address(router, model, x, args, iters=500, warmup=50):
    features = early_features(model, x)
    route_inputs = []
    handles = [
        mlp.register_forward_pre_hook(
            lambda _module, inputs: route_inputs.append(inputs[0].detach())
        )
        for mlp in atom_modules(model)
    ]
    clear_cached_routes(model)
    set_path(model, 'dense_mask')
    with autocast_context(args):
        model(x)
    for handle in handles:
        handle.remove()

    def layered_dynamic_address():
        with autocast_context(args):
            for mlp, hidden in zip(atom_modules(model), route_inputs):
                mlp._route(hidden)

    def measure(operation):
        for _ in range(warmup):
            operation()
        sync(x.device)
        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            operation()
            sync(x.device)
            samples.append(time.perf_counter() - start)
        return statistics.mean(samples) * 1000.0

    return {
        'address_projection_ms': measure(lambda: router(features)),
        'embedding_plus_address_ms': measure(lambda: router(early_features(model, x))),
        'layered_dynamic_address_ms': measure(layered_dynamic_address),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, _ = load_model(args)
    modules = atom_modules(model)
    if any(mlp.route_scope != 'sequence' for mlp in modules):
        raise ValueError('direct address experiment requires sequence routing')
    num_layers = len(modules)
    num_modes = len(modules[0].width_choices)
    train_data = np.memmap(os.path.join('data', args.dataset, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join('data', args.dataset, 'val.bin'), dtype=np.uint16, mode='r')

    train_features, train_routes = collect_examples(model, train_data, args.train_examples, args)
    val_features, val_routes = collect_examples(model, val_data, args.val_examples, args)
    if args.use_model_address:
        if model.hardware_atom_address is None:
            raise ValueError('checkpoint does not contain a jointly trained address head')
        router = BuiltinAddressAdapter(model.hardware_atom_address).to(args.device)
        train_loss = None
    else:
        router = DirectAddressRouter(model.config.n_embd, num_layers, num_modes).to(args.device)
        train_loss = train_router(router, train_features, train_routes, args)
    router.eval()

    train_plan_ids = encode_plans(train_routes, num_modes).cpu().tolist()
    counts = Counter(train_plan_ids)
    hot = counts.most_common(args.hot_k)
    hot_ids = torch.tensor([plan for plan, _ in hot], dtype=torch.long, device=args.device)
    top_coverage = []
    running = 0
    for rank, (plan, count) in enumerate(counts.most_common(), start=1):
        running += count
        if rank in (1, 2, 4, 8, 16):
            top_coverage.append({'k': rank, 'coverage': running / len(train_plan_ids)})

    metrics = address_metrics(router, val_features, val_routes, hot_ids, num_modes, args.confidence)
    quality = quality_experiment(model, router, val_data, hot_ids, args, num_modes)
    timing_x = get_batch(val_data, args.batch_size, args.block_size, args.device)
    timing = benchmark_address(router, model, timing_x, args)

    result = {
        'configuration': vars(args),
        'route_space_size': num_modes ** num_layers,
        'observed_train_plans': len(counts),
        'hot_plans': [{'id': plan, 'count': count, 'fraction': count / len(train_plan_ids)}
                      for plan, count in hot],
        'top_k_coverage': top_coverage,
        'router_train_loss': train_loss,
        'validation_address_metrics': metrics,
        'quality': quality,
        'timing': timing,
    }
    if args.use_model_address:
        router_output = os.path.join(args.out_dir, args.checkpoint)
    else:
        router_output = args.router_output or os.path.join(args.out_dir, 'direct_address_router.pt')
        torch.save({
            'model_state': router.state_dict(),
            'hidden_size': model.config.n_embd,
            'num_layers': num_layers,
            'num_modes': num_modes,
            'hot_plan_ids': hot_ids.detach().cpu(),
            'confidence': args.confidence,
        }, router_output)
    result['router_checkpoint'] = router_output
    output = args.output_json or os.path.join(args.out_dir, 'hardware_atom_address_experiment.json')
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print('===== Direct Address / Hot-Cold Experiment =====')
    print(f'observed plans: {len(counts)}/{num_modes ** num_layers}')
    print('top-k coverage: ' + ', '.join(
        f"K={item['k']} {item['coverage']:.1%}" for item in top_coverage))
    print('layer accuracy: ' + ', '.join(f'{value:.1%}' for value in metrics['per_layer_accuracy']))
    print(f"exact plan accuracy: {metrics['exact_plan_accuracy']:.1%}")
    print(f"accepted hot/confident: {metrics['accepted_fraction']:.1%}, "
          f"accepted exact accuracy: {metrics['accepted_exact_accuracy']:.1%}")
    for name, values in quality.items():
        extra = f", compute {values['mean_compute_ratio']:.1%}" if 'mean_compute_ratio' in values else ''
        print(f"{name}: PPL {values['perplexity']:.3f}{extra}")
    print(f"address projection: {timing['address_projection_ms']:.4f} ms")
    print(f"embedding + address: {timing['embedding_plus_address_ms']:.4f} ms")
    print(f"four layered dynamic addresses: {timing['layered_dynamic_address_ms']:.4f} ms")
    print(f'wrote router: {router_output}')
    print(f'wrote: {output}')


if __name__ == '__main__':
    main()
