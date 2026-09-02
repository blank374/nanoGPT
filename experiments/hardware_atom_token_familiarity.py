"""Learn bottom-up token/bigram familiarity addresses for every atom MLP."""

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
from hardware_atom_benchmark import atom_modules, clear_cached_routes, load_model, set_path, sync


class TokenFamiliarityRouter(nn.Module):
    """O(1) lexical lookup producing one route per token and MLP layer."""

    def __init__(self, vocab_size, num_layers, num_modes, bigram_buckets=512):
        super().__init__()
        self.num_layers = num_layers
        self.num_modes = num_modes
        self.bigram_buckets = bigram_buckets
        output_size = num_layers * num_modes
        self.token_routes = nn.Embedding(vocab_size, output_size)
        self.bigram_routes = nn.Embedding(bigram_buckets, output_size)
        nn.init.zeros_(self.token_routes.weight)
        nn.init.zeros_(self.bigram_routes.weight)

    def forward(self, tokens):
        previous = F.pad(tokens[:, :-1], (1, 0))
        bigram_ids = (previous * 131 + tokens) % self.bigram_buckets
        logits = self.token_routes(tokens) + self.bigram_routes(bigram_ids)
        return logits.view(*tokens.shape, self.num_layers, self.num_modes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='out-shakespeare-char-hardware-atom')
    parser.add_argument('--checkpoint', default='ckpt.pt')
    parser.add_argument('--dataset', default='shakespeare_char')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='float16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_examples', type=int, default=4096)
    parser.add_argument('--val_examples', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--router_batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=3e-3)
    parser.add_argument('--bigram_buckets', type=int, default=512)
    parser.add_argument('--confidence', type=float, default=0.70)
    parser.add_argument('--quality_batches', type=int, default=20)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--router_output', default=None)
    parser.add_argument('--seed', type=int, default=1337)
    return parser.parse_args()


def autocast_context(args):
    if not args.device.startswith('cuda') or args.dtype == 'float32':
        return torch.no_grad()
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
    return torch.amp.autocast(device_type='cuda', dtype=dtype)


def get_batch(data, count, block_size, device, targets=False):
    starts = torch.randint(len(data) - block_size - 1, (count,))
    x = np.stack([np.asarray(data[i:i + block_size], dtype=np.int64) for i in starts])
    x = torch.tensor(x, dtype=torch.long, device=device)
    if not targets:
        return x
    y = np.stack([np.asarray(data[i + 1:i + 1 + block_size], dtype=np.int64) for i in starts])
    return x, torch.tensor(y, dtype=torch.long, device=device)


@torch.no_grad()
def collect(model, data, count, args):
    all_tokens, all_routes = [], []
    clear_cached_routes(model)
    set_path(model, 'dense_mask')
    remaining = count
    while remaining:
        size = min(args.batch_size, remaining)
        tokens = get_batch(data, size, args.block_size, args.device)
        with autocast_context(args):
            model(tokens)
        routes = torch.stack(
            [mlp.last_selected_mode for mlp in atom_modules(model)], dim=2
        )
        all_tokens.append(tokens)
        all_routes.append(routes)
        remaining -= size
    return torch.cat(all_tokens), torch.cat(all_routes)


def train_router(router, tokens, routes, args):
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate)
    losses = []
    for _ in range(args.steps):
        indices = torch.randint(tokens.size(0), (args.router_batch_size,), device=tokens.device)
        logits = router(tokens.index_select(0, indices))
        labels = routes.index_select(0, indices)
        loss = F.cross_entropy(logits.flatten(0, 2), labels.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return statistics.mean(losses[-100:])


def route_plan_ids(routes, num_modes):
    powers = num_modes ** torch.arange(routes.size(-1), device=routes.device)
    return (routes * powers).sum(dim=-1)


@torch.no_grad()
def metrics(router, tokens, routes, confidence):
    logits = router(tokens)
    probabilities = logits.softmax(dim=-1)
    predicted = logits.argmax(dim=-1)
    correct = predicted == routes
    min_confidence = probabilities.max(dim=-1).values.min(dim=-1).values
    accepted = min_confidence >= confidence
    exact = correct.all(dim=-1)
    plans = route_plan_ids(routes, router.num_modes).reshape(-1).cpu().tolist()
    counts = Counter(plans)
    top = counts.most_common(16)
    running = 0
    coverage = []
    for rank, (_, count) in enumerate(top, start=1):
        running += count
        if rank in (1, 2, 4, 8, 16):
            coverage.append({'k': rank, 'coverage': running / len(plans)})
    return {
        'per_layer_accuracy': [float(value) for value in correct.float().mean(dim=(0, 1))],
        'exact_token_plan_accuracy': float(exact.float().mean()),
        'accepted_token_fraction': float(accepted.float().mean()),
        'accepted_exact_accuracy': float(exact[accepted].float().mean()) if accepted.any() else 0.0,
        'mean_confidence': float(min_confidence.mean()),
        'observed_token_plans': len(counts),
        'top_k_plan_coverage': coverage,
    }


def set_external_routes(model, route_probabilities):
    for layer, mlp in enumerate(atom_modules(model)):
        mlp.external_mode_probs = route_probabilities[:, :, layer, :]
        mlp.eval_impl = 'grouped'


def clear_external_routes(model):
    for mlp in atom_modules(model):
        mlp.external_mode_probs = None


@torch.no_grad()
def model_loss(model, x, y, args):
    with autocast_context(args):
        model(x, y)
    return float(model.last_loss_stats['task_loss'])


@torch.no_grad()
def quality(model, router, data, args):
    result = {'dynamic': [], 'token_direct': [], 'token_confident_dynamic_fallback': []}
    ratios = {name: [] for name in result}
    accepted = []
    modules = atom_modules(model)
    for _ in range(args.quality_batches):
        x, y = get_batch(data, args.batch_size, args.block_size, args.device, targets=True)
        clear_external_routes(model)
        clear_cached_routes(model)
        set_path(model, 'grouped')
        result['dynamic'].append(model_loss(model, x, y, args))
        dynamic = torch.stack([mlp.last_selected_mode for mlp in modules], dim=2)

        logits = router(x)
        probs = logits.softmax(dim=-1)
        predicted = logits.argmax(dim=-1)
        confidence = probs.max(dim=-1).values.min(dim=-1).values
        accept = confidence >= args.confidence
        accepted.append(float(accept.float().mean()))

        set_external_routes(model, probs)
        result['token_direct'].append(model_loss(model, x, y, args))

        hybrid = dynamic.clone()
        hybrid[accept] = predicted[accept]
        hybrid_probs = F.one_hot(hybrid, num_classes=router.num_modes).float()
        set_external_routes(model, hybrid_probs)
        result['token_confident_dynamic_fallback'].append(model_loss(model, x, y, args))

        for name, routes in [('dynamic', dynamic), ('token_direct', predicted),
                             ('token_confident_dynamic_fallback', hybrid)]:
            layer_costs = [
                mlp.mode_costs[routes[:, :, layer]].mean()
                for layer, mlp in enumerate(modules)
            ]
            ratios[name].append(float(torch.stack(layer_costs).mean()))
    clear_external_routes(model)
    output = {}
    for name, values in result.items():
        ce = statistics.mean(values)
        output[name] = {'cross_entropy': ce, 'perplexity': math.exp(ce),
                        'mean_compute_ratio': statistics.mean(ratios[name])}
    output['token_confident_dynamic_fallback']['accepted_token_fraction'] = statistics.mean(accepted)
    return output


@torch.no_grad()
def timing(model, router, tokens, args, warmup=50, iters=500):
    route_inputs = []
    handles = [mlp.register_forward_pre_hook(
        lambda _module, inputs: route_inputs.append(inputs[0].detach())
    ) for mlp in atom_modules(model)]
    clear_external_routes(model)
    clear_cached_routes(model)
    set_path(model, 'dense_mask')
    with autocast_context(args):
        model(tokens)
    for handle in handles:
        handle.remove()

    def dynamic_routers():
        with autocast_context(args):
            for mlp, hidden in zip(atom_modules(model), route_inputs):
                mlp._route(hidden)

    def dynamic_inference():
        clear_external_routes(model)
        with autocast_context(args):
            model.forward_inference_fast(tokens, compute_address=False)

    def direct_inference():
        route_probs = router(tokens).softmax(dim=-1)
        set_external_routes(model, route_probs)
        with autocast_context(args):
            model.forward_inference_fast(tokens, compute_address=False)
        clear_external_routes(model)

    def measure(operation):
        for _ in range(warmup):
            operation()
        sync(tokens.device)
        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            operation()
            sync(tokens.device)
            samples.append(time.perf_counter() - start)
        return statistics.mean(samples) * 1000.0

    return {'token_bigram_lookup_ms': measure(lambda: router(tokens)),
            'four_layer_dynamic_router_ms': measure(dynamic_routers),
            'dynamic_model_ms': measure(dynamic_inference),
            'token_direct_model_ms': measure(direct_inference)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, _ = load_model(args)
    modules = atom_modules(model)
    if any(mlp.route_scope != 'token' for mlp in modules):
        raise ValueError('this experiment requires a token-routed checkpoint')
    train_data = np.memmap(os.path.join('data', args.dataset, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join('data', args.dataset, 'val.bin'), dtype=np.uint16, mode='r')
    train_tokens, train_routes = collect(model, train_data, args.train_examples, args)
    val_tokens, val_routes = collect(model, val_data, args.val_examples, args)
    router = TokenFamiliarityRouter(
        model.config.vocab_size, len(modules), len(modules[0].width_choices), args.bigram_buckets
    ).to(args.device)
    train_loss = train_router(router, train_tokens, train_routes, args)
    router.eval()
    address_metrics = metrics(router, val_tokens, val_routes, args.confidence)
    quality_metrics = quality(model, router, val_data, args)
    timing_metrics = timing(model, router, val_tokens[:args.batch_size], args)

    router_output = args.router_output or os.path.join(args.out_dir, 'token_familiarity_router.pt')
    torch.save({'model_state': router.state_dict(), 'vocab_size': model.config.vocab_size,
                'num_layers': router.num_layers, 'num_modes': router.num_modes,
                'bigram_buckets': router.bigram_buckets}, router_output)
    output = args.output_json or os.path.join(args.out_dir, 'token_familiarity_experiment.json')
    result = {'configuration': vars(args), 'train_loss': train_loss,
              'address_metrics': address_metrics, 'quality': quality_metrics,
              'timing': timing_metrics, 'router_checkpoint': router_output}
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print('===== Token / MLP Familiarity Experiment =====')
    print(f"observed token plans: {address_metrics['observed_token_plans']}")
    print('top-k coverage: ' + ', '.join(
        f"K={item['k']} {item['coverage']:.1%}" for item in address_metrics['top_k_plan_coverage']))
    print('layer accuracy: ' + ', '.join(f'{value:.1%}' for value in address_metrics['per_layer_accuracy']))
    print(f"exact token plan: {address_metrics['exact_token_plan_accuracy']:.1%}")
    print(f"accepted tokens: {address_metrics['accepted_token_fraction']:.1%}, "
          f"accepted exact: {address_metrics['accepted_exact_accuracy']:.1%}")
    for name, values in quality_metrics.items():
        print(f"{name}: PPL {values['perplexity']:.3f}, compute {values['mean_compute_ratio']:.1%}")
    print(f"token+bigram lookup: {timing_metrics['token_bigram_lookup_ms']:.4f} ms")
    print(f"four dynamic routers: {timing_metrics['four_layer_dynamic_router_ms']:.4f} ms")
    print(f"dynamic full model: {timing_metrics['dynamic_model_ms']:.4f} ms")
    print(f"token-direct full model: {timing_metrics['token_direct_model_ms']:.4f} ms")
    print(f"end-to-end speedup: "
          f"{timing_metrics['dynamic_model_ms'] / timing_metrics['token_direct_model_ms']:.3f}x")
    print(f'wrote router: {router_output}')
    print(f'wrote: {output}')


if __name__ == '__main__':
    main()
