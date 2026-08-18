"""
Evaluate Dynamic Fast/Slow Path checkpoints against the full-depth baseline.

Example:
$ python experiments/dynamic_exit_eval.py --out_dir=out-shakespeare-char --device=cpu --max_new_tokens=64
"""

import argparse
import math
import os
import pickle
import sys
import time
from contextlib import nullcontext

import torch
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import GPTConfig, GPT


DEFAULT_PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "The opposite of hot is",
    "Once upon a time, in a small village",
    "In quantum chromodynamics, confinement refers to",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--confidence_method", default=None, choices=["max_prob", "entropy"])
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--entropy_threshold", type=float, default=None)
    return parser.parse_args()


def cuda_sync(device):
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_generate(fn, device):
    cuda_sync(device)
    start = time.perf_counter()
    result = fn()
    cuda_sync(device)
    elapsed = time.perf_counter() - start
    return result, elapsed


def load_model(args):
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model_args = dict(checkpoint["model_args"])
    if model_args.get("dynamic_exit", False):
        if args.confidence_method is not None:
            model_args["confidence_method"] = args.confidence_method
        if args.confidence_threshold is not None:
            model_args["confidence_threshold"] = args.confidence_threshold
        if args.entropy_threshold is not None:
            model_args["entropy_threshold"] = args.entropy_threshold

    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(args.device)
    return model, checkpoint


def build_codec(checkpoint):
    load_meta = "config" in checkpoint and "dataset" in checkpoint["config"]
    if load_meta:
        meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
        load_meta = os.path.exists(meta_path)
    if load_meta:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        return lambda s: [stoi[c] for c in s if c in stoi], lambda ids: "".join([itos[i] for i in ids])

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}), lambda ids: enc.decode(ids)


@torch.no_grad()
def prompt_loss(model, idx, ctx):
    if idx.size(1) < 2:
        return None
    was_dynamic = model.config.dynamic_exit
    model.config.dynamic_exit = False
    try:
        with ctx:
            _, loss = model(idx[:, :-1], idx[:, 1:])
    finally:
        model.config.dynamic_exit = was_dynamic
    return float(loss.item())


@torch.no_grad()
def analyze_agreement(model, idx, max_new_tokens, temperature, top_k, ctx):
    assert model.config.dynamic_exit
    stats = []
    layer_agreement = {layer: [0, 0] for layer in model.exit_layers}

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        with ctx:
            dynamic_logits, _ = model(idx_cond)
        details = model.last_exit_details
        dynamic_top1 = dynamic_logits[:, -1, :].argmax(dim=-1)

        model.config.dynamic_exit = False
        try:
            with ctx:
                full_logits, _ = model(idx_cond)
        finally:
            model.config.dynamic_exit = True
        final_top1 = full_logits[:, -1, :].argmax(dim=-1)

        logits = dynamic_logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        for b in range(idx.size(0)):
            exit_layer = int(details["exit_layer"][b, -1].item())
            agreed = bool(dynamic_top1[b].item() == final_top1[b].item())
            if exit_layer in layer_agreement:
                layer_agreement[exit_layer][1] += 1
                layer_agreement[exit_layer][0] += int(agreed)
            stats.append({
                "exit_layer": exit_layer,
                "early_exit": exit_layer < model.config.n_layer,
                "agreement": agreed,
                "false_confident_exit": exit_layer < model.config.n_layer and not agreed,
            })

        idx = torch.cat((idx, idx_next), dim=1)

    total = max(len(stats), 1)
    early = [s for s in stats if s["early_exit"]]
    return {
        "agreement_with_full_model": sum(s["agreement"] for s in stats) / total,
        "false_confident_exit_rate": sum(s["false_confident_exit"] for s in stats) / max(len(early), 1),
        "layer_agreement_rate": {
            layer: agree / count if count else 0.0
            for layer, (agree, count) in layer_agreement.items()
        },
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    model, checkpoint = load_model(args)
    encode, decode = build_codec(checkpoint)

    baseline_times = []
    dynamic_times = []
    dynamic_summaries = []
    losses = []
    agreement_summaries = []

    for prompt in args.prompts:
        start_ids = encode(prompt)
        if not start_ids:
            continue
        x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]
        loss = prompt_loss(model, x, ctx)
        if loss is not None:
            losses.append(loss)

        was_dynamic = model.config.dynamic_exit
        model.config.dynamic_exit = False
        (_, baseline_elapsed) = timed_generate(
            lambda: model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k),
            args.device,
        )
        model.config.dynamic_exit = was_dynamic
        baseline_times.append(baseline_elapsed)

        if model.config.dynamic_exit:
            ((y, _, summary), dynamic_elapsed) = timed_generate(
                lambda: model.generate_dynamic(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k),
                args.device,
            )
            dynamic_times.append(dynamic_elapsed)
            dynamic_summaries.append(summary)
            agreement_summaries.append(analyze_agreement(model, x, args.max_new_tokens, args.temperature, args.top_k, ctx))
        else:
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)

        print(f"\nPrompt: {prompt}")
        print(decode(y[0].tolist()))

    total_tokens = max(args.max_new_tokens * len(args.prompts), 1)
    baseline_time = sum(baseline_times)
    print("\n===== Dynamic Exit Evaluation =====")
    print("\nBaseline:")
    print(f"tokens/sec: {total_tokens / baseline_time:.2f}")
    print(f"ms/token: {1000 * baseline_time / total_tokens:.2f}")
    print(f"avg layers/token: {model.config.n_layer:.2f}")

    if not model.config.dynamic_exit:
        print("\nDynamic: disabled in this checkpoint")
        return

    dynamic_time = sum(dynamic_times)
    avg_layers = sum(s["avg_layers_per_token"] for s in dynamic_summaries) / len(dynamic_summaries)
    early_exit_rate = sum(s["early_exit_rate"] for s in dynamic_summaries) / len(dynamic_summaries)
    full_path_rate = sum(s["full_path_rate"] for s in dynamic_summaries) / len(dynamic_summaries)
    layer_saving = 1.0 - avg_layers / model.config.n_layer

    print("\nDynamic:")
    print(f"tokens/sec: {total_tokens / dynamic_time:.2f}")
    print(f"ms/token: {1000 * dynamic_time / total_tokens:.2f}")
    print(f"avg layers/token: {avg_layers:.2f}")
    print(f"layer saving: {100 * layer_saving:.2f}%")
    print(f"early exit rate: {100 * early_exit_rate:.2f}%")
    print(f"full path rate: {100 * full_path_rate:.2f}%")

    print("\nExit distribution:")
    for layer in model.exit_layers + [model.config.n_layer]:
        rate = sum(s["exit_distribution"].get(layer, 0.0) for s in dynamic_summaries) / len(dynamic_summaries)
        print(f"Layer {layer}: {100 * rate:.2f}%")

    print("\nQuality:")
    if losses:
        mean_loss = sum(losses) / len(losses)
        print(f"loss/perplexity: {mean_loss:.4f} / {math.exp(mean_loss):.2f}")
    agreement = sum(s["agreement_with_full_model"] for s in agreement_summaries) / len(agreement_summaries)
    false_confident = sum(s["false_confident_exit_rate"] for s in agreement_summaries) / len(agreement_summaries)
    print(f"agreement with full model: {100 * agreement:.2f}%")
    print(f"false confident exits: {100 * false_confident:.2f}%")

    print("\nLayer agreement:")
    for layer in model.exit_layers:
        rate = sum(s["layer_agreement_rate"].get(layer, 0.0) for s in agreement_summaries) / len(agreement_summaries)
        print(f"Layer {layer} vs Final: {100 * rate:.2f}%")


if __name__ == "__main__":
    main()
