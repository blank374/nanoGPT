"""Offline analysis and interventions for a trained Free-Fan-In Query model.

Outputs learned/fixed/shuffled PPL, source usage, fan-in distributions, mask
diversity, a route archive, an optional heatmap, and a coordinate-wise oracle.
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import GPT, GPTConfig

SOURCE_NAMES = ["current", "previous", "earlier", "anchor"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-free-q-binary-identity")
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--baseline_out_dir", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--num_batches", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--oracle_requests", type=int, default=4)
    parser.add_argument("--output_prefix", default="free_q")
    return parser.parse_args()


def load_model(path, device, require_free_q=True):
    checkpoint = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    prefix = "_orig_mod."
    state = {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(state)
    if require_free_q and not model.config.free_q:
        raise ValueError("checkpoint is not a Free-Q model")
    return model.to(device).eval(), checkpoint


def load_data(dataset):
    return np.memmap(os.path.join(ROOT, "data", dataset, "val.bin"), dtype=np.uint16, mode="r")


def make_batches(data, count, batch_size, block_size, device, seed):
    generator = torch.Generator().manual_seed(seed)
    result = []
    for _ in range(count):
        starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
        x = np.stack([np.asarray(data[int(i):int(i) + block_size], dtype=np.int64) for i in starts])
        y = np.stack([np.asarray(data[int(i) + 1:int(i) + block_size + 1], dtype=np.int64) for i in starts])
        result.append((torch.tensor(x, device=device), torch.tensor(y, device=device)))
    return result


def route_snapshot(model):
    return [{key: (value.detach().cpu().clone() if torch.is_tensor(value) else value)
             for key, value in record.items()}
            for record in model.free_q_route_records()]


@torch.no_grad()
def collect_learned(model, batches):
    total_nll = 0.0
    total_tokens = 0
    routes = []
    for x, y in batches:
        model.set_free_q_overrides(None)
        logits, _ = model(x, y)
        total_nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        total_tokens += y.numel()
        routes.append(route_snapshot(model))
    return total_nll / total_tokens, routes


@torch.no_grad()
def collect_nll(model, batches):
    total_nll = 0.0
    total_tokens = 0
    for x, y in batches:
        logits, _ = model(x, y)
        total_nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        total_tokens += y.numel()
    return total_nll / total_tokens


def most_common_masks(routes, n_layer, n_head):
    fixed = []
    bits = torch.tensor([1, 2, 4, 8])
    for layer in range(n_layer):
        masks = torch.cat([batch[layer]["mask"] for batch in routes], dim=0)
        heads = []
        for head in range(n_head):
            ids = (masks[:, :, head].long() * bits).sum(dim=-1).reshape(-1)
            unique_ids, counts = ids.unique(return_counts=True)
            mode = int(unique_ids[counts.argmax()].item())
            heads.append(torch.tensor([(mode >> source) & 1 for source in range(4)], dtype=torch.bool))
        fixed.append(torch.stack(heads))
    return fixed


def named_fixed_masks(routes, name, n_head):
    output = []
    requested = {
        "current": [0],
        "current_previous": [0, 1],
        "current_earlier": [0, 2],
        "all_available": [0, 1, 2, 3],
    }[name]
    for layer_record in routes[0]:
        available = layer_record["available"].bool()
        mask = torch.zeros(n_head, 4, dtype=torch.bool)
        for source in requested:
            if available[source]:
                mask[:, source] = True
        mask[:, 0] = True
        output.append(mask)
    return output


@torch.no_grad()
def evaluate_override(model, batches, override_fn):
    nll = 0.0
    tokens = 0
    for batch_index, (x, y) in enumerate(batches):
        model.set_free_q_overrides(override_fn(batch_index))
        logits, _ = model(x, y)
        nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        tokens += y.numel()
    model.set_free_q_overrides(None)
    return nll / tokens


def entropy(ids):
    counts = np.unique(ids, return_counts=True)[1].astype(np.float64)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())


def conditional_entropy(ids, tokens):
    total = len(ids)
    result = 0.0
    for token in np.unique(tokens):
        selected = ids[tokens == token]
        result += len(selected) / total * entropy(selected)
    return result


def summarize_routes(routes, batches):
    n_layer = len(routes[0])
    n_head = routes[0][0]["mask"].shape[2]
    matrix_rows = []
    fanin_rows = []
    mask_rows = []
    all_masks = []
    all_probs = []
    all_tokens = []
    tokens_by_batch = [x.detach().cpu().numpy() for x, _ in batches]
    bits = np.asarray([1, 2, 4, 8], dtype=np.int64)
    for layer in range(n_layer):
        mask = np.concatenate([record[layer]["mask"].numpy() for record in routes], axis=0)
        probs = np.concatenate([record[layer]["probs"].float().numpy() for record in routes], axis=0)
        tokens = np.concatenate(tokens_by_batch, axis=0)
        available = routes[0][layer]["available"].numpy().astype(bool)
        for head in range(n_head):
            head_mask = mask[:, :, head].reshape(-1, 4)
            head_probs = probs[:, :, head].reshape(-1, 4)
            head_tokens = tokens.reshape(-1)
            ids = (head_mask.astype(np.int64) * bits).sum(axis=-1)
            counts = Counter(ids.tolist())
            ordered = sorted(counts.values(), reverse=True)
            matrix_rows.append({
                "layer": layer, "head": head,
                **{name: float(head_mask[:, i].mean()) if available[i] else ""
                   for i, name in enumerate(SOURCE_NAMES)},
                **{f"prob_{name}": float(head_probs[:, i].mean()) if available[i] else ""
                   for i, name in enumerate(SOURCE_NAMES)},
            })
            fanin = head_mask.sum(axis=-1)
            fanin_rows.append({
                "layer": layer, "head": head,
                "average_fanin": float(fanin.mean()),
                **{f"fanin_{k}_ratio": float((fanin == k).mean()) for k in range(1, 5)},
            })
            cond = conditional_entropy(ids, head_tokens)
            mask_rows.append({
                "layer": layer, "head": head,
                "unique_masks": len(counts),
                "top1_coverage": ordered[0] / len(ids),
                "top4_coverage": sum(ordered[:4]) / len(ids),
                "mask_entropy": entropy(ids),
                "conditional_mask_entropy_token": cond,
                "mask_token_mutual_information": entropy(ids) - cond,
                "most_common_mask_id": counts.most_common(1)[0][0],
            })
            all_masks.append(head_mask)
            all_probs.append(head_probs)
            all_tokens.append(head_tokens)
    masks = np.concatenate(all_masks)
    probs = np.concatenate(all_probs)
    tokens = np.concatenate(all_tokens)
    ids = (masks.astype(np.int64) * bits).sum(axis=-1)
    counts = Counter(ids.tolist())
    ordered = sorted(counts.values(), reverse=True)
    fanin = masks.sum(axis=-1)
    cond = conditional_entropy(ids, tokens)
    summary = {
        "average_fanin": float(fanin.mean()),
        **{f"fanin_{k}_ratio": float((fanin == k).mean()) for k in range(1, 5)},
        **{f"usage_{name}": float(masks[:, i].mean()) for i, name in enumerate(SOURCE_NAMES)},
        **{f"probability_{name}": float(probs[:, i].mean()) for i, name in enumerate(SOURCE_NAMES)},
        "unique_masks": len(counts),
        "top1_coverage": ordered[0] / len(ids),
        "top4_coverage": sum(ordered[:4]) / len(ids),
        "mask_entropy": entropy(ids),
        "conditional_mask_entropy_token": cond,
        "mask_token_mutual_information": entropy(ids) - cond,
    }
    layer_rows = []
    for layer in range(n_layer):
        selected = [row for row in fanin_rows if row["layer"] == layer]
        matrix_selected = [row for row in matrix_rows if row["layer"] == layer]
        layer_rows.append({
            "layer": layer,
            "average_fanin": float(np.mean([row["average_fanin"] for row in selected])),
            **{f"fanin_{k}_ratio": float(np.mean([
                row[f"fanin_{k}_ratio"] for row in selected
            ])) for k in range(1, 5)},
            **{f"usage_{name}": float(np.mean([
                float(row[name]) for row in matrix_selected if row[name] != ""
            ])) if any(row[name] != "" for row in matrix_selected) else ""
               for name in SOURCE_NAMES},
        })
    return summary, matrix_rows, fanin_rows, mask_rows, layer_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(path, matrix_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping heatmap")
        return
    values = np.asarray([[np.nan if row[name] == "" else row[name]
                          for name in SOURCE_NAMES] for row in matrix_rows])
    labels = [f"L{row['layer']}-H{row['head']}" for row in matrix_rows]
    fig, axis = plt.subplots(figsize=(7, max(4, len(labels) * 0.32)))
    image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_xticks(range(4), SOURCE_NAMES)
    axis.set_yticks(range(len(labels)), labels)
    fig.colorbar(image, ax=axis, label="hard source usage")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def valid_masks(available):
    indices = [index for index, value in enumerate(available) if value]
    for code in range(1, 1 << len(indices)):
        mask = torch.zeros(4, dtype=torch.bool)
        for bit, source in enumerate(indices):
            mask[source] = bool(code & (1 << bit))
        yield mask


@torch.no_grad()
def coordinate_oracle(model, batches, learned_routes, request_limit):
    rows = []
    remaining = request_limit
    for batch_index, (x_batch, y_batch) in enumerate(batches):
        for request in range(x_batch.size(0)):
            if remaining <= 0:
                model.set_free_q_overrides(None)
                return rows
            x, y = x_batch[request:request + 1], y_batch[request:request + 1]
            base = [record["mask"][request:request + 1].clone()
                    for record in learned_routes[batch_index]]
            model.set_free_q_overrides(None)
            learned_logits, _ = model(x, y)
            learned_nll = F.cross_entropy(
                learned_logits.reshape(-1, learned_logits.size(-1)), y.reshape(-1)
            ).item()
            model.set_free_q_overrides(base)
            frozen_logits, _ = model(x, y)
            frozen_nll = F.cross_entropy(
                frozen_logits.reshape(-1, frozen_logits.size(-1)), y.reshape(-1)
            ).item()
            for layer in range(model.config.n_layer):
                available = learned_routes[batch_index][layer]["available"].bool()
                for head in range(model.config.n_head):
                    best_nll = float("inf")
                    best_mask = None
                    for candidate in valid_masks(available):
                        overrides = [value.clone() for value in base]
                        overrides[layer][:, :, head] = candidate.view(1, 1, 4)
                        model.set_free_q_overrides(overrides)
                        logits, _ = model(x, y)
                        nll = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                        ).item()
                        if nll < best_nll:
                            best_nll, best_mask = nll, candidate.clone()
                    rows.append({
                        "request": request_limit - remaining,
                        "layer": layer,
                        "head": head,
                        "learned_nll": learned_nll,
                        "frozen_learned_mask_nll": frozen_nll,
                        "oracle_nll": best_nll,
                        "oracle_gain_vs_frozen_mask": frozen_nll - best_nll,
                        "oracle_mask_id": sum((1 << i) for i in range(4) if best_mask[i]),
                        "oracle_fanin": int(best_mask.sum()),
                    })
            remaining -= 1
    model.set_free_q_overrides(None)
    return rows


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint_path = os.path.join(args.out_dir, args.checkpoint)
    model, checkpoint = load_model(checkpoint_path, device)
    dataset = args.dataset or checkpoint.get("config", {}).get("dataset", "shakespeare_char")
    data = load_data(dataset)
    block_size = min(args.block_size, model.config.block_size)
    batches = make_batches(data, args.num_batches, args.batch_size, block_size, device, args.seed)

    baseline_nll = None
    if args.baseline_out_dir is not None:
        baseline_path = os.path.join(args.baseline_out_dir, args.checkpoint)
        baseline, _ = load_model(baseline_path, device, require_free_q=False)
        baseline_nll = collect_nll(baseline, batches)

    learned_nll, routes = collect_learned(model, batches)
    fixed = most_common_masks(routes, model.config.n_layer, model.config.n_head)
    fixed_nll = evaluate_override(model, batches, lambda _: fixed)
    if args.batch_size < 2:
        raise ValueError("shuffle test requires --batch_size >= 2")
    shuffled_nll = evaluate_override(
        model, batches,
        lambda batch: [record["mask"].roll(1, dims=0) for record in routes[batch]],
    )
    diagnostics = {}
    for name in ("current", "current_previous", "current_earlier", "all_available"):
        masks = named_fixed_masks(routes, name, model.config.n_head)
        diagnostics[name] = evaluate_override(model, batches, lambda _, masks=masks: masks)

    summary, matrix_rows, fanin_rows, mask_rows, layer_rows = summarize_routes(routes, batches)
    quality = [{
        "baseline_nll": baseline_nll if baseline_nll is not None else "",
        "baseline_ppl": math.exp(baseline_nll) if baseline_nll is not None else "",
        "learned_nll": learned_nll, "learned_ppl": math.exp(learned_nll),
        "fixed_most_common_nll": fixed_nll, "fixed_most_common_ppl": math.exp(fixed_nll),
        "shuffled_nll": shuffled_nll, "shuffled_ppl": math.exp(shuffled_nll),
        **{f"fixed_{name}_ppl": math.exp(nll) for name, nll in diagnostics.items()},
        **summary,
    }]
    prefix = os.path.join(args.out_dir, args.output_prefix)
    write_csv(prefix + "_summary.csv", quality)
    write_csv(prefix + "_source_usage_matrix.csv", matrix_rows)
    write_csv(prefix + "_fanin_by_head.csv", fanin_rows)
    write_csv(prefix + "_fanin_by_layer.csv", layer_rows)
    write_csv(prefix + "_mask_diversity.csv", mask_rows)
    save_heatmap(prefix + "_source_usage_heatmap.png", matrix_rows)

    archive = {}
    for batch, batch_routes in enumerate(routes):
        archive[f"tokens_b{batch}"] = batches[batch][0].detach().cpu().numpy()
        for layer, record in enumerate(batch_routes):
            archive[f"mask_b{batch}_l{layer}"] = record["mask"].numpy()
            archive[f"prob_b{batch}_l{layer}"] = record["probs"].float().numpy()
    np.savez_compressed(prefix + "_routes.npz", **archive)

    if args.oracle_requests > 0:
        oracle = coordinate_oracle(model, batches, routes, args.oracle_requests)
        write_csv(prefix + "_oracle_by_request_head.csv", oracle)

    print(f"learned PPL: {math.exp(learned_nll):.4f}")
    if baseline_nll is not None:
        print(f"standard-Q baseline PPL: {math.exp(baseline_nll):.4f}")
    print(f"fixed most-common PPL: {math.exp(fixed_nll):.4f}")
    print(f"shuffled PPL: {math.exp(shuffled_nll):.4f}")
    print(f"average fan-in: {summary['average_fanin']:.4f}")
    print(f"wrote {prefix}_*.csv, route archive, and optional heatmap")


if __name__ == "__main__":
    main()
