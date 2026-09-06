"""Export and validate a training-state-free Full-Free inference artifact."""

import argparse
import hashlib
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import GPT, GPTConfig
from full_free_cell_graph_analysis import batches, evaluate


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_state(state):
    prefix = "_orig_mod."
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state.items()
    }


def load_model(state, device):
    model = GPT(GPTConfig(**state["model_args"]))
    model.load_state_dict(normalized_state(state["model"]))
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="shakespeare_char")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_batches", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=128)
    args = parser.parse_args()

    input_path = os.path.join(ROOT, args.input)
    output_path = os.path.join(ROOT, args.output)
    training = torch.load(input_path, map_location="cpu")
    weights = normalized_state(training["model"])
    artifact = {
        "format": "full-free-inference-v1-fp32-lossless",
        "model_args": training["model_args"],
        "model": weights,
        "source_sha256": sha256(input_path),
    }
    torch.save(artifact, output_path)

    exported = torch.load(output_path, map_location="cpu")
    source_weights = normalized_state(training["model"])
    export_weights = normalized_state(exported["model"])
    keys_equal = source_weights.keys() == export_weights.keys()
    tensors_equal = keys_equal and all(
        torch.equal(source_weights[key], export_weights[key]) for key in source_weights
    )

    source_model = load_model(training, args.device)
    export_model = load_model(exported, args.device)
    data = batches(args.dataset, args.num_batches, args.batch_size,
                   args.block_size, args.device, args.seed)
    source_nll, source_tokens, source_records = evaluate(source_model, data, collect=True)
    export_nll, export_tokens, export_records = evaluate(export_model, data, collect=True)
    logits_loss_equal = torch.equal(source_tokens, export_tokens)
    node_masks_equal = all(torch.equal(a["node_mask"], b["node_mask"])
                           for a, b in zip(source_records, export_records))
    edge_masks_equal = all(torch.equal(a["edge_mask"], b["edge_mask"])
                           for a, b in zip(source_records, export_records))
    accepted = bool(tensors_equal and logits_loss_equal and node_masks_equal and edge_masks_equal)
    result = {
        "accepted_lossless": accepted,
        "input": input_path,
        "output": output_path,
        "input_bytes": os.path.getsize(input_path),
        "output_bytes": os.path.getsize(output_path),
        "size_reduction_fraction": 1.0 - os.path.getsize(output_path) / os.path.getsize(input_path),
        "source_nll": source_nll,
        "export_nll": export_nll,
        "per_token_nll_bitwise_equal": logits_loss_equal,
        "weights_bitwise_equal": tensors_equal,
        "node_masks_equal": node_masks_equal,
        "edge_masks_equal": edge_masks_equal,
        "input_sha256": sha256(input_path),
        "output_sha256": sha256(output_path),
    }
    manifest_path = output_path + ".json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    if not accepted:
        raise RuntimeError("lossless inference artifact validation failed")


if __name__ == "__main__":
    main()
