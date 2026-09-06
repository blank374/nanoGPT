"""Enumerate shared Attention plans on a trained v2 checkpoint."""

import argparse
import contextlib
import itertools
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.eval_full_free_attention_v2 import (
    amp_context, load_model, sample_batch,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out-full-free-attention-v2-budget72-quality-seed1337")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batches", type=int, default=3)
    parser.add_argument("--min_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=7)
    parser.add_argument("--json", default="full_free_attention_v2_plan_search.json")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    model, _ = load_model(os.path.join(ROOT, args.out_dir, "ckpt.pt"), device)
    graph = model.cell_graph
    data = np.memmap(
        os.path.join(ROOT, "data", "shakespeare_char", "val.bin"),
        dtype=np.uint16, mode="r",
    )
    generator = torch.Generator().manual_seed(20260906)
    batches = [sample_batch(
        data, args.batch_size, model.config.block_size, device, generator
    ) for _ in range(args.eval_batches)]

    # Production inference does not collect graph-diversity diagnostics.
    model._set_cell_graph_stats = lambda valid_mask=None: None

    def score(plan):
        override = torch.zeros(graph.num_steps, dtype=torch.bool, device=device)
        override[list(plan)] = True
        graph.attention_plan_override = override
        losses = []
        for x, y in batches:
            with amp_context(device, args.dtype):
                logits, _ = model(x, y)
            losses.append(F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)
            ).item())
        return float(np.mean(losses))

    baseline_nll = score(range(graph.num_steps))
    rows = []
    for count in range(args.min_steps, args.max_steps + 1):
        candidates = []
        for plan in itertools.combinations(range(graph.num_steps), count):
            nll = score(plan)
            candidates.append({
                "plan": list(plan),
                "steps": count,
                "nll": nll,
                "ppl": math.exp(nll),
                "nll_delta": nll - baseline_nll,
            })
        candidates.sort(key=lambda row: row["nll"])
        rows.extend(candidates[:5])
        print(json.dumps({"steps": count, "best": candidates[:5]}))
    graph.attention_plan_override = None
    result = {
        "checkpoint": os.path.join(ROOT, args.out_dir, "ckpt.pt"),
        "baseline_nll": baseline_nll,
        "baseline_ppl": math.exp(baseline_nll),
        "top5_per_depth": rows,
    }
    with open(os.path.join(ROOT, args.json), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
