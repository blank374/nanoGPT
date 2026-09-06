"""Distill the quality v2 model into the learned three-Attention Fast Path.

The student starts from the B=32 sparse pilot, whose graph condensed to
Attention steps [0,4,7].  The B=72 quality model supplies full-token logits.
The student's existing joint dual budget remains active, so distillation may
spend unused Cell budget without reopening the five pruned Attention steps.
"""

import argparse
import contextlib
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

import model as frozen_model
from experiments.full_free_attention_v2 import install_into_frozen_model_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="out-full-free-attention-v2-budget8-seed1337/ckpt.pt")
    parser.add_argument("--teacher", default="out-full-free-attention-v2-budget72-quality-seed1337/ckpt.pt")
    parser.add_argument("--out_dir", default="out-full-free-attention-v2-fastpath-distilled-seed1337")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--distill_weight", type=float, default=0.6)
    parser.add_argument("--plan", default="", help="comma-separated Attention steps")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def amp_context(device, dtype):
    if device.type != "cuda" or dtype == "float32":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=getattr(torch, dtype))


def load_checkpoint(relative_path, device):
    path = os.path.join(ROOT, relative_path)
    checkpoint = torch.load(path, map_location=device)
    config = frozen_model.GPTConfig(**checkpoint["model_args"])
    model = frozen_model.GPT(config)
    state = {
        (key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(state, strict=False)
    return model.to(device), checkpoint, path


def get_batch(data, batch_size, block_size, device, generator):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = np.stack([
        np.asarray(data[int(i):int(i) + block_size], dtype=np.int64) for i in starts
    ])
    y = np.stack([
        np.asarray(data[int(i) + 1:int(i) + block_size + 1], dtype=np.int64)
        for i in starts
    ])
    return torch.tensor(x, device=device), torch.tensor(y, device=device)


@torch.no_grad()
def evaluate(model, data, args, generator):
    model.eval()
    model_device = next(model.parameters()).device
    losses = []
    active_cells = []
    active_attention = []
    for _ in range(args.eval_batches):
        x, y = get_batch(
            data, args.batch_size, model.config.block_size, model_device,
            generator,
        )
        with amp_context(model_device, args.dtype):
            logits, _ = model(x, y)
        losses.append(F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)
        ).item())
        active_cells.append(model.cell_graph.last_node_mask.float().sum(-1).mean().item())
        active_attention.append(
            model.cell_graph.last_attention_mask.float().sum(-1).mean().item()
        )
    model.train()
    nll = float(np.mean(losses))
    return {
        "nll": nll,
        "ppl": math.exp(nll),
        "active_cells": float(np.mean(active_cells)),
        "active_attention": float(np.mean(active_attention)),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    install_into_frozen_model_module(frozen_model)
    student, student_checkpoint, student_path = load_checkpoint(args.student, device)
    teacher, _, teacher_path = load_checkpoint(args.teacher, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    if args.plan:
        fast_plan = torch.zeros(
            student.cell_graph.num_steps, dtype=torch.bool, device=device
        )
        fast_plan[[int(value) for value in args.plan.split(",")]] = True
    else:
        fast_plan = student.cell_graph.attention_anchor_mask.clone()
    student.cell_graph.attention_plan_override = fast_plan
    # Start at the checkpoint's learned sparsemax temperature.
    student.cell_graph.temperature = float(student.config.cell_graph_temperature_final)
    teacher.cell_graph.temperature = float(teacher.config.cell_graph_temperature_final)
    student.config.cell_graph_dual_value = 0.0
    student.cell_graph.v2_dual_updates.zero_()

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=0.1,
        fused=(device.type == "cuda"),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.dtype == "float16"))
    train_data = np.memmap(
        os.path.join(ROOT, "data", "shakespeare_char", "train.bin"),
        dtype=np.uint16, mode="r",
    )
    val_data = np.memmap(
        os.path.join(ROOT, "data", "shakespeare_char", "val.bin"),
        dtype=np.uint16, mode="r",
    )
    train_generator = torch.Generator().manual_seed(args.seed + 1)
    val_generator = torch.Generator().manual_seed(args.seed + 2)
    history = []
    os.makedirs(os.path.join(ROOT, args.out_dir), exist_ok=True)

    for iteration in range(args.iterations + 1):
        if iteration % args.eval_interval == 0:
            metrics = evaluate(student, val_data, args, val_generator)
            metrics["iteration"] = iteration
            history.append(metrics)
            print(json.dumps(metrics))
            if iteration == args.iterations:
                break

        student.train()
        x, y = get_batch(
            train_data, args.batch_size, student.config.block_size, device, train_generator
        )
        with torch.no_grad(), amp_context(device, args.dtype):
            teacher_logits = teacher._forward_cell_graph_logits(x, all_logits=True)
        with amp_context(device, args.dtype):
            student_logits, student_loss = student(x, y)
            temp = args.temperature
            distill = F.kl_div(
                F.log_softmax(student_logits.float() / temp, dim=-1),
                F.softmax(teacher_logits.float() / temp, dim=-1),
                reduction="batchmean",
            ) * (temp * temp) / x.size(1)
            loss = (1.0 - args.distill_weight) * student_loss + args.distill_weight * distill
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        student.update_cell_graph_dual()

    model_args = dict(student_checkpoint["model_args"])
    model_args["cell_graph_temperature"] = student.cell_graph.temperature
    model_args["cell_graph_dual_value"] = student.config.cell_graph_dual_value
    output = {
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": args.iterations,
        "best_val_loss": min(row["nll"] for row in history),
        "config": vars(args),
    }
    checkpoint_path = os.path.join(ROOT, args.out_dir, "ckpt.pt")
    torch.save(output, checkpoint_path)
    summary = {
        "student_source": student_path,
        "teacher_source": teacher_path,
        "fast_attention_plan": fast_plan.nonzero().flatten().tolist(),
        "history": history,
        "checkpoint": checkpoint_path,
    }
    with open(os.path.join(ROOT, args.out_dir, "distill_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
