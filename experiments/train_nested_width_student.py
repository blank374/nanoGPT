"""
Train a nested-width student from a budget_weight=0 free-channel teacher.

The student uses prefix widths at 64-channel granularity, so eval can use true
sliced Linear. The teacher supplies two signals:
- logits distillation
- per-layer active-count targets for the student's dynamic-width router

This keeps free-channel as the discovery branch and nested-width as the
hardware-friendly acceleration branch.
"""

import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import AdaptiveWidthMLP, FreeChannelMLP, GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_out_dir", default="out-free-channel-budget-0p0")
    parser.add_argument("--teacher_checkpoint", default="ckpt.pt")
    parser.add_argument("--out_dir", default="out-nested-width-student-free-teacher-1k")
    parser.add_argument("--dataset", default="shakespeare_char")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--width_step", type=int, default=64)
    parser.add_argument("--teacher_kl_weight", type=float, default=0.5)
    parser.add_argument("--width_distill_weight", type=float, default=0.2)
    parser.add_argument("--sandwich_weight", type=float, default=0.2)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def load_teacher(args):
    ckpt = torch.load(os.path.join(args.teacher_out_dir, args.teacher_checkpoint), map_location=args.device)
    model_args = dict(ckpt["model_args"])
    model_args.setdefault("free_channel_eval_impl", "dense_mask")
    model_args.setdefault("free_channel_prefix_granularity", 64)
    teacher = GPT(GPTConfig(**model_args))
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for key, _ in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    teacher.load_state_dict(state_dict, strict=False)
    teacher.eval()
    teacher.to(args.device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    assert teacher.config.free_channel_mlp
    return teacher, ckpt


def make_student(args, vocab_size):
    max_hidden = 4 * args.n_embd
    ratios = [width / args.n_embd for width in range(args.width_step, max_hidden + 1, args.width_step)]
    config = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        bias=False,
        vocab_size=vocab_size,
        dropout=args.dropout,
        dynamic_width=True,
        dynamic_width_ratios=ratios,
        dynamic_width_cost_weight=0.0,
        dynamic_width_hard_eval=True,
        dynamic_width_temperature=1.0,
        dynamic_width_temperature_final=1.0,
        dynamic_width_temperature_anneal_iters=0,
        dynamic_width_routing="ste",
        dynamic_width_hard_loss_weight=0.0,
        dynamic_width_entropy_weight=0.0,
        dynamic_width_sliced_eval=True,
    )
    student = GPT(config).to(args.device)
    return student


def get_data(dataset):
    data_dir = os.path.join("data", dataset)
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return train_data, val_data, meta["vocab_size"]


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(np.asarray(data[i:i + block_size], dtype=np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(np.asarray(data[i + 1:i + 1 + block_size], dtype=np.int64)) for i in ix])
    return x.to(device), y.to(device)


def dynamic_width_modules(model):
    return [block.mlp for block in model.transformer.h if isinstance(block.mlp, AdaptiveWidthMLP)]


def free_channel_modules(model):
    return [block.mlp for block in model.transformer.h if isinstance(block.mlp, FreeChannelMLP)]


@torch.no_grad()
def teacher_forward_and_width_targets(teacher, x, student_modules):
    logits, _ = teacher(x)
    targets = []
    for t_mlp, s_mlp in zip(free_channel_modules(teacher), student_modules):
        active = t_mlp.last_active_channels.detach()
        width_values = s_mlp.width_values.to(active.device)
        dist = (width_values.view(1, 1, -1) - active.unsqueeze(-1)).abs()
        targets.append(dist.argmin(dim=-1).long())
    return logits.detach(), targets


def router_distill_loss(student_modules, width_targets):
    losses = []
    for mlp, target in zip(student_modules, width_targets):
        probs = mlp.last_width_probs
        logits = torch.log(probs.clamp_min(1e-9))
        losses.append(F.nll_loss(logits.reshape(-1, logits.size(-1)), target.reshape(-1)))
    return torch.stack(losses).mean()


def kl_distill_loss(student_logits, teacher_logits, temperature):
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    token_kl = F.kl_div(s, t, reduction="none").sum(dim=-1)
    return token_kl.mean() * (temperature ** 2)


def sandwich_loss(student, x, y, full_logits):
    modules = dynamic_width_modules(student)
    old = [m.force_width_index for m in modules]
    try:
        losses = []
        for width_idx in (0, len(modules[0].width_choices) - 1):
            for m in modules:
                m.force_width_index = width_idx
            logits, ce = student(x, y)
            losses.append(ce)
            if width_idx == 0:
                losses.append(kl_distill_loss(logits, full_logits.detach(), 1.0))
        return torch.stack(losses).mean()
    finally:
        for m, force_idx in zip(modules, old):
            m.force_width_index = force_idx


def get_lr(args, it):
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
    if it > args.max_iters:
        return args.min_lr
    decay_ratio = (it - args.warmup_iters) / max(args.max_iters - args.warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


@torch.no_grad()
def estimate_loss(student, data, args):
    student.eval()
    losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch(data, args.batch_size, args.block_size, args.device)
        _, loss = student(x, y)
        losses.append(loss.item())
    student.train()
    return float(np.mean(losses))


def save_checkpoint(student, optimizer, args, iter_num, best_val_loss):
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = {
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": student.config.__dict__,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": vars(args),
    }
    torch.save(ckpt, os.path.join(args.out_dir, "ckpt.pt"))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    train_data, val_data, vocab_size = get_data(args.dataset)
    teacher, teacher_ckpt = load_teacher(args)
    student = make_student(args, vocab_size)
    if args.compile:
        student = torch.compile(student)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        betas=(0.9, args.beta2),
        weight_decay=args.weight_decay,
    )
    raw_student = student._orig_mod if hasattr(student, "_orig_mod") else student
    best_val_loss = float("inf")
    t0 = time.time()

    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0:
            val_loss = estimate_loss(student, val_data, args)
            stats = raw_student.last_dynamic_width_stats or {}
            mean_width = stats.get("mean_effective_width", 0.0)
            dist = ", ".join(
                f"w{w}={stats.get('width_fractions', {}).get(str(w), 0.0):.3f}"
                for w in stats.get("width_choices", [])
            )
            print(f"step {it}: val loss {val_loss:.4f}, ppl {math.exp(val_loss):.2f}, mean_width {mean_width:.2f}, {dist}")
            if it > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(raw_student, optimizer, args, it, best_val_loss)

        if it == args.max_iters:
            break

        lr = get_lr(args, it)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.block_size, args.device)
        student.train()
        with torch.no_grad():
            teacher_logits, width_targets = teacher_forward_and_width_targets(
                teacher, x, dynamic_width_modules(raw_student)
            )
        student_logits, ce = student(x, y)
        student_modules = dynamic_width_modules(raw_student)
        width_loss = router_distill_loss(student_modules, width_targets)
        kd_loss = kl_distill_loss(student_logits, teacher_logits, args.distill_temperature)
        sw_loss = sandwich_loss(student, x, y, student_logits)
        loss = (
            ce
            + args.teacher_kl_weight * kd_loss
            + args.width_distill_weight * width_loss
            + args.sandwich_weight * sw_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()

        if it % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            stats = raw_student.last_dynamic_width_stats or {}
            print(
                f"iter {it}: loss {loss.item():.4f}, ce {ce.item():.4f}, "
                f"kd {kd_loss.item():.4f}, width {width_loss.item():.4f}, "
                f"sandwich {sw_loss.item():.4f}, mean_width {stats.get('mean_effective_width', 0.0):.2f}, "
                f"time {dt * 1000:.1f}ms"
            )

    save_checkpoint(raw_student, optimizer, args, args.max_iters, best_val_loss)
    print(f"saved checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
