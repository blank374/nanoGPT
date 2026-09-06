"""Run Natural, optional Budget sweep, analysis, and matched static baselines."""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--seeds", nargs="+", type=int, default=[1337, 2027, 3407])
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--pilot", action="store_true", help="run one 1000-step Natural seed")
    p.add_argument("--budgets", nargs="*", type=float, default=[4, 6, 8, 12, 16, 24, 32])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def run(command, dry_run=False):
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def train(a, seed, config, output, extra=None):
    checkpoint = os.path.join(ROOT, output, "ckpt.pt")
    if a.skip_existing and os.path.exists(checkpoint):
        print(f"skipping existing {checkpoint}")
        return
    command = [a.python, "train.py", config,
               f"--seed={seed}", f"--max_iters={a.max_iters}",
               f"--lr_decay_iters={a.max_iters}", f"--device={a.device}",
               f"--dtype={a.dtype}", f"--out_dir={output}"]
    if extra:
        command.extend(extra)
    run(command, a.dry_run)


if __name__ == "__main__":
    a = parse_args()
    if a.pilot:
        a.seeds = a.seeds[:1]
        a.max_iters = 1000
        a.budgets = []
    for seed in a.seeds:
        natural = f"out-full-free-natural-seed{seed}"
        train(a, seed, "config/train_shakespeare_char_full_free_natural.py", natural)
        run([a.python, "experiments/full_free_cell_graph_analysis.py",
             f"--out_dir={natural}", f"--device={a.device}", f"--seed={seed}"], a.dry_run)
        if a.pilot:
            continue

        static = f"out-full-free-static-seed{seed}"
        graph_path = f"{natural}/matched_static_graph.npz"
        train(a, seed, "config/train_shakespeare_char_full_free_static.py", static,
              [f"--cell_graph_static_graph_path={graph_path}"])
        run([a.python, "experiments/full_free_cell_graph_analysis.py",
             f"--out_dir={natural}", f"--static_out_dir={static}",
             f"--reference_out_dir={static}", f"--device={a.device}",
             f"--seed={seed}"], a.dry_run)

        for budget in a.budgets:
            label = f"{budget:g}".replace(".", "p")
            output = f"out-full-free-budget{label}-seed{seed}"
            train(a, seed, "config/train_shakespeare_char_full_free_budget.py", output,
                  [f"--cell_graph_active_cell_budget={budget}"])
            run([a.python, "experiments/full_free_cell_graph_analysis.py",
                 f"--out_dir={output}",
                 f"--reference_out_dir={static}", f"--device={a.device}",
                 f"--seed={seed}"], a.dry_run)
            budget_static = f"out-full-free-static-budget{label}-seed{seed}"
            budget_graph = f"{output}/matched_static_graph.npz"
            train(a, seed, "config/train_shakespeare_char_full_free_static.py", budget_static,
                  [f"--cell_graph_static_graph_path={budget_graph}"])
            run([a.python, "experiments/full_free_cell_graph_analysis.py",
                 f"--out_dir={output}", f"--static_out_dir={budget_static}",
                 f"--reference_out_dir={static}", f"--device={a.device}",
                 f"--seed={seed}"], a.dry_run)

    if a.budgets:
        run([a.python, "experiments/aggregate_full_free_pareto.py",
             "--seeds", *[str(value) for value in a.seeds],
             "--budgets", *[str(value) for value in a.budgets]], a.dry_run)
