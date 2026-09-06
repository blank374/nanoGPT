"""Run the three trained conditions for three seeds, then analyze interventions."""

import argparse
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {
    "fixed": "config/train_shakespeare_char_cell_graph_edges_fixed.py",
    "learned": "config/train_shakespeare_char_cell_graph_edges_learned.py",
    "budget": "config/train_shakespeare_char_cell_graph_edges_budget.py",
}


def run(command, dry_run):
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 2027, 3407])
    parser.add_argument("--max_iters", type=int, default=3000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--analysis_batches", type=int, default=32)
    return parser.parse_args()


def main_run(args):
    for seed in args.seeds:
        directories = {
            name: f"out-cell-graph-edges-{name}-seed{seed}" for name in RUNS
        }
        if not args.skip_training:
            for name, config in RUNS.items():
                checkpoint = os.path.join(ROOT, directories[name], "ckpt.pt")
                if args.skip_existing and os.path.exists(checkpoint):
                    print(f"skipping existing {checkpoint}")
                    continue
                run([
                    sys.executable, "train.py", config,
                    f"--seed={seed}", f"--max_iters={args.max_iters}",
                    f"--lr_decay_iters={args.max_iters}", f"--device={args.device}",
                    f"--dtype={args.dtype}", f"--out_dir={directories[name]}",
                ], args.dry_run)
        run([
            sys.executable, "experiments/cell_graph_free_edges_analysis.py",
            f"--learned_out_dir={directories['learned']}",
            f"--fixed_out_dir={directories['fixed']}",
            f"--budget_out_dir={directories['budget']}",
            f"--device={args.device}", f"--seed={seed}",
            f"--num_batches={args.analysis_batches}",
        ], args.dry_run)
    run([
        sys.executable, "experiments/aggregate_cell_graph_free_edges.py",
        "--seeds", *[str(seed) for seed in args.seeds],
    ], args.dry_run)


if __name__ == "__main__":
    main_run(main())
