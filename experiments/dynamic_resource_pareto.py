"""Train the required compute-penalty sweep and merge its quality/compute results."""

import argparse
import csv
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--penalties", default="0,0.01,0.05,0.10")
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--output-root", default="out-dynamic-resource-pareto")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    rows = []
    for penalty in [float(value) for value in args.penalties.split(",")]:
        label = str(penalty).replace(".", "p")
        out_dir = os.path.join(args.output_root, f"lambda-{label}")
        checkpoint = os.path.join(out_dir, "ckpt.pt")
        result_csv = os.path.join(out_dir, "dynamic_resource_evaluation.csv")
        if not args.skip_training:
            subprocess.run([
                sys.executable, os.path.join(ROOT, "train.py"),
                os.path.join(ROOT, "config", "train_shakespeare_char_dynamic_resource.py"),
                f"--dynamic_resource_compute_penalty_max={penalty}",
                f"--max_iters={args.max_iters}", f"--lr_decay_iters={args.max_iters}",
                f"--device={args.device}", f"--dtype={args.dtype}",
                f"--out_dir={out_dir}",
            ], check=True, cwd=ROOT)
        subprocess.run([
            sys.executable, os.path.join(ROOT, "experiments", "dynamic_resource_eval.py"),
            f"--checkpoint={checkpoint}", f"--device={args.device}",
            f"--batches={args.eval_batches}", f"--output-csv={result_csv}",
        ], check=True, cwd=ROOT)
        with open(result_csv, newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        row["lambda_compute"] = penalty
        rows.append(row)

    os.makedirs(args.output_root, exist_ok=True)
    output = os.path.join(args.output_root, "pareto_summary.csv")
    fields = ["lambda_compute"] + [key for key in rows[0] if key != "lambda_compute"]
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

