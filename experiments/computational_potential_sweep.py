"""Evaluate one checkpoint across a shared computational-potential price curve."""

import argparse
import csv
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cost-profile", required=True)
    parser.add_argument("--lambdas", default="0,0.1,0.25,0.5,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = args.output or os.path.join(
        os.path.dirname(args.checkpoint), "computational_potential_sweep.csv"
    )
    rows = []
    for value in [float(item) for item in args.lambdas.split(",")]:
        result = output + f".{str(value).replace('.', 'p')}.csv"
        subprocess.run([
            sys.executable, os.path.join(ROOT, "experiments", "dynamic_resource_eval.py"),
            f"--checkpoint={args.checkpoint}", f"--cost-profile={args.cost_profile}",
            f"--potential-lambda={value}", f"--device={args.device}",
            f"--batches={args.batches}", f"--batch-size={args.batch_size}",
            f"--block-size={args.block_size}", f"--output-csv={result}",
        ], check=True, cwd=ROOT)
        with open(result, newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        row["potential_lambda"] = value
        rows.append(row)
    fields = ["potential_lambda"] + [key for key in rows[0] if key != "potential_lambda"]
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
