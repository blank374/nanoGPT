"""Aggregate frozen Full-Free Natural vs independently trained matched Static runs."""

import argparse
import csv
import json
import math
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 2027, 3407])
    parser.add_argument("--output", default="full_free_multiseed.csv")
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        directory = os.path.join(ROOT, f"out-full-free-natural-seed{seed}")
        with open(os.path.join(directory, "full_free_summary.json"), encoding="utf-8") as handle:
            summary = json.load(handle)
        ppls = {}
        with open(os.path.join(directory, "full_free_interventions.csv"), encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ppls[(row["model"], row["intervention"])] = float(row["ppl"])
        dynamic = ppls[("dynamic", "learned")]
        static = ppls[("static_trained", "native")]
        rows.append({
            "seed": seed,
            "dynamic_ppl": dynamic,
            "matched_static_ppl": static,
            "dynamic_minus_static": dynamic - static,
            "dynamic_wins": int(dynamic < static),
            "avg_active_cells": summary["active_cells"],
            "avg_active_edges": summary["active_edges"],
            "avg_depth": summary["longest_path"],
            "graph_entropy": summary["graph_entropy"],
            "same_token_shuffle_ppl": ppls[("dynamic", "same_token_shuffle")],
            "same_token_position_shuffle_ppl": ppls[
                ("dynamic", "same_token_position_shuffle")
            ],
        })
    numeric = [key for key in rows[0] if key not in ("seed", "dynamic_wins")]
    mean = {"seed": "mean", "dynamic_wins": sum(row["dynamic_wins"] for row in rows)}
    std = {"seed": "sample_std", "dynamic_wins": ""}
    for key in numeric:
        values = [row[key] for row in rows]
        mean[key] = statistics.mean(values)
        std[key] = statistics.stdev(values) if len(values) > 1 else math.nan
    output = os.path.join(ROOT, args.output)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows + [mean, std])
    print(f"wrote {output}")
    print(f"Dynamic beats matched Static in {mean['dynamic_wins']}/{len(rows)} seeds")


if __name__ == "__main__":
    main()

