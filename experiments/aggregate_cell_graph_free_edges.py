"""Aggregate paired free-edge results across seeds."""

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
    parser.add_argument("--output", default="cell_graph_free_edges_multiseed.csv")
    args = parser.parse_args()

    seed_rows = []
    for seed in args.seeds:
        directory = os.path.join(ROOT, f"out-cell-graph-edges-learned-seed{seed}")
        with open(os.path.join(directory, "cell_graph_edge_summary.json"), encoding="utf-8") as handle:
            summary = json.load(handle)
        with open(os.path.join(directory, "cell_graph_edge_interventions.csv"),
                  newline="", encoding="utf-8") as handle:
            interventions = {
                (row["model"], row["intervention"]): float(row["ppl"])
                for row in csv.DictReader(handle)
            }
        seed_rows.append({
            "seed": seed,
            "learned_ppl": summary["learned_ppl"],
            "fixed_common_ppl": summary["fixed_most_common_ppl"],
            "shuffle_ppl": summary["shuffled_ppl"],
            "fixed_trained_ppl": interventions.get(("fixed_trained", "native"), float("nan")),
            "budget_trained_ppl": interventions.get(("edge_budget_trained", "native"), float("nan")),
            "learned_minus_fixed_common": summary["learned_ppl"] - summary["fixed_most_common_ppl"],
            "learned_minus_shuffle": summary["learned_ppl"] - summary["shuffled_ppl"],
            "graph_entropy": summary["entropy"],
            "top1_graph_coverage": summary["top1_coverage"],
            "top4_graph_coverage": summary["top4_coverage"],
            "top8_graph_coverage": summary["top8_coverage"],
            "mask_token_mutual_information": summary["mask_token_mutual_information"],
        })

    fields = [key for key in seed_rows[0] if key != "seed"]
    aggregate = {"seed": "mean"}
    spread = {"seed": "sample_std"}
    stderr = {"seed": "standard_error"}
    for field in fields:
        values = [row[field] for row in seed_rows if not math.isnan(row[field])]
        aggregate[field] = statistics.mean(values)
        spread[field] = statistics.stdev(values) if len(values) > 1 else 0.0
        stderr[field] = spread[field] / math.sqrt(len(values))
    seed_rows.extend([aggregate, spread, stderr])

    path = os.path.join(ROOT, args.output)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", *fields])
        writer.writeheader()
        writer.writerows(seed_rows)
    print(f"wrote {path}")
    print(
        f"learned beats fixed-common in "
        f"{sum(row['learned_minus_fixed_common'] < 0 for row in seed_rows[:-3])}/{len(args.seeds)} seeds"
    )
    print(
        f"learned beats shuffle in "
        f"{sum(row['learned_minus_shuffle'] < 0 for row in seed_rows[:-3])}/{len(args.seeds)} seeds"
    )


if __name__ == "__main__":
    main()
