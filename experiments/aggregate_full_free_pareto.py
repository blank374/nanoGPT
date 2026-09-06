"""Aggregate Full-Free dynamic/static quality-compute Pareto points."""

import argparse
import csv
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--budgets", nargs="+", type=float, required=True)
    p.add_argument("--output", default="full_free_pareto.csv")
    a = p.parse_args()
    rows = []
    for seed in a.seeds:
        for budget in a.budgets:
            label = f"{budget:g}".replace(".", "p")
            directory = os.path.join(ROOT, f"out-full-free-budget{label}-seed{seed}")
            with open(os.path.join(directory, "full_free_summary.json"), encoding="utf-8") as handle:
                summary = json.load(handle)
            dynamic_ppl = math.nan
            static_ppl = math.nan
            with open(os.path.join(directory, "full_free_interventions.csv"), encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row["model"] == "dynamic" and row["intervention"] == "learned":
                        dynamic_ppl = float(row["ppl"])
                    if row["model"] == "static_trained":
                        static_ppl = float(row["ppl"])
            rows.append({
                "seed": seed, "budget": budget,
                "dynamic_ppl": dynamic_ppl,
                "static_ppl": static_ppl,
                "avg_active_cells": summary["active_cells"],
                "avg_active_edges": summary["active_edges"],
                "avg_depth": summary["longest_path"],
                "cell_macs": summary["active_cells"] * 2 * 128 * 64,
                "router_latency_ms": summary["router_cuda_latency_ms"],
                "model_latency_ms": summary["model_cuda_latency_ms"],
            })
    with open(os.path.join(ROOT, a.output), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
        for x_name in ("avg_active_cells", "cell_macs"):
            plt.figure(figsize=(6, 4))
            for label, y_name, marker in (("Dynamic", "dynamic_ppl", "o"),
                                          ("Matched static", "static_ppl", "s")):
                plt.scatter([row[x_name] for row in rows], [row[y_name] for row in rows],
                            label=label, marker=marker)
            plt.xlabel(x_name); plt.ylabel("Validation PPL"); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(ROOT, f"full_free_ppl_vs_{x_name}.png"), dpi=160)
            plt.close()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
