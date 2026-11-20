"""Compare the baseline and the experiment on the VALIDATION set and select the final run.

Usage:
    python scripts/compare_runs.py --runs baseline_640 exp_imgsz416

Selection rule: highest validation mAP@0.50 (ties -> higher mAP@0.50:0.95).
The test set is NOT used here, so it cannot influence the choice of settings.

Outputs:
    results/comparison.md          markdown table for the README
    results/selected_run.json      which run was chosen and why
    results/learning_curves.png    both runs' losses and validation mAP per epoch
"""
from __future__ import annotations

import argparse
import csv

from common import RESULTS_DIR, load_json, save_json

TRACKED = ["imgsz", "epochs", "batch", "lr0", "optimizer", "seed"]


def read_results_csv(path):
    with open(path) as f:
        return [{k.strip(): float(v) for k, v in r.items() if v.strip()} for r in csv.DictReader(f)]


def changed_settings(infos):
    """List hyper-parameters that differ between runs (should be exactly one)."""
    keys = set().union(*(i["hyperparameters"].keys() for i in infos))
    return sorted(k for k in keys if len({str(i["hyperparameters"].get(k)) for i in infos}) > 1)


def plot_curves(runs, path):
    import matplotlib.pyplot as plt

    panels = [("train/box_loss", "Train box loss"), ("val/box_loss", "Val box loss"),
              ("metrics/mAP50(B)", "Val mAP@0.50"), ("metrics/mAP50-95(B)", "Val mAP@0.50:0.95")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for run in runs:
        rows = read_results_csv(RESULTS_DIR / "runs" / run / "results.csv")
        ep = [r["epoch"] for r in rows]
        for ax, (key, title) in zip(axes, panels):
            ax.plot(ep, [r[key] for r in rows], label=run)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["baseline_640", "exp_imgsz416"])
    args = ap.parse_args()

    infos = [load_json(RESULTS_DIR / "runs" / r / "run_info.json") for r in args.runs]
    evals = [load_json(RESULTS_DIR / "eval" / f"{r}_val.json") for r in args.runs]
    changed = changed_settings(infos)

    lines = ["| Run | Changed setting | Val precision | Val recall | Val mAP@0.50 | "
             "Val mAP@0.50:0.95 | Best epoch | Training time (min) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for info, ev in zip(infos, evals):
        setting = ", ".join(f"{k}={info['hyperparameters'].get(k)}" for k in changed) or "-"
        lines.append(f"| {info['run']} | {setting} | {ev['precision']:.3f} | {ev['recall']:.3f} | "
                     f"{ev['mAP50']:.3f} | {ev['mAP50_95']:.3f} | {info['best_epoch']['epoch']} | "
                     f"{info['training_time_min']:.1f} |")
    table = "\n".join(lines)

    best = max(zip(infos, evals), key=lambda t: (t[1]["mAP50"], t[1]["mAP50_95"]))
    selected = {"selected_run": best[0]["run"],
                "rule": "highest validation mAP@0.50, ties broken by mAP@0.50:0.95",
                "val_mAP50": best[1]["mAP50"],
                "changed_settings_between_runs": changed}
    if len(changed) != 1:
        print(f"WARNING: runs differ in {changed}; a controlled experiment changes exactly one setting")

    (RESULTS_DIR / "comparison.md").write_text(table + "\n")
    save_json(selected, RESULTS_DIR / "selected_run.json")
    plot_curves(args.runs, RESULTS_DIR / "learning_curves.png")
    print(table)
    print(f"\nSelected for test evaluation: {selected['selected_run']} ({selected['rule']})")


if __name__ == "__main__":
    main()
