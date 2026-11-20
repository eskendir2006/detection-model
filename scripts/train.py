"""Fine-tune a pretrained YOLO detector from a config file and record everything about the run.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/baseline.yaml --epochs 1 --fraction 0.05   # smoke test

Outputs:
    runs/<name>/                 full Ultralytics output incl. weights/best.pt (git-ignored)
    results/runs/<name>/         small evidence files that are committed:
        run_info.json            versions, hardware, hyper-parameters, training time, best epoch
        train_log.txt            console log of the run
        results.csv/.png         per-epoch losses and validation metrics (learning curves)
        args.yaml, *.png         the exact Ultralytics arguments and its plots
"""
from __future__ import annotations

import argparse
import platform
import re
import shutil
import sys
import time
from pathlib import Path

from common import RESULTS_DIR, RUNS_DIR, save_json


class Tee:
    """Write console output to the screen and to a log file at the same time."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s):
        self.stream.write(s)
        if not self.log.closed:  # loggers may still write at interpreter exit
            self.log.write(s)

    def flush(self):
        self.stream.flush()
        if not self.log.closed:
            self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def best_epoch(results_csv: Path) -> dict:
    """Return the row of results.csv with the highest validation fitness.

    In ultralytics 8.4.x, best.pt is saved at the epoch with the highest fitness, and
    fitness = validation mAP@0.50:0.95 (see ultralytics/utils/metrics.py, Metric.fitness).
    """
    import csv

    with open(results_csv) as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    return {"epoch": int(float(best["epoch"])),
            "val_precision": float(best["metrics/precision(B)"]),
            "val_recall": float(best["metrics/recall(B)"]),
            "val_mAP50": float(best["metrics/mAP50(B)"]),
            "val_mAP50_95": float(best["metrics/mAP50-95(B)"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, help="override (smoke tests only)")
    ap.add_argument("--fraction", type=float, help="train on a fraction of train set (smoke tests only)")
    args = ap.parse_args()

    # Start the log tee BEFORE importing ultralytics: its logger binds to sys.stdout on import.
    name = re.search(r"^name:\s*(\S+)", Path(args.config).read_text(), re.M).group(1)
    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.fraction:
        overrides["fraction"] = args.fraction
    if overrides:
        name += "_smoke"
    out = RESULTS_DIR / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.txt", "w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log), Tee(sys.__stderr__, log)

    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import YAML
    from common import ensure_dataset

    cfg = YAML.load(args.config)
    cfg.pop("name")
    model_name = cfg.pop("model")
    cfg.update(overrides)

    data_yaml = ensure_dataset()
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "CPU"
    print(f"Run '{name}' | model {model_name} | device {device}\nConfig: {cfg}")

    model = YOLO(model_name)
    t0 = time.time()
    model.train(data=str(data_yaml), project=str(RUNS_DIR), name=name, exist_ok=True, **cfg)
    train_time = time.time() - t0

    save_dir = Path(model.trainer.save_dir)
    info = {
        "run": name,
        "config_file": args.config,
        "smoke_test_overrides": overrides or None,
        "framework": f"ultralytics {ultralytics.__version__}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "device": device,
        "model": model_name,
        "initial_weights": f"{model_name} (COCO-pretrained, downloaded by Ultralytics)",
        "hyperparameters": cfg,
        "training_time_sec": round(train_time, 1),
        "training_time_min": round(train_time / 60, 2),
        "best_checkpoint": str(save_dir / "weights" / "best.pt"),
        "best_checkpoint_rule": "epoch with highest validation mAP50-95 (Ultralytics fitness)",
        "best_epoch": best_epoch(save_dir / "results.csv"),
        "save_dir": str(save_dir),
    }
    for f in list(save_dir.glob("*.csv")) + list(save_dir.glob("*.yaml")) + list(save_dir.glob("*.png")):
        shutil.copy2(f, out / f.name)
    save_json(info, out / "run_info.json")
    print(f"\nTraining time: {info['training_time_min']} min | best epoch: {info['best_epoch']}")
    print(f"Best checkpoint: {info['best_checkpoint']}\nEvidence copied to {out}")
    log.close()


if __name__ == "__main__":
    main()
