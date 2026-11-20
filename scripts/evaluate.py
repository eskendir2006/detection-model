"""Evaluate a trained run's best checkpoint on the validation or test split.

Usage:
    python scripts/evaluate.py --run baseline_640 --split val
    python scripts/evaluate.py --run baseline_640 --split test

Evaluation settings (recorded in the output JSON):
    conf = 0.001  minimum confidence kept before NMS. Very low on purpose: mAP integrates
                  over ALL confidence levels, so low-confidence boxes must not be discarded.
    iou  = 0.7    IoU threshold used by non-maximum suppression (NMS) to remove duplicates.
    A prediction counts as correct for mAP@0.50 when its IoU with a same-class ground-truth
    box is >= 0.50. Reported precision/recall are Ultralytics' values at the confidence
    threshold that maximises mean F1 (see README for a fixed-threshold version).

Output: results/eval/<run>_<split>.json (+ PR/F1 curves and confusion matrix PNGs)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import RESULTS_DIR, RUNS_DIR, ensure_dataset, load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name, e.g. baseline_640")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()

    from ultralytics import YOLO

    info = load_json(RESULTS_DIR / "runs" / args.run / "run_info.json")
    weights = Path(info["save_dir"]) / "weights" / "best.pt"
    if not weights.exists():  # e.g. results copied from another machine
        weights = RUNS_DIR / args.run / "weights" / "best.pt"
    imgsz = info["hyperparameters"]["imgsz"]
    batch = info["hyperparameters"].get("batch", 16)

    model = YOLO(str(weights))
    m = model.val(data=str(ensure_dataset()), split=args.split, imgsz=imgsz, batch=batch,
                  conf=args.conf, iou=args.iou, plots=True, project=str(RUNS_DIR),
                  name=f"{args.run}_eval_{args.split}", exist_ok=True)

    names = m.names
    per_class = {}
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                    "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)}
    result = {
        "run": args.run,
        "split": args.split,
        "weights": str(weights),
        "settings": {"imgsz": imgsz, "conf": args.conf, "nms_iou": args.iou,
                     "match_iou_for_mAP50": 0.5,
                     "precision_recall_operating_point": "confidence that maximises mean F1"},
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "per_class": per_class,
        "speed_ms_per_image": {k: round(v, 2) for k, v in m.speed.items()},
    }
    out = RESULTS_DIR / "eval"
    save_json(result, out / f"{args.run}_{args.split}.json")
    plots = out / f"{args.run}_{args.split}"
    plots.mkdir(parents=True, exist_ok=True)
    for f in Path(m.save_dir).glob("*.png"):
        shutil.copy2(f, plots / f.name)

    print(f"\n{args.run} on {args.split}: P={result['precision']:.3f} R={result['recall']:.3f} "
          f"mAP50={result['mAP50']:.3f} mAP50-95={result['mAP50_95']:.3f}")
    for c, v in per_class.items():
        print(f"  {c:10s} P={v['precision']:.3f} R={v['recall']:.3f} mAP50={v['mAP50']:.3f}")


if __name__ == "__main__":
    main()
