"""Draw test-set predictions and find failure cases for the selected run.

Usage:
    python scripts/analyze_test.py                    # uses results/selected_run.json
    python scripts/analyze_test.py --run baseline_640 --conf 0.25

Every prediction with confidence >= --conf is matched greedily (highest confidence first)
to an unmatched ground-truth box of the SAME class with IoU >= --match-iou. Errors are typed:
    missed       ground-truth box with no matching prediction            (false negative)
    wrong_class  prediction overlaps (IoU>=match-iou) a box of another class
    duplicate    second prediction on an object that was already matched
    localization prediction overlaps a same-class object, but IoU < match-iou
    background   prediction that does not overlap any object (IoU < 0.1)

Outputs (results/test_analysis/):
    predictions_grid.png     >= 6 random test images with predicted boxes, class and confidence
    failure_1..3_*.png       three failure cases (ground truth dashed green, predictions red)
    failures.md              text description of the failure cases (start of your analysis)
    per_image.csv            error counts for every test image
    summary.json             precision/recall at this fixed confidence threshold
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from common import (RESULTS_DIR, RUNS_DIR, box_iou, label_path, list_images, load_dataset,
                    load_json, read_labels, save_json, xywhn_to_xyxy)

OUT = RESULTS_DIR / "test_analysis"
ERROR_TYPES = ["missed", "wrong_class", "duplicate", "localization", "background"]


def match_image(gts, preds, match_iou):
    """gts: [(cls, box)], preds: [(cls, conf, box)] -> (tp, errors list, per-pred status)."""
    used = [False] * len(gts)
    errors, status, tp = [], [], 0
    for pc, conf, pb in sorted(preds, key=lambda p: -p[1]):
        best_j, best_iou = -1, 0.0
        for j, (gc, gb) in enumerate(gts):
            if gc == pc and not used[j]:
                iou = box_iou(pb, gb)
                if iou > best_iou:
                    best_j, best_iou = j, iou
        if best_iou >= match_iou:
            used[best_j] = True
            tp += 1
            status.append((pc, conf, pb, "TP", best_iou))
            continue
        same = max((box_iou(pb, gb) for gc, gb in gts if gc == pc), default=0.0)
        other = max(((box_iou(pb, gb), gc) for gc, gb in gts if gc != pc), default=(0.0, None))
        if same >= match_iou:
            kind = "duplicate"
        elif other[0] >= match_iou:
            kind = "wrong_class"
        elif same >= 0.1:
            kind = "localization"
        else:
            kind = "background"
        errors.append({"type": kind, "cls": pc, "conf": conf,
                       "iou_same": round(same, 2), "true_cls": other[1] if kind == "wrong_class" else None})
        status.append((pc, conf, pb, kind, max(same, other[0])))
    for j, (gc, gb) in enumerate(gts):
        if not used[j]:
            errors.append({"type": "missed", "cls": gc, "conf": None})
    return tp, errors, status


def draw(ax, img_path, gts, status, names, title, show_gt=True):
    import matplotlib.patches as patches
    from PIL import Image

    with Image.open(img_path) as im:
        ax.imshow(im.convert("RGB"))
    if show_gt:
        for gc, (x1, y1, x2, y2) in gts:
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2,
                                           ec="lime", ls="--"))
            ax.text(x1, y2, f"GT {names[gc]}", color="black", fontsize=7,
                    bbox=dict(facecolor="lime", pad=1, lw=0), va="top")
    for pc, conf, (x1, y1, x2, y2), kind, _ in status:
        col = "red" if kind != "TP" else "deepskyblue"
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2, ec=col))
        tag = "" if kind == "TP" else f" [{kind}]"
        ax.text(x1, y1, f"{names[pc]} {conf:.2f}{tag}", color="white", fontsize=7,
                bbox=dict(facecolor=col, pad=1, lw=0))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def describe(rec, names):
    parts = [f"**{rec['file']}**: {rec['n_gt']} ground-truth object(s) "
             f"({', '.join(rec['gt_classes']) or 'none'}), {rec['n_pred']} prediction(s), {rec['tp']} correct."]
    missed = Counter(names[e["cls"]] for e in rec["errors"] if e["type"] == "missed")
    for cls, n in missed.items():
        parts.append(f"- missed {n} x **{cls}**")
    for e in rec["errors"]:
        if e["type"] == "missed":
            continue
        elif e["type"] == "wrong_class":
            parts.append(f"- predicted **{names[e['cls']]}** ({e['conf']:.2f}) on a "
                         f"**{names[e['true_cls']]}** (wrong class)")
        else:
            parts.append(f"- {e['type']} false positive: **{names[e['cls']]}** ({e['conf']:.2f}), "
                         f"best IoU with a same-class object {e['iou_same']}")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold for drawing/analysis")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from PIL import Image
    from ultralytics import YOLO

    run = args.run or load_json(RESULTS_DIR / "selected_run.json")["selected_run"]
    info = load_json(RESULTS_DIR / "runs" / run / "run_info.json")
    weights = Path(info["save_dir"]) / "weights" / "best.pt"
    if not weights.exists():
        weights = RUNS_DIR / run / "weights" / "best.pt"
    imgsz = info["hyperparameters"]["imgsz"]
    data = load_dataset()
    names = {int(k): v for k, v in data["names"].items()}
    model = YOLO(str(weights))

    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for img in list_images(data, args.split):
        with Image.open(img) as im:
            w, h = im.size
        gts = [(int(r[0]), xywhn_to_xyxy(r, w, h)) for r in read_labels(label_path(img)) or []]
        res = model.predict(str(img), imgsz=imgsz, conf=args.conf, verbose=False)[0]
        preds = [(int(c), float(s), tuple(map(float, b)))
                 for c, s, b in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())]
        tp, errors, status = match_image(gts, preds, args.match_iou)
        records.append({"file": img.name, "path": img, "gts": gts, "status": status, "errors": errors,
                        "n_gt": len(gts), "n_pred": len(preds), "tp": tp,
                        "gt_classes": [names[g[0]] for g in gts]})

    # ---- per-image table and fixed-threshold precision/recall
    with open(OUT / "per_image.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "n_gt", "n_pred", "tp"] + ERROR_TYPES)
        for r in records:
            wr.writerow([r["file"], r["n_gt"], r["n_pred"], r["tp"]]
                        + [sum(e["type"] == t for e in r["errors"]) for t in ERROR_TYPES])
    tp = sum(r["tp"] for r in records)
    n_pred = sum(r["n_pred"] for r in records)
    n_gt = sum(r["n_gt"] for r in records)
    totals = {t: sum(e["type"] == t for r in records for e in r["errors"]) for t in ERROR_TYPES}
    summary = {"run": run, "split": args.split, "conf": args.conf, "match_iou": args.match_iou,
               "images": len(records), "ground_truth_objects": n_gt, "predictions": n_pred,
               "true_positives": tp,
               "precision_at_conf": round(tp / n_pred, 4) if n_pred else None,
               "recall_at_conf": round(tp / n_gt, 4) if n_gt else None,
               "error_counts": totals,
               "images_with_any_error": sum(bool(r["errors"]) for r in records)}
    save_json(summary, OUT / "summary.json")

    # ---- prediction grid (random sample, fixed seed)
    sample = random.Random(args.seed).sample(records, min(args.n_samples, len(records)))
    cols = 3
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, sample):
        draw(ax, r["path"], r["gts"], r["status"], names, r["file"], show_gt=False)
    fig.suptitle(f"{run}: test predictions (conf >= {args.conf})")
    fig.tight_layout()
    fig.savefig(OUT / "predictions_grid.png", dpi=110)
    plt.close(fig)

    # ---- pick three failure cases of different kinds (worst image per kind)
    def count(r, kinds):
        return sum(e["type"] in kinds for e in r["errors"])

    picks, chosen = [], set()
    for label, kinds in [("missed", {"missed"}), ("wrong_class", {"wrong_class"}),
                         ("false_positive", {"duplicate", "localization", "background"})]:
        cands = [r for r in records if r["file"] not in chosen and count(r, kinds) > 0]
        if cands:
            best = max(cands, key=lambda r: (count(r, kinds), len(r["errors"])))
            picks.append((label, best))
            chosen.add(best["file"])
    for r in sorted(records, key=lambda r: -len(r["errors"])):  # top up to 3 if a kind was absent
        if len(picks) >= 3 or not r["errors"]:
            break
        if r["file"] not in chosen:
            picks.append(("other", r))
            chosen.add(r["file"])

    md = [f"# Failure cases: {run} on {args.split} (conf >= {args.conf}, match IoU >= {args.match_iou})\n",
          f"Error totals: {totals}\n"]
    for k, (label, r) in enumerate(picks, 1):
        fig, ax = plt.subplots(figsize=(8, 6))
        draw(ax, r["path"], r["gts"], r["status"], names, f"Failure {k} ({label}): {r['file']}")
        fig.tight_layout()
        name = f"failure_{k}_{label}.png"
        fig.savefig(OUT / name, dpi=110)
        plt.close(fig)
        md.append(f"## Failure {k}: {label}\n\n![{name}]({name})\n\n{describe(r, names)}\n\n"
                  "Likely cause: _TODO (your analysis)_\n")
    if len(picks) < 3:
        md.append(f"Only {len(picks)} image(s) had errors at this threshold; discuss the hardest "
                  "remaining examples instead (lowest-confidence correct detections).\n")
    (OUT / "failures.md").write_text("\n".join(md))

    print(f"{run} on {args.split} @conf={args.conf}: precision={summary['precision_at_conf']} "
          f"recall={summary['recall_at_conf']}  errors={totals}")
    print(f"Saved predictions, failure figures and failures.md to {OUT}")


if __name__ == "__main__":
    main()
