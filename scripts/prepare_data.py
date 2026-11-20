"""Download the dataset, check labels, find duplicates across splits, and write statistics.

Usage:
    python scripts/prepare_data.py                      # report only
    python scripts/prepare_data.py --resolve-duplicates # also move cross-split duplicates

Outputs (results/dataset/):
    dataset_stats.json / dataset_stats.md  images and objects per split and class
    label_issues.csv                       every suspicious label found
    duplicates.csv                         near-duplicate groups (perceptual hash)
    duplicate_pairs.png                    visual check of cross-split duplicates
    gt_samples.png                         training images with ground-truth boxes
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from common import (DATASET_URL, RESULTS_DIR, SPLITS, clear_label_caches, label_path, list_images,
                    load_dataset, read_labels, save_json, xywhn_to_xyxy)

OUT = RESULTS_DIR / "dataset"


# ---------------------------------------------------------------- duplicates
def find_duplicate_groups(data: dict, max_dist: int):
    """Group images whose perceptual hashes differ by <= max_dist bits (union-find)."""
    import imagehash
    from PIL import Image

    items = []  # (split, path, hash_int)
    for split in SPLITS:
        for p in list_images(data, split):
            with Image.open(p) as im:
                h = imagehash.phash(im.convert("RGB"))
            items.append((split, p, int(str(h), 16)))

    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        hi = items[i][2]
        for j in range(i + 1, len(items)):
            if (hi ^ items[j][2]).bit_count() <= max_dist:
                parent[find(i)] = find(j)

    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[find(i)].append(it)
    return [g for g in groups.values() if len(g) > 1]


def resolve_duplicates(groups, data: dict) -> list[dict]:
    """Move every member of a cross-split group into the earliest split (train < val < test).

    This is deterministic (no randomness) and guarantees a duplicate never appears in
    two different splits. Moving into train (not out of it) avoids leaking test images.
    """
    order = {s: i for i, s in enumerate(SPLITS)}
    moves = []
    for g in groups:
        splits = {s for s, _, _ in g}
        if len(splits) < 2:
            continue
        target = min(splits, key=order.get)
        dst_img_dir = Path(data[target])
        for split, img, _ in g:
            if split == target:
                continue
            lbl = label_path(img)
            new_img = dst_img_dir / img.name
            new_lbl = label_path(new_img)
            new_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img), new_img)
            if lbl.exists():
                shutil.move(str(lbl), new_lbl)
            moves.append({"file": img.name, "from": split, "to": target})
    if moves:
        clear_label_caches(data)
    return moves


def plot_duplicate_pairs(groups, path: Path, max_pairs: int = 6):
    import matplotlib.pyplot as plt
    from PIL import Image

    cross = [g for g in groups if len({s for s, _, _ in g}) > 1][:max_pairs]
    if not cross:
        return
    fig, axes = plt.subplots(len(cross), 2, figsize=(7, 3.2 * len(cross)), squeeze=False)
    for row, g in zip(axes, cross):
        for ax, (split, p, _) in zip(row, g[:2]):
            with Image.open(p) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{split}: {p.name}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Near-duplicates found in different splits (before resolving)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------ label checks
def check_labels(data: dict):
    """Return per-split statistics and a list of suspicious labels."""
    from PIL import Image

    names = data["names"]
    nc = len(names)
    stats, issues = {}, []
    for split in SPLITS:
        imgs = list_images(data, split)
        per_class = Counter()
        n_obj = 0
        for img in imgs:
            rows = read_labels(label_path(img))
            if rows is None:
                issues.append((split, img.name, "missing label file", ""))
                continue
            if not rows:
                issues.append((split, img.name, "empty label file (no objects)", ""))
            seen = set()
            with Image.open(img) as im:
                w, h = im.size
            for k, r in enumerate(rows):
                if len(r) != 5:
                    issues.append((split, img.name, "wrong number of values", f"line {k + 1}"))
                    continue
                c, xc, yc, bw, bh = r
                if not float(c).is_integer() or not 0 <= c < nc:
                    issues.append((split, img.name, "invalid class id", f"line {k + 1}: {c}"))
                    continue
                if not all(0 <= v <= 1 for v in (xc, yc, bw, bh)):
                    issues.append((split, img.name, "coordinates outside [0,1]", f"line {k + 1}"))
                x1, y1, x2, y2 = xywhn_to_xyxy(r, w, h)
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    issues.append((split, img.name, "tiny box (<4 px)", f"line {k + 1}"))
                if tuple(r) in seen:
                    issues.append((split, img.name, "duplicate box", f"line {k + 1}"))
                seen.add(tuple(r))
                per_class[names[int(c)]] += 1
                n_obj += 1
        stats[split] = {
            "images": len(imgs),
            "objects": n_obj,
            "objects_per_class": {names[i]: per_class[names[i]] for i in range(nc)},
        }
    return stats, issues


def stats_markdown(stats: dict, names: dict) -> str:
    cls = [names[i] for i in range(len(names))]
    lines = ["| Split | Images | Objects | " + " | ".join(cls) + " |",
             "|---|---:|---:|" + "---:|" * len(cls)]
    tot = Counter()
    for split in SPLITS:
        s = stats[split]
        lines.append(f"| {split} | {s['images']} | {s['objects']} | "
                     + " | ".join(str(s["objects_per_class"][c]) for c in cls) + " |")
        tot["images"] += s["images"]
        tot["objects"] += s["objects"]
        for c in cls:
            tot[c] += s["objects_per_class"][c]
    lines.append(f"| **total** | {tot['images']} | {tot['objects']} | "
                 + " | ".join(str(tot[c]) for c in cls) + " |")
    return "\n".join(lines)


# --------------------------------------------------------- GT visualisation
def plot_ground_truth(data: dict, split: str, n: int, seed: int, path: Path):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    names = data["names"]
    colors = ["tab:orange", "tab:blue", "tab:red", "tab:green", "tab:purple"]
    imgs = list_images(data, split)
    pick = random.Random(seed).sample(imgs, min(n, len(imgs)))
    cols = 3
    rows = (len(pick) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img in zip(axes.flat, pick):
        with Image.open(img) as im:
            im = im.convert("RGB")
            w, h = im.size
            ax.imshow(im)
        for r in read_labels(label_path(img)) or []:
            x1, y1, x2, y2 = xywhn_to_xyxy(r, w, h)
            col = colors[int(r[0]) % len(colors)]
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2, ec=col))
            ax.text(x1, y1, names[int(r[0])], color="white", fontsize=8,
                    bbox=dict(facecolor=col, pad=1, lw=0))
        ax.set_title(img.name, fontsize=8)
    fig.suptitle(f"Ground-truth boxes ({split} split, seed={seed})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolve-duplicates", action="store_true",
                    help="move cross-split near-duplicates into one split")
    ap.add_argument("--hash-dist", type=int, default=5,
                    help="max perceptual-hash Hamming distance counted as near-duplicate")
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    print(f"Dataset root: {data['path']}\nClasses: {data['names']}")

    # 1) duplicates
    groups = find_duplicate_groups(data, args.hash_dist)
    cross = [g for g in groups if len({s for s, _, _ in g}) > 1]
    print(f"Near-duplicate groups: {len(groups)} total, {len(cross)} spanning >1 split")
    moves_log = OUT / "duplicate_moves.csv"
    if cross or not (OUT / "duplicates.csv").exists():
        # Only (re)write the evidence when there is something new, so a second run of
        # this script does not overwrite the "before resolving" record.
        plot_duplicate_pairs(groups, OUT / "duplicate_pairs.png")
        with open(OUT / "duplicates.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["group", "split", "file", "phash"])
            for gi, g in enumerate(groups):
                for split, p, h in g:
                    wr.writerow([gi, split, p.name, f"{h:016x}"])
    moves = resolve_duplicates(groups, data) if args.resolve_duplicates else []
    if moves:
        print(f"Moved {len(moves)} images so no duplicate group spans two splits")
        with open(moves_log, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["file", "from", "to"])
            wr.writeheader()
            wr.writerows(moves)
    elif cross:
        print("Cross-split duplicates remain: rerun with --resolve-duplicates")
    n_moved_total = (sum(1 for _ in open(moves_log)) - 1) if moves_log.exists() else 0

    # 2) labels + statistics (after any moves)
    stats, issues = check_labels(data)
    with open(OUT / "label_issues.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "file", "issue", "detail"])
        wr.writerows(issues)
    summary = {
        "dataset_source": DATASET_URL,
        "classes": data["names"],
        "annotation_format": "YOLO txt: one line per object, 'class xc yc w h', normalised to [0,1]",
        "splits": stats,
        "label_issue_counts": dict(Counter(i[2] for i in issues)),
        "duplicate_groups": len(groups),
        "cross_split_duplicate_groups_now": len(cross),
        "images_moved_to_resolve_duplicates": n_moved_total,
        "hash_dist_threshold": args.hash_dist,
    }
    save_json(summary, OUT / "dataset_stats.json")
    md = stats_markdown(stats, data["names"])
    (OUT / "dataset_stats.md").write_text(md + "\n")
    print("\n" + md)
    print("\nLabel issues:", summary["label_issue_counts"] or "none")

    # 3) ground-truth visualisation
    plot_ground_truth(data, "train", args.n_samples, args.seed, OUT / "gt_samples.png")
    print(f"\nSaved figures and tables to {OUT}")


if __name__ == "__main__":
    main()
