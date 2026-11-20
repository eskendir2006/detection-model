"""Shared paths and helpers used by every script in this repository."""
from __future__ import annotations

import json
import os
from pathlib import Path

# Scripts only save figures to files; never try to use an interactive/notebook backend.
os.environ["MPLBACKEND"] = "Agg"

REPO = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO / "datasets"          # dataset is downloaded here (git-ignored)
RUNS_DIR = REPO / "runs"                  # Ultralytics training/val outputs
RESULTS_DIR = REPO / "results"            # small evidence files that ARE committed
DATASET_ROOT = DATASETS_DIR / "african-wildlife"
DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip"
DATA_YAML = DATASETS_DIR / "african-wildlife-local.yaml"  # generated; absolute paths
CLASS_NAMES = {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"}
SPLITS = ("train", "val", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dataset() -> Path:
    """Download + unzip the dataset into ./datasets (once) and write a data YAML for it.

    We download explicitly instead of relying on Ultralytics' global `datasets_dir`
    setting, so the data always lives inside this repo and nothing global is changed.
    """
    if not (DATASET_ROOT / "images" / "train").exists():
        from ultralytics.utils.downloads import safe_download

        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        safe_download(DATASET_URL, dir=DATASETS_DIR, unzip=True, delete=True)
    cfg = {"path": str(DATASET_ROOT), "train": "images/train", "val": "images/val",
           "test": "images/test", "names": CLASS_NAMES}
    DATA_YAML.write_text(json.dumps(cfg, indent=2))  # JSON is valid YAML
    return DATA_YAML


def load_dataset() -> dict:
    """Ensure the dataset exists and return the resolved Ultralytics data dict."""
    from ultralytics.data.utils import check_det_dataset

    return check_det_dataset(str(ensure_dataset()))


def image_dir(data: dict, split: str) -> Path:
    return Path(data[split])


def label_path(img_path: Path) -> Path:
    """images/<split>/x.jpg -> labels/<split>/x.txt (standard YOLO layout)."""
    parts = list(img_path.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def list_images(data: dict, split: str) -> list[Path]:
    d = image_dir(data, split)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def read_labels(lbl: Path) -> list[list[float]] | None:
    """Return rows [cls, xc, yc, w, h] (normalised), or None if the file is missing."""
    if not lbl.exists():
        return None
    rows = []
    for line in lbl.read_text().splitlines():
        if line.strip():
            rows.append([float(v) for v in line.split()])
    return rows


def xywhn_to_xyxy(row, w: int, h: int):
    _, xc, yc, bw, bh = row[:5]
    return ((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def clear_label_caches(data: dict) -> None:
    """Ultralytics caches label scans in *.cache files; delete them after moving files."""
    root = Path(data["path"])
    for cache in root.rglob("*.cache"):
        cache.unlink()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path):
    return json.loads(Path(path).read_text())
