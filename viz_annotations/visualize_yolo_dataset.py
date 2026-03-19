#!/usr/bin/env python3
"""
YOLO dataset visualiser — bbox and/or segmentation.

Supports the standard YOLO layout:
  <dataset_root>/
  ├── data.yaml
  ├── images/
  │   ├── train/  *.jpg / *.png
  │   ├── val/    *.jpg / *.png
  │   └── test/   *.jpg / *.png
  └── labels/
      ├── train/  *.txt
      ├── val/    *.txt
      └── test/   *.txt

Usage
-----
  python visualize_yolo_dataset.py --dataset /mnt/additional-drive/FOD/fod_seg_dataset_v3_yolo
  python visualize_yolo_dataset.py --dataset /mnt/additional-drive/FOD/fod_seg_dataset_v3_yolo --split val
  python visualize_yolo_dataset.py --dataset /path/to/ds --split train --start 50

Keyboard controls
-----------------
  → / d / n / Space   next image
  ← / a / p           previous image
  r                   random image
  g                   go to image index (prompts in terminal)
  1                   toggle background classes
  2                   toggle normal/object classes
  3                   toggle line-marking classes
  + / =               increase mask opacity
  - / _               decrease mask opacity
  q / Esc             quit
"""

import argparse
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml

WINDOW_NAME = "YOLO Viewer"
DEFAULT_ALPHA = 0.425

DEFAULT_BACKGROUND_REGEX = r"(road|street|sidewalk|sky|vegetation|building|parking|terrain|drivable cobblestone|nature)"
DEFAULT_LINE_REGEX = r"(line|mark|zebra|painted)"


# ── IO ────────────────────────────────────────────────────────────────────────

def load_annotations(label_path: str, img_shape) -> Tuple[list, list, list, list]:
    """Read YOLO bbox/seg lines; return boxes, polygons, class_ids, is_seg flags."""
    h, w = img_shape[:2]
    polygons, boxes, class_ids, is_seg = [], [], [], []
    if not label_path or not os.path.exists(label_path):
        return boxes, polygons, class_ids, is_seg

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))

            if len(coords) == 4:
                cx, cy, bw, bh = coords
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                boxes.append(((x1, y1), (x2, y2)))
                polygons.append(None)
                class_ids.append(cls)
                is_seg.append(False)
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
                abs_pts = (pts * np.array([w, h])).astype(np.int32)
                polygons.append(abs_pts)
                boxes.append(None)
                class_ids.append(cls)
                is_seg.append(True)
    return boxes, polygons, class_ids, is_seg


def load_class_names(data_yaml: Path) -> Optional[List[str]]:
    with open(data_yaml) as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        max_k = max(int(k) for k in names.keys())
        return [names.get(i, names.get(str(i), str(i))) for i in range(max_k + 1)]
    return list(names) if names else None


def find_split_dirs(dataset_root: Path, split: str) -> Tuple[Path, Path]:
    """Return (images_dir, labels_dir) for the requested split."""
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {lbl_dir}")
    return img_dir, lbl_dir


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_pairs(img_dir: Path, lbl_dir: Path) -> list[Tuple[Path, Path]]:
    """Return sorted list of (image_path, label_path) with matching stems."""
    label_by_stem = {p.stem: p for p in lbl_dir.glob("*.txt")}
    pairs = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() in IMG_EXTS and img.stem in label_by_stem:
            pairs.append((img, label_by_stem[img.stem]))
    return pairs


# ── Tiers ─────────────────────────────────────────────────────────────────────

def build_tiers(class_names, bg_names_csv, line_names_csv, bg_regex, line_regex):
    bg_ids, line_ids = set(), set()
    if not class_names:
        return bg_ids, line_ids, set()

    name_to_id = {n.lower(): i for i, n in enumerate(class_names)}

    def ids_from_csv(csv_text):
        ids = set()
        if not csv_text:
            return ids
        for raw in csv_text.split(","):
            n = raw.strip().lower()
            if not n:
                continue
            if n.isdigit():
                ids.add(int(n))
            elif n in name_to_id:
                ids.add(name_to_id[n])
        return ids

    def ids_from_regex(pattern):
        ids = set()
        if not pattern:
            return ids
        rx = re.compile(pattern, re.I)
        return {i for i, n in enumerate(class_names) if rx.search(n)}

    bg_ids   |= ids_from_csv(bg_names_csv)
    line_ids |= ids_from_csv(line_names_csv)
    bg_ids   |= ids_from_regex(bg_regex or DEFAULT_BACKGROUND_REGEX)
    line_ids |= ids_from_regex(line_regex or DEFAULT_LINE_REGEX)
    bg_ids   -= line_ids

    all_ids    = set(range(len(class_names)))
    normal_ids = all_ids - bg_ids - line_ids
    return bg_ids, line_ids, normal_ids


# ── Drawing helpers ───────────────────────────────────────────────────────────

def stable_color(cls_id: int) -> Tuple[int, int, int]:
    rng = np.random.RandomState(1337 + cls_id)
    return tuple(int(x) for x in rng.randint(0, 255, size=3))


def poly_area(pts: np.ndarray) -> float:
    if pts is None or len(pts) < 3:
        return 0.0
    pts = pts.astype(np.float32)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_centroid(pts: np.ndarray) -> Tuple[int, int]:
    if pts is None or len(pts) == 0:
        return (0, 0)
    c = pts.reshape(-1, 1, 2).astype(np.int32)
    m = cv2.moments(c)
    if abs(m["m00"]) > 1e-6:
        return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
    mean = pts.mean(axis=0)
    if np.isfinite(mean).all():
        return (int(mean[0]), int(mean[1]))
    x, y, w, h = cv2.boundingRect(c)
    return (x + w // 2, y + h // 2)


def contrast_text_color(fill_bgr):
    b, g, r = fill_bgr
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 160 else (255, 255, 255)


def draw_label_centered(img, center_xy, label, fg_bgr, thickness=2):
    x, y = int(center_xy[0]), int(center_xy[1])
    base = max(16, min(img.shape[0], img.shape[1]) // 40)
    font_scale = base / 30.0
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    org = (int(x - tw / 2), int(y + th / 2))
    outline = (255, 255, 255) if fg_bgr == (0, 0, 0) else (0, 0, 0)
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg_bgr, thickness, cv2.LINE_AA)


def draw_item(overlay, poly, box, color, label, is_line):
    if poly is not None:
        if is_line:
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
        else:
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(overlay, [poly], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        draw_label_centered(overlay, polygon_centroid(poly), label, contrast_text_color(color))
    elif box is not None:
        (x1, y1), (x2, y2) = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw_label_centered(overlay, (cx, cy), label, contrast_text_color(color))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO dataset visualiser")
    parser.add_argument("--dataset", required=True,
                        help="Root of the YOLO dataset (contains data.yaml, images/, labels/)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"],
                        help="Which split to visualise (default: train)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start at this index (default: 0)")
    parser.add_argument("--bg-names",   default="", help="CSV of background class names or ids")
    parser.add_argument("--line-names", default="", help="CSV of line-marking class names or ids")
    parser.add_argument("--bg-regex",   default=None,
                        help=f"Regex for background classes (default: {DEFAULT_BACKGROUND_REGEX})")
    parser.add_argument("--line-regex", default=None,
                        help=f"Regex for line classes (default: {DEFAULT_LINE_REGEX})")
    args = parser.parse_args()

    dataset_root = Path(args.dataset).expanduser().resolve()
    if not dataset_root.exists():
        print(f"ERROR: Dataset not found: {dataset_root}")
        return 1

    # Load class names from data.yaml
    data_yaml = dataset_root / "data.yaml"
    class_names = None
    if data_yaml.exists():
        class_names = load_class_names(data_yaml)
    else:
        print(f"[warn] data.yaml not found at {data_yaml} — class ids will be shown as numbers")

    # Find split directories
    try:
        img_dir, lbl_dir = find_split_dirs(dataset_root, args.split)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    pairs = collect_pairs(img_dir, lbl_dir)

    print(f"Split      : {args.split}")
    print(f"Images dir : {img_dir}")
    print(f"Labels dir : {lbl_dir}")
    print(f"Pairs      : {len(pairs)}")
    if class_names:
        print(f"Classes    : {class_names}")
    print()
    print("Controls: →/d/n/Space=next  ←/a/p=prev  r=random  g=goto  1/2/3=toggle tiers  +/-=opacity  q/Esc=quit")

    if not pairs:
        print("No matched image↔label pairs found.")
        return 1

    bg_ids, line_ids, _ = build_tiers(
        class_names, args.bg_names, args.line_names, args.bg_regex, args.line_regex
    )

    alpha      = DEFAULT_ALPHA
    show_bg    = True
    show_normal = True
    show_lines  = True
    idx = max(0, min(args.start, len(pairs) - 1))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        img_path, lbl_path = pairs[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[warn] failed to read: {img_path}")
            idx = (idx + 1) % len(pairs)
            continue

        boxes, polygons, cls_ids, _ = load_annotations(str(lbl_path), image.shape)

        # Build render list (tier, area, cls, name, poly, box)
        items = []
        for i, cls in enumerate(cls_ids):
            name  = class_names[cls] if class_names and cls < len(class_names) else str(cls)
            poly  = polygons[i]
            box   = boxes[i]
            area  = poly_area(poly) if poly is not None else 0.0

            if cls in bg_ids:
                if not show_bg:
                    continue
                tier = 0
            elif cls in line_ids:
                if not show_lines:
                    continue
                tier = 2
            else:
                if not show_normal:
                    continue
                tier = 1

            items.append((tier, area, cls, name, poly, box))

        items.sort(key=lambda t: (t[0], t[1]))

        overlay = image.copy()
        for tier, area, cls, name, poly, box in items:
            draw_item(overlay, poly, box, stable_color(cls), name, cls in line_ids)

        canvas = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Status bar
        status = (f"[{idx + 1}/{len(pairs)}]  {img_path.name}"
                  f"  |  {len(cls_ids)} ann(s)"
                  f"  |  alpha={alpha:.2f}"
                  f"  |  BG:{show_bg} OBJ:{show_normal} LINE:{show_lines}")
        sh, sw = canvas.shape[:2]
        bar_h  = 28
        bar    = np.zeros((bar_h, sw, 3), dtype=np.uint8)
        cv2.putText(bar, status, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        canvas = np.vstack([canvas, bar])

        cv2.resizeWindow(WINDOW_NAME, sw, sh + bar_h)
        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(0) & 0xFFFF

        if key in (ord('q'), 27):                          # q / Esc → quit
            break
        elif key in (83, 65363, ord('d'), ord('n'), 32):   # → / d / n / Space → next
            idx = (idx + 1) % len(pairs)
        elif key in (81, 65361, ord('a'), ord('p')):       # ← / a / p → prev
            idx = (idx - 1 + len(pairs)) % len(pairs)
        elif key == ord('r'):                               # r → random
            idx = random.randrange(len(pairs))
        elif key == ord('g'):                               # g → goto index
            cv2.destroyWindow(WINDOW_NAME)
            try:
                raw = input(f"Go to index (0–{len(pairs)-1}): ").strip()
                idx = max(0, min(int(raw), len(pairs) - 1))
            except (ValueError, EOFError):
                pass
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        elif key == ord('1'):
            show_bg = not show_bg
        elif key == ord('2'):
            show_normal = not show_normal
        elif key == ord('3'):
            show_lines = not show_lines
        elif key in (ord('='), ord('+')):
            alpha = min(0.95, alpha + 0.05)
        elif key in (ord('-'), ord('_')):
            alpha = max(0.10, alpha - 0.05)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
