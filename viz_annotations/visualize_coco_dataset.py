#!/usr/bin/env python3
"""
COCO dataset visualiser — bbox and/or segmentation.

Supports the Roboflow COCO layout:
  <dataset_root>/
  ├── train/  _annotations.coco.json + images
  ├── valid/  _annotations.coco.json + images
  └── test/   _annotations.coco.json + images

Usage
-----
  python visualize_coco_dataset.py --dataset /mnt/additional-drive/FOD/fod2.v3i.coco
  python visualize_coco_dataset.py --dataset /mnt/additional-drive/FOD/fod2.v3i.coco --split valid
  python visualize_coco_dataset.py --dataset /path/to/ds --split train --start 50

Keyboard controls
-----------------
  → / n / Space   next image
  ← / p           previous image
  r               random image
  g               go to image index (prompts in terminal)
  q / Esc         quit

Run with the rf_detr venv:
  ../rf_detr/.venv/bin/python visualize_coco_dataset.py --dataset /path/to/dataset
"""

import argparse
import random
import sys
from pathlib import Path

# Make rfdetr venv importable without activation
_SCRIPT_DIR = Path(__file__).resolve().parent
_VENV_SITE = _SCRIPT_DIR.parent / "rf_detr" / ".venv" / "lib"
if _VENV_SITE.exists():
    for _sp in sorted(_VENV_SITE.glob("python3.*/site-packages")):
        if str(_sp) not in sys.path:
            sys.path.insert(0, str(_sp))

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_util


# ── Colour palette ────────────────────────────────────────────────────────────

def _color(cat_id: int) -> tuple:
    """Vivid, fully-saturated HSV colour keyed to category ID."""
    hue = int((cat_id * 137.508) % 180)
    hsv = np.array([[[hue, 255, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ── Annotation rendering ──────────────────────────────────────────────────────

def _decode_segmentation(seg, h, w) -> np.ndarray | None:
    """
    Decode a COCO segmentation field to a boolean (H, W) mask.
    Handles polygon lists and RLE dicts.
    Returns None if the segmentation is empty.
    """
    if not seg:
        return None

    # RLE
    if isinstance(seg, dict):
        rle = coco_mask_util.frPyObjects(seg, h, w)
        return coco_mask_util.decode(rle).astype(bool)

    # Polygon(s)
    if isinstance(seg, list):
        # Filter out degenerate polygons (< 6 coords = < 3 points)
        valid = [p for p in seg if isinstance(p, list) and len(p) >= 6]
        if not valid:
            return None
        rles = coco_mask_util.frPyObjects(valid, h, w)
        rle  = coco_mask_util.merge(rles)
        return coco_mask_util.decode(rle).astype(bool)

    return None


def _draw_mask(canvas: np.ndarray, mask: np.ndarray, color: tuple, alpha=0.50):
    overlay = canvas.copy()
    overlay[mask] = color
    result = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, color, 2)
    return result


def _draw_bbox(canvas: np.ndarray, x, y, w, h, color: tuple, label: str):
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    bg_y1 = max(0, y1 - th - 8)
    cv2.rectangle(canvas, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(canvas, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def render_image(img_path: Path, anns: list, coco: COCO,
                 show_bbox: bool, show_seg: bool) -> np.ndarray:
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        # Return a placeholder if the file is missing
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(bgr, f"Image not found: {img_path.name}",
                    (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return bgr

    h, w = bgr.shape[:2]

    # Draw masks first so boxes appear on top
    if show_seg:
        for ann in anns:
            mask = _decode_segmentation(ann.get("segmentation"), h, w)
            if mask is None:
                continue
            color = _color(ann["category_id"])
            bgr = _draw_mask(bgr, mask, color)

    if show_bbox:
        for ann in anns:
            bx, by, bw, bh = ann["bbox"]
            cat_name = coco.cats[ann["category_id"]]["name"]
            color = _color(ann["category_id"])
            _draw_bbox(bgr, bx, by, bw, bh, color, cat_name)

    return bgr


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_split(dataset_root: Path, split: str):
    """Return (COCO api, image_dir, sorted image_ids)."""
    split_dir = dataset_root / split
    ann_file  = split_dir / "_annotations.coco.json"

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    coco     = COCO(str(ann_file))
    img_ids  = sorted(coco.getImgIds())
    return coco, split_dir, img_ids


def detect_annotation_types(coco: COCO, sample_n: int = 20) -> tuple[bool, bool]:
    """Return (has_bbox, has_segmentation) by sampling annotations."""
    ann_ids = coco.getAnnIds()
    sample  = [coco.loadAnns([aid])[0]
               for aid in ann_ids[:sample_n]]
    has_bbox = any(a.get("bbox") for a in sample)
    has_seg  = any(
        a.get("segmentation") and
        (isinstance(a["segmentation"], dict) or
         (isinstance(a["segmentation"], list) and len(a["segmentation"]) > 0 and
          isinstance(a["segmentation"][0], list) and len(a["segmentation"][0]) >= 6))
        for a in sample
    )
    return has_bbox, has_seg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="COCO dataset visualiser")
    parser.add_argument("--dataset", required=True,
                        help="Root of the COCO dataset (contains train/valid/test)")
    parser.add_argument("--split",   default="train",
                        choices=["train", "valid", "test"],
                        help="Which split to visualise (default: train)")
    parser.add_argument("--start",   type=int, default=0,
                        help="Start at this index (default: 0)")
    parser.add_argument("--no-bbox", action="store_true",
                        help="Hide bounding boxes")
    parser.add_argument("--no-seg",  action="store_true",
                        help="Hide segmentation masks")
    args = parser.parse_args()

    dataset_root = Path(args.dataset).expanduser().resolve()
    if not dataset_root.exists():
        print(f"ERROR: Dataset not found: {dataset_root}")
        return 1

    print(f"Loading {args.split} split … ", end="", flush=True)
    try:
        coco, img_dir, img_ids = load_split(dataset_root, args.split)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1
    print(f"{len(img_ids)} images, {len(coco.getAnnIds())} annotations")

    has_bbox, has_seg = detect_annotation_types(coco)
    show_bbox = has_bbox and not args.no_bbox
    show_seg  = has_seg  and not args.no_seg

    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    print(f"Categories : {cats}")
    print(f"Annotations: bbox={has_bbox}  segmentation={has_seg}")
    print()
    print("Controls: →/n/Space=next  ←/p=prev  r=random  g=goto  q/Esc=quit")

    idx = max(0, min(args.start, len(img_ids) - 1))

    cv2.namedWindow("COCO Viewer", cv2.WINDOW_NORMAL)

    while True:
        img_id   = img_ids[idx]
        img_info = coco.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]

        ann_ids  = coco.getAnnIds(imgIds=img_id)
        anns     = coco.loadAnns(ann_ids)

        canvas = render_image(img_path, anns, coco, show_bbox, show_seg)

        # Status bar at bottom
        status = (f"[{idx + 1}/{len(img_ids)}]  {img_info['file_name']}"
                  f"  |  {len(anns)} ann(s)")
        sh, sw = canvas.shape[:2]
        bar_h  = 28
        bar    = np.zeros((bar_h, sw, 3), dtype=np.uint8)
        cv2.putText(bar, status, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        canvas = np.vstack([canvas, bar])

        cv2.resizeWindow("COCO Viewer", sw, sh + bar_h)
        cv2.imshow("COCO Viewer", canvas)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):           # q / Esc → quit
            break
        elif key in (83, ord('n'), 32):     # → / n / Space → next
            idx = (idx + 1) % len(img_ids)
        elif key in (81, ord('p')):         # ← / p → previous
            idx = (idx - 1) % len(img_ids)
        elif key == ord('r'):               # r → random
            idx = random.randrange(len(img_ids))
        elif key == ord('g'):               # g → goto index
            cv2.destroyWindow("COCO Viewer")
            try:
                raw = input(f"Go to index (0–{len(img_ids)-1}): ").strip()
                idx = max(0, min(int(raw), len(img_ids) - 1))
            except (ValueError, EOFError):
                pass
            cv2.namedWindow("COCO Viewer", cv2.WINDOW_NORMAL)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
