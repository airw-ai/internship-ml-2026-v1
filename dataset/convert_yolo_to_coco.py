#!/usr/bin/env python3
"""
YOLO Segmentation → COCO JSON Converter

Converts a YOLO segmentation dataset (used by the yolo/ training pipeline)
into COCO JSON format (used by the rf_detr/ training pipeline).

Input layout (YOLO):
  <yolo_dataset>/
  ├── data.yaml
  ├── images/
  │   ├── train/  *.jpg / *.png
  │   └── val/    *.jpg / *.png
  └── labels/
      ├── train/  *.txt   (class_id x1 y1 x2 y2 ... xn yn  — normalised)
      └── val/    *.txt

Output layout (COCO):
  <coco_dataset>/
  ├── train/
  │   ├── _annotations.coco.json
  │   └── *.jpg   (symlinked or copied from YOLO source)
  └── valid/
      ├── _annotations.coco.json
      └── *.jpg

Usage:
    python convert_yolo_to_coco.py \
        --yolo  ../dataset/internship_dataset \
        --coco  ../dataset/internship_dataset_coco \
        [--copy-images]   # copy instead of symlink

The COCO JSON uses bounding boxes derived from the segmentation polygon
(min/max of all polygon points).  Polygon segmentation is also stored in
the "segmentation" field so RF-DETR segmentation models can use it.
"""

import argparse
import json
import os
import shutil
import yaml
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# YOLO split name → COCO output folder name
SPLIT_MAP = {
    "train": "train",
    "val":   "valid",
    "valid": "valid",
    "test":  "test",
}


def parse_args():
    p = argparse.ArgumentParser(description="Convert YOLO segmentation dataset to COCO JSON")
    p.add_argument("--yolo",  required=True, help="Root of the YOLO dataset (contains data.yaml)")
    p.add_argument("--coco",  required=True, help="Output root for the COCO dataset")
    p.add_argument("--copy-images", action="store_true",
                   help="Copy images instead of creating symlinks (needed on Windows or remote FS)")
    return p.parse_args()


def load_class_names(data_yaml: Path) -> list[str]:
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    return list(names)


def find_images(img_dir: Path) -> list[Path]:
    return sorted(p for p in img_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def polygon_to_bbox(xs: list[float], ys: list[float],
                    img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert normalised polygon to COCO bbox [x, y, width, height] in pixels."""
    px = [x * img_w for x in xs]
    py = [y * img_h for y in ys]
    x_min, x_max = min(px), max(px)
    y_min, y_max = min(py), max(py)
    return x_min, y_min, x_max - x_min, y_max - y_min


def parse_label_file(lbl_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Parse one YOLO segmentation label file into a list of annotation dicts."""
    annotations = []
    if not lbl_path.exists():
        return annotations

    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:          # need at least class + 3 polygon points
                continue
            class_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                coords = coords[:-1]    # drop trailing odd value
            xs = coords[0::2]
            ys = coords[1::2]
            # Pixel-space polygon for COCO
            seg_px = []
            for x, y in zip(xs, ys):
                seg_px.append(round(x * img_w, 2))
                seg_px.append(round(y * img_h, 2))

            x, y, w, h = polygon_to_bbox(xs, ys, img_w, img_h)
            area = w * h

            annotations.append({
                "class_id": class_id,
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "segmentation": [seg_px],
                "area": round(area, 2),
            })
    return annotations


def get_image_size(img_path: Path) -> tuple[int, int]:
    """Return (width, height) without loading the full image."""
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.size         # (width, height)
    except ImportError:
        pass

    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except ImportError:
        pass

    raise RuntimeError(
        "Need Pillow or OpenCV to read image dimensions. "
        "Install with: pip install Pillow"
    )


def convert_split(
    yolo_root: Path,
    split: str,
    coco_split_dir: Path,
    class_names: list[str],
    copy_images: bool,
) -> None:
    img_dir = yolo_root / "images" / split
    lbl_dir = yolo_root / "labels" / split

    if not img_dir.exists():
        print(f"  [skip] images/{split}/ not found")
        return

    coco_split_dir.mkdir(parents=True, exist_ok=True)

    # Build COCO categories list
    categories = [
        {"id": i, "name": name, "supercategory": "none"}
        for i, name in enumerate(class_names)
    ]

    images_list = []
    annotations_list = []
    ann_id = 1

    image_files = find_images(img_dir)
    print(f"  Converting {len(image_files)} images in '{split}' split …")

    for img_id, img_path in enumerate(image_files, start=1):
        # Link or copy image into COCO output folder
        dest = coco_split_dir / img_path.name
        if not dest.exists():
            if copy_images:
                shutil.copy2(img_path, dest)
            else:
                dest.symlink_to(img_path.resolve())

        # Image dimensions
        width, height = get_image_size(img_path)

        images_list.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        # Annotations
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        for ann in parse_label_file(lbl_path, width, height):
            annotations_list.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann["class_id"],
                "bbox": ann["bbox"],
                "segmentation": ann["segmentation"],
                "area": ann["area"],
                "iscrowd": 0,
            })
            ann_id += 1

    coco_json = {
        "info": {"description": "Converted from YOLO segmentation format"},
        "licenses": [],
        "categories": categories,
        "images": images_list,
        "annotations": annotations_list,
    }

    out_json = coco_split_dir / "_annotations.coco.json"
    with open(out_json, "w") as f:
        json.dump(coco_json, f, indent=2)

    print(f"  Wrote {out_json.name}  "
          f"({len(images_list)} images, {len(annotations_list)} annotations)")


def main():
    args = parse_args()
    yolo_root = Path(args.yolo).expanduser().resolve()
    coco_root = Path(args.coco).expanduser().resolve()

    data_yaml = yolo_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    class_names = load_class_names(data_yaml)
    print(f"Classes ({len(class_names)}): {class_names}")

    # Detect which splits exist
    splits_found = [
        s for s in ("train", "val", "valid", "test")
        if (yolo_root / "images" / s).exists()
    ]

    if not splits_found:
        raise RuntimeError(f"No split directories found under {yolo_root}/images/")

    for split in splits_found:
        coco_split_name = SPLIT_MAP.get(split, split)
        print(f"\nSplit: {split} → {coco_split_name}/")
        convert_split(
            yolo_root=yolo_root,
            split=split,
            coco_split_dir=coco_root / coco_split_name,
            class_names=class_names,
            copy_images=args.copy_images,
        )

    print(f"\nDone! COCO dataset written to: {coco_root}")
    print("\nExpected structure for RF-DETR:")
    print(f"  {coco_root}/")
    for split in splits_found:
        coco_name = SPLIT_MAP.get(split, split)
        print(f"    {coco_name}/")
        print(f"      _annotations.coco.json")
        print(f"      *.jpg")


if __name__ == "__main__":
    main()
