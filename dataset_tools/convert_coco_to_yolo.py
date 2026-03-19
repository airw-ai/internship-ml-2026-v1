#!/usr/bin/env python3
"""
COCO JSON → YOLO Converter

Converts a COCO JSON dataset (used by the rf_detr/ training pipeline)
into YOLO format (used by the yolo/ training pipeline).

Supports both detection (bbox only) and segmentation annotations.
When segmentation polygons are present they are written as YOLO
segmentation labels; otherwise bounding boxes are used.

Input layout (COCO):
  <coco_dataset>/
  ├── train/
  │   ├── _annotations.coco.json
  │   └── *.jpg
  └── valid/      (or val/ or test/)
      ├── _annotations.coco.json
      └── *.jpg

Output layout (YOLO):
  <yolo_dataset>/
  ├── data.yaml
  ├── images/
  │   ├── train/  *.jpg  (symlinked or copied)
  │   └── val/    *.jpg
  └── labels/
      ├── train/  *.txt
      └── val/    *.txt

Usage:
    python convert_coco_to_yolo.py \\
        --coco  ../dataset/internship_dataset_coco \\
        --yolo  ../dataset/internship_dataset_yolo \\
        [--copy-images]   # copy instead of symlink
        [--seg]           # force segmentation output even when polygons absent
"""

import argparse
import json
import shutil
import yaml
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# COCO split folder name → YOLO split name
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val":   "val",
    "test":  "test",
}


def parse_args():
    p = argparse.ArgumentParser(description="Convert COCO JSON dataset to YOLO format")
    p.add_argument("--coco", required=True,
                   help="Root of the COCO dataset (contains train/, valid/, … sub-folders)")
    p.add_argument("--yolo", required=True,
                   help="Output root for the YOLO dataset")
    p.add_argument("--copy-images", action="store_true",
                   help="Copy images instead of creating symlinks")
    p.add_argument("--seg", action="store_true",
                   help="Write segmentation polygon labels (default: auto-detect)")
    return p.parse_args()


def find_annotation_file(split_dir: Path) -> Path | None:
    """Return the first _annotations.coco.json found in split_dir, or None."""
    for candidate in ("_annotations.coco.json", "annotations.json"):
        path = split_dir / candidate
        if path.exists():
            return path
    # Fall back to any .json file
    jsons = sorted(split_dir.glob("*.json"))
    return jsons[0] if jsons else None


def load_coco(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def bbox_coco_to_yolo(x: float, y: float, w: float, h: float,
                      img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert COCO [x_min, y_min, w, h] pixels to YOLO [cx, cy, w, h] normalised."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    return cx, cy, wn, hn


def polygon_to_yolo(segmentation: list, img_w: int, img_h: int) -> list[float]:
    """
    Convert a COCO segmentation polygon (flat pixel list) to a normalised
    YOLO polygon (flat list of x/y pairs in [0, 1]).
    """
    coords = segmentation[0]  # take first polygon if multiple
    result = []
    for i, v in enumerate(coords):
        result.append(round(v / img_w if i % 2 == 0 else v / img_h, 6))
    return result


def has_segmentation(annotations: list[dict]) -> bool:
    """Return True if at least one annotation has a non-empty segmentation."""
    return any(
        ann.get("segmentation") and len(ann["segmentation"]) > 0
        for ann in annotations
    )


def convert_split(
    coco_split_dir: Path,
    yolo_images_dir: Path,
    yolo_labels_dir: Path,
    cat_id_to_yolo_id: dict[int, int],
    copy_images: bool,
    use_seg: bool,
) -> int:
    """Convert one split. Returns number of images processed."""
    ann_file = find_annotation_file(coco_split_dir)
    if ann_file is None:
        print(f"  [skip] no annotation JSON found in {coco_split_dir}")
        return 0

    coco = load_coco(ann_file)

    # Auto-detect segmentation mode if not forced
    write_seg = use_seg or has_segmentation(coco.get("annotations", []))

    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)

    # Index: image_id → image metadata
    id_to_image = {img["id"]: img for img in coco.get("images", [])}

    # Index: image_id → list of annotations
    id_to_anns: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    processed = 0
    for img_meta in coco.get("images", []):
        img_id   = img_meta["id"]
        filename = img_meta["file_name"]
        img_w    = img_meta["width"]
        img_h    = img_meta["height"]

        # Locate the source image (may be directly in split_dir or sub-folder)
        src = coco_split_dir / filename
        if not src.exists():
            # Search by name only (ignore sub-path stored in file_name)
            candidates = list(coco_split_dir.rglob(Path(filename).name))
            if not candidates:
                print(f"  [warn] image not found: {filename}")
                continue
            src = candidates[0]

        # Link / copy image
        dest_img = yolo_images_dir / Path(filename).name
        if not dest_img.exists():
            if copy_images:
                shutil.copy2(src, dest_img)
            else:
                dest_img.symlink_to(src.resolve())

        # Write label file
        anns = id_to_anns.get(img_id, [])
        label_lines = []
        for ann in anns:
            cat_id   = ann["category_id"]
            yolo_cls = cat_id_to_yolo_id.get(cat_id)
            if yolo_cls is None:
                continue  # unknown category

            seg = ann.get("segmentation")
            if write_seg and seg and len(seg) > 0 and len(seg[0]) >= 6:
                coords = polygon_to_yolo(seg, img_w, img_h)
                label_lines.append(f"{yolo_cls} " + " ".join(map(str, coords)))
            else:
                x, y, w, h = ann["bbox"]
                cx, cy, wn, hn = bbox_coco_to_yolo(x, y, w, h, img_w, img_h)
                label_lines.append(
                    f"{yolo_cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}"
                )

        dest_lbl = yolo_labels_dir / (Path(filename).stem + ".txt")
        with open(dest_lbl, "w") as f:
            f.write("\n".join(label_lines))
            if label_lines:
                f.write("\n")

        processed += 1

    mode = "seg" if write_seg else "det"
    print(f"  [{mode}] {processed} images, {sum(len(v) for v in id_to_anns.values())} annotations")
    return processed


def build_category_map(coco_splits: list[dict]) -> tuple[list[str], dict[int, int]]:
    """
    Merge categories across all splits and return:
      - class_names: ordered list of names (YOLO index = position)
      - cat_id_to_yolo_id: COCO category_id → YOLO 0-based index
    """
    seen: dict[int, str] = {}
    for coco in coco_splits:
        for cat in coco.get("categories", []):
            seen[cat["id"]] = cat["name"]

    # Sort by COCO id for deterministic ordering
    sorted_cats = sorted(seen.items())
    class_names = [name for _, name in sorted_cats]
    cat_id_to_yolo_id = {cat_id: i for i, (cat_id, _) in enumerate(sorted_cats)}
    return class_names, cat_id_to_yolo_id


def main():
    args = parse_args()
    coco_root = Path(args.coco).expanduser().resolve()
    yolo_root = Path(args.yolo).expanduser().resolve()

    # Discover splits
    split_dirs: dict[str, Path] = {}
    for d in sorted(coco_root.iterdir()):
        if d.is_dir() and d.name in SPLIT_MAP:
            split_dirs[d.name] = d

    if not split_dirs:
        raise RuntimeError(
            f"No recognised split folders (train/valid/val/test) found in {coco_root}"
        )

    # Load all COCO JSONs to build a unified category map
    coco_data: dict[str, dict] = {}
    for split_name, split_dir in split_dirs.items():
        ann_file = find_annotation_file(split_dir)
        if ann_file:
            coco_data[split_name] = load_coco(ann_file)

    if not coco_data:
        raise RuntimeError("No annotation JSON files found in any split folder.")

    class_names, cat_id_to_yolo_id = build_category_map(list(coco_data.values()))
    print(f"Classes ({len(class_names)}): {class_names}")

    # Convert each split
    yolo_split_names: list[str] = []
    for coco_split, split_dir in split_dirs.items():
        yolo_split = SPLIT_MAP[coco_split]
        if yolo_split not in yolo_split_names:
            yolo_split_names.append(yolo_split)
        print(f"\nSplit: {coco_split} → {yolo_split}/")
        convert_split(
            coco_split_dir=split_dir,
            yolo_images_dir=yolo_root / "images" / yolo_split,
            yolo_labels_dir=yolo_root / "labels" / yolo_split,
            cat_id_to_yolo_id=cat_id_to_yolo_id,
            copy_images=args.copy_images,
            use_seg=args.seg,
        )

    # Write data.yaml
    data_yaml = {
        "path": str(yolo_root),
        "train": "images/train",
        "val":   "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
        "nc":    len(class_names),
    }
    if "test" in yolo_split_names:
        data_yaml["test"] = "images/test"

    yaml_path = yolo_root / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\nWrote {yaml_path}")
    print(f"\nDone! YOLO dataset written to: {yolo_root}")
    print("\nExpected structure for YOLO training:")
    print(f"  {yolo_root}/")
    print(f"    data.yaml")
    for s in yolo_split_names:
        print(f"    images/{s}/  *.jpg")
        print(f"    labels/{s}/  *.txt")


if __name__ == "__main__":
    main()
