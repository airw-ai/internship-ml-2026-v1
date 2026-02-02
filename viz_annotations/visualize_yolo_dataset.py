#!/usr/bin/env python3

import os
import re
import cv2
import yaml
import argparse
import numpy as np
from glob import glob
from typing import List, Tuple, Optional, Set

WINDOW_NAME = "YOLOv12 Detection + Segmentation Preview"
DEFAULT_ALPHA = 0.425


# -------------------- IO --------------------
def load_annotations(label_path: str, img_shape) -> Tuple[list, list, list, list]:
    """Read YOLO bbox/seg lines; return boxes, polygons, class_ids, is_seg flags."""
    h, w = img_shape[:2]
    polygons, boxes, class_ids, is_seg = [], [], [], []
    if not label_path or not os.path.exists(label_path):
        return boxes, polygons, class_ids, is_seg

    with open(label_path, "r") as f:
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


def load_class_names(names_path: Optional[str]) -> Optional[List[str]]:
    if not names_path:
        return None
    with open(names_path, "r") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        max_k = max(int(k) for k in names.keys())
        return [names.get(i, names.get(str(i), str(i))) for i in range(max_k + 1)]
    return names


# -------------------- Parse helpers --------------------
def parse_ids_from_csv(csv_text: Optional[str],
                       class_names: Optional[List[str]]) -> Set[int]:
    ids: Set[int] = set()
    if not csv_text:
        return ids
    if not class_names:
        return ids

    name_to_id = {n.lower(): i for i, n in enumerate(class_names)}
    for raw in csv_text.split(","):
        token = raw.strip()
        if token == "":
            continue
        if token.isdigit():
            ids.add(int(token))
        else:
            nid = name_to_id.get(token.lower())
            if nid is not None:
                ids.add(nid)
    return ids


# -------------------- Color utility --------------------
def stable_color_for(cls_id: int) -> Tuple[int, int, int]:
    """Deterministic pseudo-random BGR based on class id."""
    rng = np.random.RandomState(1337 + cls_id)
    return tuple(int(x) for x in rng.randint(0, 255, size=3))


# -------------------- Tiers & predicates --------------------
DEFAULT_BACKGROUND_REGEX = r"(road|street|sidewalk|sky|vegetation|building|parking|terrain|drivable cobblestone|nature)"
DEFAULT_LINE_REGEX = r"(line|mark|zebra|painted)"  # case-insensitive

def build_tiers(class_names: Optional[List[str]],
                bg_names_csv: Optional[str],
                line_names_csv: Optional[str],
                bg_regex: Optional[str],
                line_regex: Optional[str]):
    bg_ids, line_ids = set(), set()

    if class_names:
        name_to_id = {n.lower(): i for i, n in enumerate(class_names)}
        def ids_from_csv(csv_text):
            ids = set()
            if not csv_text:
                return ids
            for raw in csv_text.split(","):
                n = raw.strip().lower()
                if n == "":
                    continue
                if n.isdigit():
                    ids.add(int(n))
                elif n in name_to_id:
                    ids.add(name_to_id[n])
            return ids

        bg_ids |= ids_from_csv(bg_names_csv)
        line_ids |= ids_from_csv(line_names_csv)

        def ids_from_regex(pattern):
            ids = set()
            if not pattern:
                return ids
            rx = re.compile(pattern, re.I)
            for i, n in enumerate(class_names):
                if rx.search(n):
                    ids.add(i)
            return ids

        bg_ids |= ids_from_regex(bg_regex or DEFAULT_BACKGROUND_REGEX)
        line_ids |= ids_from_regex(line_regex or DEFAULT_LINE_REGEX)

    bg_ids -= line_ids

    all_ids = set(range(len(class_names))) if class_names else set()
    normal_ids = all_ids - bg_ids - line_ids if all_ids else set()

    return bg_ids, line_ids, normal_ids


# -------------------- Geometry & text helpers --------------------
def poly_area(pts: np.ndarray) -> float:
    if pts is None or len(pts) < 3:
        return 0.0
    pts = pts.astype(np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_centroid(pts: np.ndarray) -> Tuple[int, int]:
    """
    Robust centroid for simple polygons.
    Tries image moments; falls back to mean and then bounding-rect center.
    """
    if pts is None or len(pts) == 0:
        return (0, 0)
    c = pts.reshape(-1, 1, 2).astype(np.int32)
    m = cv2.moments(c)
    if abs(m["m00"]) > 1e-6:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        return (cx, cy)
    # fallback: arithmetic mean
    mean = pts.mean(axis=0)
    if np.isfinite(mean).all():
        return (int(mean[0]), int(mean[1]))
    # final fallback: bounding rect center
    x, y, w, h = cv2.boundingRect(c)
    return (x + w // 2, y + h // 2)


def center_of_box(box: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[int, int]:
    (x1, y1), (x2, y2) = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def contrast_text_color(fill_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Choose black or white text color based on luminance of the background fill.
    Uses ITU-R BT.601 luma approximation.
    """
    b, g, r = fill_bgr
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luma > 160 else (255, 255, 255)


def draw_label_centered(img: np.ndarray,
                        center_xy: Tuple[int, int],
                        label: str,
                        fg_bgr: Tuple[int, int, int],
                        thickness: int = 2):
    """
    Draw label centered on (x,y) with a thin outline for legibility.
    """
    x, y = int(center_xy[0]), int(center_xy[1])

    # Estimate a sensible font scale from image size
    base = max(16, min(img.shape[0], img.shape[1]) // 40)  # ~2.5% of min dimension
    font_scale = base / 30.0

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    # Center the text box on (x, y)
    org = (int(x - tw / 2), int(y + th / 2))

    # Outline (opposite color) for readability
    outline_col = (255, 255, 255) if fg_bgr == (0, 0, 0) else (0, 0, 0)
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_col, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg_bgr, thickness, cv2.LINE_AA)


def font_scale_for(area_or_size: float) -> float:
    """
    Optional hook if you want font to scale with object size.
    Currently unused in favor of image-based scaling in draw_label_centered.
    """
    return 1.0


# -------------------- Draw helpers --------------------
def draw_item(overlay, cls_id, poly, box, color, label, is_line, alpha):
    """
    Draw the filled shape/outline, then center the label inside it.
    """
    if poly is not None:
        if is_line:
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
        else:
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(overlay, [poly], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # Centered label for polygon
        cx, cy = polygon_centroid(poly)
        text_color = contrast_text_color(color)
        draw_label_centered(overlay, (cx, cy), label, text_color, thickness=2)

    elif box is not None:
        (x1, y1), (x2, y2) = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Centered label for bbox
        cx, cy = center_of_box(box)
        text_color = contrast_text_color(color)
        draw_label_centered(overlay, (cx, cy), label, text_color, thickness=2)


# -------------------- Main visualization --------------------
def visualize(images_dir, labels_dir, names_path=None,
              bg_names=None, line_names=None, bg_regex=None, line_regex=None,
              ignore_csv: Optional[str] = None):
    image_paths = sorted(
        glob(os.path.join(images_dir, "*.jpg")) +
        glob(os.path.join(images_dir, "*.jpeg")) +
        glob(os.path.join(images_dir, "*.png"))
    )

    label_files = glob(os.path.join(labels_dir, "*.txt"))
    labels_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in label_files}

    pairs = [(ip, labels_by_stem[os.path.splitext(os.path.basename(ip))[0]])
             for ip in image_paths if os.path.splitext(os.path.basename(ip))[0] in labels_by_stem]

    print("=== Dataset Summary ===")
    print(f"Images found: {len(image_paths)}")
    print(f"Label files found: {len(label_files)}")
    print(f"Matched image↔label pairs: {len(pairs)}")
    print("========================")
    if not pairs:
        return

    class_names = load_class_names(names_path)
    if ignore_csv and not class_names:
        print("[warn] --ignore-classes was provided but --data-yaml was not (or names failed to load). Ignoring.")
        ignore_ids = set()
    else:
        ignore_ids = parse_ids_from_csv(ignore_csv, class_names) if class_names else set()

    if class_names and ignore_ids:
        ignored_desc = ", ".join(
            f"{i}:{class_names[i] if i < len(class_names) else str(i)}" for i in sorted(ignore_ids)
        )
        print(f"Ignoring classes: {ignored_desc}")

    bg_ids, line_ids, normal_ids = build_tiers(class_names, bg_names, line_names, bg_regex, line_regex)

    alpha = DEFAULT_ALPHA
    show_bg, show_normal, show_lines = True, True, True

    idx = 0
    while True:
        img_path, lbl_path = pairs[idx]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: failed to read image: {img_path}")
            idx = (idx + 1) % len(pairs)
            continue

        boxes, polygons, cls_ids, is_seg = load_annotations(lbl_path, image.shape)

        items = []
        for i, cls in enumerate(cls_ids):
            if class_names and cls in ignore_ids:
                continue

            poly = polygons[i]
            box = boxes[i]
            area = poly_area(poly) if poly is not None else 0.0

            if class_names:
                name = class_names[cls] if cls < len(class_names) else str(cls)
            else:
                name = str(cls)

            if cls in bg_ids:
                tier = 0
                visible = show_bg
            elif cls in line_ids:
                tier = 2
                visible = show_lines
            else:
                tier = 1
                visible = show_normal

            if not visible:
                continue

            items.append((tier, area, cls, name, poly, box))

        items.sort(key=lambda t: (t[0], t[1]))

        overlay = image.copy()
        for tier, area, cls, name, poly, box in items:
            color = stable_color_for(cls)
            is_line = (cls in line_ids)
            draw_item(overlay, cls, poly, box, color, name, is_line, alpha)

        blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        footer = f"{os.path.basename(img_path)} | alpha={alpha:.2f} | BG:{show_bg} OBJ:{show_normal} LINE:{show_lines}"
        cv2.putText(blended, footer, (20, blended.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        display = cv2.resize(blended, (1280, 720)) if blended.shape[1] > 1280 else blended
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(0) & 0xFFFF
        if key in (ord("q"), 27):  # q or ESC
            break
        elif key in (ord("d"), 83, 65363):   # next
            idx = (idx + 1) % len(pairs)
        elif key in (ord("a"), 81, 65361):   # prev
            idx = (idx - 1 + len(pairs)) % len(pairs)
        elif key == ord("1"):
            show_bg = not show_bg
        elif key == ord("2"):
            show_normal = not show_normal
        elif key == ord("3"):
            show_lines = not show_lines
        elif key == ord("l"):
            show_bg, show_normal, show_lines = (False, False, True)
        elif key == ord("b"):
            show_bg = not show_bg
        elif key == ord("=") or key == ord("+"):
            alpha = min(0.95, alpha + 0.05)
        elif key == ord("-") or key == ord("_"):
            alpha = max(0.10, alpha - 0.05)

    cv2.destroyAllWindows()


# -------------------- CLI --------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="Path to images folder")
    p.add_argument("--labels", required=True, help="Path to labels folder")
    p.add_argument("--data-yaml", help="Optional path to data.yaml for class names")

    p.add_argument("--bg-names", default="", help="CSV of background class names or ids")
    p.add_argument("--line-names", default="", help="CSV of line-marking class names or ids")
    p.add_argument("--bg-regex", default=None, help=f"Regex for background classes (default: {DEFAULT_BACKGROUND_REGEX})")
    p.add_argument("--line-regex", default=None, help=f"Regex for line classes (default: {DEFAULT_LINE_REGEX})")
    p.add_argument("--ignore-classes", default="",
                   help="CSV of class names or ids to ignore (requires --data-yaml)")

    args = p.parse_args()
    try:
        visualize(args.images, args.labels, args.data_yaml,
                  args.bg_names, args.line_names, args.bg_regex, args.line_regex,
                  ignore_csv=args.ignore_classes)
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
        cv2.destroyAllWindows()
