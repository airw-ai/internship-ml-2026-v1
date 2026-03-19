#!/usr/bin/env python3
import os
import re
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set
from tqdm import tqdm

# ------------------------ Helpers ------------------------
IMG_EXTS_DEFAULT = ["jpg", "jpeg", "png"]

def parse_args():
    p = argparse.ArgumentParser(
        description="Split a YOLO-style image/label pool into train/val with class-balanced (multi-label) stratification."
    )
    p.add_argument("-r", "--dataset-root", required=True,
                   help="Root of the YOLO dataset (where images/ and labels/ live).")
    p.add_argument("--pool-images", default="images/train",
                   help="Pool images dir relative to dataset root (default: images/train).")
    p.add_argument("--pool-labels", default="labels/train",
                   help="Pool labels dir relative to dataset root (default: labels/train).")
    p.add_argument("-v", "--val-ratio", type=float, default=0.20,
                   help="Validation ratio (default: 0.20).")
    p.add_argument("-s", "--seed", type=int, default=42,
                   help="Random seed (default: 42).")
    p.add_argument("--mode", choices=["move", "copy", "symlink"], default="move",
                   help="How to populate split dirs (default: move).")
    p.add_argument("--ext", default=",".join(IMG_EXTS_DEFAULT),
                   help=f"Comma-separated image extensions (default: {','.join(IMG_EXTS_DEFAULT)}).")
    p.add_argument("--no-yaml", action="store_true", default=False,
                   help="Skip updating data.yaml train/val paths.")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Plan only; do not write files.")
    p.add_argument("--no-stratify", action="store_true", default=False,
                   help="Disable stratified split; use random selection only.")
    return p.parse_args()

def norm_exts(ext_csv: str):
    exts = {e.lower().lstrip(".") for e in ext_csv.split(",") if e.strip()}
    return {e for e in exts if re.fullmatch(r"[a-z0-9]+", e)}

def find_images(pool_images_dir: Path, exts: Set[str]) -> List[Path]:
    imgs = []
    for ext in exts:
        imgs.extend(pool_images_dir.glob(f"*.{ext}"))
        imgs.extend(pool_images_dir.glob(f"*/*.{ext}"))  # one level deep allowed
    return sorted(set(imgs))

def ensure_dirs(root: Path):
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

def same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return str(a) == str(b)

def place(src: Path, dst: Path, mode: str, dry_run: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    if mode == "move":
        if same_file(src, dst):
            return
        shutil.move(str(src), str(dst))
    elif mode == "copy":
        if src.is_file():
            shutil.copy2(str(src), str(dst))
    elif mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

def delete_if_exists(p: Path, dry_run: bool):
    if p.exists() or p.is_symlink():
        if not dry_run:
            p.unlink()

def update_data_yaml(dataset_root: Path):
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        print("data.yaml not found. Skipping update.")
        return
    lines = yaml_path.read_text().splitlines()
    new_lines = []
    for line in lines:
        k = line.strip().split(":", 1)[0]
        if k == "train":
            new_lines.append("train: images/train")
        elif k == "val":
            new_lines.append("val: images/val")
        else:
            new_lines.append(line)
    yaml_path.write_text("\n".join(new_lines) + "\n")
    print("Updated data.yaml with train/val relative paths.")

# ---------- YOLO helpers ----------
def read_image_classes(label_path: Path) -> set[int]:
    cls_set = set()
    if not label_path.exists():
        return cls_set
    with label_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cid = int(parts[0])
            except Exception:
                continue
            cls_set.add(cid)
    return cls_set

def summarize_distribution(imgs: list[Path], labels_dir: Path) -> Counter:
    c = Counter()
    for p in imgs:
        s = read_image_classes(labels_dir / f"{p.stem}.txt")
        for cid in s:
            c[cid] += 1
    return c

# ---------- Multi-label stratified split ----------
def stratified_split(
    all_images: List[Path],
    labels_dir: Path,
    val_ratio: float,
    seed: int
) -> Set[Path]:
    """
    Iterative greedy multi-label stratification:
    - Target per-class validation image counts = round(val_ratio * class_image_count)
    - Repeatedly pick the class with highest remaining need, choose the candidate image that best
      reduces the total remaining need with minimal overshoot, assign to val.
    - Fill remaining val slots randomly.
    """
    rnd = random.Random(seed)

    # Map image -> classes
    img2cls: Dict[Path, Set[int]] = {}
    class2imgs: Dict[int, Set[Path]] = defaultdict(set)
    for img in all_images:
        s = read_image_classes(labels_dir / f"{img.stem}.txt")
        img2cls[img] = s
        for cid in s:
            class2imgs[cid].add(img)

    # Count per-class images (images where class appears at least once)
    class_counts = {cid: len(imgs) for cid, imgs in class2imgs.items()}
    # Desired per-class validation image counts
    desired = {cid: int(round(val_ratio * cnt)) for cid, cnt in class_counts.items()}

    # Total validation images target
    val_target = int(round(len(all_images) * val_ratio))

    # Current achieved counts in val
    achieved = Counter()
    in_val: Set[Path] = set()
    unassigned: Set[Path] = set(all_images)

    # Helper to score a candidate image for the current needs
    def overshoot_after(img: Path) -> int:
        # Sum of overshoot across classes if we add this image
        score = 0
        for cid in img2cls[img]:
            new = achieved[cid] + 1
            over = new - desired.get(cid, 0)
            if over > 0:
                score += over
        return score

    # Loop while we still have classes needing validation coverage and slots left
    # Work list of classes ordered by remaining need (desc) and rarity (desc)
    def most_needed_class() -> int | None:
        remaining = []
        for cid, tgt in desired.items():
            need = tgt - achieved[cid]
            if need > 0:
                remaining.append((need, class_counts[cid], cid))
        if not remaining:
            return None
        # Prefer higher need, then rarer classes (higher class_counts gives more options,
        # but picking rarer earlier tends to stabilize)
        remaining.sort(key=lambda x: (-x[0], x[1]))
        return remaining[0][2]

    # Main greedy assignment
    while len(in_val) < val_target:
        cid = most_needed_class()
        if cid is None:
            break  # all class targets satisfied; fill remainder randomly
        # Candidates: unassigned images that contain this class
        candidates = [img for img in class2imgs.get(cid, set()) if img in unassigned]
        if not candidates:
            # Cannot satisfy this class further
            # Zero out its remaining desired to prevent infinite loop
            desired[cid] = achieved[cid]
            continue

        # Among candidates, prefer images that cover other needed classes too,
        # and that minimize overshoot.
        # Score = (overshoot_penalty, -num_needed_classes_covered, random_tiebreaker)
        needed_set = {k for k, v in desired.items() if v - achieved[k] > 0}
        best = None
        best_key = None
        for img in candidates:
            img_cls = img2cls[img]
            overs = overshoot_after(img)
            needed_cov = len(img_cls & needed_set)
            key = (overs, -needed_cov, rnd.random())
            if best is None or key < best_key:
                best, best_key = img, key

        # Assign best
        in_val.add(best)
        unassigned.remove(best)
        for k in img2cls[best]:
            achieved[k] += 1

    # If we still need more validation images to meet the global ratio, fill from leftovers
    if len(in_val) < val_target and unassigned:
        filler = list(unassigned)
        rnd.shuffle(filler)
        need = val_target - len(in_val)
        for img in filler[:need]:
            in_val.add(img)
            unassigned.remove(img)

    return in_val

def print_distribution_report(
    train_imgs: list[Path],
    val_imgs: list[Path],
    train_labels_dir: Path,
    val_labels_dir: Path,
):
    train_dist = summarize_distribution(train_imgs, train_labels_dir)
    val_dist   = summarize_distribution(val_imgs,   val_labels_dir)

    all_classes = sorted(set(train_dist.keys()) | set(val_dist.keys()))
    total_train = max(1, len(train_imgs))
    total_val   = max(1, len(val_imgs))

    print("\n=== Class distribution (image-level presence) ===")
    print(f"{'Class':>7} | {'Train n':>7} {'Train%':>7} | {'Val n':>7} {'Val%':>7} | ratio Val/Train")
    print("-"*68)
    for cid in all_classes:
        tn = train_dist.get(cid, 0)
        vn = val_dist.get(cid, 0)
        tp = 100.0 * tn / total_train
        vp = 100.0 * vn / total_val
        ratio = (vp / tp) if tp > 0 else (float('inf') if vp > 0 else 1.0)
        print(f"{cid:7d} | {tn:7d} {tp:6.2f}% | {vn:7d} {vp:6.2f}% | {ratio:6.2f}x")
    print("-"*68)
    print(f"Images: train={len(train_imgs)}  val={len(val_imgs)}\n")

# ------------------------ Main ------------------------
def main():
    args = parse_args()

    ROOT = Path(args.dataset_root).expanduser().resolve()
    pool_images = (ROOT / args.pool_images).resolve()
    pool_labels = (ROOT / args.pool_labels).resolve()
    dest_images_train = ROOT / "images" / "train"
    dest_images_val   = ROOT / "images" / "val"
    dest_labels_train = ROOT / "labels" / "train"
    dest_labels_val   = ROOT / "labels" / "val"

    exts = norm_exts(args.ext)
    ensure_dirs(ROOT)

    # Enumerate pool images
    all_images = find_images(pool_images, exts)
    if not all_images:
        print(f"No images found in {pool_images} with extensions {sorted(exts)}")
        return

    random.seed(args.seed)

    # Choose validation set (stratified or random)
    if args.no_stratify:
        random.shuffle(all_images)
        val_count = int(round(len(all_images) * args.val_ratio))
        val_images = set(all_images[:val_count])
    else:
        val_images = stratified_split(all_images, pool_labels, args.val_ratio, args.seed)

    train_images = [img for img in all_images if img not in val_images]
    val_images_list = sorted(val_images)

    print(f"Total images: {len(all_images)}  |  "
          f"Train target: {len(train_images)}  |  Val target: {len(val_images_list)}")
    if args.dry_run:
        print("Dry-run: no files will be written.")

    pool_is_dest_train = same_file(pool_images, dest_images_train)

    # -------- 1) Populate VAL --------
    for img_path in tqdm(val_images_list, desc=f"{args.mode.upper()} val images"):
        label_path = pool_labels / f"{img_path.stem}.txt"
        dest_img = dest_images_val / img_path.name
        dest_lbl = dest_labels_val / label_path.name
        place(img_path, dest_img, args.mode, args.dry_run)
        if label_path.exists():
            place(label_path, dest_lbl, args.mode, args.dry_run)

    # If pool is images/train, enforce authoritative split:
    if pool_is_dest_train:
        for img_path in tqdm(val_images_list, desc="Pruning val picks from images/train"):
            delete_if_exists(img_path, args.dry_run)
        for img_path in tqdm(val_images_list, desc="Pruning val picks from labels/train"):
            lbl = pool_labels / f"{img_path.stem}.txt"
            delete_if_exists(lbl, args.dry_run)

    # -------- 2) Populate TRAIN (only needed if pool != images/train OR mode != move) --------
    if not (pool_is_dest_train and args.mode == "move"):
        for img_path in tqdm(train_images, desc=f"{args.mode.UPPER() if hasattr(args.mode,'UPPER') else args.mode.upper()} train images"):
            label_path = pool_labels / f"{img_path.stem}.txt"
            dest_img = dest_images_train / img_path.name
            dest_lbl = dest_labels_train / label_path.name
            place(img_path, dest_img, args.mode, args.dry_run)
            if label_path.exists():
                place(label_path, dest_lbl, args.mode, args.dry_run)

    if not args.no_yaml:
        update_data_yaml(ROOT)

    ## Final sanity counts
    train_imgs_final = sorted([p for p in dest_images_train.glob("*") if p.is_file()])
    val_imgs_final   = sorted([p for p in dest_images_val.glob("*")   if p.is_file()])
    print(f"Done. images/train: {len(train_imgs_final)} | images/val: {len(val_imgs_final)}")

    # Accurate distribution report from destination label dirs
    print_distribution_report(
        train_imgs_final,
        val_imgs_final,
        dest_labels_train,
        dest_labels_val,
    )

if __name__ == "__main__":
    main()
