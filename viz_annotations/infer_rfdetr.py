#!/usr/bin/env python3
"""
RF-DETR inference visualiser — image and video.

Reads config.yaml from ../rf_detr/ to pick the model variant and default
checkpoint directory, then runs inference and displays / saves the result
with bounding boxes and (for seg variants) instance masks.

Usage
-----
# Single image
python infer_rfdetr.py --image /path/to/image.jpg

# Video — display only
python infer_rfdetr.py --video /path/to/video.mp4

# Video — save annotated output
python infer_rfdetr.py --video input.mp4 --save out.mp4

# Override checkpoint / threshold
python infer_rfdetr.py --image img.jpg --checkpoint ../runs/rf_detr/checkpoint_best_ema.pth --threshold 0.4

Video keyboard controls
-----------------------
  Space       pause / resume
  →           step one frame (while paused)
  q / Esc     quit

Run with the rf_detr venv:
  ../rf_detr/.venv/bin/python infer_rfdetr.py --image /path/to/image.jpg
  ../rf_detr/.venv/bin/python infer_rfdetr.py --video /path/to/video.mp4
"""

import argparse
import json
import sys
from pathlib import Path

# ── Make rfdetr importable without activating the venv ───────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_RF_DETR_DIR = _SCRIPT_DIR.parent / "rf_detr"
_VENV_SITE = _RF_DETR_DIR / ".venv" / "lib"
if _VENV_SITE.exists():
    for _sp in sorted(_VENV_SITE.glob("python3.*/site-packages")):
        if str(_sp) not in sys.path:
            sys.path.insert(0, str(_sp))

import cv2
import numpy as np
import yaml
from PIL import Image

# ── Model variant registry (mirrors train.py) ─────────────────────────────────
_MODEL_VARIANTS = {
    "nano":       "rfdetr.RFDETRNano",
    "small":      "rfdetr.RFDETRSmall",
    "medium":     "rfdetr.RFDETRMedium",
    "base":       "rfdetr.RFDETRBase",
    "large":      "rfdetr.RFDETRLarge",
    "seg-nano":   "rfdetr.RFDETRSegNano",
    "seg-small":  "rfdetr.RFDETRSegSmall",
    "seg-medium": "rfdetr.RFDETRSegMedium",
    "seg-large":  "rfdetr.RFDETRSegLarge",
}

_CHECKPOINT_PRIORITY = [
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "checkpoint.pth",
]


def _import_class(dotted: str):
    module_name, class_name = dotted.rsplit(".", 1)
    import importlib
    return getattr(importlib.import_module(module_name), class_name)


def _load_config():
    cfg_path = _RF_DETR_DIR / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def _find_checkpoint(runs_dir: Path) -> Path | None:
    for name in _CHECKPOINT_PRIORITY:
        p = runs_dir / name
        if p.exists():
            return p
    candidates = sorted(runs_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_class_names(dataset_dir: Path | None) -> dict[int, str]:
    if dataset_dir is None or not dataset_dir.exists():
        return {}
    for split in ("valid", "validation", "val", "train"):
        ann = dataset_dir / split / "_annotations.coco.json"
        if ann.exists():
            try:
                data = json.loads(ann.read_text())
                return {c["id"]: c["name"] for c in data.get("categories", [])}
            except Exception:
                pass
    return {}


def _color(class_id: int):
    hue = int((class_id * 137.508) % 180)
    hsv = np.array([[[hue, 255, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _draw_mask(image_bgr, mask, color, alpha=0.55):
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    overlay = image_bgr.copy()
    overlay[mask] = color
    result = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    return result


def annotate_frame(bgr: np.ndarray, predictions, class_names: dict) -> np.ndarray:
    """Draw predictions onto a BGR frame and return it."""
    if not hasattr(predictions, "xyxy"):
        return bgr

    xyxy       = predictions.xyxy
    confidence = getattr(predictions, "confidence", None)
    class_ids  = getattr(predictions, "class_id",  None)
    masks      = getattr(predictions, "mask",       None)

    n = len(xyxy)

    # Masks first so boxes sit on top
    if masks is not None:
        for i in range(n):
            cid  = int(class_ids[i]) if class_ids is not None else 0
            mask = np.asarray(masks[i])
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.dtype != np.bool_:
                mask = mask > 0.5
            bgr = _draw_mask(bgr, mask, _color(cid))

    for i in range(n):
        x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
        cid   = int(class_ids[i])   if class_ids  is not None else -1
        score = float(confidence[i]) if confidence is not None else None
        color = _color(cid)

        label = class_names.get(cid, f"class_{cid}")
        if score is not None:
            label += f" {score:.2f}"

        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bg_y1 = max(0, y1 - th - 8)
        cv2.rectangle(bgr, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(bgr, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return bgr


# ── Image mode ────────────────────────────────────────────────────────────────

def run_image(args, model, class_names):
    image_pil = Image.open(args.image).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    predictions = model.predict(image_pil, threshold=args.threshold)
    print(f"Detections: {len(predictions.xyxy)}")

    image_bgr = annotate_frame(image_bgr, predictions, class_names)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), image_bgr)
        print(f"Saved: {out}")

    if not args.no_show:
        cv2.namedWindow("RF-DETR Inference", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RF-DETR Inference", image_bgr.shape[1], image_bgr.shape[0])
        cv2.imshow("RF-DETR Inference", image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── Video mode ────────────────────────────────────────────────────────────────

def run_video(args, model, class_names):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h}  {fps:.1f} fps  {total_frames} frames")

    writer = None
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        print(f"Saving to: {out_path}")

    if not args.no_show:
        cv2.namedWindow("RF-DETR Inference", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RF-DETR Inference", w, h)

    paused      = False
    frame_idx   = 0
    frame_budget_ms = 1000 / fps   # target ms per frame

    print("Space=pause/resume  →=step  q/Esc=quit")

    while True:
        if not paused:
            ret, bgr = cap.read()
            if not ret:
                break  # end of video
            frame_idx += 1

            t_frame_start = cv2.getTickCount()

            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            predictions = model.predict(pil, threshold=args.threshold)
            annotated   = annotate_frame(bgr.copy(), predictions, class_names)

            # Overlay frame counter + live fps
            elapsed_ms = (cv2.getTickCount() - t_frame_start) / cv2.getTickFrequency() * 1000
            live_fps   = 1000 / elapsed_ms if elapsed_ms > 0 else 0
            status = (f"frame {frame_idx}/{total_frames}"
                      f"  dets={len(predictions.xyxy)}"
                      f"  {elapsed_ms:.0f}ms ({live_fps:.1f}fps)")
            cv2.putText(annotated, status, (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(annotated)

            if not args.no_show:
                cv2.imshow("RF-DETR Inference", annotated)

            # Wait only the remaining budget so display tracks source fps
            wait_ms = max(1, int(frame_budget_ms - elapsed_ms))
        else:
            wait_ms = 1

        # Keyboard handling
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord('q'), 27):           # q / Esc → quit
            break
        elif key == ord(' '):               # Space → pause/resume
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == 83 and paused:          # → while paused → step one frame
            ret, bgr = cap.read()
            if ret:
                frame_idx += 1
                pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                predictions = model.predict(pil, threshold=args.threshold)
                annotated   = annotate_frame(bgr.copy(), predictions, class_names)
                status = f"frame {frame_idx}/{total_frames}  dets={len(predictions.xyxy)}  [paused]"
                cv2.putText(annotated, status, (8, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
                if writer is not None:
                    writer.write(annotated)
                if not args.no_show:
                    cv2.imshow("RF-DETR Inference", annotated)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved: {args.save}")
    if not args.no_show:
        cv2.destroyAllWindows()
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    cfg = _load_config()
    model_cfg = cfg.get("model", {})
    local_cfg = cfg.get("local", {})

    default_variant = str(model_cfg.get("variant", "seg-nano")).lower()
    default_runs    = (_RF_DETR_DIR / Path(local_cfg.get("output_dir", "../runs/rf_detr"))).resolve()
    default_dataset = local_cfg.get("dataset_dir")

    parser = argparse.ArgumentParser(description="RF-DETR inference visualiser")

    # Input — mutually exclusive
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--image", help="Input image path")
    inp.add_argument("--video", help="Input video path")

    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint .pth (default: best in runs dir)")
    parser.add_argument("--variant", default=default_variant,
                        choices=list(_MODEL_VARIANTS), help="Model variant")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Confidence threshold (default: 0.3)")
    parser.add_argument("--save", default=None,
                        help="Save annotated output (image or video path)")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip display window (useful with --save)")
    args = parser.parse_args()

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    checkpoint = Path(args.checkpoint) if args.checkpoint else _find_checkpoint(default_runs)
    if checkpoint is None or not checkpoint.exists():
        print(f"ERROR: No checkpoint found in {default_runs}")
        print("Pass --checkpoint /path/to/checkpoint.pth explicitly.")
        return 1

    print(f"Variant   : {args.variant}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Threshold : {args.threshold}")

    # ── Class names ───────────────────────────────────────────────────────────
    dataset_dir = Path(default_dataset).expanduser() if default_dataset else None
    class_names = _load_class_names(dataset_dir)
    if class_names:
        print(f"Classes   : {class_names}")

    # ── Load model ────────────────────────────────────────────────────────────
    ModelClass = _import_class(_MODEL_VARIANTS[args.variant])
    model = ModelClass(pretrain_weights=str(checkpoint))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.optimize_for_inference()

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.image:
        return run_image(args, model, class_names) or 0
    else:
        return run_video(args, model, class_names) or 0


if __name__ == "__main__":
    sys.exit(main())
