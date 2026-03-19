#!/usr/bin/env python3
"""
YOLO inference visualiser — image and video.

Reads config.yaml from ../yolo/ to pick the model task and default
checkpoint directory, then runs inference and displays / saves the result
with bounding boxes and (for seg variants) instance masks.

Usage
-----
# Single image
python infer_yolo.py --image /path/to/image.jpg

# Video — display only
python infer_yolo.py --video /path/to/video.mp4

# Video — save annotated output
python infer_yolo.py --video input.mp4 --save out.mp4

# Override checkpoint / threshold / input size
python infer_yolo.py --image img.jpg --checkpoint ../runs/train/run/weights/best.pt --threshold 0.4 --imgsz 640

Video keyboard controls
-----------------------
  Space       pause / resume
  →           step one frame (while paused)
  q / Esc     quit

Run with the yolo venv:
  ../yolo/.venv/bin/python infer_yolo.py --image /path/to/image.jpg
  ../yolo/.venv/bin/python infer_yolo.py --video /path/to/video.mp4
"""

import argparse
import sys
from pathlib import Path

# ── Make ultralytics importable without activating the venv ──────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_YOLO_DIR = _SCRIPT_DIR.parent / "yolo"
_VENV_SITE = _YOLO_DIR / ".venv" / "lib"
if _VENV_SITE.exists():
    for _sp in sorted(_VENV_SITE.glob("python3.*/site-packages")):
        if str(_sp) not in sys.path:
            sys.path.insert(0, str(_sp))

import cv2
import numpy as np
import yaml


def _load_config() -> dict:
    cfg_path = _YOLO_DIR / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def _find_checkpoint(runs_dir: Path) -> Path | None:
    """Search runs/train/* subdirectories for best.pt, preferring the newest run."""
    train_dir = runs_dir / "runs" / "train"
    if not train_dir.exists():
        return None

    # Collect all best.pt / last.pt candidates across run subdirs
    for name in ("best.pt", "last.pt"):
        candidates = sorted(
            train_dir.glob(f"*/weights/{name}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def _color(class_id: int) -> tuple[int, int, int]:
    hue = int((class_id * 137.508) % 180)
    hsv = np.array([[[hue, 255, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _draw_mask(image_bgr: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.55) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    overlay = image_bgr.copy()
    overlay[mask] = color
    result = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    return result


def annotate_frame(bgr: np.ndarray, result, class_names: dict) -> np.ndarray:
    """Draw ultralytics Result predictions onto a BGR frame and return it."""
    boxes = result.boxes
    masks = result.masks  # None for detection models

    if boxes is None or len(boxes) == 0:
        return bgr

    h, w = bgr.shape[:2]

    xyxy       = boxes.xyxy.cpu().numpy()       # (N, 4)
    confidence = boxes.conf.cpu().numpy()        # (N,)
    class_ids  = boxes.cls.cpu().numpy().astype(int)  # (N,)

    n = len(xyxy)

    # Masks first so boxes sit on top
    if masks is not None:
        mask_data = masks.data.cpu().numpy()  # (N, H_mask, W_mask)
        for i in range(n):
            cid  = class_ids[i]
            mask = mask_data[i]
            # Resize mask to frame size if needed
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            bgr = _draw_mask(bgr, mask > 0.5, _color(cid))

    for i in range(n):
        x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
        cid   = class_ids[i]
        score = float(confidence[i])
        color = _color(cid)

        label = class_names.get(cid, f"class_{cid}")
        label += f" {score:.2f}"

        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bg_y1 = max(0, y1 - th - 8)
        cv2.rectangle(bgr, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(bgr, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return bgr


# ── Image mode ────────────────────────────────────────────────────────────────

def run_image(args, model, class_names: dict) -> int:
    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"ERROR: Cannot read image: {args.image}")
        return 1

    predict_kwargs = dict(conf=args.threshold, verbose=False)
    if args.imgsz is not None:
        predict_kwargs["imgsz"] = args.imgsz
    results = model.predict(bgr, **predict_kwargs)
    result  = results[0]
    n_dets  = len(result.boxes) if result.boxes is not None else 0
    print(f"Detections: {n_dets}")

    annotated = annotate_frame(bgr.copy(), result, class_names)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), annotated)
        print(f"Saved: {out}")

    if not args.no_show:
        cv2.namedWindow("YOLO Inference", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Inference", annotated.shape[1], annotated.shape[0])
        cv2.imshow("YOLO Inference", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


# ── Video mode ────────────────────────────────────────────────────────────────

def run_video(args, model, class_names: dict) -> int:
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
        cv2.namedWindow("YOLO Inference", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Inference", w, h)

    paused          = False
    frame_idx       = 0
    frame_budget_ms = 1000 / fps

    predict_kwargs = dict(conf=args.threshold, verbose=False)
    if args.imgsz is not None:
        predict_kwargs["imgsz"] = args.imgsz

    print("Space=pause/resume  →=step  q/Esc=quit")

    while True:
        if not paused:
            ret, bgr = cap.read()
            if not ret:
                break
            frame_idx += 1

            t_frame_start = cv2.getTickCount()

            results  = model.predict(bgr, **predict_kwargs)
            result   = results[0]
            n_dets   = len(result.boxes) if result.boxes is not None else 0
            annotated = annotate_frame(bgr.copy(), result, class_names)

            elapsed_ms = (cv2.getTickCount() - t_frame_start) / cv2.getTickFrequency() * 1000
            live_fps   = 1000 / elapsed_ms if elapsed_ms > 0 else 0
            status = (f"frame {frame_idx}/{total_frames}"
                      f"  dets={n_dets}"
                      f"  {elapsed_ms:.0f}ms ({live_fps:.1f}fps)")
            cv2.putText(annotated, status, (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(annotated)

            if not args.no_show:
                cv2.imshow("YOLO Inference", annotated)

            wait_ms = max(1, int(frame_budget_ms - elapsed_ms))
        else:
            wait_ms = 1

        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == 83 and paused:   # → while paused → step one frame
            ret, bgr = cap.read()
            if ret:
                frame_idx += 1
                results   = model.predict(bgr, **predict_kwargs)
                result    = results[0]
                n_dets    = len(result.boxes) if result.boxes is not None else 0
                annotated = annotate_frame(bgr.copy(), result, class_names)
                status = f"frame {frame_idx}/{total_frames}  dets={n_dets}  [paused]"
                cv2.putText(annotated, status, (8, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
                if writer is not None:
                    writer.write(annotated)
                if not args.no_show:
                    cv2.imshow("YOLO Inference", annotated)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved: {args.save}")
    if not args.no_show:
        cv2.destroyAllWindows()
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    cfg       = _load_config()
    local_cfg = cfg.get("local", {})

    default_runs = (_YOLO_DIR / Path(local_cfg.get("output_dir", "./runs"))).resolve()

    parser = argparse.ArgumentParser(description="YOLO inference visualiser")

    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--image", help="Input image path")
    inp.add_argument("--video", help="Input video path")

    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint .pt path (default: newest best.pt in runs dir)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Confidence threshold (default: 0.3)")
    parser.add_argument("--save", default=None,
                        help="Save annotated output (image or video path)")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip display window (useful with --save)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Inference input size in pixels (e.g. 320, 640, 1280). "
                             "Overrides the model's default. Must be a multiple of 32.")
    args = parser.parse_args()

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    checkpoint = Path(args.checkpoint) if args.checkpoint else _find_checkpoint(default_runs)
    if checkpoint is None or not checkpoint.exists():
        print(f"ERROR: No checkpoint found in {default_runs}")
        print("Pass --checkpoint /path/to/best.pt explicitly.")
        return 1

    print(f"Checkpoint: {checkpoint}")
    print(f"Threshold : {args.threshold}")
    print(f"Imgsz     : {args.imgsz or 'model default'}")

    # ── Load model ────────────────────────────────────────────────────────────
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))

    # ── Class names ───────────────────────────────────────────────────────────
    class_names: dict[int, str] = model.names or {}
    if class_names:
        print(f"Classes   : {class_names}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.image:
        return run_image(args, model, class_names)
    else:
        return run_video(args, model, class_names)


if __name__ == "__main__":
    sys.exit(main())
