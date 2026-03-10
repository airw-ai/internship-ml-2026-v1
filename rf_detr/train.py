#!/usr/bin/env python3
"""
RF-DETR Training Script
Training pipeline for RF-DETR object detection / instance segmentation models.
Dataset format: COCO JSON (see ../dataset/convert_yolo_to_coco.py to convert from YOLO).
"""

import os
import warnings
import yaml
from pathlib import Path

# Suppress the benign numpy RuntimeWarning that rfdetr's evaluation emits when
# np.nanmean is called on an all-NaN precision array (happens for classes that
# have no valid AP yet, e.g. early in training).  The calling code already
# checks np.isnan() on the result and skips those classes safely.
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in",
    category=RuntimeWarning,
)

# Model variant map
MODEL_VARIANTS = {
    "nano":   ("rfdetr.RFDETRNano",   False),
    "small":  ("rfdetr.RFDETRSmall",  False),
    "medium": ("rfdetr.RFDETRMedium", False),
    "base":   ("rfdetr.RFDETRBase",   False),
    "large":  ("rfdetr.RFDETRLarge",  False),
    # segmentation variants
    "seg-nano":    ("rfdetr.RFDETRSegNano",    True),
    "seg-small":   ("rfdetr.RFDETRSegSmall",   True),
    "seg-medium":  ("rfdetr.RFDETRSegMedium",  True),
    "seg-large":   ("rfdetr.RFDETRSegLarge",   True),
    "seg-xlarge":  ("rfdetr.RFDETRSegXLarge",  True),
    "seg-2xlarge": ("rfdetr.RFDETRSeg2XLarge", True),
}


class TrainingConfig:
    """Load and validate training configuration from YAML."""

    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config["model"]
        self.variant = str(model_cfg["variant"]).lower()

        local_cfg = self.config["local"]
        self.dataset_dir = Path(local_cfg["dataset_dir"]).expanduser().resolve()
        self.output_dir = Path(local_cfg["output_dir"]).expanduser().resolve()

        self.train_args = self.config.get("training", {})
        self.export_cfg = self.config.get("export", {})

    def validate(self):
        if self.variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model variant '{self.variant}'. "
                f"Choose from: {list(MODEL_VARIANTS.keys())}"
            )
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")


def _import_model_class(dotted_name: str):
    """Import a class from a dotted module path."""
    module_name, class_name = dotted_name.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def main():
    config_path = Path(__file__).resolve().parent / "config.yaml"

    if not config_path.exists():
        print(f"Error: config.yaml not found at {config_path}")
        print("Copy config.example.yaml to config.yaml and edit it.")
        return 1

    cfg = TrainingConfig(config_path)

    try:
        cfg.validate()
    except (ValueError, FileNotFoundError) as e:
        print(f"Configuration error: {e}")
        return 1

    dotted_name, is_seg = MODEL_VARIANTS[cfg.variant]
    ModelClass = _import_model_class(dotted_name)

    print("=" * 60)
    print(f"RF-DETR Training  |  variant: {cfg.variant}")
    print(f"Dataset : {cfg.dataset_dir}")
    print(f"Output  : {cfg.output_dir}")
    print("=" * 60)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    model = ModelClass()

    train_kwargs = {
        "dataset_dir":  str(cfg.dataset_dir),
        "dataset_file": "roboflow",   # roboflow layout: train/_annotations.coco.json
        "output_dir":   str(cfg.output_dir),
    }
    train_kwargs.update(cfg.train_args)

    model.train(**train_kwargs)

    # ── ONNX export ──────────────────────────────────────────────────────────
    # Only the main process exports (rank 0 in DDP; always rank 0 in single-GPU).
    # After model.train() the model already holds the best weights (RF-DETR
    # swaps in the EMA/best state at the end of the training loop).
    is_main = os.environ.get("RANK", "0") == "0"
    export_enabled = cfg.export_cfg.get("enabled", True)

    if is_main and export_enabled:
        onnx_dir = cfg.output_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("Exporting to ONNX...")
        print(f"Output : {onnx_dir}")
        print("=" * 60)

        try:
            model.export(
                output_dir=str(onnx_dir),
                opset_version=cfg.export_cfg.get("opset_version", 17),
                batch_size=cfg.export_cfg.get("batch_size", 1),
                simplify=cfg.export_cfg.get("simplify", False),
            )
            print(f"ONNX export saved to: {onnx_dir}/inference_model.onnx")
        except Exception as e:
            print(f"ONNX export failed: {e}")
            print("Training weights are still saved in the output directory.")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Outputs saved to: {cfg.output_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
