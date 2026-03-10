#!/usr/bin/env python3
"""
Local YOLO Training Application
Simplified training pipeline for YOLO models without cloud dependencies.
"""

import os
import csv
import yaml
import torch
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from ultralytics import YOLO
from tqdm import tqdm


# ---------------------------
# Configuration
# ---------------------------

class TrainingConfig:
    """Centralized configuration management"""

    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Model settings
        self.model_type = str(self.config["model"]["type"]).lower()
        self.model_task = self._normalize_task(str(self.config["model"]["task"]).lower())

        # Local paths
        local_cfg = self.config["local"]
        self.dataset_dir = Path(local_cfg["dataset_dir"]).expanduser().resolve()
        self.weights_dir = Path(local_cfg["weights_dir"]).expanduser().resolve()
        self.output_dir = Path(local_cfg["output_dir"]).expanduser().resolve()

        # Dataset settings
        dataset_cfg = self.config["dataset"]
        self.relative_train = dataset_cfg["relative_train_img_path"]
        self.relative_val = dataset_cfg["relative_val_img_path"]
        self.balance_config = dataset_cfg.get("balance", {})

        # Training settings
        train_cfg = self.config["training"]
        self.weights_version = train_cfg["weights_version"]
        self.training_args = {k: v for k, v in train_cfg.items() if k != "weights_version"}

        # Derived paths
        self.data_yaml_path = self.dataset_dir / "data.yaml"
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.run_name = f"{self.model_type}_{self.timestamp}"

    def _normalize_task(self, task: str) -> str:
        """Normalize task names to standard format"""
        task = task.strip().lower()
        if task in {"detect", "detection", "det"}:
            return "detect"
        if task in {"segment", "seg", "segmentation"}:
            return "segment"
        if task in {"obb", "oriented", "oriented_box", "oriented_bbox"}:
            return "obb"
        return task


# ---------------------------
# Dataset Balancing
# ---------------------------

class DatasetBalancer:
    """Handle dataset balancing via oversampling"""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, dataset_root: Path, data_yaml_path: Path):
        self.dataset_root = dataset_root
        self.data_yaml_path = data_yaml_path
        self.names = self._read_class_names()

    def _read_class_names(self) -> list[str]:
        """Read class names from data.yaml"""
        with open(self.data_yaml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[i] for i in range(len(names))]
        return list(names)

    def balance_dataset(
        self,
        train_img_root: Path,
        config: dict,
        run_name: str
    ) -> Path | None:
        """
        Create balanced training list via oversampling.

        Returns path to balanced list file or None if balancing disabled.
        """
        if not config.get("enabled", False):
            return None

        if config.get("mode", "oversample").lower() != "oversample":
            return None

        print("[Balance] Building balanced training list...")

        # Find all images and their classes
        img_to_classes, class_to_imgs = self._analyze_dataset(train_img_root)

        if not img_to_classes:
            print("[Balance] No images found")
            return None

        # Count images per class
        base_counts = Counter({cid: len(imgs) for cid, imgs in class_to_imgs.items()})
        for cid in range(len(self.names)):
            base_counts.setdefault(cid, 0)

        # Determine target count
        target = self._calculate_target(base_counts, config)
        print(f"[Balance] Target images per class: {target}")

        # Build balanced list with oversampling
        balanced_list = self._oversample_images(
            img_to_classes,
            class_to_imgs,
            base_counts,
            target,
            config.get("max_dup_per_image", 10)
        )

        # Save results
        debug_dir = self.dataset_root / "runs/train/balance_debug" / run_name
        if config.get("save_debug_hist", True):
            self._save_histograms(debug_dir, base_counts, balanced_list, img_to_classes)

        list_path = self.dataset_root / "lists" / f"train_balanced_{run_name}.txt"
        self._write_image_list(balanced_list, list_path)

        print(f"[Balance] Original: {len(img_to_classes)} -> Balanced: {len(balanced_list)}")
        return list_path

    def _analyze_dataset(self, train_img_root: Path) -> tuple[dict, dict]:
        """Analyze dataset to find image-class relationships"""
        img_to_classes = {}
        class_to_imgs = defaultdict(list)

        for img in self._find_images(train_img_root):
            lbl_path = self._get_label_path(img)
            class_ids = self._parse_label_file(lbl_path)
            img_to_classes[img] = class_ids
            for cid in class_ids:
                class_to_imgs[cid].append(img)

        return img_to_classes, class_to_imgs

    def _find_images(self, root: Path) -> list[Path]:
        """Find all images in directory"""
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in self.IMG_EXTS]

    def _get_label_path(self, img_path: Path) -> Path:
        """Convert image path to label path"""
        parts = list(img_path.parts)
        try:
            idx = parts.index("images")
            parts[idx] = "labels"
            parts[-1] = img_path.stem + ".txt"
            return Path(*parts)
        except ValueError:
            # Fallback
            rel = img_path.relative_to(self.dataset_root)
            return self.dataset_root / "labels" / rel.parents[1].name / (img_path.stem + ".txt")

    def _parse_label_file(self, lbl_path: Path) -> set[int]:
        """Parse YOLO label file and return class IDs"""
        class_ids = set()
        if not lbl_path.exists():
            return class_ids

        with open(lbl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pieces = line.split()
                    class_ids.add(int(float(pieces[0])))
                except Exception:
                    continue

        return class_ids

    def _calculate_target(self, counts: Counter, config: dict) -> int:
        """Calculate target number of images per class"""
        nonzero = [c for c in counts.values() if c > 0]
        if not nonzero:
            return 0

        target_field = config.get("target", "max")

        if isinstance(target_field, int):
            return int(target_field)

        target_str = str(target_field).lower()
        if target_str == "max":
            return max(nonzero)
        elif target_str == "median":
            return int(sorted(nonzero)[len(nonzero) // 2])
        else:
            return max(nonzero)

    def _oversample_images(
        self,
        img_to_classes: dict,
        class_to_imgs: dict,
        base_counts: Counter,
        target: int,
        max_dup: int
    ) -> list[Path]:
        """Oversample minority classes to reach target"""
        balanced = list(img_to_classes.keys())
        rng = random.Random(20250831)

        for cid in range(len(self.names)):
            have = base_counts.get(cid, 0)
            need = max(0, target - have)
            if need == 0:
                continue

            pool = class_to_imgs.get(cid, [])
            if not pool:
                continue

            dup_counter = Counter()
            added = 0

            while added < need:
                img = rng.choice(pool)
                if dup_counter[img] < max_dup:
                    balanced.append(img)
                    dup_counter[img] += 1
                    added += 1

                # Check if all images are saturated
                if len(dup_counter) == len(pool) and all(v >= max_dup for v in dup_counter.values()):
                    break

            if added < need:
                cname = self.names[cid] if cid < len(self.names) else f"class_{cid}"
                print(f"[Balance] Warning: class {cid} '{cname}' added {added}/{need} (hit duplication limit)")

        return balanced

    def _save_histograms(self, debug_dir: Path, before: Counter, balanced_list: list, img_to_classes: dict):
        """Save before/after histograms"""
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Before histogram
        self._write_histogram_csv(debug_dir / "hist_before.csv", before)

        # After histogram
        after = Counter({cid: 0 for cid in range(len(self.names))})
        cache = {}
        for img in balanced_list:
            if img not in cache:
                cache[img] = img_to_classes.get(img, set())
            for cid in cache[img]:
                after[cid] += 1

        self._write_histogram_csv(debug_dir / "hist_after.csv", after)

    def _write_histogram_csv(self, path: Path, counts: Counter):
        """Write histogram to CSV"""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "image_count"])
            for i in range(len(self.names)):
                name = self.names[i] if i < len(self.names) else f"class_{i}"
                writer.writerow([i, name, counts.get(i, 0)])

    def _write_image_list(self, paths: list[Path], dest: Path):
        """Write image list to text file"""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            for p in paths:
                f.write(str(p) + "\n")
        print(f"[Balance] Wrote list: {dest} ({len(paths)} images)")


# ---------------------------
# Model Training
# ---------------------------

class YOLOTrainer:
    """Handle YOLO model training workflow"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.run_dir = None
        self.model = None

    def prepare_environment(self):
        """Prepare local directories and data.yaml"""
        print("=" * 60)
        print("Preparing environment...")
        print("=" * 60)

        # Verify dataset exists
        if not self.config.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.config.dataset_dir}")

        if not self.config.data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {self.config.data_yaml_path}")

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Update data.yaml paths
        self._update_data_yaml()

        print(f"Dataset: {self.config.dataset_dir}")
        print(f"Output: {self.config.output_dir}")

    def _update_data_yaml(self):
        """Update data.yaml with absolute paths"""
        with open(self.config.data_yaml_path, "r") as f:
            data = yaml.safe_load(f)

        data["train"] = str(self.config.dataset_dir / self.config.relative_train)
        data["val"] = str(self.config.dataset_dir / self.config.relative_val)

        with open(self.config.data_yaml_path, "w") as f:
            yaml.dump(data, f)

    def apply_balancing(self) -> bool:
        """Apply dataset balancing if configured"""
        print("=" * 60)
        print("Checking dataset balancing...")
        print("=" * 60)

        balancer = DatasetBalancer(self.config.dataset_dir, self.config.data_yaml_path)
        train_img_root = self.config.dataset_dir / self.config.relative_train

        balanced_list = balancer.balance_dataset(
            train_img_root,
            self.config.balance_config,
            self.config.run_name
        )

        if balanced_list:
            # Update data.yaml to use balanced list
            with open(self.config.data_yaml_path, "r") as f:
                data = yaml.safe_load(f)
            data["train"] = str(balanced_list)
            with open(self.config.data_yaml_path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            print(f"[Balance] Using balanced list: {balanced_list}")
            return True
        else:
            print("[Balance] Using original dataset (balancing disabled or not applicable)")
            return False

    def load_model(self):
        """Load YOLO model with appropriate weights"""
        print("=" * 60)
        print("Loading model...")
        print("=" * 60)

        weights_file = f"{self.config.weights_version}.pt"
        weights_path = self.config.weights_dir / weights_file

        # Check if weights exist locally
        if not weights_path.exists():
            print(f"Weights not found at {weights_path}")
            print(f"Attempting to download from Ultralytics: {self.config.weights_version}.pt")
            # Ultralytics will automatically download if not found
            weights_path = self.config.weights_version + ".pt"
        else:
            weights_path = str(weights_path)

        has_seg_suffix = self.config.weights_version.endswith("-seg")
        has_obb_suffix = self.config.weights_version.endswith("-obb")

        if self.config.model_task == "segment":
            if has_seg_suffix:
                print(f"Loading segmentation model from: {weights_file}")
                self.model = YOLO(str(weights_path), task="segment")
            else:
                # Weights don't have -seg suffix, check for yaml architecture file
                # Extract base model name without size suffix (e.g., yolo12n -> yolo12)
                base_model = self.config.weights_version.rstrip("nslmx")
                yaml_file = f"{base_model}-seg.yaml"
                yaml_path = self.config.weights_dir / yaml_file

                if yaml_path.exists():
                    # Use yaml file to convert detection weights to segmentation
                    print(f"Loading segmentation model from: {yaml_file} + {weights_file}")
                    self.model = YOLO(str(yaml_path)).load(str(weights_path))
                else:
                    # Yaml not found, try loading detection weights directly with task="segment"
                    print(f"Loading segmentation model (converting detection weights directly)")
                    print(f"Note: {yaml_file} not found, attempting direct conversion")
                    self.model = YOLO(str(weights_path), task="segment")

        elif self.config.model_task == "obb":
            if has_obb_suffix:
                print(f"Loading OBB model from: {weights_file}")
                self.model = YOLO(weights_path, task="obb")
            else:
                print(f"Loading OBB model (converting detection weights)")
                yaml_file = f"{self.config.weights_version}-obb.yaml"
                yaml_path = self.config.weights_dir / yaml_file
                if yaml_path.exists():
                    self.model = YOLO(str(yaml_path)).load(weights_path)
                else:
                    print(f"Warning: {yaml_file} not found, using detection weights directly")
                    self.model = YOLO(weights_path, task="obb")

        else:  # detection
            print(f"Loading detection model from: {weights_file}")
            self.model = YOLO(weights_path)

    def train(self):
        """Run training"""
        print("=" * 60)
        print("Starting training...")
        print("=" * 60)

        torch.cuda.empty_cache()

        # Prepare training arguments
        train_args = self.config.training_args.copy()
        train_args.update({
            "data": str(self.config.data_yaml_path),
            "project": str(self.config.output_dir / "runs/train"),
            "name": self.config.run_name,
        })

        if "task" not in train_args:
            train_args["task"] = self.config.model_task

        # Train model
        self.model.train(**train_args)

        # Set run directory
        self.run_dir = self.config.output_dir / "runs/train" / self.config.run_name

    def export_onnx(self) -> bool:
        """Export model to ONNX format"""
        print("=" * 60)
        print("Exporting to ONNX...")
        print("=" * 60)

        try:
            # Get the trained model path
            best_pt = self.run_dir / "weights" / "best.pt"

            # Load the trained model for export
            export_model = YOLO(str(best_pt))

            # Export with appropriate settings
            export_model.export(
                format="onnx",
                imgsz=self.config.training_args.get("imgsz", 640),
                simplify=True,
                opset=17,
                dynamic=False
            )

            onnx_path = self.run_dir / "weights" / "best.onnx"
            if onnx_path.exists():
                print(f"ONNX export successful: {onnx_path}")
                return True
            else:
                print("ONNX export completed but file not found")
                return False

        except Exception as e:
            print(f"ONNX export failed: {e}")
            return False

    def save_artifacts(self):
        """Save training artifacts information"""
        print("=" * 60)
        print("Training artifacts saved locally")
        print("=" * 60)

        if self.run_dir and self.run_dir.exists():
            print(f"Run directory: {self.run_dir}")

            weights_dir = self.run_dir / "weights"
            if weights_dir.exists():
                pt_path = weights_dir / "best.pt"
                onnx_path = weights_dir / "best.onnx"

                print(f"\nModel files:")
                if pt_path.exists():
                    print(f"  - PyTorch: {pt_path}")
                if onnx_path.exists():
                    print(f"  - ONNX: {onnx_path}")

            plots = list(self.run_dir.glob("*.png"))
            if plots:
                print(f"\nPlots: {len(plots)} files in {self.run_dir}")
        else:
            print("Run directory not found")


# ---------------------------
# Main Entry Point
# ---------------------------

def main():
    """Main training workflow"""
    config_path = Path(__file__).resolve().parent / "config.yaml"

    if not config_path.exists():
        print(f"Error: config.yaml not found at {config_path}")
        print("Please create config.yaml with your training settings")
        return 1

    # Initialize components
    config = TrainingConfig(config_path)
    trainer = YOLOTrainer(config)

    # Execute training pipeline
    try:
        trainer.prepare_environment()
        trainer.apply_balancing()
        trainer.load_model()
        trainer.train()
        trainer.export_onnx()
        trainer.save_artifacts()

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
