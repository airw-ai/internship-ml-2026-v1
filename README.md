# Internship ML Project 2026

Machine Learning internship project for training YOLO segmentation models with comprehensive dataset balancing and augmentation support.

## Project Structure

This project provides a complete YOLO segmentation training pipeline with visualization tools:

```
internship-ml-2026-v1/
├── model_training/            # YOLO segmentation training pipeline
│   ├── train.py              # Main training script
│   ├── config.example.yaml   # Comprehensive config template with all parameters
│   ├── requirements.txt      # Python dependencies
│   └── README.md             # Detailed training guide
│
├── viz_annotations/           # Dataset visualization tools
│   └── visualize_yolo_dataset.py  # Visualize YOLO segmentation annotations
│
├── dataset/                   # Place your datasets here (download from Google Drive)
│   └── internship_dataset/   # Your YOLO segmentation dataset
│
└── weights/                   # Place pretrained weights here (download from Google Drive)
    └── yolo11s-seg.pt        # Example: YOLO11s segmentation weights
```

## Project Overview

### YOLO Segmentation Training

A comprehensive local training pipeline for YOLO segmentation models with advanced features:

- **Segmentation Focus**: Train models to predict precise pixel-level masks
- **Dataset Class Balancing**: Automatic oversampling to balance class distribution
- **Comprehensive Augmentation**: 30+ augmentation parameters (geometric, color, advanced)
- **Flexible Configuration**: YAML-based configuration with 80+ tunable parameters
- **Model Export**: Automatic ONNX export for deployment
- **Training Monitoring**: Real-time metrics, plots, and validation
- **Multi-GPU Support**: Scale training across multiple GPUs
- **Resume Training**: Continue from checkpoints

**Location**: `model_training/`

**See**: [model_training/README.md](model_training/README.md) for detailed instructions.

### Dataset Visualization

Tools for visualizing and inspecting YOLO segmentation dataset annotations:

- Visualize segmentation masks with color-coded classes
- Verify dataset integrity before training
- Inspect class distribution and annotation quality
- Preview augmentations

**Location**: `viz_annotations/`

**See**: [Dataset Visualization](#dataset-visualization-1) section below for usage instructions.

## Dataset Visualization

Before training, it's **highly recommended** to visualize your dataset to verify annotations and understand your data quality.

### Using the Visualization Tool

The `viz_annotations/visualize_yolo_dataset.py` script provides an interactive viewer for your YOLO segmentation dataset.

#### Basic Usage

```bash
cd viz_annotations

# Visualize your dataset
python3 visualize_yolo_dataset.py \
  --images ../dataset/internship_dataset/images/train \
  --labels ../dataset/internship_dataset/labels/train \
  --data-yaml ../dataset/internship_dataset/data.yaml
```

#### Features

- **Color-coded segmentation masks** with transparent overlays
- **Class labels** displayed at object centroids
- **Tiered rendering** (background → objects → lines) for proper overlap
- **Interactive controls** for navigating and filtering
- **Auto-contrast text** for readability on any color
- **Adjustable transparency** to see underlying image

#### Interactive Controls

| Key | Action |
|-----|--------|
| `d` or `→` | Next image |
| `a` or `←` | Previous image |
| `q` or `ESC` | Quit |
| `1` | Toggle background classes |
| `2` | Toggle object classes |
| `3` | Toggle line marking classes |
| `b` | Toggle background only |
| `l` | Show lines only |
| `+` or `=` | Increase mask opacity |
| `-` or `_` | Decrease mask opacity |

#### Advanced Options

**Ignore specific classes:**
```bash
python3 visualize_yolo_dataset.py \
  --images ../dataset/internship_dataset/images/train \
  --labels ../dataset/internship_dataset/labels/train \
  --data-yaml ../dataset/internship_dataset/data.yaml \
  --ignore-classes "sky,building"
```

**Custom class grouping:**
```bash
# Specify which classes are backgrounds vs lines
python3 visualize_yolo_dataset.py \
  --images ../dataset/internship_dataset/images/train \
  --labels ../dataset/internship_dataset/labels/train \
  --data-yaml ../dataset/internship_dataset/data.yaml \
  --bg-names "road,sidewalk,parking" \
  --line-names "lane_marking,crosswalk"
```

**Use regex for automatic grouping:**
```bash
# Automatically detect backgrounds and lines using patterns
python3 visualize_yolo_dataset.py \
  --images ../dataset/internship_dataset/images/train \
  --labels ../dataset/internship_dataset/labels/train \
  --data-yaml ../dataset/internship_dataset/data.yaml \
  --bg-regex "(road|sidewalk|sky|vegetation)" \
  --line-regex "(line|marking|zebra)"
```

#### What to Look For

✅ **Good annotations:**
- Masks follow object boundaries precisely
- No overlapping polygons of same class
- Consistent labeling across images
- No missing objects

## Data Augmentation

Data augmentation is a critical technique for improving model generalization and robustness by artificially expanding the training dataset through various transformations. The training pipeline supports a comprehensive set of augmentation techniques that can be configured in `model_training/config.yaml`.

### Why Data Augmentation Matters

- **Prevents Overfitting**: Increases training data diversity without collecting new data
- **Improves Generalization**: Helps models perform well on unseen data with different conditions
- **Simulates Real-World Variations**: Mimics lighting, perspective, and orientation changes
- **Increases Robustness**: Makes models resilient to image quality variations and challenging scenarios

### Augmentation Categories

#### Color Augmentation
Modifies color properties to simulate different lighting and environmental conditions:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `hsv_h` | float | 0.015 | 0.0-1.0 | Adjusts hue by fraction of color wheel for color variability |
| `hsv_s` | float | 0.7 | 0.0-1.0 | Alters saturation to simulate environmental conditions |
| `hsv_v` | float | 0.4 | 0.0-1.0 | Modifies brightness for various lighting conditions |

#### Geometric Augmentation
Applies spatial transformations to teach the model object invariance:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `degrees` | float | 0.0 | 0.0-180 | Rotates images randomly for orientation invariance |
| `translate` | float | 0.1 | 0.0-1.0 | Translates images to learn partially visible objects |
| `scale` | float | 0.5 | 0.0-1.0 | Scales images to simulate different distances |
| `shear` | float | 0.0 | -180 to +180 | Shears images to mimic different viewing angles |
| `perspective` | float | 0.0 | 0.0-0.001 | Applies perspective transformation for 3D understanding |

#### Flip Augmentation
Mirror transformations for data diversity:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `flipud` | float | 0.0 | 0.0-1.0 | Vertical flip probability for data variability |
| `fliplr` | float | 0.5 | 0.0-1.0 | Horizontal flip for symmetrical objects and diversity |

#### Advanced Augmentation
Complex transformations that combine multiple images:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mosaic` | float | 1.0 | 0.0-1.0 | Combines four images for complex scene understanding |
| `mixup` | float | 0.0 | 0.0-1.0 | Blends two images to introduce label noise and variability |

#### Segmentation-Specific Augmentation

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `copy_paste` | float | 0.0 | 0.0-1.0 | Copies and pastes objects to increase instances |
| `copy_paste_mode` | str | flip | flip, mixup | Strategy for copy-paste augmentation |

### Augmentation Best Practices

1. **Start Conservative**: Begin with default values and increase gradually
2. **Task-Specific Tuning**: Different tasks benefit from different augmentation strategies
3. **Monitor Validation**: Watch validation metrics to avoid over-augmentation
4. **Combine Strategically**: Use complementary augmentations (e.g., color + geometric)
5. **Consider Data Characteristics**: Match augmentations to expected real-world variations

### Example Configurations

**Light Augmentation (Quick Training):**
```yaml
training:
  degrees: 5.0
  fliplr: 0.5
  hsv_h: 0.01
  mosaic: 0.5
```

**Heavy Augmentation (Complex Scenes):**
```yaml
training:
  degrees: 15.0
  translate: 0.2
  scale: 0.7
  mosaic: 1.0
  mixup: 0.2
  hsv_h: 0.02
  hsv_s: 0.8
  hsv_v: 0.5
```

**Segmentation (Recommended):**
```yaml
training:
  copy_paste: 0.3
  copy_paste_mode: flip
  mosaic: 1.0
  degrees: 10.0
  scale: 0.5
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
```

See `model_training/config.example.yaml` for all available augmentation parameters with detailed comments.

## Getting Started

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU (recommended) or CPU
- PyTorch >= 2.5.1
- 8GB+ RAM (16GB+ recommended for larger models)
- Google Drive access for downloading datasets and weights

### Quick Start

1. **Clone or navigate to this repository**:
   ```bash
   cd /path/to/internship-ml-2026-v1
   ```

2. **Set up virtual environment**:
   ```bash
   # Create python3 virtual environment
   python3 -m venv .venv

   # Activate the environment
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   cd model_training
   pip install -r requirements.txt
   ```

4. **Download dataset and weights from Google Drive**:

   The dataset and pretrained weights are shared via Google Drive. Download them into the appropriate folders:

   ```bash
   # Create directories if they don't exist
   mkdir -p ../dataset
   mkdir -p ../weights
   ```

   **Download links will be provided by your instructor. After downloading:**

   ```bash
   # Example: Place your downloaded dataset
   # Unzip to: internship-ml-2026-v1/dataset/internship_dataset

   # Example: Place your downloaded weights
   # Unzip to: internship-ml-2026-v1/weights
   ```

   **Expected structure after download:**
   ```
   internship-ml-2026-v1/
   ├── dataset/
   │   └── internship_dataset/
   │       ├── data.yaml
   │       ├── images/
   │       │   ├── train/
   │       │   └── val/
   │       └── labels/
   │           ├── train/
   │           └── val/
   └── weights/
       └── yolo11s-seg.pt (or other model weights)
   ```

5. **Visualize your dataset (recommended)**:
   ```bash
   cd ../viz_annotations
   python3 visualize_yolo_dataset.py
   ```

   This will:
   - Show sample images with segmentation masks overlaid
   - Display class distribution statistics
   - Help verify dataset integrity before training

6. **Configure training**:
   ```bash
   cd ../model_training

   # Copy the example config and customize it
   cp config.example.yaml config.yaml

   # Edit config.yaml with your settings
   nano config.yaml  # or use your preferred editor
   ```

7. **Start training**:
   ```bash
   python3 train.py
   ```

   The script will automatically:
   - Load your dataset and apply balancing (if enabled)
   - Load pretrained weights from the weights folder
   - Train the segmentation model with your configuration
   - Export to ONNX format
   - Save all results to the output directory

## Detailed Documentation

- **[model_training/README.md](model_training/README.md)**: Complete training guide
  - Installation instructions
  - Configuration reference
  - Training examples for different tasks (detection, segmentation, OBB)
  - Dataset balancing guide
  - YOLO model selection and comparison
  - Troubleshooting and performance tips

- **[model_training/config.example.yaml](model_training/config.example.yaml)**: Comprehensive configuration template
  - All 80+ available parameters documented
  - Parameter descriptions, ranges, and defaults
  - 10 example configurations for different use cases
  - Task-specific parameter notes

## Dataset Format Reference

### Input: YOLO Bounding Box Format

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── ...
│   └── val/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img1.txt  (class_id x_center y_center width height)
│   │   └── ...
│   └── val/
│       └── ...
└── data.yaml
```

### YOLO Segmentation Format

**Label format**: Each line in a .txt file contains:
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```
Where:
- `class_id`: Integer class identifier (0-indexed)
- `x1 y1 x2 y2 ... xn yn`: Normalized polygon points (0.0-1.0) defining the segmentation mask

**Example label file** (`labels/train/image001.txt`):
```
0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4
1 0.5 0.5 0.7 0.5 0.7 0.7 0.5 0.7
```

**data.yaml** format:
```yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 3
names: ['class1', 'class2', 'class3']
```

## Common Issues and Solutions

### CUDA Out of Memory
Reduce memory usage:
```yaml
training:
  batch: 8          # Reduce batch size
  imgsz: 480        # Reduce image size
  cache: false      # Disable caching
  workers: 4        # Reduce workers
```

Or train on CPU:
```yaml
training:
  device: cpu
```

### Slow Training
Enable optimizations:
```yaml
training:
  cache: ram        # Cache images in RAM
  amp: true         # Use mixed precision
  workers: 8        # Increase data loading workers
```

### Model Not Converging
Adjust learning rate or optimizer:
```yaml
training:
  optimizer: AdamW
  lr0: 0.0001       # Lower learning rate
  cos_lr: true      # Enable cosine scheduler
  patience: 50      # Increase patience
```

### Dataset Not Found
Ensure your `config.yaml` has the correct absolute path:
```yaml
local:
  dataset_dir: "/absolute/path/to/your/dataset"
```

## Key Features

### Dataset Class Balancing

Automatically balance class distribution via oversampling:
- Configurable target: majority class, median class, or custom count
- Prevents overfitting on majority classes
- Generates debug histograms for before/after comparison
- Limits maximum duplications per image

### Training Monitoring

- Real-time metrics (loss, mAP, precision, recall)
- Automatic plot generation (training curves, confusion matrix, PR curves)
- Validation during training
- Early stopping with patience
- Checkpoint saving (best and last)

## Resources

### Ultralytics YOLO Resources
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [Training Mode Guide](https://docs.ultralytics.com/modes/train/)
- [Augmentation Settings](https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters)
- [YOLO Dataset Format](https://docs.ultralytics.com/datasets/)

### PyTorch Resources
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [CUDA Setup Guide](https://pytorch.org/get-started/locally/)

### ONNX Resources
- [ONNX Documentation](https://onnx.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)

## Support

For issues:
1. Check the [troubleshooting section](#common-issues-and-solutions) above
2. Review [model_training/README.md](model_training/README.md) for detailed troubleshooting
3. Check [Ultralytics documentation](https://docs.ultralytics.com/)
4. Verify PyTorch/CUDA compatibility: `python3 -c "import torch; print(torch.cuda.is_available())"`

## License

This project uses:
- Ultralytics YOLO (AGPL-3.0 / Enterprise license available)
- PyTorch (BSD-style license)

Refer to individual library licenses for specific terms.
