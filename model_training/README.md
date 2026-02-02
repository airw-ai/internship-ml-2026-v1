# YOLO Segmentation Training - Local Setup

Designed for training pixel-perfect segmentation models on your local machine with advanced features.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Virtual environment (recommended)

## Installation

1. **Navigate to the model_training directory:**
   ```bash
   cd model_training
   ```

2. **Install dependencies:**
   ```bash
   # Using the project virtual environment
   source ../.venv/bin/activate
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Download Dataset and Weights from Google Drive

**Your instructor will provide Google Drive links for:**
- Pretrained segmentation weights (e.g., `yolo11s-seg.pt`)
- Training dataset (YOLO segmentation format)

**After downloading, place files in:**
```bash
# Weights go here
weights/yolo11s-seg.pt

# Dataset goes here
dataset/internship_dataset/
  ├── data.yaml
  ├── images/
  └── labels/
```

### 2. Configure Training

Copy the example configuration and customize it:

```bash
# Copy the comprehensive example config
cp config.example.yaml config.yaml

# Edit with your settings
nano config.yaml  # or use your preferred editor
```

**Minimal required configuration:**

```yaml
model:
  type: pavement        # Your model name (used in output paths)
  task: segment         # Must be 'segment' for segmentation

local:
  dataset_dir: "./dataset/internship_dataset"  # Path to your dataset
  weights_dir: "./weights"                     # Path to pretrained weights
  output_dir: "./runs"                         # Where to save results

training:
  weights_version: yolo11s-seg    # Segmentation model to use
  epochs: 150
  batch: 16
  imgsz: 640
  device: 0                       # GPU ID or 'cpu'
```

**Note**: See `config.example.yaml` for all 80+ available parameters with descriptions and examples.

### 3. Run Training

```bash
python3 train.py
```

The script will:
1. Load and validate the dataset
2. Apply class balancing (if enabled)
3. Download/load the pre-trained model (auto-downloads if not found)
4. Train the model with your configuration
5. Export to ONNX format (fixed for segmentation models)
6. Save all artifacts to the output directory

## Configuration Guide

**Complete Reference**: See [config.example.yaml](config.example.yaml) for all 80+ parameters with detailed documentation.

Below are the most commonly used settings:

### Model Settings

```yaml
model:
  type: pavement        # Model identifier (used in filenames and paths)
  task: segment         # Must be 'segment' for segmentation training
```

### Local Paths

```yaml
local:
  dataset_dir: "/path/to/your/dataset"  # YOLO dataset directory
  weights_dir: "./weights"               # Pre-trained weights directory
  output_dir: "./runs"                   # Output directory for training runs
```

### Dataset Balancing

Balance class distribution via oversampling:

```yaml
dataset:
  balance:
    enabled: true           # Enable balancing
    mode: oversample        # Method (only oversample supported)
    target: median          # "max", "median", or integer
    max_dup_per_image: 10   # Max duplications per image
    save_debug_hist: true   # Save before/after histograms
```

**Target options:**
- `max`: Balance to majority class count
- `median`: Balance to median class count
- `500`: Balance to specific count (integer)

### Training Settings

#### Basic Settings

```yaml
training:
  weights_version: yolo11n  # Base model: yolo11n/s/m/l/x
  epochs: 100               # Training epochs
  batch: 16                 # Batch size
  imgsz: 640                # Image size
  device: null              # null=auto, 0=GPU0, cpu=CPU
```

#### Learning Rate

```yaml
training:
  lr0: 0.01                 # Initial learning rate
  lrf: 0.01                 # Final LR (lr0 * lrf)
  warmup_epochs: 3.0        # Warmup epochs
  cos_lr: false             # Cosine LR scheduler
```

#### Augmentation

```yaml
training:
  # Color augmentation
  hsv_h: 0.015              # Hue variation
  hsv_s: 0.7                # Saturation variation
  hsv_v: 0.4                # Brightness variation

  # Geometric augmentation
  degrees: 7.5              # Rotation (0-180)
  translate: 0.1            # Translation
  scale: 0.5                # Scale variation
  shear: 2.5                # Shear transformation
  perspective: 0.0004       # Perspective distortion
  flipud: 0.0               # Vertical flip (usually 0 for roads)
  fliplr: 0.5               # Horizontal flip

  # Advanced augmentation
  mosaic: 0.97              # Mosaic probability (4 images combined)
  mixup: 0.06               # Mixup probability (blend 2 images)

  # Segmentation-specific
  copy_paste: 0.23          # Copy-paste objects between images
  copy_paste_mode: flip     # Strategy: 'flip' or 'mixup'
```

## Model Selection

### Segmentation Models

All models are segmentation-specific with the `-seg` suffix:

```yaml
training:
  weights_version: yolo11s-seg  # Recommended for most use cases
```

## Output Structure

After training, the output directory will contain:

```
runs/
└── train/
    └── my_model_2026-01-30T14-30-00/
        ├── weights/
        │   ├── best.pt          # Best model (PyTorch)
        │   ├── best.onnx        # Best model (ONNX)
        │   └── last.pt          # Last checkpoint
        ├── results.csv          # Training metrics
        ├── results.png          # Training curves
        ├── confusion_matrix.png # Confusion matrix
        ├── F1_curve.png         # F1 score curve
        ├── P_curve.png          # Precision curve
        ├── R_curve.png          # Recall curve
        ├── PR_curve.png         # Precision-Recall curve
        └── ...                  # Additional plots
```

## Example Configurations

### Quick Test Run

```yaml
training:
  device: cpu
  epochs: 10
  batch: 4
  imgsz: 320
```

### Production Training

```yaml
training:
  weights_version: yolo11s-seg
  epochs: 300
  batch: 32
  patience: 50
  optimizer: AdamW
  cos_lr: true
```

### Heavy Augmentation

```yaml
training:
  mosaic: 1.0
  mixup: 0.2
  degrees: 10.0
  hsv_h: 0.02
  hsv_s: 0.8
  hsv_v: 0.5
```

### Segmentation with Balancing (Recommended)

```yaml
model:
  task: segment

dataset:
  balance:
    enabled: true
    target: max

training:
  weights_version: yolo11s-seg
  copy_paste: 0.3
  mosaic: 1.0
  mixup: 0.1
```

## Troubleshooting

### Out of Memory

**Symptoms**: CUDA out of memory errors during training

**Solutions**:
```yaml
training:
  batch: 8              # Reduce batch size (try 8, 4, or even 2)
  imgsz: 416            # Reduce image size (try 416, 320)
  cache: false          # Disable caching
  workers: 4            # Reduce workers
  amp: true             # Enable mixed precision (reduces memory)
```

Or train on CPU (much slower):
```yaml
training:
  device: cpu
```

### Slow Training

**Solutions**:
```yaml
training:
  cache: ram            # Cache images in RAM (requires sufficient memory)
  workers: 8            # Increase data loading workers
  amp: true             # Use Automatic Mixed Precision
```

Or use a smaller model:
```yaml
training:
  weights_version: yolo11n  # Use nano instead of larger models
```

### Model Not Converging

**Symptoms**: Loss not decreasing, poor validation metrics

**Solutions**:
```yaml
training:
  lr0: 0.001            # Try lower learning rate
  cos_lr: true          # Enable cosine scheduler
  optimizer: AdamW      # Try different optimizer
  patience: 50          # Increase patience for early stopping
  warmup_epochs: 5.0    # Increase warmup
```

Also check:
- Dataset quality and annotations
- Class balance (enable balancing if needed)
- Reduce augmentation if too aggressive

### Weights Not Found

The script automatically downloads weights from Ultralytics if not found locally. Ensure you have:
- Internet connectivity on first run
- Write permissions in the weights directory

To use local weights:
```yaml
local:
  weights_dir: "/path/to/weights"

training:
  weights_version: yolo11n  # Must have yolo11n.pt in weights_dir
```

### Dataset Not Found / Path Errors

**Solution**: Always use absolute paths in config.yaml:
```yaml
local:
  dataset_dir: "/home/user/datasets/my_dataset"  # Not ./dataset or ~/dataset
```

Verify paths:
```bash
ls -la /home/user/datasets/my_dataset/data.yaml
ls -la /home/user/datasets/my_dataset/images/train/
ls -la /home/user/datasets/my_dataset/labels/train/
```

### Validation Metrics Not Improving

Check:
1. **Augmentation too aggressive**: Reduce augmentation parameters
2. **Learning rate too high**: Lower lr0
3. **Class imbalance**: Enable dataset balancing
4. **Model too small**: Try a larger model (s)

### Training Crashes During Augmentation

If crashes occur during mosaic or mixup:
```yaml
training:
  mosaic: 0.0           # Disable mosaic
  mixup: 0.0            # Disable mixup
  close_mosaic: 0       # Disable mosaic from start
```

## Advanced Usage

### Resume Training

```yaml
training:
  resume: true
```

This will resume from the last checkpoint in the run directory.

### Multi-GPU Training

```yaml
training:
  device: "0,1,2,3"  # Use GPUs 0, 1, 2, and 3
  batch: 64           # Increase batch size accordingly
```

### Custom Validation Split

```yaml
training:
  split: test  # Use test split for validation instead of val
```

### Freeze Backbone for Fine-tuning

```yaml
training:
  freeze: 10            # Freeze first 10 layers
  lr0: 0.0001           # Use lower learning rate
```

### Time-Limited Training

```yaml
training:
  time: 2.0             # Stop after 2 hours (overrides epochs)
```

## Complete Parameter Reference

See [config.example.yaml](config.example.yaml) for comprehensive documentation of all available parameters, including:

- **Core Training**: epochs, batch, imgsz, device, workers, patience
- **Optimization**: optimizer, learning rates, momentum, weight decay
- **Augmentation**: geometric (30+ options), color, advanced techniques
- **Loss Weights**: box, cls, dfl, pose, kobj, angle
- **Advanced**: caching, AMP, multi-scale, compilation
- **Validation**: split selection, max detections

Each parameter includes:
- Description and purpose
- Default value
- Valid range
- Applicable tasks
- Usage notes

## Performance Tips

1. **Use GPU**: Training on GPU is 10-100x faster than CPU
2. **Enable caching**: Speeds up training by caching images in RAM/disk
3. **Increase batch size**: Larger batches = better GPU utilization (if memory allows)
4. **Use multi-scale**: Improves model robustness at cost of speed
5. **Enable AMP**: Automatic Mixed Precision reduces memory usage and speeds up training

## Additional Resources

### Documentation
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Train Settings Reference](https://docs.ultralytics.com/modes/train/#train-settings)
- [Augmentation Guide](https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters)
- [Dataset Format Guide](https://docs.ultralytics.com/datasets/)

### Related Files
- [config.example.yaml](config.example.yaml) - Comprehensive configuration template
- [../README.md](../README.md) - Project overview and data augmentation guide

## Support

### For Training Issues

Check in this order:
1. **Dataset format**: Verify YOLO format with proper data.yaml
2. **Paths**: Use absolute paths in config.yaml
3. **Dependencies**: Ensure all requirements installed (`pip install -r requirements.txt`)
4. **Environment**: Virtual environment activated
5. **GPU**: Check CUDA availability (`python3 -c "import torch; print(torch.cuda.is_available())"`)

### For Ultralytics YOLO Issues

Refer to [official documentation](https://docs.ultralytics.com/) and [GitHub issues](https://github.com/ultralytics/ultralytics/issues).

### Quick Diagnostics

```bash
# Check Python version
python3 --version

# Check PyTorch and CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check Ultralytics
python3 -c "from ultralytics import YOLO; print('Ultralytics installed successfully')"

# Test dataset loading
python3 -c "from ultralytics import YOLO; model = YOLO('yolo11n.pt'); model.val(data='path/to/data.yaml')"
```
