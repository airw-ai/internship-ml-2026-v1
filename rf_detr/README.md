# RF-DETR Training

Training pipeline for [RF-DETR](https://github.com/roboflow/rf-detr) — Roboflow's real-time transformer-based object detection and instance segmentation model built on a DINOv2 ViT backbone.

## Prerequisites

- Python >= 3.10
- CUDA-capable GPU (recommended)

## Installation

```bash
# From the repo root
source .venv/bin/activate

# Detection + standard segmentation models
pip install -r rf_detr/requirements.txt

```

## Dataset Preparation

RF-DETR expects **COCO JSON** in the Roboflow directory layout (`train/_annotations.coco.json`).
If your dataset is in YOLO format (used by the `yolo/` pipeline), convert it first:

```bash
python dataset/convert_yolo_to_coco.py \
    --yolo  dataset/internship_dataset \
    --coco  dataset/internship_dataset_coco \
    --copy-images
```

### Expected COCO Layout

```
dataset/internship_dataset_coco/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
└── valid/
    ├── _annotations.coco.json
    └── *.jpg
```

The `_annotations.coco.json` file follows the standard COCO schema:

```json
{
  "categories": [{"id": 0, "name": "class_name"}],
  "images":     [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
  "annotations":[{"id": 1, "image_id": 1, "category_id": 0,
                  "bbox": [x, y, w, h], "segmentation": [[x1,y1,...]], "area": 1234}]
}
```

## Quick Start

```bash
cd rf_detr

# Copy and edit the config
cp config.example.yaml config.yaml
nano config.yaml          # set dataset_dir and other settings

# Train
python train.py
```

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| `model.variant` | Model size: `nano/small/medium/base/large` or `seg-{nano..large}` | `base` |
| `local.dataset_dir` | Path to COCO dataset root | — |
| `local.output_dir` | Where to write checkpoints and logs | `../runs/rf_detr` |
| `training.epochs` | Training epochs | `100` |
| `training.batch_size` | Batch size per step | `4` |
| `training.grad_accum_steps` | Gradient accumulation steps | `4` |
| `training.lr` | Base learning rate | `1e-4` |
| `training.use_ema` | Exponential Moving Average | `true` |
| `training.early_stopping` | Stop when mAP plateaus | `false` |
| `training.tensorboard` | TensorBoard logging | `true` |
| `training.wandb` | Weights & Biases logging | `false` |

See `config.example.yaml` for the full parameter reference with descriptions.

## Model Variants

### Detection

| Variant | Parameters | Notes |
|---------|-----------|-------|
| `nano`  | ~6M  | Fastest, lowest accuracy |
| `small` | ~12M | Good speed/accuracy balance |
| `medium`| ~20M | Recommended for most tasks |
| `base`  | ~29M | Default, strong general-purpose |
| `large` | ~42M | Highest accuracy |

### Segmentation

| Variant | Notes |
|---------|-------|
| `seg-nano` through `seg-large` | Standard license (Apache 2.0) |

## Output Structure

```
runs/rf_detr/
├── checkpoint_best_regular.pth   # Best non-EMA checkpoint
├── checkpoint_best_ema.pth       # Best EMA checkpoint (if use_ema: true)
├── checkpoint_best_total.pth     # Overall best (used for final export)
├── checkpoint_epoch_N.pth        # Periodic checkpoints
├── onnx/
│   └── inference_model.onnx      # Exported ONNX model (after training)
├── metrics/                      # Training & validation plots
└── logs/                         # TensorBoard event files
```

## Augmentation

RF-DETR uses [albumentations](https://albumentations.ai/) for training-time image augmentation. Augmentation is applied only to training images — the validation split always sees the original (unaugmented) images.

Configure it via the `augmentation:` section in `config.yaml`.

### Presets

| Preset | Description | Best for |
|--------|-------------|---------|
| `default` | HorizontalFlip (p=0.5) | General starting point |
| `conservative` | Gentle brightness/contrast + flip | Small datasets < 500 images |
| `aggressive` | Spatial + colour (flip, rotate, affine, jitter) | Large datasets 2000+ images |
| `aerial` | Horizontal + vertical flip + 90° rotation + brightness | Satellite / overhead imagery |
| `industrial` | Brightness + Gaussian blur + noise | Manufacturing / inspection |
| `custom` | You define every transform | Full control |
| `disabled` | No augmentation | Debugging, overfit test |

```yaml
augmentation:
  preset: aggressive
```

### Custom Transforms

Set `preset: custom` and list any albumentations transform under `transforms:`:

```yaml
augmentation:
  preset: custom
  transforms:
    HorizontalFlip: {p: 0.5}
    ColorJitter:    {brightness: 0.3, contrast: 0.3, saturation: 0.2, hue: 0.05, p: 0.6}
    GaussianBlur:   {blur_limit: 3, p: 0.2}
    Rotate:         {limit: 20, p: 0.4}
    GaussNoise:     {std_range: [0.01, 0.05], p: 0.3}
```

**Geometric vs. pixel-level** — RF-DETR detects this automatically:
- **Geometric** (`Rotate`, `Affine`, `Flip`, `Perspective`, `ElasticTransform`, …) — wrapped with `BboxParams` so bounding boxes and masks transform with the image.
- **Pixel-level** (`ColorJitter`, `GaussNoise`, `GaussianBlur`, `RandomBrightnessContrast`, …) — applied to pixel values only; no coordinate adjustment needed.

Container transforms are also supported:

```yaml
augmentation:
  preset: custom
  transforms:
    HorizontalFlip: {p: 0.5}
    OneOf:
      - GaussianBlur: {blur_limit: 3, p: 1.0}
      - GaussNoise:   {std_range: [0.01, 0.05], p: 1.0}
    p: 0.3
```

### Choosing a Preset

```
Dataset size       Recommended preset
────────────────────────────────────────
< 500 images       conservative
500 – 2000         default or conservative
2000+              aggressive
Satellite/aerial   aerial
Manufacturing      industrial
Unknown            start with default, evaluate, then step up
```

---

## Interpreting Training Output

### metrics_plot.png

After each epoch RF-DETR writes `metrics_plot.png` to `output_dir`. It is a 2×2 grid of subplots — all x-axes are epoch number.

```
┌──────────────────────────┬──────────────────────────┐
│  Training & Validation   │    AP @ 0.50             │
│  Loss                    │                          │
├──────────────────────────┼──────────────────────────┤
│  AP @ 0.50:0.95          │    AR @ 0.50:0.95        │
│                          │                          │
└──────────────────────────┴──────────────────────────┘
```

#### Top-left — Training and Validation Loss

- **Solid line**: training loss (averaged over all batches in the epoch).
- **Dashed line**: validation loss (computed on the `valid/` split at the end of each epoch).

What to look for:

| Pattern | Meaning |
|---------|---------|
| Both lines decrease steadily | Good — model is learning |
| Training loss drops but validation loss plateaus or rises | Overfitting — try lower LR, more regularisation, or fewer epochs |
| Both losses plateau quickly | Underfitting — try more epochs, higher LR, or a larger model variant |
| Validation loss is *lower* than training loss early on | Normal — training loss includes augmentation noise; validation is clean |
| Large gap between train and val loss | Overfitting or distribution mismatch between splits |

#### Top-right — AP @ 0.50

Average Precision at IoU threshold 0.50 (the classic PASCAL VOC metric). A detection counts as correct if the predicted box/mask overlaps the ground-truth by ≥ 50 %.

- **Solid line**: base model AP50.
- **Dashed line**: EMA model AP50.

Higher is better (0–100 %). The EMA curve is usually smoother and slightly higher than the base curve because it averages weights over recent steps.

#### Bottom-left — AP @ 0.50:0.95

The primary COCO metric — Average Precision averaged over IoU thresholds 0.50, 0.55, …, 0.95. Much stricter than AP50; a model that predicts roughly-correct boxes will score well on AP50 but poorly here.

Watch this panel most closely: the checkpoint saved as `checkpoint_best_*.pth` is selected by the metric that maximises this value (base model or EMA, whichever is higher).

#### Bottom-right — AR @ 0.50:0.95

Average Recall averaged over IoU thresholds 0.50–0.95. Measures the model's ability to *find* objects (not just rank them correctly).

A high AR with a lower AP usually means the model finds most objects but ranks some false positives highly. If AR improves but AP stalls, try tightening the confidence threshold or increasing `num_select`.

### Healthy vs. Unhealthy Training at a Glance

| Metric | Healthy | Warning signs |
|--------|---------|---------------|
| Loss | Smooth downward trend, val ≈ train | Val loss diverges upward; extreme spikes |
| AP50 | Rises quickly in early epochs, then gradually improves | Flat from epoch 1; oscillates wildly |
| AP50:95 | Rises more slowly than AP50, tracks it from below | Never rises above ~0.05; diverges from AP50 after mid-training |
| AR50:95 | Rises in parallel with AP50:95 | Rises but AP50:95 does not (false-positive heavy model) |
| Base vs. EMA | EMA slightly above base after warm-up | EMA and base diverge dramatically (EMA decay may be too high) |

### Other Output Files

| File | Contents |
|------|---------|
| `log.txt` | One JSON object per line, appended each epoch. Contains `train_loss`, `test_loss`, all AP/AR values, and epoch number. Useful for post-hoc analysis or plotting with custom scripts. |
| `results.json` | Full COCO evaluation results on the test split (only written when `run_test: true`). Contains per-category AP/AR breakdown. |
| `results_mask.json` | Same as `results.json` but for segmentation masks (segmentation model variants only). |

### TensorBoard

When `tensorboard: true` (default), TensorBoard event files are written to `<output_dir>/logs/`. Launch with:

```bash
tensorboard --logdir runs/rf_detr/logs
```

TensorBoard provides the same loss and metric curves as `metrics_plot.png` but is interactive and updates in real time during training.

---

## ONNX Export

ONNX export runs automatically after training completes. RF-DETR restores the best checkpoint (EMA or regular, whichever scored higher) before exporting, so the ONNX model always reflects the best training result.

**Output names in the ONNX graph:**

| Model type | Outputs |
|-----------|---------|
| Detection | `dets` (boxes), `labels` (logits) |
| Segmentation | `dets`, `labels`, `masks` |

### Export Configuration

```yaml
export:
  enabled: true       # set false to skip
  opset_version: 17   # ONNX opset (17 recommended)
  batch_size: 1       # static batch size baked into the graph
  simplify: false     # graph simplification via onnxsim (see below)
```

### Graph Simplification

Simplification folds constants, removes redundant ops, and reduces file size. It requires extra dependencies:

```bash
pip install rfdetr[onnxexport]   # installs onnx + onnxsim
```

Then enable in config:

```yaml
export:
  simplify: true
```

### Export Without Training

To export from an existing checkpoint without re-training, run a small script from the repo root:

```python
# export_onnx.py  (run from repo root: python export_onnx.py)
from rfdetr import RFDETRSegSmall   # match your trained variant

model = RFDETRSegSmall(pretrain_weights="runs/rf_detr/checkpoint_best_total.pth")
model.export(output_dir="runs/rf_detr/onnx", opset_version=17, batch_size=1)
```

Replace `RFDETRSegSmall` with the variant you trained (`RFDETRNano`, `RFDETRBase`, `RFDETRSegMedium`, etc.). The class names are derived from the checkpoint itself, so no other configuration is needed.

## Multi-GPU Training

RF-DETR uses **PyTorch DDP (DistributedDataParallel)**. Distributed mode is activated automatically when the script is launched with `torchrun` — no config flag is needed. The library reads the `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` environment variables that `torchrun` sets.

### Launch Commands

```bash
# Single GPU (standard)
python train.py

# Multi-GPU — single machine, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Multi-node — 2 nodes × 4 GPUs each (run on every node)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=<0|1> \
         --master_addr=<node0_ip> --master_port=29500 train.py
```

### Effective Batch Size

> **Effective batch = `batch_size` × `n_gpus` × `grad_accum_steps`**

When scaling to multiple GPUs, reduce `batch_size` proportionally to keep the same effective batch size — or accept the larger batch and scale the learning rate accordingly.

| Setup | `batch_size` | `n_gpus` | `grad_accum_steps` | Effective batch |
|-------|-------------|----------|-------------------|-----------------|
| 1 GPU | 4 | 1 | 4 | 16 |
| 4 GPU | 4 | 4 | 4 | 64 |
| 4 GPU (same eff.) | 1 | 4 | 4 | 16 |

### Multi-GPU Config Parameters

These are set under `training:` in your `config.yaml`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `sync_bn` | `true` | `SyncBatchNorm` — syncs BN statistics across GPUs. Recommended for DDP. |
| `dist_url` | `"env://"` | Rendezvous URL. `"env://"` reads `MASTER_ADDR`/`MASTER_PORT` from `torchrun`. Use `"tcp://<host>:<port>"` for manual setup. |
| `gradient_checkpointing` | `false` | Recomputes activations during backward pass. Reduces VRAM ~30–40% at ~20% speed cost. Use when you can't lower `batch_size` further. |
| `num_workers` | `2` | DataLoader workers **per GPU process**. Increase with more CPU cores. |
| `device` | `"cuda"` | Target device. In DDP mode, each process auto-assigns its GPU from `LOCAL_RANK` — leave as `"cuda"`. |

### Multi-GPU Config Example

```yaml
# config.yaml — 4-GPU training
training:
  device: cuda
  batch_size: 4              # per-GPU; effective batch = 4 × 4 GPUs × 4 accum = 64
  grad_accum_steps: 4
  sync_bn: true
  num_workers: 4             # increase with more CPU cores
  gradient_checkpointing: false
```

```bash
torchrun --nproc_per_node=4 train.py
```

### Memory-Constrained Multi-GPU

If you're still running out of VRAM after reducing `batch_size`:

```yaml
training:
  batch_size: 2
  grad_accum_steps: 8        # effective batch = 2 × 2 GPUs × 8 = 32
  gradient_checkpointing: true   # ~30–40% less VRAM, ~20% slower
```

## Example Configurations

### Quick CPU Test

```yaml
model:
  variant: nano
training:
  epochs: 5
  batch_size: 2
  grad_accum_steps: 1
  num_workers: 0
```

### Production Detection

```yaml
model:
  variant: large
training:
  epochs: 100
  batch_size: 8
  grad_accum_steps: 2
  lr: 5.0e-5
  early_stopping: true
  early_stopping_patience: 15
```

### Segmentation

```yaml
model:
  variant: seg-medium
training:
  epochs: 80
  batch_size: 4
  cls_loss_coef: 5.0
```

## Troubleshooting

### CUDA Out of Memory

Reduce memory usage (in order of preference):

```yaml
training:
  batch_size: 2
  grad_accum_steps: 8            # keeps effective batch = 16
  gradient_checkpointing: true   # ~30-40% VRAM reduction
```

### Poor Convergence

```yaml
training:
  warmup_epochs: 3.0
  lr: 5.0e-5
  early_stopping: false
  epochs: 150
```

### Model Not Downloading

RF-DETR downloads pretrained DINOv2 weights on first run. Ensure internet access, or pre-download and set `TORCH_HOME`.

## Resources

- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [RF-DETR Documentation](https://rfdetr.roboflow.com)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
