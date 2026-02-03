# Training Run Analysis Guide

## Overview
The training process generates multiple graphs, metrics, and visualizations to help you understand model performance and identify areas for improvement.

Training runs typically include:
- **Task**: Instance Segmentation
- **Model**: YOLO11-seg (various sizes: s, m, l, x)
- **Metrics**: Detection and segmentation performance curves
- **Visualizations**: Sample predictions and ground truth comparisons

---

## Understanding the Graphs

### 1. **results.png** - Training Progress Overview
**Purpose**: Comprehensive view of all metrics across training epochs.

This multi-panel graph shows:
- **Training losses** (box_loss, seg_loss, cls_loss, dfl_loss) - Should decrease over time
- **Validation losses** - Should decrease and stabilize
- **Detection metrics** - Precision, Recall, mAP50, mAP50-95 for bounding boxes
- **Segmentation metrics** - Precision, Recall, mAP50, mAP50-95 for masks

**How to interpret**:
- **Good signs**: Smooth downward trend in losses, upward trend in metrics
- **Warning signs**:
  - Validation loss increasing while training loss decreases indicates overfitting
  - Metrics plateauing early suggests model may need different hyperparameters
  - Erratic fluctuations suggest learning rate may be too high or data augmentation too aggressive

---

### 2. Detection Performance Curves (Box Metrics)

#### **BoxP_curve.png** - Precision-Confidence Curve
Shows how precision changes with different confidence thresholds for bounding box predictions.

- **X-axis**: Confidence threshold (0.0 to 1.0)
- **Y-axis**: Precision (proportion of correct predictions)
- **Interpretation**:
  - Higher curve = better precision
  - The curve typically drops as you lower confidence (accepting more predictions)
  - Look for the "knee" where precision drops sharply - good threshold point

#### **BoxR_curve.png** - Recall-Confidence Curve
Shows how recall changes with different confidence thresholds.

- **X-axis**: Confidence threshold
- **Y-axis**: Recall (proportion of ground truth objects detected)
- **Interpretation**:
  - Higher curve = better recall
  - Recall increases as confidence threshold drops (model accepts more detections)
  - Trade-off: Lower threshold → higher recall but lower precision

#### **BoxPR_curve.png** - Precision-Recall Curve
Shows the relationship between precision and recall across all confidence thresholds.

- **X-axis**: Recall
- **Y-axis**: Precision
- **Area Under Curve (AUC)**: This is your mAP (mean Average Precision)
- **Interpretation**:
  - Ideal curve hugs the top-right corner (high precision AND high recall)
  - Larger area under curve = better overall performance
  - Compare curves between training runs to see which model performs better

#### **BoxF1_curve.png** - F1-Confidence Curve
F1 score is the harmonic mean of precision and recall (2 × P × R / (P + R)).

- **X-axis**: Confidence threshold
- **Y-axis**: F1 score
- **Interpretation**:
  - Peak of curve indicates optimal confidence threshold
  - Higher peak = better balanced performance
  - Use this to select your inference confidence threshold

---

### 3. Segmentation Performance Curves (Mask Metrics)

These are identical to box metrics but measure mask segmentation quality:

- **MaskP_curve.png**: Mask precision vs confidence
- **MaskR_curve.png**: Mask recall vs confidence
- **MaskPR_curve.png**: Mask precision-recall curve (AUC = mask mAP)
- **MaskF1_curve.png**: Mask F1 score vs confidence

**Key difference from box metrics**:
- Box metrics only check if bounding box IoU > threshold
- Mask metrics require pixel-level accuracy in the segmentation

**Typical observation**: Mask metrics are usually slightly lower than box metrics because segmentation is harder than detection.

---

### 4. Confusion Matrices

#### **confusion_matrix.png** - Raw Counts
Shows actual counts of predictions vs ground truth labels.

- **Rows**: True labels (ground truth)
- **Columns**: Predicted labels
- **Diagonal**: Correct predictions
- **Off-diagonal**: Confusion between classes

#### **confusion_matrix_normalized.png** - Percentage View
Same as above but normalized to percentages (0-1 or 0-100%).

**How to interpret**:
- Darker diagonal = better performance
- Strong off-diagonal elements = specific confusion patterns
  - Example: If class A is often predicted as class B, model struggles to distinguish them
- **Background (bg) class**: Shows false positives (predicted object where none exists) and false negatives (missed detections)

---

### 5. Visual Comparisons

#### **labels.jpg**
Sample of ground truth annotations from training data.
- Shows how your data is labeled
- Check for labeling consistency and quality

#### **train_batch0.jpg, train_batch1.jpg, train_batch2.jpg**
Sample training images with applied augmentations.
- Shows what the model actually sees during training
- Check if augmentations are reasonable (not too aggressive)
- Verify images are properly preprocessed

#### **val_batch[0-2]_labels.jpg**
Ground truth annotations on validation images.

#### **val_batch[0-2]_pred.jpg**
Model predictions on the same validation images.

**How to compare**:
- Open labels and pred files side-by-side
- Look for:
  - Tight mask overlaps indicate good segmentation
  - Missed objects indicate false negatives
  - Hallucinated objects indicate false positives
  - Poor mask boundaries suggest needs for more training or different architecture

---

## Key Metrics Explained

### Loss Functions

#### **box_loss** (Bounding Box Loss)
Measures how well predicted boxes match ground truth boxes.
- **Lower is better**
- Typically uses IoU loss or CIoU loss
- Good models achieve values below 1.0

#### **seg_loss** (Segmentation Loss)
Measures pixel-level mask prediction accuracy.
- **Lower is better**
- Usually binary cross-entropy or focal loss on mask pixels
- Expected to be higher than box loss, values around 1.5-2.5 are typical

#### **cls_loss** (Classification Loss)
Measures how well the model classifies detected objects.
- **Lower is better**
- Uses binary cross-entropy for objectness/class prediction
- Good models achieve values below 1.0

#### **dfl_loss** (Distribution Focal Loss)
Advanced loss for box regression that treats box coordinates as distributions.
- **Lower is better**
- Helps with precise localization
- Typically decreases to values around 1.0-1.5

### Detection Metrics

#### **mAP50 (mean Average Precision @ IoU 0.5)**
Average precision when considering a detection correct if IoU > 0.5.
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**:
  - mAP50 > 0.5: Excellent performance
  - mAP50 0.3-0.5: Good performance
  - mAP50 0.2-0.3: Fair performance
  - mAP50 < 0.2: Needs improvement

#### **mAP50-95 (mean Average Precision @ IoU 0.5:0.95)**
Average of mAP at IoU thresholds from 0.5 to 0.95 (step 0.05).
- **More strict metric** - requires precise localization
- **Range**: 0.0 to 1.0 (higher is better)
- **Rule of thumb**: mAP50-95 is typically 50-60% of mAP50 for well-trained models
- Lower ratios indicate the model struggles with precise localization

#### **Precision**
Of all predictions, what percentage were correct?
- **Formula**: TP / (TP + FP)
- **High precision** = Few false positives
- Important when false alarms are costly

#### **Recall**
Of all ground truth objects, what percentage were detected?
- **Formula**: TP / (TP + FN)
- **High recall** = Few false negatives
- Important when missing objects is costly

---

## How to Determine if Model is Improving

### Compare Between Training Runs

1. **Check mAP50-95 (most important metric)**
   - Higher = better overall performance
   - Compare final values: Model A vs Model B
   - Look at trajectory: Does it plateau early or keep improving?

2. **Examine loss convergence**
   - Compare final training losses
   - Lower and more stable = better
   - Check if validation loss follows training loss

3. **Look at Precision-Recall trade-off**
   - Neither should be extremely low
   - Balance depends on your application:
     - **Safety-critical**: Favor higher recall (catch all objects)
     - **Low false-alarm tolerance**: Favor higher precision

4. **Visual inspection**
   - Compare val_batch predictions side-by-side
   - Better model should have:
     - Tighter masks
     - Fewer missed objects
     - Fewer false positives

### Signs of a Good Model

**Positive indicators**:
- Validation losses decrease steadily with training losses
- mAP50 > 0.5 (excellent), > 0.3 (good), > 0.2 (acceptable)
- mAP50-95 is 50-70% of mAP50
- Precision and recall are balanced (neither below 0.3)
- Visual predictions closely match ground truth

### Signs of Problems

**Overfitting**:
- Training loss continues decreasing while validation loss increases
- **Solution**: More data, stronger augmentation, regularization, or early stopping

**Underfitting**:
- Both training and validation losses remain high
- Metrics plateau at low values
- **Solution**: Larger model, more training epochs, adjust learning rate

**Class Imbalance**:
- Good metrics overall but confusion matrix shows poor performance on specific classes
- **Solution**: Weighted loss, oversampling rare classes, or more diverse data

**Poor Segmentation Quality**:
- Box metrics good but mask metrics much worse
- **Solution**: Increase mask_ratio, use better mask annotations, train longer

---

## Using results.csv for Analysis

The CSV file contains all metrics for every epoch. Use it to:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results.csv')
df.columns = df.columns.str.strip()  # Clean column names

# Plot training vs validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(df['epoch'], df['train/box_loss'], label='Train')
plt.plot(df['epoch'], df['val/box_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Box Loss')
plt.legend()
plt.title('Box Loss Over Time')

# Plot mAP metrics
plt.subplot(1, 3, 2)
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='Box mAP50')
plt.plot(df['epoch'], df['metrics/mAP50(M)'], label='Mask mAP50')
plt.xlabel('Epoch')
plt.ylabel('mAP50')
plt.legend()
plt.title('mAP50 Over Time')

# Plot precision and recall
plt.subplot(1, 3, 3)
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.title('Precision vs Recall')

plt.tight_layout()
plt.show()
```

---

## Model Comparison Checklist

When comparing multiple training runs:

- [ ] **Higher mAP50-95** - Better overall performance
- [ ] **Lower validation losses** - Better generalization
- [ ] **Smooth learning curves** - Stable training
- [ ] **Balanced precision/recall** - Well-calibrated model
- [ ] **Better visual predictions** - Practical performance
- [ ] **Faster convergence** - More efficient training
- [ ] **Higher F1 score** - Better precision-recall balance

---

## Next Steps for Improvement

Common strategies to improve model performance:

1. **Train longer**: If early stopping triggered, model may benefit from more epochs
2. **Try larger model**: YOLOv11m-seg or YOLOv11l-seg for better capacity
3. **Adjust confidence threshold**: Use F1 curve to find optimal value for your use case
4. **Improve data quality**: Review labels.jpg and fix any annotation errors
5. **Increase image size**: Try 800 or 1024 instead of 640 for better resolution
6. **Fine-tune hyperparameters**: Experiment with learning rate and augmentation strength
7. **Ensemble models**: Combine predictions from multiple runs for better robustness
8. **Add more training data**: Especially for underperforming classes
9. **Adjust class weights**: If dealing with class imbalance
10. **Review augmentation**: Ensure augmentations match deployment conditions

---

## Files Reference

| File | Description |
|------|-------------|
| `results.png` | All training metrics in one view |
| `results.csv` | Raw metric values per epoch |
| `args.yaml` | Complete training configuration |
| `Box*_curve.png` | Detection performance curves |
| `Mask*_curve.png` | Segmentation performance curves |
| `confusion_matrix*.png` | Classification performance |
| `labels.jpg` | Training data sample |
| `train_batch*.jpg` | Augmented training samples |
| `val_batch*_labels.jpg` | Validation ground truth |
| `val_batch*_pred.jpg` | Validation predictions |
| `weights/` | Saved model checkpoints |

---

## Interpreting Your Results

After reviewing all the graphs and metrics, consider these questions:

1. **Is the model learning?**
   - Check if training and validation losses are decreasing
   - Verify that mAP metrics are increasing over epochs

2. **Is there overfitting?**
   - Compare training vs validation loss curves
   - Look for divergence where val loss increases while train loss decreases

3. **What is the optimal confidence threshold?**
   - Check BoxF1_curve.png for the peak F1 score
   - Use this threshold for inference

4. **Which classes need attention?**
   - Review confusion matrix for weak performance
   - Check validation predictions for common failure patterns

5. **Should I continue training?**
   - If metrics are still improving steadily, consider training longer
   - If metrics have plateaued, try architectural or data improvements

6. **How does this compare to other runs?**
   - Use mAP50-95 as the primary comparison metric
   - Compare final validation losses
   - Review visual predictions side-by-side
