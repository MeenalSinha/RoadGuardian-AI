# Training Guide - RoadGuardian AI

## Table of Contents
- [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Training Configuration](#training-configuration)
- [Training Process](#training-process)
- [Monitoring Training](#monitoring-training)
- [Model Evaluation](#model-evaluation)
- [Optimization Tips](#optimization-tips)

## Prerequisites

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GTX 1660 (6GB VRAM)
- RAM: 16GB
- Storage: 20GB free

**Recommended:**
- GPU: NVIDIA RTX 3060 or better (8GB+ VRAM)
- RAM: 32GB
- Storage: 50GB SSD

**Training Time Estimates:**
- YOLOv8n (nano): 8-10 hours for 100 epochs
- YOLOv8s (small): 12-15 hours for 100 epochs
- YOLOv8m (medium): 20-25 hours for 100 epochs

### Software Requirements

```bash
pip install ultralytics>=8.0.196
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install opencv-python>=4.8.0
pip install albumentations>=1.3.0
pip install pandas numpy matplotlib seaborn
```

## Dataset Preparation

### 1. Dataset Structure

Organize your dataset in YOLO format:

```
data/
└── yolo/
    ├── dataset.yaml          # Dataset configuration
    ├── train/
    │   ├── images/           # Training images
    │   │   ├── img001.jpg
    │   │   ├── img002.jpg
    │   │   └── ...
    │   └── labels/           # Training labels
    │       ├── img001.txt
    │       ├── img002.txt
    │       └── ...
    └── val/
        ├── images/           # Validation images
        └── labels/           # Validation labels
```

### 2. Dataset Configuration (dataset.yaml)

```yaml
# data/yolo/dataset.yaml

# UPDATE THIS PATH to match your system
path: /path/to/your/dataset  # Absolute path to dataset root

# Platform-specific examples:
# Windows: D:/pothole_detection_project/data/yolo
# Linux:   /home/user/pothole-detection/data/yolo
# macOS:   /Users/user/pothole-detection/data/yolo

train: train/images  # Train images (relative to path)
val: val/images      # Validation images (relative to path)

# Classes
nc: 3  # Number of classes
names:
  0: pothole
  1: crack
  2: damage
```

### 3. Label Format

YOLO format: `class x_center y_center width height`

Example label file (`img001.txt`):
```
0 0.5 0.3 0.2 0.15  # pothole at center
1 0.7 0.8 0.1 0.05  # crack at bottom-right
2 0.3 0.4 0.25 0.2  # damage on left
```

All values are normalized (0-1):
- `class`: Class ID (0, 1, or 2)
- `x_center`: Horizontal center (0-1)
- `y_center`: Vertical center (0-1)
- `width`: Box width (0-1)
- `height`: Box height (0-1)

### 4. Data Augmentation (Optional)

Already included in YOLOv8 training, but you can customize:

```python
# In training script
augmentation_config = {
    'hsv_h': 0.015,      # Hue augmentation
    'hsv_s': 0.7,        # Saturation
    'hsv_v': 0.4,        # Value
    'degrees': 10.0,     # Rotation
    'translate': 0.1,    # Translation
    'scale': 0.5,        # Scaling
    'shear': 0.0,        # Shear
    'perspective': 0.0,  # Perspective
    'flipud': 0.0,       # Vertical flip
    'fliplr': 0.5,       # Horizontal flip
    'mosaic': 1.0,       # Mosaic augmentation
    'mixup': 0.1,        # Mixup augmentation
}
```

### 5. Dataset Statistics

Check your dataset before training:

```python
# check_dataset.py
from pathlib import Path
import yaml

# Load dataset config
with open('data/yolo/dataset.yaml', 'r') as f:
    data = yaml.safe_load(f)

# Count files
train_images = list(Path(data['path']).joinpath(data['train']).glob('*.jpg'))
val_images = list(Path(data['path']).joinpath(data['val']).glob('*.jpg'))

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Total: {len(train_images) + len(val_images)}")
print(f"Split: {len(train_images)/(len(train_images)+len(val_images))*100:.1f}% train")
```

**Recommended split:** 80% train, 20% validation

## Training Configuration

### Basic Training (Quick Start)

```python
# train_basic.py
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # Start with pretrained weights

# Train
results = model.train(
    data='data/yolo/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda',  # or 'cpu'
    project='outputs/training',
    name='pothole_detection',
)
```

### Advanced Training (Production Quality)

```python
# train_advanced.py
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # or yolov8m.pt for better accuracy

# Training configuration
results = model.train(
    # Dataset
    data='data/yolo/dataset.yaml',
    
    # Training duration
    epochs=150,
    
    # Image & batch
    imgsz=1280,              # Larger for better large object detection
    batch=8,                 # Adjust based on GPU memory
    
    # Optimization
    optimizer='AdamW',       # AdamW optimizer
    lr0=0.001,              # Initial learning rate
    lrf=0.01,               # Final learning rate (fraction of lr0)
    momentum=0.937,          # SGD momentum
    weight_decay=0.0005,     # Weight decay
    
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    
    # Training settings
    patience=50,             # Early stopping patience
    save_period=10,          # Save checkpoint every N epochs
    cos_lr=True,            # Cosine LR scheduler
    close_mosaic=10,        # Disable mosaic in last N epochs
    amp=True,               # Automatic Mixed Precision
    multi_scale=True,       # Multi-scale training
    
    # Validation
    val=True,
    
    # Device & workers
    device='cuda',
    workers=4,
    
    # Output
    project='outputs/training',
    name='pothole_detection_v2',
    exist_ok=True,
    verbose=True,
)
```

### Model Selection

| Model | Parameters | Speed | mAP (expected) | Use Case |
|-------|-----------|-------|----------------|----------|
| YOLOv8n | 3.2M | 2.4ms | 80-82% | Production (current) |
| YOLOv8s | 11.2M | 5ms | 83-85% | Balanced |
| YOLOv8m | 25.9M | 10ms | 85-90% | High accuracy |
| YOLOv8l | 43.7M | 15ms | 88-92% | Research |

**Recommendation:** Start with YOLOv8n, upgrade to YOLOv8m if needed.

## Training Process

### Step 1: Prepare Environment

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 2: Run Training

```bash
# Using the pipeline script
python pothole_detection_pipeline.py --full --train --epochs 150

# Or custom training script
python train_advanced.py
```

### Step 3: Monitor Progress

Training outputs will be saved to:
```
outputs/training/pothole_detection_v2/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   ├── last.pt          # Latest checkpoint
│   └── epoch_*.pt       # Periodic checkpoints
├── results.csv          # Training metrics per epoch
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
└── val_batch*.jpg       # Validation visualizations
```

## Monitoring Training

### Real-time Monitoring

**Option 1: Terminal Output**
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/150     5.2G      1.234      0.567      0.890        128        640
  2/150     5.2G      1.189      0.543      0.876        128        640
  ...
```

**Option 2: TensorBoard (Optional)**
```bash
# Install tensorboard
pip install tensorboard

# Log training with tensorboard
# Add to training config:
# callbacks=['tensorboard']

# View logs
tensorboard --logdir outputs/training/
```

**Option 3: Weights & Biases (Optional)**
```bash
# Install wandb
pip install wandb

# Login
wandb login

# Add to training config:
# project='pothole-detection'
# name='experiment-1'
```

### Training Metrics to Watch

**Loss curves (should decrease):**
- `box_loss`: Bounding box regression loss
- `cls_loss`: Classification loss
- `dfl_loss`: Distribution focal loss

**Validation metrics (should increase):**
- `mAP@0.5`: Mean Average Precision at IoU=0.5
- `mAP@0.5:0.95`: Mean Average Precision at IoU=0.5-0.95
- `Precision`: Positive predictive value
- `Recall`: True positive rate

**Healthy training signs:**
- Losses decreasing smoothly
- mAP increasing steadily
- No overfitting (train/val gap small)
- Converges around epoch 80-120

## Model Evaluation

### After Training Completes

```python
# evaluate_model.py
from ultralytics import YOLO

# Load best model
model = YOLO('outputs/training/pothole_detection_v2/weights/best.pt')

# Validate on test set
metrics = model.val(
    data='data/yolo/dataset.yaml',
    split='val',
    imgsz=640,
    batch=16,
    conf=0.001,  # Low threshold for full evaluation
    iou=0.7,
    device='cuda'
)

# Print results
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

### Per-Class Performance

```python
# Check per-class metrics
for i, name in enumerate(model.names.values()):
    print(f"{name}:")
    print(f"  Precision: {metrics.box.class_result(i)[0]:.4f}")
    print(f"  Recall: {metrics.box.class_result(i)[1]:.4f}")
    print(f"  mAP@0.5: {metrics.box.class_result(i)[2]:.4f}")
```

### Test on Real Images

```python
# test_inference.py
from ultralytics import YOLO
import cv2
from pathlib import Path

model = YOLO('outputs/training/pothole_detection_v2/weights/best.pt')

# Test on sample images
test_images = Path('data/sample_images').glob('*.jpg')

for img_path in test_images:
    results = model.predict(
        str(img_path),
        conf=0.15,
        save=True,
        project='outputs/test_results'
    )
    
    print(f"{img_path.name}: {len(results[0].boxes)} detections")
```

## Optimization Tips

### 1. Improve Accuracy

**If mAP < 75%:**
- Increase epochs (100 → 150)
- Use larger model (YOLOv8n → YOLOv8m)
- Increase image size (640 → 1280)
- Add more training data
- Check label quality

**If overfitting (train >> val):**
- Increase augmentation
- Add dropout/regularization
- Reduce model size
- Add more validation data

### 2. Speed Up Training

```python
# Faster training config
results = model.train(
    data='data/yolo/dataset.yaml',
    epochs=100,
    imgsz=640,        # Smaller images
    batch=32,         # Larger batch (if GPU allows)
    workers=8,        # More data loading threads
    cache=True,       # Cache images in RAM
    amp=True,         # Mixed precision
    multi_scale=False # Disable multi-scale
)
```

### 3. Handle GPU Memory Issues

**If CUDA Out of Memory:**
```python
# Reduce batch size
batch=8  # or 4

# Reduce image size
imgsz=640  # instead of 1280

# Disable some augmentations
mosaic=0.0
mixup=0.0
```

### 4. Resume Training

**If training interrupted:**
```python
# Resume from last checkpoint
model = YOLO('outputs/training/pothole_detection_v2/weights/last.pt')

model.train(
    resume=True,  # Resume training
    # ... same config as before
)
```

### 5. Fine-tuning Existing Model

```python
# Start from your trained model
model = YOLO('model/best.pt')

# Fine-tune on new data
model.train(
    data='data/new_dataset/dataset.yaml',
    epochs=50,        # Fewer epochs
    lr0=0.0001,      # Lower learning rate
    freeze=10,        # Freeze first 10 layers
)
```

## Best Practices

### Dataset Quality
- ✅ Minimum 10,000 images for production
- ✅ At least 100 examples per class
- ✅ Diverse: angles, lighting, weather, road types
- ✅ High-quality labels (double-check annotations)
- ✅ Balanced classes (not too imbalanced)

### Training Strategy
- ✅ Start with pretrained weights
- ✅ Use cosine LR scheduler
- ✅ Enable automatic mixed precision (AMP)
- ✅ Monitor validation metrics closely
- ✅ Use early stopping (patience=50)
- ✅ Save checkpoints regularly

### Hyperparameter Tuning
```python
# Good starting points
epochs = 100-150
batch = 8-16 (based on GPU)
imgsz = 640-1280
lr0 = 0.001
patience = 50
```

## Expected Results

### Training Progression

| Epoch | mAP@0.5 | box_loss | cls_loss | Notes |
|-------|---------|----------|----------|-------|
| 1-20 | 40-50% | 1.2 | 0.8 | Fast initial learning |
| 21-50 | 60-70% | 0.8 | 0.4 | Steady improvement |
| 51-100 | 75-82% | 0.5 | 0.2 | Convergence |
| 101-150 | 80-82% | 0.4 | 0.15 | Fine-tuning |

### Final Metrics (YOLOv8n)
- mAP@0.5: 80-82%
- Precision: 78-80%
- Recall: 75-78%
- Speed: 2-3ms

### Final Metrics (YOLOv8m)
- mAP@0.5: 85-90%
- Precision: 85-88%
- Recall: 82-85%
- Speed: 8-12ms

## Troubleshooting

### Training Not Converging

**Symptoms:** Loss not decreasing, mAP stuck
**Solutions:**
- Lower learning rate
- Increase batch size
- Check dataset labels
- Try different optimizer

### Poor Validation Performance

**Symptoms:** Good train, poor validation
**Solutions:**
- More augmentation
- More validation data
- Reduce model complexity
- Check for data leakage

### Slow Training

**Symptoms:** <1 it/s training speed
**Solutions:**
- Enable AMP (amp=True)
- Increase workers
- Cache images (cache=True)
- Use smaller image size
- Check GPU utilization

## Next Steps

After successful training:

1. **Evaluate:** Test on diverse real-world images
2. **Export:** Convert to ONNX/TensorRT for deployment
3. **Deploy:** See [DEPLOYMENT.md](DEPLOYMENT.md)
4. **Monitor:** Track performance in production
5. **Iterate:** Collect edge cases and retrain

---

**Training Guide Version:** 1.0  
**Last Updated:** 2024-11-30  
**Compatible with:** YOLOv8, Ultralytics 8.0+
