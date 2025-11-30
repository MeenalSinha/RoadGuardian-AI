# Setup Guide - RoadGuardian AI

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Model Download](#model-download)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Quick Start](#quick-start)

## System Requirements

### Hardware Requirements

**Minimum (CPU Only):**
- CPU: Intel i5 / AMD Ryzen 5 (4 cores)
- RAM: 8GB
- Storage: 5GB free space
- OS: Windows 10/11, Ubuntu 20.04+, macOS 10.15+

**Recommended (GPU):**
- CPU: Intel i7 / AMD Ryzen 7 (8 cores)
- GPU: NVIDIA GTX 1660 or better (6GB+ VRAM)
- RAM: 16GB
- Storage: 10GB free space
- OS: Windows 10/11, Ubuntu 20.04+

**Optimal (Production):**
- CPU: Intel i9 / AMD Ryzen 9
- GPU: NVIDIA RTX 3060 or better (8GB+ VRAM)
- RAM: 32GB
- Storage: 20GB SSD
- OS: Ubuntu 22.04 LTS

### Software Requirements

- **Python:** 3.8, 3.9, 3.10, or 3.11
- **CUDA:** 11.7+ (for GPU support)
- **Git:** Latest version

## Installation

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/pothole-detection.git
cd pothole-detection

# Or download as ZIP and extract
```

### Step 2: Create Virtual Environment

**On Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**On Linux/macOS:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 3: Install Dependencies

**Option A: All Dependencies (Recommended)**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Option B: Minimal Installation**
```bash
# Core dependencies only
pip install ultralytics==8.0.196
pip install streamlit==1.28.0
pip install opencv-python==4.8.1.78
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Option C: CPU-Only Installation**
```bash
# For systems without GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics streamlit opencv-python
```

### Step 4: Verify GPU Support (Optional)

```python
# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output (if GPU available):
```
CUDA Available: True
CUDA Version: 11.8
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
```

## Model Download

### Option 1: Download Pre-trained Model

**From GitHub Releases:**
```bash
# Download from releases
wget https://github.com/yourusername/pothole-detection/releases/download/v2.0/best.pt

# Or using curl
curl -L https://github.com/yourusername/pothole-detection/releases/download/v2.0/best.pt -o model/best.pt
```

**From Google Drive:**
```bash
# Install gdown
pip install gdown

# Download model
gdown https://drive.google.com/uc?id=YOUR_FILE_ID -O model/best.pt
```

### Option 2: Use Sample Model (Testing Only)

```bash
# The repository includes a sample model for testing
# Located at: model/best.pt (if included)
```

### Option 3: Train Your Own Model

See [TRAINING.md](TRAINING.md) for detailed instructions on training from scratch.

## Project Structure

After setup, your directory should look like:

```
pothole-detection/
├── app.py                          # Streamlit application
├── pothole_detection_pipeline.py   # Training pipeline
├── requirements.txt                # Dependencies
├── model/
│   └── best.pt                     # Trained model (6MB)
├── data/
│   └── sample_images/              # Test images
├── outputs/
│   ├── metrics_summary.json        # Model metrics
│   └── sample_results/             # Example outputs
└── docs/
    ├── SETUP.md                    # This file
    ├── TRAINING.md                 # Training guide
    └── DEPLOYMENT.md               # Deployment guide
```

## Verification

### Test Installation

**1. Verify Python Version:**
```bash
python --version
# Should show: Python 3.8.x or higher
```

**2. Test Package Imports:**
```bash
python -c "import streamlit; import cv2; import torch; from ultralytics import YOLO; print('✅ All packages imported successfully!')"
```

**3. Test Model Loading:**
```python
# test_setup.py
from ultralytics import YOLO
import cv2
import numpy as np

print("🔍 Testing model setup...")

# Load model
try:
    model = YOLO('model/best.pt')
    print("✅ Model loaded successfully!")
    print(f"   Classes: {model.names}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit(1)

# Test prediction on dummy image
try:
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(dummy_image, verbose=False)
    print("✅ Inference test passed!")
    print(f"   Inference time: {results[0].speed}")
except Exception as e:
    print(f"❌ Inference test failed: {e}")
    exit(1)

print("\n🎉 Setup verification complete!")
print("   Ready to run: streamlit run app.py")
```

**Run verification:**
```bash
python test_setup.py
```

### Test Web Application

```bash
# Run Streamlit app
streamlit run app.py

# Should open browser at http://localhost:8501
```

**Expected behavior:**
- Application loads without errors
- Can upload test images
- Model makes predictions
- Results are displayed correctly

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce batch size in config
batch_size = 4  # Instead of 16

# Or use CPU
device = 'cpu'

# Or reduce image size
imgsz = 640  # Instead of 1280
```

#### 2. Module Not Found

**Problem:**
```
ModuleNotFoundError: No module named 'ultralytics'
```

**Solution:**
```bash
# Ensure virtual environment is activated
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 3. Model File Not Found

**Problem:**
```
FileNotFoundError: model/best.pt not found
```

**Solution:**
```bash
# Create model directory
mkdir -p model

# Download model (see Model Download section)
# Or train your own model (see TRAINING.md)
```

#### 4. Streamlit Won't Start

**Problem:**
```
Command 'streamlit' not found
```

**Solution:**
```bash
# Reinstall streamlit
pip install --upgrade streamlit

# Or use python module syntax
python -m streamlit run app.py
```

#### 5. OpenCV Import Error

**Problem:**
```
ImportError: libGL.so.1: cannot open shared object file
```

**Solution (Linux):**
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

#### 6. Slow Inference on CPU

**Expected behavior:** CPU inference is slower (~100ms vs 2.4ms on GPU)

**Solutions:**
- Use GPU if available
- Reduce image size: `imgsz=640`
- Reduce max detections: `max_det=100`
- Use lighter model (already using YOLOv8n)

### Platform-Specific Issues

#### Windows

**Issue: Permission denied**
```bash
# Run as administrator or adjust folder permissions
```

**Issue: Long path names**
```bash
# Enable long paths in Windows
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

#### Linux

**Issue: Missing system libraries**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

#### macOS

**Issue: SSL certificate error**
```bash
# Install certificates
/Applications/Python\ 3.x/Install\ Certificates.command
```

## Quick Start

Once setup is complete:

### 1. Run Web Application

```bash
streamlit run app.py
```

### 2. Test with Sample Image

```bash
# Use provided sample images
python test_model.py --image data/sample_images/pothole1.jpg
```

### 3. Process Your Own Images

```python
# simple_test.py
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('model/best.pt')

# Load image
image = cv2.imread('your_road_image.jpg')

# Run prediction
results = model.predict(
    image,
    conf=0.15,  # Confidence threshold
    save=True   # Save annotated image
)

print(f"Found {len(results[0].boxes)} defects!")
```

## Environment Variables (Optional)

Create `.env` file for custom configuration:

```bash
# .env
MODEL_PATH=model/best.pt
CONFIDENCE_THRESHOLD=0.15
IOU_THRESHOLD=0.45
MAX_DETECTIONS=300
IMAGE_SIZE=640
DEVICE=cuda  # or 'cpu'
```

Load in your code:
```python
from dotenv import load_dotenv
import os

load_dotenv()
model_path = os.getenv('MODEL_PATH', 'model/best.pt')
```

## Next Steps

After successful setup:

1. **Test the Application:** Try the web interface with sample images
2. **Read Training Guide:** See [TRAINING.md](TRAINING.md) to train on your data
3. **Deployment:** Check [DEPLOYMENT.md](DEPLOYMENT.md) for production setup
4. **Customize:** Modify confidence thresholds and parameters as needed

## Getting Help

**Documentation:**
- Training Guide: [TRAINING.md](TRAINING.md)
- Deployment Guide: [DEPLOYMENT.md](DEPLOYMENT.md)
- Model Card: [model_card.md](../model/model_card.md)

**Community:**
- GitHub Issues: https://github.com/yourusername/pothole-detection/issues
- Discussions: https://github.com/yourusername/pothole-detection/discussions

**Contact:**
- Email: your.email@example.com
- GitHub: @yourusername

---

**Setup Guide Version:** 1.0  
**Last Updated:** 2024-11-30  
**Compatibility:** RoadGuardian AI v2.0
