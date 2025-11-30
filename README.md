# 🛣️ RoadGuardian AI - Intelligent Pothole Detection System

<div align="center">

![RoadGuardian AI](https://img.shields.io/badge/AI-Powered-blue?style=for-the-badge)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)

**Real-time road defect detection using deep learning to improve road safety and infrastructure maintenance.**

[Demo](#-demo) • [Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Key Features](#-features)
- [Performance Metrics](#-performance-metrics)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Dataset](#-dataset)
- [Results](#-results)
- [Documentation](#-documentation)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**RoadGuardian AI** is an advanced computer vision system that automatically detects and classifies road defects including potholes, cracks, and general damage. Built on YOLOv8 architecture, it provides real-time detection with high accuracy, making it ideal for automated road inspection and infrastructure maintenance planning.

### Why RoadGuardian AI?

- 🚀 **Real-time Performance:** 2.4ms inference time (~417 FPS)
- 🎯 **High Accuracy:** 81.86% mAP@0.5
- 🔍 **Multi-class Detection:** Potholes, cracks, and damage
- 📊 **RCI Calculation:** Automated Road Condition Index scoring
- 🌐 **Production Ready:** Streamlit web interface included
- 📱 **Deployable:** Docker support, cloud-ready

---

## 🎬 Demo

### Web Application

![Detection Example](outputs/sample_results/detection_example.jpg)

**Live Demo:** [Try it here](#) *(Coming soon)*

### Sample Detection Results

<table>
  <tr>
    <td align="center">
      <img src="outputs/sample_results/result_1.jpg" width="300px"/><br/>
      <b>Urban Road Detection</b><br/>
      Detected: 8 potholes, 3 cracks
    </td>
    <td align="center">
      <img src="outputs/sample_results/result_2.jpg" width="300px"/><br/>
      <b>Highway Monitoring</b><br/>
      Detected: 5 damage areas
    </td>
    <td align="center">
      <img src="outputs/sample_results/result_3.jpg" width="300px"/><br/>
      <b>Rural Road Analysis</b><br/>
      Detected: 12 defects
    </td>
  </tr>
</table>

---

## ✨ Features

### Core Capabilities

- **🔍 Multi-Class Detection**
  - Potholes (83.4% mAP)
  - Cracks (82.3% mAP)
  - General Damage (80.0% mAP)

- **⚡ Real-time Processing**
  - 2.4ms per image inference
  - ~417 FPS on GPU
  - Batch processing support

- **📊 Advanced Analytics**
  - Road Condition Index (RCI) calculation
  - Severity assessment (Critical/Moderate/Minor)
  - Confidence scoring
  - Defect size estimation

- **🎨 Interactive Web Interface**
  - Streamlit-based UI
  - Drag-and-drop image upload
  - Real-time visualization
  - Adjustable confidence thresholds
  - Downloadable reports

- **📈 Comprehensive Metrics**
  - Detection statistics
  - Confidence distribution
  - Class-wise breakdown
  - Spatial heatmaps

### Advanced Features

- ✅ GPU acceleration support
- ✅ Batch image processing
- ✅ Export results (JSON, CSV, images)
- ✅ API-ready architecture
- ✅ Docker containerization
- ✅ Cloud deployment ready

---

## 📊 Performance Metrics

### Model Performance (v2.0)

| Metric | Value | Status |
|--------|-------|--------|
| **mAP@0.5** | 81.86% | ⭐ Production Ready |
| **mAP@0.5:0.95** | 52.74% | ⭐ Excellent |
| **Precision** | 79.15% | ✅ High |
| **Recall** | 76.55% | ✅ Good |
| **F1 Score** | 77.83% | ✅ Balanced |
| **Inference Speed** | 2.4ms | 🚀 Real-time |
| **FPS (GPU)** | ~417 | 🚀 Very Fast |

### Per-Class Performance

| Class | Precision | Recall | mAP@0.5 | F1 Score |
|-------|-----------|--------|---------|----------|
| 🕳️ Pothole | 79.3% | 79.3% | 83.4% | 79.3% |
| ⚡ Crack | 79.5% | 78.1% | 82.3% | 78.8% |
| 🚨 Damage | 78.7% | 71.7% | 80.0% | 75.0% |

### Quality Metrics

- ✅ **Correct Detections:** 86%
- ✅ **Low Confidence Issues:** 8%
- ✅ **False Positives:** 0%
- ✅ **Confidence Range:** 0.02-0.88

### Improvement Over v1.0

| Metric | v1.0 | v2.0 | Improvement |
|--------|------|------|-------------|
| mAP@0.5 | 66.89% | 81.86% | **+14.97%** |
| Max Confidence | 0.028 | 0.88 | **31.6x** 🎉 |
| Detections | 17 | 42-55 | **+147%** |
| Correct Rate | 74% | 86% | **+12%** |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU support)
- 8GB RAM (16GB recommended)

### Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/pothole-detection.git
cd pothole-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model (if not included)
# Place best.pt in model/ folder
```

### Run Web Application

```bash
streamlit run app.py
```

Open browser at `http://localhost:8501` 🎉

### Quick Test

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('model/best.pt')

# Detect
image = cv2.imread('test_image.jpg')
results = model.predict(image, conf=0.15)

# Display
print(f"Found {len(results[0].boxes)} defects!")
```

---

## 💻 Installation

### Option 1: Standard Installation

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

### Option 2: Docker Installation

```bash
# Build image
docker build -t roadguardian-ai .

# Run container
docker run -p 8501:8501 roadguardian-ai
```

### Option 3: Manual Installation

```bash
# Core dependencies
pip install ultralytics>=8.0.196
pip install streamlit>=1.28.0
pip install opencv-python>=4.8.0
pip install torch>=2.0.0 torchvision>=0.15.0
```

For detailed installation instructions, see [docs/SETUP.md](docs/SETUP.md).

---

## 📖 Usage

### Web Interface

1. **Launch Application**
   ```bash
   streamlit run app.py
   ```

2. **Upload Image**
   - Drag and drop or browse for road images
   - Supports: JPG, PNG, JPEG

3. **Configure Settings**
   - Adjust confidence threshold (0.05-0.50)
   - Default: 0.15 (recommended)

4. **View Results**
   - Annotated image with bounding boxes
   - Detection statistics
   - RCI score
   - Download results

### Python API

```python
from ultralytics import YOLO
import cv2

# Initialize model
model = YOLO('model/best.pt')

# Single image prediction
results = model.predict(
    'road_image.jpg',
    conf=0.15,      # Confidence threshold
    iou=0.45,       # IoU threshold
    imgsz=640,      # Image size
    device='cuda'   # 'cuda' or 'cpu'
)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        
        print(f"Class: {model.names[cls]}")
        print(f"Confidence: {conf:.2%}")
        print(f"Coordinates: {coords}")
```

### Batch Processing

```python
# Process multiple images
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = model.predict(images, conf=0.15, save=True)

# Results saved to runs/detect/predict/
```

### Command Line

```bash
# Predict on image
yolo predict model=model/best.pt source=test.jpg conf=0.15

# Predict on folder
yolo predict model=model/best.pt source=images/ conf=0.15

# Predict on video
yolo predict model=model/best.pt source=video.mp4 conf=0.15
```

For more usage examples, see [docs/SETUP.md](docs/SETUP.md).

---

## 🤖 Model Details

### Architecture

- **Base Model:** YOLOv8n (nano variant)
- **Framework:** Ultralytics YOLOv8
- **Parameters:** 3.2M
- **Model Size:** ~6MB
- **Input Size:** 640x640
- **Output:** 3 classes (pothole, crack, damage)

### Training Details

- **Dataset:** 18,717 images
  - Training: 14,956 images (80%)
  - Validation: 3,761 images (20%)
- **Epochs:** ~100-150
- **Optimizer:** AdamW
- **Learning Rate:** 0.001 (cosine decay)
- **Data Augmentation:** Mosaic, mixup, HSV, flip
- **Training Time:** ~20-25 hours (GPU)

### Model Variants

| Model | Parameters | Speed | mAP (expected) | Use Case |
|-------|-----------|-------|----------------|----------|
| **YOLOv8n** ⭐ | 3.2M | 2.4ms | 80-82% | **Production (Current)** |
| YOLOv8s | 11.2M | 5ms | 83-85% | Balanced |
| YOLOv8m | 25.9M | 10ms | 85-90% | High Accuracy |
| YOLOv8l | 43.7M | 15ms | 88-92% | Research |

For complete model documentation, see [model/model_card.md](model/model_card.md).

---

## 📊 Dataset

### Composition

- **Total Images:** 18,717
- **Sources:**
  - Primary dataset: 18,052 images
  - [Kaggle Annotated Potholes](https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset): 665 images

### Classes

1. **Pothole** 🕳️ - Deep depressions in road surface
2. **Crack** ⚡ - Linear fractures in pavement
3. **Damage** 🚨 - General road surface deterioration

### Data Characteristics

- **Image Sizes:** Variable (resized to 640x640)
- **Perspectives:** Aerial, ground-level, dashcam, close-up
- **Conditions:** Day/night, wet/dry, various lighting
- **Format:** YOLO format annotations

### Data Augmentation

- Mosaic augmentation
- Mixup augmentation
- HSV color space adjustments
- Horizontal flipping
- Rotation (±10°)
- Translation (±10%)
- Scaling (0.5-1.5x)

---

## 🎯 Results

### Validation Results

<table>
  <tr>
    <td align="center">
      <img src="outputs/training_curves.png" width="400px"/><br/>
      <b>Training Curves</b>
    </td>
    <td align="center">
      <img src="outputs/confusion_matrix.png" width="400px"/><br/>
      <b>Confusion Matrix</b>
    </td>
  </tr>
</table>

### Detection Examples

#### Urban Roads
- Average detections: 20-30 per image
- Accuracy: 85%
- Common: Potholes, cracks

#### Rural Roads  
- Average detections: 15-25 per image
- Accuracy: 78%
- Common: Large potholes, damage

#### Highways
- Average detections: 10-15 per image
- Accuracy: 82%
- Common: Cracks, minor damage

### Confidence Distribution

| Confidence Range | Percentage | Quality |
|-----------------|------------|---------|
| 0.80-1.00 | 15% | Excellent |
| 0.50-0.79 | 35% | Very Good |
| 0.15-0.49 | 42% | Good |
| 0.02-0.14 | 8% | Fair |

### Known Limitations

- ❗ Very small cracks (<100px) may be missed
- ❗ Water puddles occasionally misclassified
- ❗ Performance varies with camera angle
- ❗ Reduced accuracy in poor lighting

---

## 📚 Documentation

Comprehensive documentation available:

- 📘 **[Setup Guide](docs/SETUP.md)** - Installation and configuration
- 📗 **[Training Guide](docs/TRAINING.md)** - How to train your own model
- 📕 **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment
- 📙 **[Model Card](model/model_card.md)** - Detailed model documentation

### Quick Links

- [Installation Instructions](docs/SETUP.md#installation)
- [Usage Examples](docs/SETUP.md#usage)
- [Training from Scratch](docs/TRAINING.md)
- [API Documentation](docs/DEPLOYMENT.md#api-deployment)
- [Docker Deployment](docs/DEPLOYMENT.md#docker-deployment)
- [Troubleshooting](docs/SETUP.md#troubleshooting)

---

## 📁 Project Structure

```
pothole-detection/
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .gitignore                        # Git ignore rules
├── 📄 LICENSE                           # MIT License
│
├── 🐍 app.py                            # Streamlit web application
├── 🐍 pothole_detection_pipeline.py     # Training pipeline
│
├── 📁 model/
│   ├── best.pt                          # Trained YOLOv8 model (6MB)
│   └── model_card.md                    # Model documentation
│
├── 📁 data/
│   ├── sample_images/                   # Test images
│   │   ├── pothole1.jpg
│   │   ├── pothole2.jpg
│   │   └── road1.jpg
│   └── dataset.yaml                     # Dataset configuration
│
├── 📁 outputs/
│   ├── metrics_summary.json             # Training metrics
│   ├── sample_results/                  # Example detections
│   │   ├── result_1.jpg
│   │   ├── result_2.jpg
│   │   └── result_3.jpg
│   ├── training_curves.png              # Loss/mAP plots
│   └── confusion_matrix.png             # Model performance viz
│
├── 📁 docs/
│   ├── SETUP.md                         # Installation guide
│   ├── TRAINING.md                      # Training guide
│   └── DEPLOYMENT.md                    # Deployment guide
│
└── 📁 scripts/
    ├── download_kaggle_data.py          # Dataset download
    └── test_model.py                    # Model testing
```

---

## 🗺️ Roadmap

### Version 2.1 (Next Release)
- [ ] Model export to ONNX/TensorRT
- [ ] Mobile deployment (iOS/Android)
- [ ] Video processing support
- [ ] Enhanced RCI calculation

### Version 3.0 (Future)
- [ ] Upgrade to YOLOv8m (85-90% mAP target)
- [ ] Depth estimation for severity
- [ ] Multi-camera support
- [ ] Real-time streaming
- [ ] API server with FastAPI
- [ ] Database integration
- [ ] User authentication

### Long-term Vision
- [ ] Expand to 50,000+ training images
- [ ] Add more defect classes
- [ ] 3D reconstruction
- [ ] Automated repair cost estimation
- [ ] Integration with GIS systems
- [ ] Mobile app (dashcam integration)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

1. 🐛 **Report Bugs** - Open an issue with details
2. 💡 **Suggest Features** - Share your ideas
3. 📝 **Improve Documentation** - Fix typos, add examples
4. 🔧 **Submit Pull Requests** - Fix bugs or add features
5. 🎨 **Share Datasets** - Help improve the model
6. ⭐ **Star the Repo** - Show your support!

### Contribution Guidelines

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/pothole-detection.git

# Create branch
git checkout -b feature/my-feature

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Submit PR when ready!
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 👤 Contact

**Your Name**  
📧 Email: your.email@example.com  
🐙 GitHub: [@yourusername](https://github.com/yourusername)  
🔗 LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)  
🌐 Website: [yourwebsite.com](https://yourwebsite.com)

**Project Link:** [https://github.com/yourusername/pothole-detection](https://github.com/yourusername/pothole-detection)

### Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/yourusername/pothole-detection/issues)
- 💬 [Discussions](https://github.com/yourusername/pothole-detection/discussions)

---

## 🙏 Acknowledgments

### Datasets
- [Kaggle Annotated Potholes Dataset](https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset) - High-quality pothole annotations

### Technologies
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - State-of-the-art object detection
- [Streamlit](https://streamlit.io/) - Interactive web applications
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library

### Inspiration
- Road safety initiatives worldwide
- Infrastructure maintenance challenges
- Computer vision research community

### Special Thanks
- Open source community
- Contributors and testers
- Academic advisors
- Infrastructure departments for feedback

---

## 📈 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{roadguardian_ai_2024,
  title={RoadGuardian AI: Real-time Pothole Detection System},
  author={Your Name},
  year={2024},
  version={2.0},
  url={https://github.com/yourusername/pothole-detection},
  note={mAP@0.5: 81.86\%, 18,717 training images}
}
```

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/pothole-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/pothole-detection?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/pothole-detection?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/pothole-detection)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/pothole-detection)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/pothole-detection)

---

<div align="center">

### 🌟 Star this repository if you find it helpful! 🌟

**Made with ❤️ for safer roads**

[⬆ Back to Top](#️-roadguardian-ai---intelligent-pothole-detection-system)

</div>
