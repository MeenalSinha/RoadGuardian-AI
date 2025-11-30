# Deployment Guide - RoadGuardian AI

## Table of Contents
- [Deployment Options](#deployment-options)
- [Local Deployment](#local-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Docker Deployment](#docker-deployment)
- [Production Considerations](#production-considerations)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Scaling](#scaling)

## Deployment Options

### Quick Comparison

| Option | Complexity | Cost | Performance | Scalability | Best For |
|--------|-----------|------|-------------|-------------|----------|
| **Local** | Low | Free | High | Low | Testing, demos |
| **Streamlit Cloud** | Low | Free-Low | Medium | Low | Prototypes |
| **Docker** | Medium | Variable | High | Medium | Production |
| **AWS/GCP/Azure** | High | Medium-High | Very High | Very High | Enterprise |
| **Edge** | Medium | Hardware | Very High | Low | IoT, vehicles |

## Local Deployment

### For Development & Testing

**Simple Streamlit App:**
```bash
# Activate environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Run app
streamlit run app.py

# Access at http://localhost:8501
```

**Custom port:**
```bash
streamlit run app.py --server.port 8080
```

**Network access:**
```bash
streamlit run app.py --server.address 0.0.0.0
# Access from other devices: http://YOUR_IP:8501
```

### Production-like Local Setup

```bash
# Use gunicorn (for API)
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 api:app
```

## Cloud Deployment

### Option 1: Streamlit Cloud (Easiest)

**Steps:**
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Deploy!

**Limitations:**
- Free tier: 1GB RAM, limited resources
- Not suitable for heavy production load
- Good for demos and prototypes

**Requirements file (`requirements.txt`):**
```txt
streamlit==1.28.0
ultralytics==8.0.196
opencv-python-headless==4.8.1.78  # Use headless for cloud
torch==2.0.1
torchvision==0.15.2
pillow==10.0.1
pandas==2.0.3
plotly==5.17.0
pydeck==0.8.0
```

### Option 2: Heroku

**1. Create Heroku app:**
```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create roadguardian-ai

# Set buildpacks
heroku buildpacks:add --index 1 heroku/python
```

**2. Create `Procfile`:**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**3. Create `setup.sh`:**
```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your.email@example.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

**4. Deploy:**
```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### Option 3: AWS EC2

**1. Launch EC2 instance:**
- Instance type: t3.medium (CPU) or g4dn.xlarge (GPU)
- OS: Ubuntu 22.04 LTS
- Storage: 20GB+
- Security group: Allow port 8501

**2. SSH into instance:**
```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
```

**3. Setup environment:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python & dependencies
sudo apt install python3-pip python3-venv -y

# Clone repository
git clone https://github.com/yourusername/pothole-detection.git
cd pothole-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**4. Run as service (systemd):**

Create `/etc/systemd/system/roadguardian.service`:
```ini
[Unit]
Description=RoadGuardian AI Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/pothole-detection
Environment="PATH=/home/ubuntu/pothole-detection/venv/bin"
ExecStart=/home/ubuntu/pothole-detection/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0

[Install]
WantedBy=multi-user.target
```

**5. Start service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable roadguardian
sudo systemctl start roadguardian
sudo systemctl status roadguardian
```

**6. Setup nginx (optional reverse proxy):**
```bash
sudo apt install nginx -y

# Create nginx config
sudo nano /etc/nginx/sites-available/roadguardian
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/roadguardian /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Option 4: Google Cloud Platform (GCP)

**Using Cloud Run (serverless):**

**1. Create Dockerfile (see Docker section)**

**2. Build and push:**
```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Build image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/roadguardian

# Deploy to Cloud Run
gcloud run deploy roadguardian \
  --image gcr.io/YOUR_PROJECT_ID/roadguardian \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Option 5: Azure

**Using Azure Container Instances:**

```bash
# Login
az login

# Create resource group
az group create --name roadguardian-rg --location eastus

# Deploy container
az container create \
  --resource-group roadguardian-rg \
  --name roadguardian \
  --image YOUR_DOCKERHUB/roadguardian:latest \
  --dns-name-label roadguardian \
  --ports 8501 \
  --cpu 2 \
  --memory 4
```

## Docker Deployment

### Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t roadguardian-ai:v2.0 .

# Run container
docker run -d \
  --name roadguardian \
  -p 8501:8501 \
  --restart unless-stopped \
  roadguardian-ai:v2.0

# Check logs
docker logs -f roadguardian

# Stop container
docker stop roadguardian

# Remove container
docker rm roadguardian
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./model:/app/model
      - ./data:/app/data
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**Run with compose:**
```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### GPU Support (Docker)

For GPU-accelerated inference:

```dockerfile
# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3-pip

# ... rest of Dockerfile
```

**Run with GPU:**
```bash
docker run -d \
  --gpus all \
  --name roadguardian \
  -p 8501:8501 \
  roadguardian-ai:v2.0
```

## Production Considerations

### 1. Security

**Environment Variables:**
```bash
# Don't hardcode secrets
# Use environment variables

# .env file (DO NOT commit to git)
API_KEY=your_secret_key
DATABASE_URL=postgresql://...
MODEL_PATH=model/best.pt
```

**Load in Python:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')
```

**HTTPS/SSL:**
```bash
# Use Let's Encrypt for free SSL
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 2. Performance Optimization

**Model Optimization:**
```python
# Export to ONNX for faster inference
from ultralytics import YOLO

model = YOLO('model/best.pt')
model.export(format='onnx')  # Creates best.onnx

# Use ONNX model
model = YOLO('model/best.onnx')
```

**Caching:**
```python
import streamlit as st

@st.cache_resource
def load_model():
    return YOLO('model/best.pt')

model = load_model()  # Only loads once
```

**Async Processing:**
```python
# For API deployments
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def process_image(image):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, model.predict, image)
    return result
```

### 3. Load Balancing

**Using nginx:**
```nginx
upstream roadguardian {
    server localhost:8501;
    server localhost:8502;
    server localhost:8503;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://roadguardian;
    }
}
```

### 4. Database Integration

**For storing results:**
```python
import psycopg2
from datetime import datetime

def save_detection(image_id, detections):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO detections (image_id, timestamp, count, results)
        VALUES (%s, %s, %s, %s)
    """, (image_id, datetime.now(), len(detections), json.dumps(detections)))
    
    conn.commit()
    cur.close()
    conn.close()
```

### 5. API Deployment (Alternative to Streamlit)

**Using FastAPI:**

```python
# api.py
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO('model/best.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Predict
    results = model.predict(image, conf=0.15)
    
    # Format response
    detections = []
    for box in results[0].boxes:
        detections.append({
            'class': model.names[int(box.cls[0])],
            'confidence': float(box.conf[0]),
            'bbox': box.xyxy[0].tolist()
        })
    
    return {'count': len(detections), 'detections': detections}

@app.get("/health")
async def health():
    return {'status': 'healthy'}
```

**Run FastAPI:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Monitoring & Maintenance

### 1. Logging

**Configure logging:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use in code
logger.info(f"Processed image: {image_id}")
logger.error(f"Prediction failed: {error}")
```

### 2. Metrics Collection

**Track performance:**
```python
import time
from prometheus_client import Counter, Histogram

# Metrics
predictions_total = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

# Use
@prediction_duration.time()
def predict_image(image):
    results = model.predict(image)
    predictions_total.inc()
    return results
```

### 3. Error Tracking

**Use Sentry (optional):**
```bash
pip install sentry-sdk
```

```python
import sentry_sdk

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    traces_sample_rate=1.0
)
```

### 4. Health Checks

**Endpoint for monitoring:**
```python
@app.get("/health")
def health_check():
    try:
        # Test model
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(dummy, verbose=False)
        return {'status': 'healthy', 'model': 'loaded'}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}, 500
```

### 5. Automated Backups

```bash
#!/bin/bash
# backup.sh

# Backup model
cp model/best.pt backups/best_$(date +%Y%m%d).pt

# Backup database (if applicable)
pg_dump database_name > backups/db_$(date +%Y%m%d).sql

# Keep only last 7 days
find backups/ -name "*.pt" -mtime +7 -delete
```

## Scaling

### Horizontal Scaling

**Using Kubernetes:**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: roadguardian
spec:
  replicas: 3
  selector:
    matchLabels:
      app: roadguardian
  template:
    metadata:
      labels:
        app: roadguardian
    spec:
      containers:
      - name: app
        image: roadguardian-ai:v2.0
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Auto-scaling

**GCP Cloud Run auto-scaling:**
```bash
gcloud run deploy roadguardian \
  --min-instances 1 \
  --max-instances 10 \
  --cpu 2 \
  --memory 4Gi
```

## Deployment Checklist

### Pre-Deployment
- [ ] Model trained and validated (mAP > 80%)
- [ ] Code tested locally
- [ ] Dependencies documented (requirements.txt)
- [ ] Environment variables configured
- [ ] Security review completed
- [ ] Documentation updated

### Deployment
- [ ] Choose deployment platform
- [ ] Setup CI/CD pipeline (optional)
- [ ] Configure domain/DNS
- [ ] Setup SSL/HTTPS
- [ ] Deploy application
- [ ] Test production endpoint

### Post-Deployment
- [ ] Monitor performance metrics
- [ ] Setup logging and alerts
- [ ] Test with real users
- [ ] Document known issues
- [ ] Plan for updates/maintenance

## Troubleshooting

### Common Issues

**1. High Memory Usage**
- Reduce batch size
- Use model optimization (ONNX)
- Enable model quantization

**2. Slow Response Times**
- Use GPU acceleration
- Implement caching
- Optimize image preprocessing

**3. Model Loading Fails**
- Check file permissions
- Verify model path
- Ensure dependencies installed

**4. Connection Issues**
- Check firewall rules
- Verify security groups
- Test network connectivity

## Next Steps

After successful deployment:

1. **Monitor:** Track performance and errors
2. **Optimize:** Improve speed and accuracy
3. **Scale:** Add more resources as needed
4. **Update:** Regular model retraining
5. **Maintain:** Bug fixes and improvements

---

**Deployment Guide Version:** 1.0  
**Last Updated:** 2024-11-30  
**Compatible with:** RoadGuardian AI v2.0
