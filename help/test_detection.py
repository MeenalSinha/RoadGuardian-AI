from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# Test with your trained model
model_path = 'model/best.pt'

print(f"Loading model: {model_path}")
model = YOLO(model_path)

# Check what classes it knows
print(f"Model classes: {model.names}")

# Find a test image (adjust path to your actual image)
# Try multiple possible locations
possible_images = [
    'road.jpg',
    'test_images/road.jpg',
    'app/sample_images/road.jpg',
    'data/yolo/val/images/image_00000.jpg',  # From your dataset
]

img_path = None
for path in possible_images:
    if os.path.exists(path):
        img_path = path
        break

if img_path is None:
    # If no image found, just use first image from validation set
    val_dir = Path('data/yolo/val/images')
    if val_dir.exists():
        images = list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png'))
        if images:
            img_path = str(images[0])

if img_path is None:
    print("\n❌ No test image found!")
    print("Please provide an image path or place 'road.jpg' in the project folder")
    exit()

print(f"\n📷 Testing with image: {img_path}")

# Test detection with multiple confidence thresholds
for conf_thresh in [0.01, 0.10, 0.25, 0.50]:
    print(f"\n{'='*60}")
    print(f"Testing with confidence threshold: {conf_thresh}")
    print('='*60)
    
    results = model.predict(img_path, conf=conf_thresh, verbose=False)
    
    print(f"Detections found: {len(results[0].boxes)}")
    
    if len(results[0].boxes) > 0:
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            print(f"\n  Detection {i+1}:")
            print(f"    Class: {cls_name}")
            print(f"    Confidence: {confidence:.4f}")
            print(f"    BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            print(f"    Area: {area:.0f} px²")
    else:
        print("  ❌ No detections")

# Save annotated image with lowest threshold
print(f"\n{'='*60}")
print("Saving annotated image with conf=0.01...")
results = model.predict(img_path, conf=0.01, verbose=False)
annotated = results[0].plot()
cv2.imwrite('test_detection_result.jpg', annotated)
print("✅ Saved: test_detection_result.jpg")

# Also save with conf=0.25 (default in Streamlit)
results_025 = model.predict(img_path, conf=0.25, verbose=False)
annotated_025 = results_025[0].plot()
cv2.imwrite('test_detection_conf025.jpg', annotated_025)
print("✅ Saved: test_detection_conf025.jpg")