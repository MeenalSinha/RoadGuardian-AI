"""
Interactive Model Test - RoadGuardian AI
=========================================

This script helps you test your model by asking for paths interactively.

Usage:
    python test_model_interactive.py
"""

import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import json
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Install with: pip install ultralytics opencv-python numpy")
    sys.exit(1)


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def find_files(extension, description):
    """Search for files with given extension"""
    print(f"Searching for {description}...")
    files = []
    
    for root, dirs, filenames in os.walk('.'):
        # Skip common folders
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv', 'node_modules']]
        
        for filename in filenames:
            if filename.endswith(extension):
                full_path = os.path.join(root, filename)
                files.append(full_path)
    
    return files


def find_image_folders():
    """Find folders containing images"""
    print("Searching for image folders...")
    image_folders = []
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv', 'node_modules']]
        
        img_count = sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        
        if img_count > 0:
            image_folders.append((root, img_count))
    
    return image_folders


def choose_from_list(items, item_type):
    """Let user choose from a list"""
    if not items:
        print(f"\n❌ No {item_type} found!")
        return None
    
    print(f"\nFound {len(items)} {item_type}:\n")
    
    for i, item in enumerate(items, 1):
        if isinstance(item, tuple):
            path, count = item
            print(f"{i}. {path} ({count} images)")
        else:
            print(f"{i}. {item}")
    
    print(f"\n0. Enter path manually")
    
    while True:
        try:
            choice = input(f"\nChoose {item_type} (0-{len(items)}): ").strip()
            
            if choice == '0':
                manual_path = input("Enter path: ").strip().strip('"')
                return manual_path
            
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                if isinstance(items[idx], tuple):
                    return items[idx][0]
                return items[idx]
            else:
                print(f"Invalid choice. Enter 0-{len(items)}")
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            return None


def test_model(model_path, images_path, conf_threshold=0.15):
    """Test the model"""
    print_header("TESTING MODEL")
    
    # Load model
    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Get images
    images_path = Path(images_path)
    if not images_path.exists():
        print(f"❌ Path not found: {images_path}")
        return False
    
    if images_path.is_file():
        images = [images_path]
    else:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for ext in extensions:
            images.extend(list(images_path.glob(ext)))
    
    if not images:
        print(f"❌ No images found in: {images_path}")
        return False
    
    print(f"Found {len(images)} images\n")
    
    # Test on a few images
    print("Testing model...")
    test_count = min(len(images), 10)
    
    all_detections = []
    
    for i, img_path in enumerate(images[:test_count], 1):
        print(f"[{i}/{test_count}] {img_path.name}...", end=' ')
        
        img = cv2.imread(str(img_path))
        if img is None:
            print("SKIP (failed to load)")
            continue
        
        results = model.predict(img, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes
        
        detections = len(boxes)
        all_detections.append(detections)
        
        print(f"{detections} detections")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY".center(80))
    print("="*80 + "\n")
    
    print(f"Images tested: {test_count}")
    print(f"Total detections: {sum(all_detections)}")
    print(f"Average per image: {np.mean(all_detections):.1f}")
    print(f"Min: {min(all_detections)}, Max: {max(all_detections)}")
    
    print("\n✅ Test completed successfully!")
    print("\nTo test all images and save results, run:")
    print(f'python test_model.py --model "{model_path}" --images "{images_path}" --save-annotated')
    
    return True


def main():
    """Main function"""
    print_header("INTERACTIVE MODEL TESTER")
    
    print("This tool will help you test your model interactively.\n")
    
    # Find model
    print("="*80)
    print("STEP 1: SELECT MODEL")
    print("="*80)
    
    models = find_files('.pt', 'model files')
    model_path = choose_from_list(models, 'model')
    
    if not model_path:
        print("\n❌ No model selected. Exiting.")
        return
    
    print(f"\n✅ Selected model: {model_path}")
    
    # Find images
    print("\n" + "="*80)
    print("STEP 2: SELECT IMAGE FOLDER")
    print("="*80)
    
    image_folders = find_image_folders()
    images_path = choose_from_list(image_folders, 'image folder')
    
    if not images_path:
        print("\n❌ No images selected. Exiting.")
        return
    
    print(f"\n✅ Selected images: {images_path}")
    
    # Confidence threshold
    print("\n" + "="*80)
    print("STEP 3: CONFIDENCE THRESHOLD")
    print("="*80)
    
    print("\nConfidence threshold (0.05-0.50):")
    print("  - 0.15 (recommended, default)")
    print("  - 0.25 (higher quality, fewer detections)")
    print("  - 0.05 (more detections, may have false positives)")
    
    conf_input = input("\nEnter confidence threshold [0.15]: ").strip()
    
    try:
        conf_threshold = float(conf_input) if conf_input else 0.15
        conf_threshold = max(0.05, min(0.50, conf_threshold))
    except ValueError:
        conf_threshold = 0.15
    
    print(f"\n✅ Using confidence threshold: {conf_threshold}")
    
    # Run test
    input("\nPress Enter to start testing...")
    
    test_model(model_path, images_path, conf_threshold)
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
