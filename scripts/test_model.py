"""
Model Testing Script - RoadGuardian AI
=======================================

This script tests the trained YOLOv8 model on sample images and generates
a comprehensive test report with visualizations.

Requirements:
- Trained model (model/best.pt)
- Test images in data/sample_images/

Usage:
    python scripts/test_model.py
    python scripts/test_model.py --model model/best.pt --images data/sample_images/
    python scripts/test_model.py --conf 0.15 --iou 0.45 --save-annotated

Author: Your Name
Date: 2024
"""

import argparse
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: ultralytics not installed")
    print("Install with: pip install ultralytics")
    sys.exit(1)


# Class names and colors
CLASS_NAMES = {0: 'pothole', 1: 'crack', 2: 'damage'}
CLASS_COLORS = {
    'pothole': (0, 0, 255),      # Red
    'crack': (0, 255, 255),       # Yellow
    'damage': (0, 100, 255)       # Orange
}


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def print_step(step, text):
    """Print formatted step"""
    print(f"[{step}] {text}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test RoadGuardian AI model on sample images"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='model/best.pt',
        help='Path to trained model (default: model/best.pt)'
    )
    parser.add_argument(
        '--images',
        type=str,
        default='data/sample_images',
        help='Path to test images directory (default: data/sample_images)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_results',
        help='Output directory for results (default: test_results)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.15,
        help='Confidence threshold (default: 0.15)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IOU threshold for NMS (default: 0.45)'
    )
    parser.add_argument(
        '--save-annotated',
        action='store_true',
        help='Save annotated images'
    )
    
    return parser.parse_args()


def load_model(model_path):
    """Load YOLO model"""
    print_step("LOAD", f"Loading model from: {model_path}")
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
        print(f"   Type: YOLOv8n")
        print(f"   Classes: {len(model.names)}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return None


def get_test_images(images_dir):
    """Get list of test images"""
    print_step("SCAN", f"Scanning for images in: {images_dir}")
    
    images_path = Path(images_dir)
    if not images_path.exists():
        print(f"❌ Directory not found: {images_dir}")
        return []
    
    # Supported formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    
    for ext in extensions:
        images.extend(list(images_path.glob(ext)))
    
    print(f"✅ Found {len(images)} images")
    return sorted(images)


def annotate_image(image, boxes, model):
    """Annotate image with detection boxes"""
    annotated = image.copy()
    
    for box in boxes:
        # Extract box information
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = CLASS_NAMES[cls]
        
        # Get color
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Prepare label (no emoji, only text)
        label = f"{class_name} | {conf:.2%}"
        
        # Calculate label size
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw label background
        cv2.rectangle(
            annotated,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 10, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return annotated


def test_single_image(model, image_path, conf_threshold, iou_threshold):
    """Test model on a single image"""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Run inference
    results = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    result = results[0]
    boxes = result.boxes
    
    # Extract detection information
    detections = []
    class_counts = {'pothole': 0, 'crack': 0, 'damage': 0}
    confidences = []
    
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = CLASS_NAMES[cls]
        
        detections.append({
            'class': class_name,
            'confidence': conf,
            'bbox': box.xyxy[0].tolist()
        })
        
        class_counts[class_name] += 1
        confidences.append(conf)
    
    # Calculate statistics
    stats = {
        'total_detections': len(boxes),
        'class_counts': class_counts,
        'confidences': confidences,
        'avg_confidence': np.mean(confidences) if confidences else 0.0,
        'max_confidence': max(confidences) if confidences else 0.0,
        'min_confidence': min(confidences) if confidences else 0.0
    }
    
    return {
        'image_path': str(image_path),
        'image_name': image_path.name,
        'detections': detections,
        'stats': stats,
        'annotated_image': annotate_image(image, boxes, model)
    }


def test_model(model, images, conf_threshold, iou_threshold, output_dir, save_annotated):
    """Test model on all images"""
    print_step("TEST", f"Testing model on {len(images)} images...")
    
    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for annotated images
    if save_annotated:
        annotated_dir = output_path / "annotated_images"
        annotated_dir.mkdir(exist_ok=True)
    
    for i, image_path in enumerate(images, 1):
        print(f"   Testing {i}/{len(images)}: {image_path.name}")
        
        result = test_single_image(model, image_path, conf_threshold, iou_threshold)
        
        if result is None:
            print(f"   ⚠️  Skipped (failed to load)")
            continue
        
        results.append(result)
        
        # Save annotated image
        if save_annotated:
            annotated_path = annotated_dir / f"annotated_{result['image_name']}"
            cv2.imwrite(str(annotated_path), result['annotated_image'])
        
        # Print summary
        stats = result['stats']
        print(f"      Detections: {stats['total_detections']} "
              f"(P:{stats['class_counts']['pothole']}, "
              f"C:{stats['class_counts']['crack']}, "
              f"D:{stats['class_counts']['damage']}) "
              f"| Avg conf: {stats['avg_confidence']:.2%}")
    
    print(f"\n✅ Testing complete: {len(results)}/{len(images)} images processed")
    return results


def generate_summary(results, conf_threshold, iou_threshold):
    """Generate summary statistics"""
    print_step("SUMMARY", "Generating test summary...")
    
    total_images = len(results)
    total_detections = sum(r['stats']['total_detections'] for r in results)
    
    # Class counts
    total_potholes = sum(r['stats']['class_counts']['pothole'] for r in results)
    total_cracks = sum(r['stats']['class_counts']['crack'] for r in results)
    total_damage = sum(r['stats']['class_counts']['damage'] for r in results)
    
    # Confidence statistics
    all_confidences = []
    for r in results:
        all_confidences.extend(r['stats']['confidences'])
    
    summary = {
        'test_info': {
            'timestamp': datetime.now().isoformat(),
            'confidence_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'total_images': total_images
        },
        'detection_summary': {
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / total_images if total_images > 0 else 0,
            'class_distribution': {
                'pothole': total_potholes,
                'crack': total_cracks,
                'damage': total_damage
            },
            'class_percentages': {
                'pothole': (total_potholes / total_detections * 100) if total_detections > 0 else 0,
                'crack': (total_cracks / total_detections * 100) if total_detections > 0 else 0,
                'damage': (total_damage / total_detections * 100) if total_detections > 0 else 0
            }
        },
        'confidence_stats': {
            'mean': float(np.mean(all_confidences)) if all_confidences else 0.0,
            'median': float(np.median(all_confidences)) if all_confidences else 0.0,
            'std': float(np.std(all_confidences)) if all_confidences else 0.0,
            'min': float(np.min(all_confidences)) if all_confidences else 0.0,
            'max': float(np.max(all_confidences)) if all_confidences else 0.0,
            'range': float(np.max(all_confidences) - np.min(all_confidences)) if all_confidences else 0.0
        },
        'per_image_results': [
            {
                'image': r['image_name'],
                'detections': r['stats']['total_detections'],
                'avg_confidence': r['stats']['avg_confidence']
            }
            for r in results
        ]
    }
    
    return summary


def print_summary(summary):
    """Print summary to console"""
    print_header("TEST SUMMARY")
    
    # Test info
    print("📋 Test Information:")
    print(f"   Timestamp: {summary['test_info']['timestamp']}")
    print(f"   Confidence threshold: {summary['test_info']['confidence_threshold']}")
    print(f"   IOU threshold: {summary['test_info']['iou_threshold']}")
    print(f"   Images tested: {summary['test_info']['total_images']}")
    
    # Detection summary
    print("\n📊 Detection Summary:")
    det = summary['detection_summary']
    print(f"   Total detections: {det['total_detections']}")
    print(f"   Avg per image: {det['avg_detections_per_image']:.1f}")
    print(f"\n   Class Distribution:")
    print(f"      Pothole: {det['class_distribution']['pothole']} "
          f"({det['class_percentages']['pothole']:.1f}%)")
    print(f"      Crack:   {det['class_distribution']['crack']} "
          f"({det['class_percentages']['crack']:.1f}%)")
    print(f"      Damage:  {det['class_distribution']['damage']} "
          f"({det['class_percentages']['damage']:.1f}%)")
    
    # Confidence stats
    print("\n🎯 Confidence Statistics:")
    conf = summary['confidence_stats']
    print(f"   Mean:   {conf['mean']:.4f} ({conf['mean']*100:.2f}%)")
    print(f"   Median: {conf['median']:.4f} ({conf['median']*100:.2f}%)")
    print(f"   Std:    {conf['std']:.4f}")
    print(f"   Range:  {conf['min']:.4f} - {conf['max']:.4f}")
    
    # Performance assessment
    print("\n⭐ Performance Assessment:")
    avg_conf = conf['mean']
    if avg_conf >= 0.7:
        print("   Confidence: EXCELLENT ✅")
    elif avg_conf >= 0.5:
        print("   Confidence: GOOD ✅")
    elif avg_conf >= 0.3:
        print("   Confidence: FAIR ⚠️")
    else:
        print("   Confidence: LOW ❌")
    
    if conf['max'] >= 0.8:
        print("   Max confidence: EXCELLENT (≥80%) ✅")
    elif conf['max'] >= 0.5:
        print("   Max confidence: GOOD (≥50%) ✅")
    else:
        print("   Max confidence: NEEDS IMPROVEMENT ⚠️")


def save_results(results, summary, output_dir):
    """Save test results to JSON"""
    print_step("SAVE", "Saving results...")
    
    output_path = Path(output_dir)
    
    # Save detailed results
    results_file = output_path / "test_results_detailed.json"
    with open(results_file, 'w') as f:
        # Remove annotated images from JSON (too large)
        results_to_save = []
        for r in results:
            r_copy = r.copy()
            r_copy.pop('annotated_image', None)
            results_to_save.append(r_copy)
        
        json.dump(results_to_save, f, indent=2)
    print(f"✅ Detailed results: {results_file}")
    
    # Save summary
    summary_file = output_path / "test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary: {summary_file}")


def main():
    """Main execution function"""
    print_header("ROADGUARDIAN AI - MODEL TESTING SCRIPT")
    
    # Parse arguments
    args = parse_arguments()
    
    print("⚙️  Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Images: {args.images}")
    print(f"   Output: {args.output}")
    print(f"   Confidence threshold: {args.conf}")
    print(f"   IOU threshold: {args.iou}")
    print(f"   Save annotated: {args.save_annotated}")
    
    # Load model
    model = load_model(args.model)
    if model is None:
        sys.exit(1)
    
    # Get test images
    images = get_test_images(args.images)
    if not images:
        print("❌ No test images found!")
        sys.exit(1)
    
    # Test model
    results = test_model(
        model, images, args.conf, args.iou, args.output, args.save_annotated
    )
    
    if not results:
        print("❌ No results generated!")
        sys.exit(1)
    
    # Generate summary
    summary = generate_summary(results, args.conf, args.iou)
    
    # Print summary
    print_summary(summary)
    
    # Save results
    save_results(results, summary, args.output)
    
    # Final message
    print_header("TESTING COMPLETE!")
    print(f"📁 Results saved to: {args.output}")
    print("\nFiles generated:")
    print(f"   - test_summary.json")
    print(f"   - test_results_detailed.json")
    if args.save_annotated:
        print(f"   - annotated_images/ (annotated images)")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
