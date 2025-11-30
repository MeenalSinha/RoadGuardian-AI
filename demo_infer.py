#!/usr/bin/env python3
"""Pothole Detection CLI Inference"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

def log(msg, status="info"):
    icons = {"info": "ℹ️", "ok": "✅", "error": "❌", "rocket": "🚀"}
    print(f"{icons.get(status, 'ℹ️')} {msg}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Input image')
    parser.add_argument('--model', default='model/best.pt', help='Model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence')
    parser.add_argument('--output', default='outputs/predictions', help='Output dir')
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        log(f"Image not found: {args.image}", "error")
        return
    
    log(f"Loading: {args.model}", "rocket")
    model = YOLO(args.model)
    
    log(f"Inference: {args.image}", "rocket")
    results = model(args.image, conf=args.conf)
    boxes = results[0].boxes
    log(f"Detected: {len(boxes)} pothole(s)", "ok")
    
    for i, box in enumerate(boxes):
        print(f"  {i+1}. Conf: {float(box.conf[0]):.3f}")
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) / f"result_{Path(args.image).name}"
    cv2.imwrite(str(output_path), results[0].plot())
    log(f"Saved: {output_path}", "ok")

if __name__ == "__main__":
    main()
