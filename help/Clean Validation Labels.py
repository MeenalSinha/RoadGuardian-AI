import os
import re
from pathlib import Path

labels_dir = Path("data/yolo/val/labels")
images_dir = Path("data/yolo/val/images")

removed = 0
for file in labels_dir.glob("*.txt"):
    with open(file) as f:
        lines = f.readlines()
    
    # Check for invalid class IDs (> 2)
    invalid = any(re.match(r'^[3-9]\b', line) for line in lines)
    
    if invalid:
        print(f"🗑️ Removing: {file.name}")
        file.unlink()
        
        # Remove corresponding image
        img_path = images_dir / file.name.replace('.txt', '.jpg')
        if img_path.exists():
            img_path.unlink()
        removed += 1

print(f"✅ Removed {removed} invalid validation samples")