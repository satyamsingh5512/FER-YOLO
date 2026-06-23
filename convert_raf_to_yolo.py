#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert RAF-DB dataset to YOLO annotation format.

RAF-DB emotion labels:
    1=Surprise, 2=Fear, 3=Disgust, 4=Happiness, 5=Sadness, 6=Anger, 7=Neutral

SFEW class indices (sfew_classes.txt order):
    0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Neutral, 5=Sad, 6=Surprise
"""

import os
import argparse

RAF_TO_SFEW = {
    1: 6,  # Surprise  -> 6
    2: 2,  # Fear      -> 2
    3: 1,  # Disgust   -> 1
    4: 3,  # Happiness -> 3
    5: 5,  # Sadness   -> 5
    6: 0,  # Anger     -> 0
    7: 4,  # Neutral   -> 4
}


def _bbox_from_file(bbox_file):
    """Read x1,y1,x2,y2 from a RAF-DB boundingbox txt file."""
    with open(bbox_file, 'r') as f:
        parts = f.readline().strip().split()
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = int(float(parts[0])), int(float(parts[1])), \
                          int(float(parts[2])), int(float(parts[3]))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
        return x1, y1, x2, y2
    except ValueError:
        return None


def _bbox_from_image(img_path):
    """Fall back: use full image dimensions as bounding box."""
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            w, h = img.size
        return 0, 0, w, h
    except Exception:
        return None


def convert(dataset_root, output_dir):
    label_file = os.path.join(dataset_root, 'EmoLabel', 'list_patition_label.txt')
    bbox_dir   = os.path.join(dataset_root, 'Annotation', 'boundingbox')
    image_dir  = os.path.join(dataset_root, 'Image', 'aligned')

    # Validate required paths
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Aligned image dir not found: {image_dir}")

    # Check if bbox files are available
    bbox_available = os.path.isdir(bbox_dir) and any(
        f.endswith('_boundingbox.txt') for f in os.listdir(bbox_dir)
    ) if os.path.isdir(bbox_dir) else False

    if bbox_available:
        print(f"Bounding box files found in {bbox_dir}")
    else:
        print(f"WARNING: No bounding box files found in {bbox_dir}")
        print(f"         Falling back to full-image bbox (valid for 100x100 aligned face crops)")

    with open(label_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    train_ann, test_ann = [], []
    skipped = 0

    print(f"Processing {len(lines)} images from: {dataset_root}")

    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            skipped += 1
            continue

        img_name, emo = parts
        emo = int(emo)
        if emo not in RAF_TO_SFEW:
            skipped += 1
            continue

        class_id  = RAF_TO_SFEW[emo]
        base_name = img_name.replace('.jpg', '')
        aligned   = base_name + '_aligned.jpg'
        img_path  = os.path.join(image_dir, aligned)

        # Get bounding box: prefer file, fall back to full image
        bbox = None
        if bbox_available:
            bbox_file = os.path.join(bbox_dir, base_name + '_boundingbox.txt')
            if os.path.exists(bbox_file):
                bbox = _bbox_from_file(bbox_file)

        if bbox is None:
            # Full-image bbox fallback — valid since aligned images are face crops
            if not os.path.exists(img_path):
                skipped += 1
                continue
            bbox = _bbox_from_image(img_path)

        if bbox is None:
            skipped += 1
            continue

        x1, y1, x2, y2 = bbox
        ann_line = f"{img_path} {x1},{y1},{x2},{y2},{class_id}\n"

        if img_name.startswith('train_'):
            train_ann.append(ann_line)
        elif img_name.startswith('test_'):
            test_ann.append(ann_line)
        else:
            skipped += 1

    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, 'raf_train.txt')
    val_file   = os.path.join(output_dir, 'raf_val.txt')

    with open(train_file, 'w') as f:
        f.writelines(train_ann)
    with open(val_file, 'w') as f:
        f.writelines(test_ann)

    print(f"\nDone!")
    print(f"  Train samples : {len(train_ann)}")
    print(f"  Val   samples : {len(test_ann)}")
    print(f"  Skipped       : {skipped}")
    return train_file, val_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='basic-20260621T140957Z-3-001/basic')
    parser.add_argument('--output_dir',   default='./')
    args = parser.parse_args()
    convert(args.dataset_root, args.output_dir)
