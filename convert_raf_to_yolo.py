#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert RAF-DB dataset to YOLO annotation format.

RAF-DB emotion labels:
    1=Surprise, 2=Fear, 3=Disgust, 4=Happiness, 5=Sadness, 6=Anger, 7=Neutral

SFEW class indices (sfew_classes.txt order):
    0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Neutral, 5=Sad, 6=Surprise

Usage:
    # Local
    python3 convert_raf_to_yolo.py --dataset_root basic-20260621T140957Z-3-001/basic

    # Kaggle (run inside notebook)
    python3 convert_raf_to_yolo.py \
        --dataset_root /kaggle/input/raf-db-basic/basic-20260621T140957Z-3-001/basic \
        --output_dir /kaggle/working
"""

import os
import argparse

# RAF-DB label -> SFEW class index
RAF_TO_SFEW = {
    1: 6,  # Surprise  -> 6
    2: 2,  # Fear      -> 2
    3: 1,  # Disgust   -> 1
    4: 3,  # Happiness -> 3
    5: 5,  # Sadness   -> 5
    6: 0,  # Anger     -> 0
    7: 4,  # Neutral   -> 4
}


def convert(dataset_root, output_dir):
    label_file = os.path.join(dataset_root, 'EmoLabel', 'list_patition_label.txt')
    bbox_dir   = os.path.join(dataset_root, 'Annotation', 'boundingbox')
    image_dir  = os.path.join(dataset_root, 'Image', 'aligned')

    for path, name in [(label_file, 'label file'), (bbox_dir, 'bbox dir'), (image_dir, 'image dir')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name}: {path}")

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
            print(f"  Warning: unknown label {emo} for {img_name}, skipping")
            skipped += 1
            continue

        class_id   = RAF_TO_SFEW[emo]
        base_name  = img_name.replace('.jpg', '')
        aligned    = base_name + '_aligned.jpg'
        bbox_file  = os.path.join(bbox_dir, base_name + '_boundingbox.txt')

        if not os.path.exists(bbox_file):
            skipped += 1
            continue

        with open(bbox_file, 'r') as f:
            bbox_line = f.readline().strip()

        parts_bb = bbox_line.split()
        if len(parts_bb) != 4:
            skipped += 1
            continue

        try:
            x1, y1, x2, y2 = int(float(parts_bb[0])), int(float(parts_bb[1])), \
                              int(float(parts_bb[2])), int(float(parts_bb[3]))
        except ValueError:
            skipped += 1
            continue

        # Clamp to valid range
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)

        img_path = os.path.join(image_dir, aligned)
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
    print(f"  Train file    : {train_file}")
    print(f"  Val   file    : {val_file}")
    return train_file, val_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert RAF-DB to YOLO annotation format')
    parser.add_argument(
        '--dataset_root',
        default='basic-20260621T140957Z-3-001/basic',
        help='Root of the RAF-DB basic folder (contains EmoLabel/, Annotation/, Image/)'
    )
    parser.add_argument(
        '--output_dir',
        default='./',
        help='Where to write raf_train.txt and raf_val.txt'
    )
    args = parser.parse_args()
    convert(args.dataset_root, args.output_dir)
