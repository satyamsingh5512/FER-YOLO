#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert ImageFolder-style FER dataset to FER-YOLO-Mamba annotation format.

Input layout:
    split_FER2013_original/
        train/
            angry/      *.jpg
            disgust/    *.jpg
            fear/       *.jpg
            happy/      *.jpg
            neutral/    *.jpg
            sad/        *.jpg
            surprise/   *.jpg
        val/   (same structure)
        test/  (same structure)

Output line format (per image, full-image bbox since these are face crops):
    /abs/path/img.jpg  x1,y1,x2,y2,class_id

Class mapping (matches model_data/sfew_classes.txt / fer_classes.txt):
    0=angry  1=disgust  2=fear  3=happy  4=neutral  5=sad  6=surprise
"""

import os
import argparse
from PIL import Image

CLASS_MAP = {
    'angry':    0,
    'disgust':  1,
    'fear':     2,
    'happy':    3,
    'neutral':  4,
    'sad':      5,
    'surprise': 6,
}

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def _convert_split(split_dir, out_file):
    lines_out = []
    skipped   = 0

    class_dirs = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

    print(f"  Classes found: {class_dirs}")

    for cls_name in class_dirs:
        cls_id = CLASS_MAP.get(cls_name.lower())
        if cls_id is None:
            print(f"  WARNING: unknown class folder '{cls_name}', skipping")
            skipped += 1
            continue

        cls_dir = os.path.join(split_dir, cls_name)
        images  = [f for f in sorted(os.listdir(cls_dir)) if f.lower().endswith(IMG_EXTS)]

        for img_file in images:
            img_path = os.path.abspath(os.path.join(cls_dir, img_file))
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                skipped += 1
                continue

            # Full image as bounding box (images are already face crops)
            ann_line = f"{img_path} 0,0,{W},{H},{cls_id}\n"
            lines_out.append(ann_line)

    with open(out_file, 'w') as f:
        f.writelines(lines_out)

    print(f"  -> {out_file}")
    print(f"     samples : {len(lines_out)}  |  skipped : {skipped}")
    return len(lines_out)


def convert(dataset_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(dataset_root, 'train')
    val_dir   = os.path.join(dataset_root, 'val')
    test_dir  = os.path.join(dataset_root, 'test')

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train/ not found in {dataset_root}")

    print(f"Converting ImageFolder dataset: {dataset_root}\n")

    train_file = os.path.join(output_dir, 'raf_train.txt')
    val_file   = os.path.join(output_dir, 'raf_val.txt')

    print("train/")
    n_train = _convert_split(train_dir, train_file)

    if os.path.isdir(val_dir):
        print("val/")
        n_val = _convert_split(val_dir, val_file)
    elif os.path.isdir(test_dir):
        print("test/  (no val/ found, using test/ for validation)")
        n_val = _convert_split(test_dir, val_file)
    else:
        raise FileNotFoundError("Neither val/ nor test/ folder found.")

    print(f"\nDone!  train={n_train}  val={n_val}")

    if n_train == 0:
        raise RuntimeError("0 training samples. Check that class folders contain images.")

    return train_file, val_file


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--output_dir', default='./')
    args = ap.parse_args()
    convert(args.dataset_root, args.output_dir)
