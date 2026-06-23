#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a YOLO-format detection dataset (Ultralytics layout) into the
annotation format FER-YOLO-Mamba expects.

Expected input layout (auto-detected, both supported):

    A) split_FER2013_original/
         train/images/*.jpg   train/labels/*.txt
         val/images/*.jpg     val/labels/*.txt
         test/images/*.jpg    test/labels/*.txt

    B) split_FER2013_original/
         train/*.jpg + train/*.txt   (images and labels mixed in one folder)
         val/...   test/...

YOLO label line format (normalized 0-1):
    class_id  x_center  y_center  width  height

Output (written to output_dir):
    raf_train.txt   <- from train/
    raf_val.txt     <- from val/  (falls back to test/ if val is missing)

Output line format (absolute pixel coords, what the repo's dataloader reads):
    /abs/path/img.jpg  x1,y1,x2,y2,cls  x1,y1,x2,y2,cls ...
"""

import os
import argparse

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def _resolve_split_dirs(split_dir):
    """Return (img_dir, lbl_dir) for a split folder, supporting both layouts."""
    img_sub = os.path.join(split_dir, 'images')
    lbl_sub = os.path.join(split_dir, 'labels')
    if os.path.isdir(img_sub) and os.path.isdir(lbl_sub):
        return img_sub, lbl_sub
    # flat layout: images and labels in the same folder
    return split_dir, split_dir


def _label_path_for(image_path, img_dir, lbl_dir):
    """Find the .txt label that corresponds to an image."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(lbl_dir, base + '.txt')


def _convert_split(split_dir, out_file):
    img_dir, lbl_dir = _resolve_split_dirs(split_dir)

    images = [
        os.path.join(img_dir, f)
        for f in sorted(os.listdir(img_dir))
        if f.lower().endswith(IMG_EXTS)
    ]

    from PIL import Image

    lines_out = []
    n_boxes = 0
    n_no_label = 0
    n_empty = 0

    for img_path in images:
        lbl_path = _label_path_for(img_path, img_dir, lbl_dir)
        if not os.path.exists(lbl_path):
            n_no_label += 1
            continue

        with open(lbl_path) as f:
            label_lines = [l.strip() for l in f if l.strip()]

        if not label_lines:
            n_empty += 1
            continue

        # need image size to un-normalize
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            continue

        boxes = []
        for ll in label_lines:
            parts = ll.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            # normalized -> absolute pixel corners
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(f"{x1},{y1},{x2},{y2},{cls}")

        if not boxes:
            n_empty += 1
            continue

        lines_out.append(os.path.abspath(img_path) + ' ' + ' '.join(boxes) + '\n')
        n_boxes += len(boxes)

    with open(out_file, 'w') as f:
        f.writelines(lines_out)

    print(f"  {os.path.basename(split_dir):6} -> {out_file}")
    print(f"        images with boxes : {len(lines_out)}")
    print(f"        total boxes       : {n_boxes}")
    print(f"        missing label     : {n_no_label}")
    print(f"        empty label       : {n_empty}")
    return len(lines_out)


def convert(dataset_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(dataset_root, 'train')
    val_dir   = os.path.join(dataset_root, 'val')
    test_dir  = os.path.join(dataset_root, 'test')

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train/ folder not found in {dataset_root}")

    print(f"Converting YOLO dataset: {dataset_root}\n")

    train_file = os.path.join(output_dir, 'raf_train.txt')
    val_file   = os.path.join(output_dir, 'raf_val.txt')

    n_train = _convert_split(train_dir, train_file)

    # validation: prefer val/, fall back to test/
    if os.path.isdir(val_dir):
        n_val = _convert_split(val_dir, val_file)
    elif os.path.isdir(test_dir):
        print("  (no val/ folder, using test/ for validation)")
        n_val = _convert_split(test_dir, val_file)
    else:
        raise FileNotFoundError("Neither val/ nor test/ folder found.")

    print(f"\nDone!  train={n_train}  val={n_val}")
    if n_train == 0:
        raise RuntimeError(
            "0 training samples produced. Check that label .txt files exist "
            "next to (or in a labels/ folder beside) the images."
        )
    return train_file, val_file


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True,
                    help='Root of the split dataset (contains train/, val/, test/)')
    ap.add_argument('--output_dir', default='./')
    args = ap.parse_args()
    convert(args.dataset_root, args.output_dir)
