#!/bin/bash
# ============================================================
# RunPod Training — FER-YOLO-Mamba on RTX 6000
#
# Usage:
#   bash run_training.sh /path/to/basic
#
# Example:
#   bash run_training.sh /workspace/dataset/basic
#
# If no argument given, looks for the dataset next to this script.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Dataset root — the folder containing EmoLabel/, Annotation/, Image/
if [ -n "$1" ]; then
    DATASET_ROOT="$1"
else
    DATASET_ROOT="${SCRIPT_DIR}/basic-20260621T140957Z-3-001/basic"
fi

echo "=============================="
echo " Config"
echo "=============================="
echo "Project dir  : ${SCRIPT_DIR}"
echo "Dataset root : ${DATASET_ROOT}"

# Validate dataset
if [ ! -d "${DATASET_ROOT}/EmoLabel" ]; then
    echo ""
    echo "ERROR: EmoLabel/ not found inside ${DATASET_ROOT}"
    echo "Usage: bash run_training.sh /path/to/basic"
    exit 1
fi

cd "${SCRIPT_DIR}"

echo ""
echo "=============================="
echo " Step 1 — Generate annotations"
echo "=============================="
python3 convert_raf_to_yolo.py \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${SCRIPT_DIR}"

# Verify
TRAIN_COUNT=$(wc -l < "${SCRIPT_DIR}/raf_train.txt")
VAL_COUNT=$(wc -l < "${SCRIPT_DIR}/raf_val.txt")
echo "Train samples : ${TRAIN_COUNT}"
echo "Val   samples : ${VAL_COUNT}"

if [ "$TRAIN_COUNT" -eq 0 ]; then
    echo "ERROR: 0 training samples generated. Check dataset path."
    exit 1
fi

echo ""
echo "=============================="
echo " Step 2 — Start training"
echo " RTX 6000: 640x640, batch 64/32, fp16, 8 workers"
echo "=============================="
python3 train_kaggle.py

echo ""
echo "=============================="
echo " Training complete!"
echo " Checkpoints saved in: ${SCRIPT_DIR}/logs/"
echo "=============================="
