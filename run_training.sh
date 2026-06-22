#!/bin/bash
# ============================================================
# Training launcher — FER-YOLO-Mamba (RunPod / generic GPU)
# Usage:  bash run_training.sh [path-to-dataset-root]
#
# Example:
#   bash run_training.sh /workspace/basic-20260621T140957Z-3-001/basic
#
# If no argument is given, uses the dataset folder beside this script.
#
# Training config auto-scales to your GPU VRAM. Override anything via env vars:
#   EPOCHS=2 UNFREEZE_BATCH=4 bash run_training.sh /path/to/basic   # quick test
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Dataset root (folder containing EmoLabel/, Annotation/, Image/) ──
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

if [ ! -d "${DATASET_ROOT}/EmoLabel" ]; then
    echo ""
    echo "ERROR: Dataset not found at ${DATASET_ROOT}"
    echo "Expected sub-folders: EmoLabel/, Annotation/, Image/"
    echo ""
    echo "Usage: bash run_training.sh /path/to/basic"
    exit 1
fi

cd "${SCRIPT_DIR}"

echo ""
echo "=============================="
echo " Step 1 - Generate annotations"
echo "=============================="
python3 convert_raf_to_yolo.py \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${SCRIPT_DIR}"

echo ""
echo "=============================="
echo " Step 2 - Start training (config auto-scales to GPU VRAM)"
echo "=============================="
python3 train_kaggle.py

echo ""
echo "=============================="
echo " Training complete!"
echo " Checkpoints saved in: ${SCRIPT_DIR}/logs/"
echo "=============================="
