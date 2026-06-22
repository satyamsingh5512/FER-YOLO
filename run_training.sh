#!/bin/bash
# ============================================================
# RunPod training launcher — FER-YOLO-Mamba
# ============================================================
#
# Usage (all equivalent — dataset is auto-located or downloaded):
#
#   bash run_training.sh                         # auto-detect / auto-download
#   bash run_training.sh /path/to/basic          # explicit dataset root
#
# Training config auto-scales to GPU VRAM. Override via env vars:
#   EPOCHS=2 UNFREEZE_BATCH=4 bash run_training.sh   # quick smoke test
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=============================="
echo " FER-YOLO-Mamba Training"
echo "=============================="
echo "Project dir : ${SCRIPT_DIR}"
echo ""

# ── Step 0: Locate (or download) the RAF-DB dataset ─────────────────────────

RAF_ROOT=""

# Priority 1: explicit argument
if [ -n "$1" ] && [ -d "$1/EmoLabel" ]; then
    RAF_ROOT="$1"
    echo "Dataset : ${RAF_ROOT}  (from argument)"

# Priority 2: .dataset_path file (written by download_dataset.py)
elif [ -f "${SCRIPT_DIR}/.dataset_path" ]; then
    SAVED=$(cat "${SCRIPT_DIR}/.dataset_path")
    if [ -d "${SAVED}/EmoLabel" ]; then
        RAF_ROOT="${SAVED}"
        echo "Dataset : ${RAF_ROOT}  (from .dataset_path)"
    fi
fi

# Priority 3: scan common local locations
if [ -z "${RAF_ROOT}" ]; then
    RAF_ROOT=$(python3 - << 'PYEOF'
import os, sys

def find_raf(start):
    required = {"EmoLabel", "Annotation", "Image"}
    for root, dirs, files in os.walk(start):
        if required <= set(dirs):
            return root
        # don't descend into hidden or cache dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
    return None

# Search script dir and parent (covers typical RunPod /workspace layout)
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
for search in [script_dir, os.path.dirname(script_dir), "/workspace"]:
    r = find_raf(search)
    if r:
        print(r)
        sys.exit(0)
PYEOF
    )
    if [ -n "${RAF_ROOT}" ] && [ -d "${RAF_ROOT}/EmoLabel" ]; then
        echo "Dataset : ${RAF_ROOT}  (auto-detected)"
    else
        RAF_ROOT=""
    fi
fi

# Priority 4: download from Google Drive
if [ -z "${RAF_ROOT}" ]; then
    echo ""
    echo "Dataset not found locally."
    echo "Downloading RAF-DB from Google Drive (~1.8 GB) …"
    echo ""
    python3 download_dataset.py --output "${SCRIPT_DIR}/dataset"
    if [ -f "${SCRIPT_DIR}/.dataset_path" ]; then
        RAF_ROOT=$(cat "${SCRIPT_DIR}/.dataset_path")
    fi
fi

# Verify we have a valid root
if [ -z "${RAF_ROOT}" ] || [ ! -d "${RAF_ROOT}/EmoLabel" ]; then
    echo ""
    echo "ERROR: Could not locate RAF-DB dataset (EmoLabel/ not found)."
    echo ""
    echo "Options:"
    echo "  1. Re-run with an explicit path:"
    echo "       bash run_training.sh /path/to/basic"
    echo "  2. Upload the dataset manually then re-run:"
    echo "       scp -r basic/ root@<POD_IP>:${SCRIPT_DIR}/"
    echo "       bash run_training.sh ${SCRIPT_DIR}/basic"
    echo "  3. Run the downloader directly:"
    echo "       python3 download_dataset.py --output ${SCRIPT_DIR}/dataset"
    exit 1
fi

echo ""
echo "=============================="
echo " Step 1 — Generate annotations"
echo "=============================="
python3 convert_raf_to_yolo.py \
    --dataset_root "${RAF_ROOT}" \
    --output_dir   "${SCRIPT_DIR}"

echo ""
echo "=============================="
echo " Step 2 — Start training  (config auto-scales to GPU VRAM)"
echo "=============================="
python3 train_kaggle.py

echo ""
echo "=============================="
echo " Training complete!"
echo " Checkpoints saved in: ${SCRIPT_DIR}/logs/"
echo "=============================="
