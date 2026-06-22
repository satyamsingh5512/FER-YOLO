#!/bin/bash
# ============================================================
# RunPod ONE-CLICK setup + train — FER-YOLO-Mamba
# ============================================================
#
# On a fresh RunPod GPU pod, this single script does everything:
#   1. Show GPU info
#   2. Install Python deps (requirements.txt)
#   3. Install CUDA extensions (causal-conv1d + mamba-ssm prebuilt wheels)
#   4. Download RAF-DB dataset from Google Drive (~1.8 GB)
#   5. Generate annotation files
#   6. Start training (config auto-scales to your GPU VRAM)
#
# Recommended RunPod template: PyTorch 2.6 – 2.10 on CUDA 12.x
# Works on: T4 / V100 / 3090 / 4090 / L4 / A10 / A100 / RTX 6000
#
# Usage
# -----
#   # Clone and run in one shot:
#   git clone https://github.com/satyamsingh5512/FER-YOLO /workspace/FER-YOLO
#   cd /workspace/FER-YOLO
#   bash runpod_setup.sh
#
#   # Already have the dataset locally? Pass its path to skip download:
#   bash runpod_setup.sh /path/to/basic
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Optional: user-supplied dataset path skips the Google Drive download
USER_DATASET_PATH="${1:-}"

# ── 1. GPU info ───────────────────────────────────────────────────────────────
echo "================================================"
echo " Step 1/6 — GPU Info"
echo "================================================"
nvidia-smi || { echo "WARNING: nvidia-smi not found."; echo "Ensure you started a GPU pod."; }
echo ""

# ── 2. Python deps ────────────────────────────────────────────────────────────
echo "================================================"
echo " Step 2/6 — Python dependencies"
echo "================================================"
pip install -q -r requirements.txt
echo "Done."
echo ""

# ── 3. CUDA extensions ───────────────────────────────────────────────────────
echo "================================================"
echo " Step 3/6 — CUDA extensions (causal-conv1d + mamba-ssm)"
echo "           Prebuilt wheels — no source compile needed"
echo "================================================"
python3 install_mamba.py
echo ""

# ── 4. Dataset ────────────────────────────────────────────────────────────────
echo "================================================"
echo " Step 4/6 — RAF-DB Dataset"
echo "================================================"

if [ -n "${USER_DATASET_PATH}" ] && [ -d "${USER_DATASET_PATH}/EmoLabel" ]; then
    # User supplied path — save it and skip download
    echo "Using supplied dataset: ${USER_DATASET_PATH}"
    echo "${USER_DATASET_PATH}" > "${SCRIPT_DIR}/.dataset_path"

elif [ -f "${SCRIPT_DIR}/.dataset_path" ]; then
    SAVED=$(cat "${SCRIPT_DIR}/.dataset_path")
    if [ -d "${SAVED}/EmoLabel" ]; then
        echo "Dataset already present: ${SAVED}"
    else
        python3 download_dataset.py --output "${SCRIPT_DIR}/dataset"
    fi

else
    echo "Downloading RAF-DB from Google Drive (~1.8 GB) …"
    python3 download_dataset.py --output "${SCRIPT_DIR}/dataset"
fi

RAF_ROOT=$(cat "${SCRIPT_DIR}/.dataset_path" 2>/dev/null || echo "")

if [ -z "${RAF_ROOT}" ] || [ ! -d "${RAF_ROOT}/EmoLabel" ]; then
    echo ""
    echo "ERROR: Dataset not found after download step."
    echo "See the manual-download instructions printed above, then re-run:"
    echo "  bash runpod_setup.sh /path/to/basic"
    exit 1
fi
echo "RAF-DB root : ${RAF_ROOT}"
echo ""

# ── 5. Generate annotations ───────────────────────────────────────────────────
echo "================================================"
echo " Step 5/6 — Generate annotation files"
echo "================================================"
python3 convert_raf_to_yolo.py \
    --dataset_root "${RAF_ROOT}" \
    --output_dir   "${SCRIPT_DIR}"
echo ""

# ── 6. Train ──────────────────────────────────────────────────────────────────
echo "================================================"
echo " Step 6/6 — Training  (auto-scaled to GPU VRAM)"
echo "================================================"
python3 train_kaggle.py

echo ""
echo "================================================"
echo " All done!  Checkpoints → ${SCRIPT_DIR}/logs/"
echo "================================================"
