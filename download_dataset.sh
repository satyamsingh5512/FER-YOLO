#!/bin/bash
# ============================================================
# Download RAF-DB dataset from Google Drive
# Run once before training:  bash download_dataset.sh
# ============================================================
set -e

GDRIVE_FILE_ID="1NxXfJM4m8H3tnzKipySJ1mzIPzesO3IC"
DOWNLOAD_DIR="/workspace/dataset"
ZIP_PATH="${DOWNLOAD_DIR}/rafdb.zip"

echo "=============================="
echo " Downloading RAF-DB dataset"
echo " from Google Drive"
echo "=============================="

mkdir -p "${DOWNLOAD_DIR}"

# Install gdown (handles large GDrive files + confirmation tokens)
pip install -q gdown

echo "Downloading..."
gdown --id "${GDRIVE_FILE_ID}" -O "${ZIP_PATH}"

echo ""
echo "=============================="
echo " Extracting..."
echo "=============================="
unzip -q "${ZIP_PATH}" -d "${DOWNLOAD_DIR}"
rm "${ZIP_PATH}"

echo ""
echo "Contents of ${DOWNLOAD_DIR}:"
ls -la "${DOWNLOAD_DIR}"

echo ""
echo "=============================="
echo " Finding dataset root..."
echo "=============================="
# Find the folder containing EmoLabel/ + Annotation/ + Image/
DATASET_ROOT=$(python3 - <<PYEOF
import os
for root, dirs, files in os.walk('${DOWNLOAD_DIR}'):
    if all(m in dirs for m in ['EmoLabel', 'Annotation', 'Image']):
        print(root)
        break
PYEOF
)

if [ -z "$DATASET_ROOT" ]; then
    echo "ERROR: Could not find EmoLabel/+Annotation/+Image/ inside ${DOWNLOAD_DIR}"
    echo "Contents:"
    find "${DOWNLOAD_DIR}" -maxdepth 4 -type d
    exit 1
fi

echo "Dataset root: ${DATASET_ROOT}"
echo ""
echo "Download complete. Now run:"
echo "  bash run_training.sh ${DATASET_ROOT}"
