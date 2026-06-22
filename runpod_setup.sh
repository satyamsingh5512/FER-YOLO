#!/bin/bash
# ============================================================
# RunPod / generic CUDA-GPU setup — FER-YOLO-Mamba
# Run once after starting your pod:  bash runpod_setup.sh
#
# Recommended RunPod template: any "PyTorch 2.6 - 2.10" image on CUDA 12.x.
# Works on T4/V100/3090/4090/L4/A10/A100/RTX 6000 (config auto-scales to VRAM).
# ============================================================

set -e   # stop on any error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=============================="
echo " GPU Info"
echo "=============================="
nvidia-smi || echo "WARNING: nvidia-smi not found — is this a GPU pod?"

echo ""
echo "=============================="
echo " Installing python dependencies"
echo "=============================="
# Lightweight pure-python deps (torch is already in the image — do NOT reinstall it)
pip install -q -r requirements.txt

echo ""
echo "=============================="
echo " Installing CUDA extensions (causal-conv1d + mamba-ssm)"
echo " Uses prebuilt wheels matched to this torch/CUDA — no slow source build"
echo "=============================="
python3 install_mamba.py

echo ""
echo "=============================="
echo " Final verification"
echo "=============================="
python3 -c "
import torch
print(f'PyTorch  : {torch.__version__}')
print(f'CUDA     : {torch.version.cuda}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU      : {p.name}')
    print(f'VRAM     : {p.total_memory / 1e9:.1f} GB')
else:
    print('GPU      : NOT AVAILABLE — training needs a CUDA GPU')
import selective_scan_cuda  # noqa: F401
print('selective_scan_cuda : OK')
import timm
print(f'timm     : {timm.__version__}')
"

echo ""
echo "Setup complete. Now run:  bash run_training.sh /path/to/basic"
