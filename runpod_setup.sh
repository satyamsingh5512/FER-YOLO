#!/bin/bash
# ============================================================
# RunPod Setup — FER-YOLO-Mamba on RTX 6000
# Run once after starting your pod:  bash runpod_setup.sh
# ============================================================
set -e

echo "=============================="
echo " GPU Info"
echo "=============================="
nvidia-smi

echo ""
echo "=============================="
echo " Setting CUDA_HOME"
echo "=============================="
for p in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.9; do
    if [ -d "$p" ]; then
        export CUDA_HOME=$p
        export PATH=$CUDA_HOME/bin:$PATH
        echo "CUDA_HOME = $CUDA_HOME"
        break
    fi
done
export MAX_JOBS=4

echo ""
echo "=============================="
echo " Installing dependencies"
echo "=============================="
pip install -q packaging einops
pip install -q "timm>=0.6.13,<0.9.0"

# causal-conv1d is skipped — only selective_scan_cuda from mamba-ssm is needed
pip install --no-deps "mamba-ssm>=2.0.3" || \
pip install --no-deps --no-build-isolation "mamba-ssm>=2.0.3" || \
pip install "mamba-ssm>=2.0.3"

echo ""
echo "=============================="
echo " Verifying"
echo "=============================="
python3 - <<'PYEOF'
import torch
import selective_scan_cuda
import timm, einops

p = torch.cuda.get_device_properties(0)
print(f"GPU     : {p.name}")
print(f"VRAM    : {p.total_memory/1e9:.1f} GB")
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"timm    : {timm.__version__}")
print(f"selective_scan_cuda : OK")
print("\nSetup complete.")
PYEOF
