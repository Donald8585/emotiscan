#!/bin/bash
# EmotiScan GPU Setup Script
# Run this ONCE to install all dependencies with CUDA GPU support.
# Usage (Git Bash on Windows):
#   cd /c/Users/logos/Downloads/emotiscan
#   bash setup_gpu.sh

PYTHON="/d/Python/python.exe"

echo "=== EmotiScan GPU Setup ==="
echo ""

# Step 1: Install CUDA PyTorch FIRST (before ultralytics pulls CPU-only torch)
# Try cu126 first (broadest compatibility), fall back to cu128
echo "[1/4] Installing PyTorch with CUDA 12.6..."
$PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if [ $? -ne 0 ]; then
    echo "cu126 failed, trying CUDA 12.8..."
    $PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

# Step 2: Install remaining requirements (ultralytics will see torch already installed)
echo ""
echo "[2/4] Installing remaining dependencies..."
$PYTHON -m pip install -r requirements.txt

# Step 3: Verify GPU detection
echo ""
echo "[3/4] Verifying GPU detection..."
$PYTHON -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device:      {torch.cuda.get_device_name(0)}')
    print(f'CUDA version:    {torch.version.cuda}')
    print()
    print('=== GPU READY ===')
else:
    print()
    print('WARNING: CUDA not detected! You may have CPU-only torch.')
    print('Try:  $PYTHON -m pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu121')
"

# Step 4: Quick test
echo ""
echo "[4/4] Testing YOLOv8 + HSEmotion..."
$PYTHON -c "
from config import GPU_AVAILABLE, GPU_DEVICE_NAME, YOLO_DEVICE
print(f'GPU_AVAILABLE: {GPU_AVAILABLE}')
print(f'GPU_DEVICE:    {GPU_DEVICE_NAME}')
print(f'YOLO_DEVICE:   {YOLO_DEVICE}')
if YOLO_DEVICE == 'cuda':
    print()
    print('=== ALL GOOD - EmotiScan will use your GPU! ===')
else:
    print()
    print('Something is wrong - YOLO_DEVICE should be cuda.')
"

echo ""
echo "Done! Start the app with:"
echo "  $PYTHON -m streamlit run streamlit_app.py"
