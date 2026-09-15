#!/bin/bash
# RunPod Setup Script - Run this on RunPod to execute the full pipeline

set -e  # Exit on error

echo "========================================================================"
echo "Cross-Generator AI Image Detector - RunPod Setup"
echo "========================================================================"

# Navigate to project directory
cd /workspace/cross-generator-ai-detector || cd $(pwd)

echo ""
echo "[1/5] Installing Python dependencies..."
pip install -q -r requirements.txt

echo ""
echo "[2/5] Verifying CUDA/GPU..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA: {torch.cuda.is_available()}'); print(f'✓ GPU: {torch.cuda.get_device_name()}')"

echo ""
echo "[3/5] Creating .env file..."
if [ ! -f .env ]; then
    cp .env.runpod .env
    echo "✓ .env created from .env.runpod"
else
    echo "✓ .env already exists"
fi

echo ""
echo "[4/5] Verifying directories..."
mkdir -p data/raw data/processed models outputs logs
echo "✓ All directories ready"

echo ""
echo "[5/5] Starting pipeline..."
echo "========================================================================"
echo "Configuration:"
echo "  - SDXL: U-Net Diffusion (1024×1024)"
echo "  - FLUX.1-schnell: Diffusion Transformer (1024×1024)"
echo "  - SD 1.5: Classic U-Net (512×512)"
echo "  - COCO: Real photos (ground truth)"
echo "  - Cross-validation: 4-fold (hold out each AI generator)"
echo "========================================================================"
echo ""

# Run the full pipeline
python main.py full -y

echo ""
echo "========================================================================"
echo "✓ Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - Images: data/raw/{sdxl,flux,sd15,coco}/"
echo "  - Features: data/processed/*.npy"
echo "  - Results: outputs/cv_results_*.json"
echo ""
