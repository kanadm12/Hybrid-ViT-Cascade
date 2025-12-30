#!/bin/bash

# Optimized CT Reconstruction Training
# Single GPU, all optimizations enabled

set -e

echo "======================================"
echo "Optimized CT Reconstruction Training"
echo "======================================"
echo ""
echo "Optimizations enabled:"
echo "  ✓ Cascaded Group Attention (30-40% faster)"
echo "  ✓ CNN + ViT Hybrid Architecture"
echo "  ✓ Learnable Depth Priors with Uncertainty"
echo "  ✓ Sandwich Layout (2 Attn + 4 FFN blocks)"
echo "  ✓ Hierarchical Adaptive Conditioning"
echo ""
echo "Hardware: 1x GPU"
echo "Expected time: ~45 minutes (15 epochs)"
echo "Expected PSNR: >25 dB (vs ~22 dB baseline)"
echo ""

# Navigate to optimized model directory
cd "$(dirname "$0")"

# Check if CUDA is available
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"

echo "CUDA device: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "GPU Memory: $(python3 -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")')"
echo ""

# Run training
echo "Starting training..."
python3 train_optimized.py

echo ""
echo "======================================"
echo "Training completed!"
echo "Check best_model_optimized.pth for best model"
echo "======================================"
