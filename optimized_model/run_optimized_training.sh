#!/bin/bash

# Run Optimized Model Training
# This script trains the optimized CT reconstruction model with all 6 improvements

echo "=========================================="
echo "   Optimized CT Regression Training"
echo "=========================================="
echo ""
echo "Model Features:"
echo "  ✓ Cascaded Group Attention (CGA)"
echo "  ✓ CNN + ViT Hybrid Architecture"
echo "  ✓ Learnable Depth Priors"
echo "  ✓ Sandwich Layout (Attn → FFN → Attn)"
echo "  ✓ Multi-Scale Attention"
echo "  ✓ Hierarchical Conditioning"
echo ""
echo "Expected Parameters: ~180M"
echo "Expected Training Time: ~45 minutes (15 epochs)"
echo "Target PSNR: 25-28 dB"
echo ""
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "train_optimized.py" ]; then
    echo "Error: train_optimized.py not found!"
    echo "Please run this script from the optimized_model directory"
    exit 1
fi

# Check if config exists
if [ ! -f "config_optimized.json" ]; then
    echo "Error: config_optimized.json not found!"
    exit 1
fi

# Create checkpoints directory
mkdir -p checkpoints_optimized

# Run training
echo "Starting training..."
echo ""

python3 train_optimized.py --config config_optimized.json

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Check results in:"
echo "  - checkpoints_optimized/best_model.pt"
echo "  - checkpoints_optimized/training_log.json"
echo "  - feature_visualizations/"
echo ""
