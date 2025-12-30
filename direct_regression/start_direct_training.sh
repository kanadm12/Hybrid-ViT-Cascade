#!/bin/bash
# Start Direct Regression Training - Single GPU
# This tests if the architecture can learn WITHOUT diffusion

set -e

echo "=========================================="
echo "DIRECT CT REGRESSION (NO DIFFUSION)"
echo "=========================================="

cd /workspace/Hybrid-ViT-Cascade/direct_regression

echo ""
echo "This will train a direct X-ray → CT model"
echo "NO diffusion, NO timesteps, NO noise"
echo ""
echo "Expected results:"
echo "  Epoch 3: PSNR >15 dB (proof architecture works)"
echo "  Epoch 10: PSNR >20 dB (success!)"
echo ""
echo "If PSNR stays <15 dB: Architecture is broken"
echo "If PSNR >20 dB: Diffusion was the problem"
echo ""
echo "Training on 1 GPU, 10 epochs (~30 minutes)"
echo "=========================================="
echo ""

python3 train_direct.py --config config_direct.json
