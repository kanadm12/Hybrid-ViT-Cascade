#!/bin/bash
# Fresh Training Startup Script
# Run this on RunPod to start training from scratch with all fixes

set -e  # Exit on any error

echo "=================================="
echo "Starting Fresh Training Setup"
echo "=================================="

# Navigate to workspace
cd /workspace/Hybrid-ViT-Cascade

# Pull latest changes
echo ""
echo "[1/4] Pulling latest code from git..."
git pull origin main

# Delete old checkpoints
echo ""
echo "[2/4] Removing old checkpoints..."
if [ -d "checkpoints" ]; then
    rm -rf checkpoints/*
    echo "✓ Deleted all checkpoints"
else
    echo "✓ No checkpoints directory found"
fi

# Create fresh checkpoints directory
mkdir -p checkpoints

# Verify configuration
echo ""
echo "[3/4] Verifying training configuration..."
echo "Config: config/runpod_config.json"
echo "Max Patients: $(grep -o '"max_patients": [0-9]*' config/runpod_config.json | grep -o '[0-9]*')"
echo "Epochs per Stage: $(grep -o '"num_epochs": [0-9]*' config/runpod_config.json | head -1 | grep -o '[0-9]*')"
echo "Physics Loss: $(grep -o '"physics_weight": [0-9.]*' config/runpod_config.json | head -1 | grep -o '[0-9.]*')"
echo "Learning Rate: $(grep -o '"learning_rate": [0-9.e-]*' config/runpod_config.json | head -1 | grep -o '[0-9.e-]*')"

# Start training
echo ""
echo "[4/4] Starting distributed training on 4 GPUs..."
echo "=================================="
echo "Training will proceed in 3 stages:"
echo "  Stage 1: 64³ resolution (~3 hours)"
echo "  Stage 2: 128³ resolution (~6 hours)"
echo "  Stage 3: 256³ resolution (~9 hours)"
echo "=================================="
echo ""
echo "Expected Results:"
echo "  Epoch 1: PSNR >18 dB (proof fixes work!)"
echo "  Epoch 10: PSNR >27 dB"
echo "  Stage 1 Final: PSNR ~30 dB"
echo "  Stage 2 Final: PSNR ~35 dB"
echo "  Stage 3 Final: PSNR 38-42 dB"
echo "=================================="
echo ""

torchrun --nproc_per_node=4 training/train_distributed.py --config config/runpod_config.json
