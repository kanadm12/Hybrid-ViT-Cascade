#!/bin/bash
# Fresh Training Startup Script - SINGLE GPU
# Run this on RunPod to start training from scratch with all fixes

set -e  # Exit on any error

echo "=================================="
echo "Starting Fresh Training Setup (1 GPU)"
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
echo "Learning Rate: $(grep -o '"learning_rate": [0-9.e-]*' config/runpod_config.json | head -1 | grep -o '[0-9.e-]*')"

# Start training
echo ""
echo "[4/4] Starting training on 1 GPU..."
echo "=================================="
echo "Training will proceed in 3 stages:"
echo "  Stage 1: 64³ resolution (~12 hours)"
echo "  Stage 2: 128³ resolution (~24 hours)"
echo "  Stage 3: 256³ resolution (~36 hours)"
echo "=================================="
echo ""
echo "Expected Results:"
echo "  Epoch 1: PSNR >15 dB (proof BatchNorm fix works!)"
echo "  Epoch 5: PSNR >22 dB"
echo "  Epoch 30: PSNR 28-32 dB (Stage 1 final)"
echo "=================================="
echo ""

torchrun --nproc_per_node=1 training/train_distributed.py --config config/runpod_config.json
