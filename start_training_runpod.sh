#!/bin/bash
#
# Complete 4-Stage Training Setup for RunPod
# Trains all stages sequentially with best checkpoint saving
#

set -e  # Exit on error

echo "============================================================"
echo "Hybrid-ViT Cascade 4-Stage Training"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  - Stage 1: 64³ (batch=32, 50 epochs)"
echo "  - Stage 2: 128³ (batch=16, 50 epochs)"
echo "  - Stage 3: 256³ (batch=8, 50 epochs)"
echo "  - Stage 4: 512³ (batch=4, 50 epochs)"
echo "  - Total: 200 epochs across 4 stages"
echo ""

# Stop any existing training
echo "Stopping any existing training processes..."
pkill -9 -f train_distributed || true

# Navigate to workspace
cd /workspace/Hybrid-ViT-Cascade

# Pull latest code
echo ""
echo "Pulling latest code from GitHub..."
git pull

# Clear Python cache
echo ""
echo "Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Clear GPU memory
echo ""
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" || true

# Check dataset
echo ""
echo "Checking dataset..."
if [ -d "/workspace/drr_patient_data" ]; then
    num_patients=$(ls -d /workspace/drr_patient_data/patient_* 2>/dev/null | wc -l)
    echo "  Found $num_patients patient datasets"
else
    echo "  ERROR: Dataset directory not found at /workspace/drr_patient_data"
    exit 1
fi

# Check checkpoints directory
echo ""
echo "Checking checkpoints..."
mkdir -p checkpoints
if [ -f "checkpoints/stage1_best.pt" ]; then
    echo "  ✓ Stage 1 checkpoint exists (will resume)"
else
    echo "  ✗ No Stage 1 checkpoint (will train from scratch)"
fi

# Check TensorBoard logs
echo ""
echo "Setting up TensorBoard logs..."
mkdir -p runs

# Check feature maps directory
echo ""
echo "Setting up feature maps directory..."
mkdir -p /workspace/feature_maps

echo ""
echo "============================================================"
echo "Starting Training"
echo "============================================================"
echo ""
echo "Training will proceed through all 4 stages automatically."
echo "Checkpoints will be saved to: checkpoints/"
echo "TensorBoard logs will be saved to: runs/"
echo "Feature maps will be saved to: /workspace/feature_maps/"
echo ""
echo "To monitor training in another terminal:"
echo "  tensorboard --logdir=/workspace/Hybrid-ViT-Cascade/runs --host=0.0.0.0 --port=6006"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Start training
echo ""
echo "Launching distributed training on 4 GPUs..."
echo ""

# Check if we should resume from checkpoint
RESUME_FLAG=""
if [ -f "checkpoints/stage1_best.pt" ]; then
    RESUME_FLAG="--resume_from checkpoints/stage1_best.pt"
    echo "Resuming from checkpoints/stage1_best.pt"
fi

torchrun --nproc_per_node=4 training/train_distributed.py \
  --config config/runpod_config.json \
  --tensorboard \
  $RESUME_FLAG \
  2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo ""
echo "Checkpoints saved:"
ls -lh checkpoints/*.pt
echo ""
echo "To run inference:"
echo "  python run_inference.py"
echo ""
