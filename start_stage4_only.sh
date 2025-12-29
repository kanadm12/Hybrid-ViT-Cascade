#!/bin/bash
#
# Start Training from Stage 4 Only
# Requires stage3_best.pt checkpoint to exist
#

set -e  # Exit on error

echo "============================================================"
echo "Starting Stage 4 Training (512³ resolution)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  - Resolution: 512³ voxels"
echo "  - Batch size: 1 per GPU (4 total)"
echo "  - Epochs: 50"
echo "  - Physics weight: 0.6"
echo ""

# Navigate to workspace
cd /workspace/Hybrid-ViT-Cascade

# Pull latest code
echo "Pulling latest code..."
git pull

# Clear Python cache
echo "Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Clear GPU memory
echo "Clearing GPU memory..."
pkill -9 -f train_distributed || true
sleep 3
python -c "import torch; torch.cuda.empty_cache()" || true

# Check for stage3 checkpoint
echo ""
echo "Checking for stage 3 checkpoint..."
if [ ! -f "checkpoints/stage3_best.pt" ]; then
    echo "ERROR: stage3_best.pt not found!"
    echo "Stage 4 requires stage3_best.pt to start."
    exit 1
fi
echo "  ✓ Found checkpoints/stage3_best.pt"

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

# Setup directories
echo ""
echo "Setting up directories..."
mkdir -p checkpoints
mkdir -p runs
mkdir -p /workspace/feature_maps

echo ""
echo "============================================================"
echo "Launching Stage 4 Training"
echo "============================================================"
echo ""
echo "This will train ONLY stage 4 starting from stage3_best.pt"
echo "Expected time: ~14-18 hours for 50 epochs"
echo ""
echo "To monitor progress:"
echo "  - Watch log: tail -f training_stage4_*.log"
echo "  - TensorBoard: tensorboard --logdir=runs --host=0.0.0.0 --port=6006"
echo "  - GPU usage: nvidia-smi"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Start training from stage 4
echo ""
echo "Launching distributed training on 4 GPUs (stage 4 only)..."
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 training/train_distributed.py \
  --config config/runpod_config.json \
  --resume_from checkpoints/stage3_best.pt \
  --start_stage 4 \
  --tensorboard \
  2>&1 | tee training_stage4_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "============================================================"
echo "Stage 4 Training Complete!"
echo "============================================================"
echo ""
echo "Checkpoint saved:"
ls -lh checkpoints/stage4_best.pt
echo ""
echo "To run full inference through all 4 stages:"
echo "  python run_inference.py"
echo ""
