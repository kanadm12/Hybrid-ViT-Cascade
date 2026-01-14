#!/bin/bash
# Phase 2: Fine-tune all layers end-to-end

set -e

DATASET="/workspace/drr_patient_data_expanded"
PHASE1_CHECKPOINT="checkpoints_direct256_b200_phase1/direct256_best_psnr.pth"
CHECKPOINT_DIR="checkpoints_direct256_b200_phase2"

echo "================================"
echo " Phase 2: Fine-Tuning           "
echo " Unfreeze all, train end-to-end "
echo "================================"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Phase 1 checkpoint: $PHASE1_CHECKPOINT"
echo "  Output: $CHECKPOINT_DIR"
echo "  Epochs: 100"
echo "  Batch size: 1"
echo "  Learning rate: 5e-5"
echo ""

if [ ! -f "$PHASE1_CHECKPOINT" ]; then
    echo "ERROR: Phase 1 checkpoint not found!"
    echo "  Expected: $PHASE1_CHECKPOINT"
    echo ""
    echo "Run Phase 1 first:"
    echo "  ./run_phase1_b200.sh"
    echo ""
    exit 1
fi

python transfer_128_to_256_b200.py \
    --dataset_path "$DATASET" \
    --resume_256 "$PHASE1_CHECKPOINT" \
    --batch_size 1 \
    --num_workers 4 \
    --num_epochs 100 \
    --lr 5e-5 \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo ""
echo "================================"
echo " Phase 2 Complete!              "
echo "================================"
echo ""
echo "Best checkpoints:"
echo "  Loss: ${CHECKPOINT_DIR}/direct256_best_loss.pth"
echo "  PSNR: ${CHECKPOINT_DIR}/direct256_best_psnr.pth"
echo "  SSIM: ${CHECKPOINT_DIR}/direct256_best_ssim.pth"
echo ""
echo "Training log: ${CHECKPOINT_DIR}/training_log.csv"
echo ""
echo "Expected final PSNR: ~30-31 dB"
echo "Expected final SSIM: ~0.75-0.80"
echo ""
echo "Use best PSNR checkpoint for inference!"
echo ""
