#!/bin/bash
# Direct 256³ training from scratch on B200 GPU

set -e

DATASET="/workspace/drr_patient_data"
CHECKPOINT_DIR="checkpoints_direct256_scratch"

echo "================================"
echo " Direct 256³ Training (Scratch) "
echo " B200 GPU (180GB)               "
echo "================================"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Output: $CHECKPOINT_DIR"
echo "  Epochs: 200"
echo "  Batch size: 1"
echo "  Learning rate: 1e-4"
echo "  Training: From scratch (no transfer learning)"
echo ""

python train_direct256_scratch.py \
    --dataset_path "$DATASET" \
    --batch_size 1 \
    --num_workers 4 \
    --num_epochs 200 \
    --lr 1e-4 \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo ""
echo "================================"
echo " Training Complete!             "
echo "================================"
echo ""
echo "Best checkpoints:"
echo "  Loss: ${CHECKPOINT_DIR}/direct256_best_loss.pth"
echo "  PSNR: ${CHECKPOINT_DIR}/direct256_best_psnr.pth"
echo "  SSIM: ${CHECKPOINT_DIR}/direct256_best_ssim.pth"
echo ""
echo "Training log: ${CHECKPOINT_DIR}/training_log.csv"
echo ""
echo "Expected final PSNR: ~28-29 dB"
echo ""
