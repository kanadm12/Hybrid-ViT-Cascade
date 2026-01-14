#!/bin/bash
# Phase 1: Train 256³ layers with frozen 128³ backbone

set -e

DATASET="/workspace/drr_patient_data_expanded"
CHECKPOINT_128="/workspace/Hybrid-ViT-Cascade/direct_regression/progressive_cascade/direct128_best_psnr_resumed.pth"
CHECKPOINT_DIR="checkpoints_direct256_b200_phase1"

echo "================================"
echo " Phase 1: Transfer Learning     "
echo " Freeze 128³, Train 256³        "
echo "================================"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  128³ checkpoint: $CHECKPOINT_128"
echo "  Output: $CHECKPOINT_DIR"
echo "  Epochs: 20"
echo "  Batch size: 1"
echo "  Learning rate: 1e-4"
echo ""

python transfer_128_to_256_b200.py \
    --dataset_path "$DATASET" \
    --checkpoint_128 "$CHECKPOINT_128" \
    --freeze_128 \
    --batch_size 1 \
    --num_workers 4 \
    --num_epochs 20 \
    --lr 1e-4 \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo ""
echo "================================"
echo " Phase 1 Complete!              "
echo "================================"
echo ""
echo "Best checkpoint: ${CHECKPOINT_DIR}/direct256_best_psnr.pth"
echo "Training log: ${CHECKPOINT_DIR}/training_log.csv"
echo ""
echo "Expected PSNR: ~28.5-29 dB"
echo ""
echo "Next step: Run Phase 2 fine-tuning"
echo "  ./run_phase2_b200.sh"
echo ""
