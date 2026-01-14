#!/bin/bash
# Two-phase transfer learning: 128³ → 256³ on B200

set -e

DATASET="/workspace/drr_patient_data"
CHECKPOINT_128="/workspace/Hybrid-ViT-Cascade/direct_regression/progressive_cascade/direct128_best_psnr_resumed.pth"
CHECKPOINT_DIR="checkpoints_direct256_b200"

echo "============================"
echo " Transfer Learning Pipeline "
echo " 128³ → 256³ on B200 (180GB)"
echo "============================"
echo ""

# Phase 1: Freeze 128³, train 256³ only (20 epochs)
echo "Phase 1: Training 256³ layers (128³ frozen) - 20 epochs"
echo "------------------------------------------------------------"
python transfer_128_to_256_b200.py \
    --dataset_path "$DATASET" \
    --checkpoint_128 "$CHECKPOINT_128" \
    --freeze_128 \
    --batch_size 1 \
    --num_workers 4 \
    --num_epochs 20 \
    --lr 1e-4 \
    --checkpoint_dir "${CHECKPOINT_DIR}_phase1"

echo ""
echo "Phase 1 complete! Best checkpoint saved."
echo ""

# Phase 2: Fine-tune all layers (100 epochs)
echo "Phase 2: Fine-tuning all layers end-to-end - 100 epochs"
echo "------------------------------------------------------------"
PHASE1_BEST="${CHECKPOINT_DIR}_phase1/direct256_best_psnr.pth"

python transfer_128_to_256_b200.py \
    --dataset_path "$DATASET" \
    --resume_256 "$PHASE1_BEST" \
    --batch_size 1 \
    --num_workers 4 \
    --num_epochs 100 \
    --lr 5e-5 \
    --checkpoint_dir "${CHECKPOINT_DIR}_phase2"

echo ""
echo "======================================"
echo " Transfer learning complete! "
echo "======================================"
echo ""
echo "Results:"
echo "  Phase 1 checkpoints: ${CHECKPOINT_DIR}_phase1/"
echo "  Phase 2 checkpoints: ${CHECKPOINT_DIR}_phase2/"
echo ""
echo "Expected performance:"
echo "  Phase 1 end: ~28.5-29 dB PSNR"
echo "  Phase 2 end: ~30-31 dB PSNR"
echo ""
