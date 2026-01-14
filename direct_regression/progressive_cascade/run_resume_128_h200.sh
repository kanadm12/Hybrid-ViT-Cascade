#!/bin/bash
# Resume Direct128 H200 Training from Best Checkpoint
# Current: 27.98 dB PSNR, 0.50 SSIM at epoch 71
# Target: Continue training to improve performance

CHECKPOINT="/workspace/Hybrid-ViT-Cascade/direct_regression/progressive_cascade/direct128_best_psnr_resumed.pth"
DATASET="/workspace/drr_patient_data"
SAVE_DIR="checkpoints_direct128_h200_continued"
BATCH_SIZE=2
NUM_EPOCHS=200  # Total epochs (will continue from checkpoint epoch)
LR=3e-5  # Lower LR for fine-tuning
NUM_WORKERS=8

echo "================================"
echo "Resume Direct128 H200 Training"
echo "================================"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $DATASET"
echo "Save Dir: $SAVE_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Total Epochs: $NUM_EPOCHS"
echo "Learning Rate: $LR"
echo "================================"

python resume_direct128.py \
    --checkpoint "$CHECKPOINT" \
    --dataset_path "$DATASET" \
    --save_dir "$SAVE_DIR" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --num_workers $NUM_WORKERS
