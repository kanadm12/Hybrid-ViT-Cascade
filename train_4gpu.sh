#!/bin/bash
# Training script for 4 A100 GPUs (80GB each)
# Easy-to-use wrapper for distributed training

# Configuration
CONFIG="config/multi_view_config.json"
CHECKPOINT_DIR="checkpoints_4gpu"
NUM_GPUS=4
METHOD="torchrun"  # Options: "torchrun" or "accelerate"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB="--wandb"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Training Hybrid-ViT Cascade on 4 A100 GPUs"
echo "=========================================="
echo "Config: $CONFIG"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "Method: $METHOD"
echo "=========================================="

# Method 1: Using torchrun (PyTorch native DDP)
if [ "$METHOD" == "torchrun" ]; then
    echo "Using torchrun with DDP..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        training/train_distributed.py \
        --config "$CONFIG" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        $USE_WANDB
fi

# Method 2: Using Accelerate (simpler, recommended)
if [ "$METHOD" == "accelerate" ]; then
    echo "Using Accelerate..."
    accelerate launch \
        --multi_gpu \
        --num_processes=$NUM_GPUS \
        --mixed_precision=fp16 \
        training/train_accelerate.py \
        --config "$CONFIG" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        $USE_WANDB
fi

echo "Training completed!"
