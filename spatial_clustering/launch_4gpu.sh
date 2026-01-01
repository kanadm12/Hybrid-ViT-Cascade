#!/bin/bash
# Launch script for 4-GPU training

# Set number of GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch with torchrun (recommended for PyTorch 1.9+)
torchrun --nproc_per_node=4 \
    --master_port=29500 \
    train_spatial_clustering_4gpu.py

# Alternative: Use torch.distributed.launch (older PyTorch versions)
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=29500 \
#     train_spatial_clustering_4gpu.py
