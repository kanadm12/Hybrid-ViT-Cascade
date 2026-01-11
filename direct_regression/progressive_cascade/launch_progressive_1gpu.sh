#!/bin/bash
# Single GPU Training Script for Progressive Cascade
# 100 Patients, 100 Epochs per Stage

echo "=========================================="
echo "Progressive Cascade Training - Single GPU"
echo "100 Patients | 100 Epochs per Stage"
echo "=========================================="
echo ""

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "Starting training..."
echo "Logs will be saved to training_1gpu.log"
echo ""

# Run training with logging
python train_progressive_1gpu.py 2>&1 | tee training_1gpu.log

echo ""
echo "Training complete! Check training_1gpu.log for full details."
