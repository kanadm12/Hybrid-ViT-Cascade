#!/bin/bash

# Inference script for Direct Regression CT Reconstruction
# Generates CT volumes from X-ray pairs using trained model

set -e

echo "=========================================="
echo "DIRECT REGRESSION CT INFERENCE"
echo "=========================================="
echo ""

# Navigate to direct regression directory
cd "$(dirname "$0")"

# Check if model checkpoint exists
if [ ! -f "best_model_direct.pth" ]; then
    echo "Error: Model checkpoint not found!"
    echo "Expected: best_model_direct.pth"
    echo "Please train the model first or provide checkpoint path"
    exit 1
fi

echo "Model checkpoint: best_model_direct.pth"
echo "Dataset: /workspace/drr_patient_data"
echo "Split: validation set"
echo "Output: inference_results/"
echo ""

# Run inference (process 10 samples by default)
echo "Running inference..."
python3 inference_direct.py \
    --checkpoint best_model_direct.pth \
    --data_dir /workspace/drr_patient_data \
    --split val \
    --max_samples 10 \
    --batch_size 1

echo ""
echo "=========================================="
echo "Inference completed!"
echo "Results saved to: inference_results/"
echo ""
echo "Files generated per sample:"
echo "  - sample_XXX_visualization.png (comprehensive view)"
echo "  - sample_XXX_predicted.npy (numpy array)"
echo "  - sample_XXX_predicted.nii.gz (NIfTI format)"
echo "=========================================="
