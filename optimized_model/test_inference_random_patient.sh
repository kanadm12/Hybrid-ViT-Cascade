#!/bin/bash
# Quick inference test on a random patient from the dataset

DATASET_PATH="/workspace/drr_patient_data"
CHECKPOINT_PATH="checkpoints_optimized/best_model.pt"
OUTPUT_DIR="inference_results"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Find a random patient folder
echo "Searching for patient folders..."
PATIENT_FOLDERS=($(ls -d ${DATASET_PATH}/patient_* 2>/dev/null | head -5))

if [ ${#PATIENT_FOLDERS[@]} -eq 0 ]; then
    echo "Error: No patient folders found in $DATASET_PATH"
    exit 1
fi

# Pick the first one (or random)
PATIENT_FOLDER=${PATIENT_FOLDERS[0]}
PATIENT_NAME=$(basename ${PATIENT_FOLDER})

echo "Selected patient: ${PATIENT_NAME}"
echo "Patient folder: ${PATIENT_FOLDER}"

# Find DRR files
echo "Looking for DRR files..."
DRR_FILES=($(find ${PATIENT_FOLDER} -name "drr_*.png" -o -name "drr_*.nii*" | head -2))

if [ ${#DRR_FILES[@]} -eq 0 ]; then
    echo "Error: No DRR files found in ${PATIENT_FOLDER}"
    echo "Contents of folder:"
    ls -la ${PATIENT_FOLDER}
    exit 1
fi

echo "Found ${#DRR_FILES[@]} DRR files:"
for drr in "${DRR_FILES[@]}"; do
    echo "  - $(basename $drr)"
done

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Warning: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please specify checkpoint path or wait for training to complete"
    exit 1
fi

# Run inference
OUTPUT_PATH="${OUTPUT_DIR}/${PATIENT_NAME}_predicted.nii.gz"

echo ""
echo "========================================"
echo "Running Inference"
echo "========================================"
echo "Patient: ${PATIENT_NAME}"
echo "DRRs: ${DRR_FILES[@]}"
echo "Output: ${OUTPUT_PATH}"
echo ""

python3 inference_optimized.py \
    --checkpoint ${CHECKPOINT_PATH} \
    --xrays ${DRR_FILES[@]} \
    --output ${OUTPUT_PATH} \
    --upscale 256 256 256 \
    --device cuda

echo ""
echo "========================================"
echo "Inference Complete!"
echo "========================================"
echo "Output saved to: ${OUTPUT_PATH}"
echo "Visualization: ${OUTPUT_DIR}/${PATIENT_NAME}_predicted_visualization.png"
echo ""
echo "To view with Python:"
echo "  python3 -c \"import nibabel as nib; import numpy as np; vol = nib.load('${OUTPUT_PATH}').get_fdata(); print(f'Shape: {vol.shape}, Range: [{vol.min():.3f}, {vol.max():.3f}]')\""
