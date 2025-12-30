#!/bin/bash
# Example inference commands for the optimized model

# ============================================
# BASIC USAGE
# ============================================

# Generate CT from 2 X-ray views (64³ resolution)
python3 inference_optimized.py \
    --checkpoint checkpoints_optimized/best_model.pt \
    --xrays xray_view1.png xray_view2.png \
    --output results/predicted_ct.nii.gz

# ============================================
# HIGH RESOLUTION (256³)
# ============================================

# Upscale to 256x256x256 for better detail
python3 inference_optimized.py \
    --checkpoint checkpoints_optimized/best_model.pt \
    --xrays xray_view1.png xray_view2.png \
    --output results/predicted_ct_256.nii.gz \
    --upscale 256 256 256

# ============================================
# ULTRA HIGH RESOLUTION (512³)
# ============================================

# Upscale to 512x512x512 for maximum detail
# Requires ~16GB GPU memory
python3 inference_optimized.py \
    --checkpoint checkpoints_optimized/best_model.pt \
    --xrays xray_view1.png xray_view2.png \
    --output results/predicted_ct_512.nii.gz \
    --upscale 512 512 512

# ============================================
# SINGLE VIEW (if only one X-ray available)
# ============================================

python3 inference_optimized.py \
    --checkpoint checkpoints_optimized/best_model.pt \
    --xrays single_xray.png \
    --output results/predicted_ct_single_view.nii.gz \
    --upscale 256 256 256

# ============================================
# CPU INFERENCE (slower but no GPU needed)
# ============================================

python3 inference_optimized.py \
    --checkpoint checkpoints_optimized/best_model.pt \
    --xrays xray_view1.png xray_view2.png \
    --output results/predicted_ct_cpu.nii.gz \
    --device cpu

# ============================================
# NIFTI INPUT (if X-rays are in NIFTI format)
# ============================================

python3 inference_optimized.py \
    --checkpoint checkpoints_optimized/best_model.pt \
    --xrays xray1.nii.gz xray2.nii.gz \
    --output results/predicted_ct.nii.gz \
    --upscale 256 256 256

# ============================================
# BATCH PROCESSING
# ============================================

# Process multiple patients
for patient in patient_*/; do
    echo "Processing ${patient}..."
    python3 inference_optimized.py \
        --checkpoint checkpoints_optimized/best_model.pt \
        --xrays "${patient}xray1.png" "${patient}xray2.png" \
        --output "results/${patient%/}_ct.nii.gz" \
        --upscale 256 256 256
done

echo "All inference examples listed above!"
