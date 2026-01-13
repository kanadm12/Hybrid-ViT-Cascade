#!/bin/bash
# Inference commands for Direct128 H200 Model

# Example 1: Random patient with best PSNR checkpoint, output 512³
python inference_direct128.py \
    --checkpoint checkpoints_direct128_h200/best_psnr_checkpoint.pth \
    --dataset_path /workspace/drr_patient_data \
    --output_dir inference_results \
    --output_size 512

# Example 2: Specific patient with best SSIM checkpoint
# python inference_direct128.py \
#     --checkpoint checkpoints_direct128_h200/best_ssim_checkpoint.pth \
#     --dataset_path /workspace/drr_patient_data \
#     --patient_id patient_001 \
#     --output_dir inference_results \
#     --output_size 512

# Example 3: Lower resolution output (256³) for faster testing
# python inference_direct128.py \
#     --checkpoint checkpoints_direct128_h200/best_psnr_checkpoint.pth \
#     --dataset_path /workspace/drr_patient_data \
#     --output_size 256

# Example 4: Run on all validation patients
# for patient in patient_001 patient_002 patient_003; do
#     python inference_direct128.py \
#         --checkpoint checkpoints_direct128_h200/best_psnr_checkpoint.pth \
#         --dataset_path /workspace/drr_patient_data \
#         --patient_id $patient \
#         --output_dir inference_results
# done
