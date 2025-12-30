"""
Quick test: Run inference on a random patient from the dataset
"""
import os
import sys
from pathlib import Path
import random

# Find dataset
DATASET_PATH = Path("/workspace/drr_patient_data")
if not DATASET_PATH.exists():
    # Try local path
    DATASET_PATH = Path("../data")
    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

print(f"Dataset path: {DATASET_PATH}")

# Find patient folders (any subdirectory)
patient_folders = sorted([d for d in DATASET_PATH.iterdir() if d.is_dir()])
if not patient_folders:
    print(f"Error: No patient folders found in {DATASET_PATH}")
    sys.exit(1)

print(f"Found {len(patient_folders)} patient folders")

# Pick first patient (or random)
patient_folder = patient_folders[0]
patient_name = patient_folder.name

print(f"\nSelected: {patient_name}")
print(f"Path: {patient_folder}")

# Find DRR files (look for *_drr.png pattern)
drr_files = list(patient_folder.glob("*_drr.png")) + \
            list(patient_folder.glob("drr_*.png")) + \
            list(patient_folder.glob("*_drr.nii*")) + \
            list(patient_folder.glob("drr_*.nii*"))

if not drr_files:
    print(f"\nError: No DRR files found in {patient_folder}")
    print("\nFolder contents:")
    for item in patient_folder.iterdir():
        print(f"  - {item.name}")
    sys.exit(1)

# Take first 2 DRRs
drr_files = sorted(drr_files)[:2]

print(f"\nFound {len(drr_files)} DRR files:")
for drr in drr_files:
    print(f"  - {drr.name}")

# Check for checkpoint
checkpoint_path = Path("checkpoints_optimized/best_model.pt")
if not checkpoint_path.exists():
    # Try epoch checkpoints
    checkpoint_path = Path("checkpoints_optimized/epoch_15.pt")
    if not checkpoint_path.exists():
        print("\n" + "="*50)
        print("Warning: No trained checkpoint found!")
        print("="*50)
        print("Expected locations:")
        print("  - checkpoints_optimized/best_model.pt")
        print("  - checkpoints_optimized/epoch_15.pt")
        print("\nPlease wait for training to complete first.")
        sys.exit(1)

print(f"\nCheckpoint: {checkpoint_path}")

# Prepare output
output_dir = Path("inference_results")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / f"{patient_name}_predicted.nii.gz"

# Build command
cmd = [
    "python3", "inference_optimized.py",
    "--checkpoint", str(checkpoint_path),
    "--xrays", *[str(f) for f in drr_files],
    "--output", str(output_path),
    "--upscale", "256", "256", "256",
    "--device", "cuda"
]

print("\n" + "="*50)
print("Running Inference Command:")
print("="*50)
print(" ".join(cmd))
print()

# Run inference
os.system(" ".join(cmd))

print("\n" + "="*50)
print("Inference Complete!")
print("="*50)
print(f"Output: {output_path}")
print(f"Visualization: {output_path.parent / (output_path.stem + '_visualization.png')}")
