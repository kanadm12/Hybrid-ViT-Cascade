# Inference Guide - Hybrid-ViT Cascade

This guide explains how to use your trained model to reconstruct 3D CT volumes from X-ray images.

## Prerequisites

- Trained model checkpoint (e.g., `stage3_best.pt`)
- Test X-ray images (dual-view DRRs)
- Same configuration file used during training

## Quick Start

### Basic Inference (RunPod)

```bash
cd /workspace/Hybrid-ViT-Cascade

# Run inference on 5 test samples
python inference.py \
  --checkpoint /workspace/checkpoints/hybrid_vit_cascade/stage3_best.pt \
  --config config/runpod_config.json \
  --data_dir /workspace/drr_patient_data \
  --output_dir /workspace/inference_results \
  --stage stage3 \
  --num_samples 5 \
  --num_steps 50
```

### Local Inference (Windows)

```powershell
cd C:\Users\Kanad\Desktop\x2ctpa\hybrid_vit_cascade

python inference.py `
  --checkpoint path\to\stage3_best.pt `
  --config config\runpod_config.json `
  --data_dir path\to\test_data `
  --output_dir .\inference_results `
  --stage stage3 `
  --num_samples 5 `
  --num_steps 50
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Path to trained model checkpoint (`.pt` file) | **Required** |
| `--config` | Path to config file (same as training) | **Required** |
| `--data_dir` | Path to test data directory | **Required** |
| `--output_dir` | Directory to save inference results | `./inference_results` |
| `--stage` | Which cascade stage to use (`stage1`, `stage2`, `stage3`) | `stage3` |
| `--num_samples` | Number of test samples to process | `5` |
| `--num_steps` | Number of diffusion sampling steps (higher = better quality but slower) | `50` |
| `--device` | Device to run on (`cuda` or `cpu`) | `cuda` |

## Sampling Steps Guide

The `--num_steps` parameter controls the quality-speed tradeoff:

| Steps | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| 20 | Fast (~10s) | Decent | Quick preview |
| 50 | Medium (~25s) | Good | Standard inference |
| 100 | Slow (~50s) | Better | High-quality results |
| 250 | Very Slow (~2min) | Best | Publication-quality |

*Timing estimates on H200 GPU*

## Output Structure

For each test sample, the script creates:

```
inference_results/
├── sample_000_patient_001/
│   ├── reconstructed.nii.gz       # Predicted CT volume
│   ├── ground_truth.nii.gz        # Ground truth CT (for comparison)
│   └── visualization.png          # Visual comparison
├── sample_001_patient_002/
│   ├── ...
└── ...
```

### Output Files

1. **`reconstructed.nii.gz`**: Reconstructed 3D CT volume in NIfTI format
   - Can be viewed in medical imaging software (ITK-SNAP, 3D Slicer, etc.)
   - Shape: (256, 256, 256) for stage3

2. **`ground_truth.nii.gz`**: Original CT volume for comparison
   - Same format as reconstructed volume

3. **`visualization.png`**: Multi-panel visualization showing:
   - Input X-rays (frontal + lateral views)
   - Reconstructed CT slices (axial, coronal, sagittal)

## Example Workflows

### 1. Quick Test (Fast Preview)

```bash
# Test on 3 samples with fast sampling
python inference.py \
  --checkpoint /workspace/checkpoints/hybrid_vit_cascade/stage3_best.pt \
  --config config/runpod_config.json \
  --data_dir /workspace/drr_patient_data \
  --num_samples 3 \
  --num_steps 20
```

### 2. High-Quality Reconstruction

```bash
# Full quality on 10 samples
python inference.py \
  --checkpoint /workspace/checkpoints/hybrid_vit_cascade/stage3_best.pt \
  --config config/runpod_config.json \
  --data_dir /workspace/drr_patient_data \
  --num_samples 10 \
  --num_steps 100
```

### 3. Process Entire Test Set

```bash
# Process all available test samples
python inference.py \
  --checkpoint /workspace/checkpoints/hybrid_vit_cascade/stage3_best.pt \
  --config config/runpod_config.json \
  --data_dir /workspace/drr_patient_data \
  --num_samples 1000 \
  --num_steps 50
```

### 4. Compare Different Stages

```bash
# Stage 1 (64³ resolution)
python inference.py \
  --checkpoint /workspace/checkpoints/hybrid_vit_cascade/stage1_best.pt \
  --config config/runpod_config.json \
  --data_dir /workspace/drr_patient_data \
  --stage stage1 \
  --num_samples 5

# Stage 2 (128³ resolution)
python inference.py \
  --checkpoint /workspace/checkpoints/hybrid_vit_cascade/stage2_best.pt \
  --config config/runpod_config.json \
  --data_dir /workspace/drr_patient_data \
  --stage stage2 \
  --num_samples 5

# Stage 3 (256³ resolution)
python inference.py \
  --checkpoint /workspace/checkpoints/hybrid_vit_cascade/stage3_best.pt \
  --config config/runpod_config.json \
  --data_dir /workspace/drr_patient_data \
  --stage stage3 \
  --num_samples 5
```

## Viewing Results

### Using ITK-SNAP (Recommended)

1. Download ITK-SNAP: http://www.itksnap.org/
2. Open reconstructed volume: `File` → `Open Main Image` → Select `.nii.gz` file
3. Overlay ground truth: `Segmentation` → `Open Segmentation` → Select ground truth
4. Navigate slices using arrow keys or scroll wheel

### Using 3D Slicer

1. Download 3D Slicer: https://www.slicer.org/
2. Drag and drop `.nii.gz` files into Slicer window
3. Use slice viewers to inspect reconstruction quality

### Using Python

```python
import nibabel as nib
import matplotlib.pyplot as plt

# Load volumes
pred = nib.load('inference_results/sample_000/reconstructed.nii.gz').get_fdata()
gt = nib.load('inference_results/sample_000/ground_truth.nii.gz').get_fdata()

# Display middle slice
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(pred[128, :, :], cmap='bone')
axes[0].set_title('Reconstructed')
axes[1].imshow(gt[128, :, :], cmap='bone')
axes[1].set_title('Ground Truth')
plt.show()
```

## Performance Metrics

The script computes and displays:

- **MSE (Mean Squared Error)**: Lower is better
- **MAE (Mean Absolute Error)**: Lower is better

Typical values for well-trained models:
- Stage 1 (64³): MSE ~0.5-1.0
- Stage 2 (128³): MSE ~0.3-0.7
- Stage 3 (256³): MSE ~0.2-0.5

## Troubleshooting

### Out of Memory

If you get OOM errors:

```bash
# Reduce batch size (process one at a time)
python inference.py \
  --checkpoint ... \
  --num_samples 1  # Process one sample at a time

# Or reduce sampling steps
python inference.py \
  --checkpoint ... \
  --num_steps 20  # Faster, uses less memory
```

### Slow Inference

- Reduce `--num_steps` (20-30 is usually sufficient for preview)
- Ensure you're using GPU (`--device cuda`)
- Check GPU utilization with `nvidia-smi`

### Poor Quality Reconstructions

- Increase `--num_steps` (try 100-250)
- Verify you're using the correct checkpoint (stage3_best.pt)
- Check that config matches training setup
- Ensure test data is properly preprocessed

## Custom X-ray Input

To run inference on your own X-ray images (not from the test set):

1. **Prepare your data**:
   - Dual-view X-rays (frontal + lateral)
   - Format: PNG or DICOM
   - Resolution: Match training data (e.g., 256×256)

2. **Create a custom dataset loader**:
   ```python
   # Modify utils/dataset.py or create new loader
   # Ensure X-rays are normalized [-1, 1]
   ```

3. **Run inference with custom data path**

See `utils/dataset.py` for data format requirements.

## Next Steps

- **Evaluate Quality**: Compute PSNR, SSIM metrics on test set
- **Clinical Assessment**: Have radiologists evaluate reconstructions
- **Optimize Speed**: Profile and optimize inference pipeline
- **Deploy**: Package model for clinical deployment

## Citation

If you use this model for research, please cite:
```bibtex
@article{your_paper,
  title={Hybrid-ViT Cascade for X-ray to CT Reconstruction},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```
