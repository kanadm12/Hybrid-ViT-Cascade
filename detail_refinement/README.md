# Detail Refinement Network

**Two-Stage CT Reconstruction for Clinical-Quality Sharpness**

## Overview

This module implements a **detail refinement network** that adds high-frequency details to coarse CT predictions, achieving clinical-quality sharpness without retraining the base model.

### Architecture

**Stage 1**: Base Model (frozen)
- Input: Dual-view X-rays
- Output: Coarse 64³ CT volume (structure/anatomy)
- Status: Already trained (13.77 dB PSNR)

**Stage 2**: Detail Refinement Network (trainable)
- Input: Coarse CT + Dual-view X-rays
- Output: Refined 64³ CT volume (sharp details)
- Size: ~5M parameters
- Target: +2-3 dB PSNR gain, sharper edges

## Key Features

1. **Frequency-Domain Attention**
   - Learns to emphasize high frequencies
   - FFT-based frequency manipulation
   - Preserves phase information

2. **High-Pass Filtering**
   - 3D Laplacian kernel for edge extraction
   - Separates low-freq (structure) from high-freq (details)

3. **Multi-Component Loss**
   - L1 Loss: Basic reconstruction
   - Frequency Loss: Match high-frequency components
   - Gradient Loss: Sharp edges (Laplacian)
   - Perceptual Loss: Texture/detail similarity
   - Consistency Loss: Don't deviate too much from coarse

4. **X-ray Detail Guidance**
   - Extracts detail cues from dual-view X-rays
   - Cross-view fusion for depth information
   - Lifts 2D details to 3D volume

## Training

### Quick Start

```bash
cd /workspace/Hybrid-ViT-Cascade/detail_refinement

# Train on 4 GPUs
python train_refinement_4gpu.py --config config_refinement.json --world_size 4
```

### Configuration

Edit `config_refinement.json`:

```json
{
  "model": {
    "base_model_checkpoint": "direct_regression/checkpoints_enhanced/best_model.pt",
    "hidden_channels": 64
  },
  "training": {
    "batch_size": 4,
    "num_epochs": 40,
    "learning_rate": 5e-5
  },
  "loss": {
    "l1_weight": 1.0,
    "frequency_weight": 0.5,
    "gradient_weight": 0.3,
    "perceptual_weight": 0.2,
    "consistency_weight": 0.1
  }
}
```

### Training Progress

Expected PSNR progression:
- Coarse (base model): 13.77 dB
- Epoch 10: ~14.5 dB (+0.7 dB)
- Epoch 20: ~15.2 dB (+1.4 dB)
- Epoch 40: ~16.0 dB (+2.2 dB) ← Target

## Inference

### Compare Coarse vs. Refined

```bash
python inference_refinement.py \
  --base_checkpoint ../direct_regression/checkpoints_enhanced/best_model.pt \
  --refinement_checkpoint checkpoints/best_model.pt \
  --data_path /workspace/drr_patient_data \
  --num_patients 20 \
  --output_dir inference_results \
  --save_volumes
```

### Output Structure

```
inference_results/
├── metrics.txt                    # Overall PSNR/SSIM metrics
├── patient_000/
│   ├── coarse.nii.gz             # Base model prediction
│   ├── refined.nii.gz            # Detail-refined prediction
│   ├── ground_truth.nii.gz       # Target CT
│   └── patient_000_comparison.png # Visual comparison
├── patient_001/
│   └── ...
```

## Technical Details

### Why Two-Stage?

1. **Stability**: Base model provides reliable structure
2. **Efficiency**: Only train small refinement network (5M vs 16M params)
3. **Safety**: Medical imaging requires stable anatomy reconstruction
4. **Focus**: Refinement network specializes in details/sharpness

### High-Frequency Enhancement

**Frequency Domain Loss**:
```python
# FFT to frequency domain
pred_fft = torch.fft.fftn(pred)
target_fft = torch.fft.fftn(target)

# High-pass filter (emphasize high frequencies)
high_pass_weight = sigmoid(10 * (freq_magnitude - 0.1))

# Weighted loss
loss = |pred_fft * high_pass_weight - target_fft * high_pass_weight|
```

**Gradient Loss (Edges)**:
```python
# 3D Laplacian for edge detection
high_freq_pred = Laplacian_3D(pred)
high_freq_target = Laplacian_3D(target)

loss = L1(high_freq_pred, high_freq_target)
```

### Comparison with Other Approaches

| Approach | Pros | Cons | PSNR Gain |
|----------|------|------|-----------|
| **Detail Refinement (This)** | Safe, stable, fast training | Requires base model | +2-3 dB |
| Attention Model | End-to-end | Underperformed (12.56 dB) | -1.2 dB |
| Full GAN | Very sharp | Unstable, can hallucinate | +3-5 dB |
| Larger Resolution (128³) | More voxels | Insufficient capacity | -6 dB |

## Expected Results

### Quantitative

- **Coarse PSNR**: 13.77 dB (base model)
- **Refined PSNR**: 16.0 dB (target)
- **Gain**: +2.2 dB
- **Training time**: ~1.5 hours (40 epochs, 4 GPUs)

### Qualitative

**Before (Coarse)**:
- ✓ Correct anatomy/structure
- ✗ Blurry organ boundaries
- ✗ Smooth bone edges
- ✗ Loss of fine vessels

**After (Refined)**:
- ✓ Correct anatomy/structure
- ✓ Sharp organ boundaries
- ✓ Crisp bone edges
- ✓ Visible fine details

## Files

- `model_refinement.py`: DetailRefinementNetwork + Loss functions
- `train_refinement_4gpu.py`: DDP training script
- `inference_refinement.py`: Comparison inference
- `config_refinement.json`: Training configuration

## Citation

Two-stage approach inspired by:
- Unsharp masking (traditional image sharpening)
- Super-resolution networks (residual learning)
- Medical image enhancement (detail-preserving refinement)
