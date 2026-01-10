# Progressive Multi-Scale CT Reconstruction

## Overview

A progressive cascade system for X-ray to CT reconstruction that combines proven baseline architecture with novel multi-resolution refinement. The system progressively generates CT volumes at 64³→128³→256³ resolutions with frequency-aware training and X-ray geometric consistency.

## Architecture

### Multi-Scale Cascade
- **Stage 1 (64³)**: Base reconstruction focusing on coarse structure
- **Stage 2 (128³)**: Refinement adding texture and medium-frequency details  
- **Stage 3 (256³)**: Final refinement capturing fine details and edges

### Key Features
- **Shared X-ray Encoder**: Multi-scale feature extraction (512→256→128 progressive downsampling)
- **Progressive Training**: Stage-by-stage training with frozen earlier stages
- **Frequency-Aware Losses**: Stage-specific loss functions
- **Geometric Consistency**: DRR reprojection loss ensuring X-ray alignment
- **Memory Efficient**: Gradient checkpointing for 256³ volumes

## Loss Functions

### Stage 1: Structure (L1 + SSIM)
```
Loss = 1.0 × L1 + 0.5 × SSIM
```
Focuses on overall anatomy and intensity distribution.

### Stage 2: Texture (L1 + SSIM + VGG)
```
Loss = 1.0 × L1 + 0.5 × SSIM + 0.1 × VGG_perceptual
```
Adds tri-planar VGG perceptual loss for realistic texture.

### Stage 3: Details (L1 + SSIM + VGG + Gradient + DRR)
```
Loss = 1.0 × L1 + 0.5 × SSIM + 0.1 × VGG + 0.2 × Gradient + 0.3 × DRR_reprojection
```
Captures fine details, edges, and ensures geometric consistency with input X-rays.

## Installation

```bash
# Navigate to progressive_cascade directory
cd progressive_cascade

# Install required packages (if needed)
pip install torch torchvision numpy matplotlib tqdm
pip install nibabel  # For NIfTI export (optional)
```

## Training

### Progressive Training (Recommended)

Train all stages sequentially with automatic checkpoint loading:

```bash
# 4-GPU training
python train_progressive_4gpu.py

# The script will:
# 1. Train Stage 1 (64³) for 50 epochs
# 2. Freeze Stage 1, train Stage 2 (128³) for 30 epochs
# 3. Freeze Stages 1+2, train Stage 3 (256³) for 20 epochs
```

### Configuration

Edit [config_progressive.json](config_progressive.json) to customize:
- Model architecture (voxel_dim, depth, heads)
- Training hyperparameters (epochs, batch_size, learning_rate)
- Loss weights per stage
- Data paths and splits

### Memory Requirements

| Stage | Resolution | Batch Size | Memory/GPU | Training Time |
|-------|-----------|------------|------------|---------------|
| 1     | 64³       | 8          | ~10GB      | ~6 hours      |
| 2     | 128³      | 4          | ~16GB      | ~12 hours     |
| 3     | 256³      | 2          | ~35GB      | ~24 hours     |

**Note**: Stage 3 uses gradient checkpointing to fit in 40GB GPUs.

## Inference

### Single Sample Inference

```bash
python inference_progressive.py \
  --checkpoint checkpoints_progressive/stage3_best.pth \
  --mode single \
  --sample-idx 0 \
  --output-dir outputs_progressive \
  --save-nifti
```

### Full Dataset Evaluation

```bash
python inference_progressive.py \
  --checkpoint checkpoints_progressive/stage3_best.pth \
  --mode evaluate \
  --num-samples 100 \
  --output-dir outputs_progressive
```

## Expected Performance

Based on initial architecture design:

| Stage | Resolution | PSNR (dB) | SSIM  | Description |
|-------|-----------|-----------|-------|-------------|
| 1     | 64³       | 28-30     | 0.85-0.90 | Coarse structure |
| 2     | 128³      | 32-35     | 0.92-0.95 | Added texture |
| 3     | 256³      | 35-38     | 0.95-0.97 | Fine details |

## File Structure

```
progressive_cascade/
├── model_progressive.py        # Multi-scale architecture
├── loss_multiscale.py         # Frequency-aware loss system
├── train_progressive_4gpu.py  # Progressive training pipeline
├── inference_progressive.py   # Inference and evaluation
├── config_progressive.json    # Configuration file
├── README.md                  # This file
└── utils.py                   # Utility functions
```

## Key Components

### model_progressive.py
- `MultiScaleXrayEncoder`: Progressive X-ray feature extraction
- `Stage1Base64`: Base 64³ reconstruction module
- `Stage2Refiner128`: 128³ refinement with residual connections
- `Stage3Refiner256`: 256³ refinement with detail enhancement
- `ProgressiveCascadeModel`: Full cascade with stage freezing

### loss_multiscale.py
- `Stage1Loss`: L1 + SSIM for structure
- `Stage2Loss`: + VGG perceptual for texture
- `Stage3Loss`: + Gradient + DRR for details and consistency
- `DRRReprojectionLoss`: Differentiable X-ray reprojection

### train_progressive_4gpu.py
- Distributed training with PyTorch DDP
- Mixed precision with torch.amp
- Stage-by-stage progressive training
- Automatic checkpoint management

## Alternative Configurations

### Memory-Constrained (GPUs with <32GB)
```json
{
  "training": {
    "stage3": {
      "batch_size": 1,
      "voxel_dim": 192
    }
  }
}
```

### Fast Prototyping
```json
{
  "training": {
    "stage1": {"num_epochs": 20},
    "stage2": {"num_epochs": 15},
    "stage3": {"num_epochs": 10}
  },
  "data": {
    "max_patients": 50
  }
}
```

### High Quality (Maximum PSNR/SSIM)
```json
{
  "training": {
    "stage1": {"num_epochs": 100},
    "stage2": {"num_epochs": 50},
    "stage3": {"num_epochs": 30, "voxel_dim": 320}
  }
}
```

## Advanced Features

### Gradient Checkpointing
Enabled automatically for Stage 3 to reduce memory:
```python
use_gradient_checkpointing=True  # In Stage3Refiner256
```

### DRR Reprojection Loss
Ensures geometric consistency with input X-rays:
```python
drr_ap = generate_drr(predicted_ct, view_angle=0)
drr_lateral = generate_drr(predicted_ct, view_angle=90)
loss = L1(drr_ap, input_xray_ap) + L1(drr_lateral, input_xray_lateral)
```

### Multi-Scale Cross-Attention
X-ray features match CT resolution at each stage:
- Stage 1 (64³): 128×128 X-ray features
- Stage 2 (128³): 256×256 X-ray features  
- Stage 3 (256³): 512×512 X-ray features

## Troubleshooting

### Out of Memory on Stage 3
- Reduce batch_size to 1
- Reduce voxel_dim from 256 to 192
- Enable gradient checkpointing (should be on by default)

### Training Unstable
- Reduce learning rates by 50%
- Increase gradient_clip threshold
- Check data normalization

### Poor PSNR/SSIM on Stage 1
- Train longer (increase num_epochs for stage1)
- Increase model capacity (voxel_dim, vit_depth)
- Adjust loss weights (increase ssim_weight)

## Citation

If you use this work, please cite:
```
Progressive Multi-Scale CT Reconstruction from X-rays
Combining proven baseline architecture with novel multi-resolution refinement
```

## License

See parent repository for license information.
