# Enhanced Direct Regression Model - 256³ CT Volume Reconstruction

## Overview

This is a two-stage training approach for high-resolution 256³ CT volume reconstruction from dual-view X-rays:

**Stage 1 (Base Model)**: Train deep 64³ model with enhanced losses (50 epochs)
**Stage 2 (Refinement)**: Train lightweight 64³→256³ upsampling network (20 epochs)

## Model Architecture

### Stage 1: Enhanced Direct Model (64³)
- **Encoder**: 4-stage deep CNN (512×512 → 32×32 features)
- **Decoder**: Multi-scale outputs with residual blocks (32, 64, 128 channels)
- **Parameters**: ~15-20M
- **Output**: 64³ CT volume

### Stage 2: Refinement Network (256³)
- **Architecture**: ESRGAN-inspired upsampling with PixelShuffle
- **Upsampling**: 64³ → 128³ → 256³
- **Parameters**: ~5-8M (lightweight)
- **Output**: 256³ high-resolution CT

## Enhanced Losses

### Stage 1 Losses:
1. **L1 Loss** (weight=1.0): Pixel-wise accuracy
2. **SSIM Loss** (weight=0.5): Structural similarity
3. **Perceptual Loss** (weight=0.1): VGG16 features at 3 layers
4. **Edge-Aware Loss** (weight=0.1): Sobel filters for sharp boundaries
5. **Multi-scale Loss** (weight=0.3): Supervise at 32, 64, 128 channel outputs

### Stage 2 Losses:
1. **L1 Loss** (weight=1.0)
2. **Perceptual Loss** (weight=0.1): For anatomical correctness
3. **Edge Loss** (weight=0.1): For sharp details at 256³

## Quick Start

### 1. Test Model Architecture
```bash
cd direct_regression
python test_enhanced.py
```
This will verify:
- Model forward/backward pass
- Loss functions
- Memory usage (~4-6 GB per GPU expected)
- Mixed precision training

### 2. Stage 1: Train Base Model (64³)
```bash
# Launch 4 GPU training
launch_4gpu.bat

# Or manually:
python train_enhanced_4gpu.py
```

**Expected Results:**
- Epoch 1-10: Rapid improvement
- Epoch 20-30: PSNR ~22-25 dB
- Epoch 40-50: PSNR ~25-28 dB (target)

**Training Time:** ~2-3 hours on 4×A100 GPUs

### 3. Stage 2: Train Refinement (256³)
After Stage 1 completes with PSNR >25 dB:

```bash
python train_refinement_4gpu.py
```

**Expected Results:**
- Epoch 5: PSNR ~26-28 dB
- Epoch 10-15: PSNR ~28-30 dB
- Epoch 20: PSNR ~28-32 dB (target)

**Training Time:** ~1-2 hours on 4×A100 GPUs

## Configuration

### config_enhanced.json (Stage 1)
```json
{
  "model": {
    "volume_size": [64, 64, 64],
    "base_channels": 64
  },
  "training": {
    "num_epochs": 50,
    "batch_size": 4,
    "learning_rate": 0.0001
  },
  "loss": {
    "l1_weight": 1.0,
    "ssim_weight": 0.5,
    "perceptual_weight": 0.1,
    "edge_weight": 0.1,
    "multiscale_weight": 0.3
  }
}
```

### Hyperparameter Tuning

**If PSNR <20 dB after 20 epochs:**
- Increase `learning_rate` to 2e-4
- Increase `l1_weight` to 2.0
- Train for 70-100 epochs

**If training is unstable:**
- Decrease `learning_rate` to 5e-5
- Increase `grad_clip_max_norm` to 2.0
- Reduce `batch_size` to 2

**If edges are blurry:**
- Increase `edge_weight` to 0.2
- Increase `perceptual_weight` to 0.15

## Memory Requirements

### Stage 1 (64³):
- **Batch size 4**: ~6 GB per GPU
- **Batch size 8**: ~10 GB per GPU
- **Recommended**: Batch size 4-6 on A100 80GB

### Stage 2 (256³):
- **Batch size 2**: ~15 GB per GPU
- **Batch size 4**: ~25 GB per GPU
- **Recommended**: Batch size 2 on A100 80GB

## Monitoring Training

### Key Metrics to Watch:
1. **Total Loss**: Should decrease steadily
2. **PSNR**: Target >25 dB for Stage 1, >28 dB for Stage 2
3. **Perceptual Loss**: Indicates anatomical accuracy
4. **Edge Loss**: Indicates boundary sharpness

### Example Training Output:
```
Epoch 1 [10/250] Loss: 0.3245 | L1: 0.1523 | SSIM: 0.0823 | Perc: 0.0456 | Edge: 0.0312 | 8.32 samples/s

Epoch 1 Training Summary:
  Total Loss: 0.2891
  L1: 0.1234
  SSIM: 0.0723
  Perceptual: 0.0412
  Edge: 0.0289
  Multiscale: 0.0233

Validation Results:
  Total Loss: 0.2456
  PSNR: 23.45 dB
  ✓ New best model saved! PSNR: 23.45 dB
```

## Checkpoints

### Stage 1:
- `checkpoints_enhanced/best_model.pt` - Best PSNR model
- `checkpoints_enhanced/epoch_X.pt` - Periodic checkpoints (every 5 epochs)

### Stage 2:
- `checkpoints_refinement/best_refinement.pt` - Best refinement model

## Inference

After both stages are trained, use the complete pipeline:

```python
import torch
from model_enhanced import EnhancedDirectModel, RefinementNetwork

# Load base model
base_model = EnhancedDirectModel(volume_size=(64, 64, 64))
base_checkpoint = torch.load('checkpoints_enhanced/best_model.pt')
base_model.load_state_dict(base_checkpoint['model_state_dict'])
base_model.eval().cuda()

# Load refinement
refinement = RefinementNetwork()
ref_checkpoint = torch.load('checkpoints_refinement/best_refinement.pt')
refinement.load_state_dict(ref_checkpoint['refinement_state_dict'])
refinement.eval().cuda()

# Inference
with torch.no_grad():
    xrays = torch.randn(1, 2, 512, 512).cuda()  # Your X-rays
    
    # Stage 1: 64³ prediction
    predicted_64, _ = base_model(xrays)
    
    # Stage 2: Refine to 256³
    predicted_256 = refinement(predicted_64)

print(f"Final output shape: {predicted_256.shape}")  # [1, 1, 256, 256, 256]
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Enable gradient checkpointing (add to model)
- Use `batch_size=1` with `gradient_accumulation_steps=4`

### Loss is NaN
- Check data normalization (should be [0, 1])
- Reduce `learning_rate` to 5e-5
- Check for invalid values in dataset

### Poor Quality (PSNR <20 dB)
- Verify dataset is loading correctly
- Check X-ray and CT alignment
- Increase training epochs to 100
- Tune loss weights

### Training is Slow
- Increase `num_workers` in config
- Use `prefetch_factor=4` in DataLoader
- Check GPU utilization with `nvidia-smi`

## Performance Targets

| Stage | Epochs | PSNR (dB) | Training Time |
|-------|--------|-----------|---------------|
| Stage 1 (64³) | 50 | 25-28 | 2-3 hours |
| Stage 2 (256³) | 20 | 28-32 | 1-2 hours |
| **Total** | **70** | **28-32** | **3-5 hours** |

## Next Steps After Training

1. **Evaluate on test set**: Calculate PSNR, SSIM on held-out data
2. **Visual inspection**: Check anatomical correctness
3. **Clinical validation**: Compare with ground truth CTs
4. **Optimize inference**: Export to ONNX/TensorRT for deployment
5. **Further refinement**: Train Stage 2 longer if needed

## Files

- `model_enhanced.py` - Model architectures and losses
- `train_enhanced_4gpu.py` - Stage 1 distributed training
- `train_refinement_4gpu.py` - Stage 2 refinement training
- `test_enhanced.py` - Test model before training
- `config_enhanced.json` - Stage 1 configuration
- `config_refinement.json` - Stage 2 configuration (auto-generated)
- `launch_4gpu.bat` - Launch script for Windows

## Why Two-Stage Training?

**Memory Efficiency**: Training directly at 256³ would require 64× memory (OOM on most GPUs)

**Better Quality**: 
- Stage 1 learns anatomical structure with rich losses
- Stage 2 focuses purely on upsampling details

**Flexibility**:
- Can use 64³ output if 256³ is not needed
- Can retrain only refinement without retraining base
- Can experiment with different upsampling strategies

**Faster Training**:
- Stage 1 converges faster at 64³
- Stage 2 only trains lightweight network (5-8M params)
