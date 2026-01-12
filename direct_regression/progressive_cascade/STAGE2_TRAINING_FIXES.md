# Stage 2 Training Fixes - Applied on RunPod

## Changes Made to Fix Training

### 1. Reduced Gradient Loss Weight (Prevent NaN)
**File:** `direct_regression/progressive_cascade/loss_multiscale.py`
```python
# OLD: gradient_weight=0.15
# NEW: gradient_weight=0.05
class Stage2Loss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=0.5, vgg_weight=0.1, gradient_weight=0.05):
```

### 2. Reduced ViT Complexity (Fit in Memory)
**File:** `direct_regression/progressive_cascade/model_progressive.py`
```python
# Stage 2 changes:
vit_depth=4,  # Was 6
num_heads=4,  # Was 8
```

### 3. Set Stage 2 Target Size to 16³ (4096 tokens)
**File:** `models/hybrid_vit_backbone.py`
```python
elif D <= 128:
    target_size = 16  # Stage 2: 128³ → 16³ (8x downsample)
```

### 4. Reduced Batch Size to 1
**File:** `direct_regression/progressive_cascade/config_progressive.json`
```json
"batch_size": 1  // Stage 2
```

### 5. Allow Stage 2 Checkpoint Loading
**File:** `direct_regression/progressive_cascade/train_progressive_1gpu.py`
```python
# Line 205:
if stage > 3:  # Changed from: if stage > 2
```

## Current Training Status
- **Stage 1**: ✅ Completed 100 epochs (27.13 dB PSNR)
- **Stage 2**: 🔄 Training from epoch 21 with fixed gradient weight
- **Checkpoint**: `stage1_best.pth` (actually Stage 2 epoch 21)
- **Expected**: ~3 hours to complete 100 epochs total

## Training Command (On RunPod)
```bash
cd /workspace/Hybrid-ViT-Cascade/direct_regression/progressive_cascade
python train_progressive_1gpu.py
```

## Key Improvements
1. ✅ Gradient loss added (0.05 weight) - sharpens edges
2. ✅ 100 epochs total (vs old 20) - better convergence  
3. ✅ Stable training (no NaN with reduced gradient weight)
4. ✅ 16³ tokens (4096) - fits in memory reliably

## Expected Results
- **PSNR**: 28-30 dB (vs old 27.63 dB)
- **SSIM**: 0.52-0.60 (vs old 0.4975)
- **Quality**: Sharper edges, better organ boundaries
