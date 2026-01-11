# Stage 2 Quality Improvement Plan

## Problem Diagnosis

Your Stage 2 results (PSNR: 27.63 dB, SSIM: 0.4975) show blurry reconstruction without fine anatomical details (bones, organs, vessels). This is caused by:

### 1. **Insufficient Training** ⚠️ PRIMARY ISSUE
- Only **20 epochs** is far too short for medical image reconstruction
- Medical DL models typically need 50-200 epochs to capture fine details
- The model hasn't converged yet - loss was still decreasing

### 2. **Loss Function Gaps**
- Previous Stage 2 only had: L1 + SSIM + VGG
- Missing **gradient loss** for sharp edges (bones, vessels)
- VGG trained on natural images, not medical scans

### 3. **Architecture Bottleneck** (Less critical)
- Stage 2 uses 24³ ≈ 13,800 tokens (5.3x downsample)
- This is actually reasonable - not the main issue
- More training will help the model use this capacity better

## Solutions Implemented

### ✅ 1. Extended Training (100 epochs)
**Modified Files:**
- [train_progressive_1gpu.py](train_progressive_1gpu.py) - Changed from 20→100 epochs per stage
- [train_stage2_extended.py](train_stage2_extended.py) - NEW dedicated Stage 2 training script

**Key Changes:**
- 100 epochs with cosine annealing LR schedule
- Better convergence for capturing fine anatomical structures
- Periodic checkpointing every 20 epochs

### ✅ 2. Improved Loss Function
**Modified Files:**
- [loss_multiscale.py](loss_multiscale.py) - Added gradient loss to Stage 2

**New Stage 2 Loss:**
```python
Total Loss = L1 + 0.5×SSIM + 0.1×VGG + 0.15×Gradient
```

**Why This Helps:**
- **Gradient Loss**: Penalizes blurry edges → sharper bones/vessels
- **SSIM**: Structural similarity → better organ shapes
- **VGG**: Texture preservation → soft tissue detail
- **L1**: Overall intensity accuracy

### ✅ 3. Training Optimizations
- **Cosine Annealing**: LR 1e-4 → 1e-6 for smooth convergence
- **Gradient Checkpointing**: Already enabled for memory efficiency
- **PSNR-based Checkpointing**: Saves best model based on image quality

## Next Steps

### Option A: Resume Stage 2 Training (RECOMMENDED)
Use the new extended training script:

```bash
# On RunPod
cd /workspace/x2ctpa/hybrid_vit_cascade/direct_regression/progressive_cascade
python train_stage2_extended.py
```

**Expected Results:**
- **Training Time**: ~3 hours (100 epochs)
- **Expected PSNR**: 30-35 dB (vs current 27.63 dB)
- **Expected SSIM**: 0.6-0.7 (vs current 0.4975)
- **Visual Quality**: Visible vertebrae, sharper organ boundaries, clearer vessels

### Option B: Full Pipeline with New Settings
Retrain from scratch with 100 epochs per stage:

```bash
# Edit train_progressive_1gpu.py line 450: start_stage = 1
python train_progressive_1gpu.py
```

**Training Time**: ~10 hours total (Stage 1: 3h, Stage 2: 3h, Stage 3: 4h)

### Option C: Further Improvements (Future Work)

If Option A still doesn't capture enough detail:

1. **Reduce Downsampling Further**
   - Change Stage 2: 24³ → 32³ tokens (5.3x → 4x downsample)
   - More spatial resolution but higher memory usage

2. **Add Medical-Specific Losses**
   - Bone segmentation loss
   - Vessel enhancement loss
   - Anatomical landmark constraints

3. **Multi-View Input**
   - Use 4+ X-ray views instead of 2
   - Better geometric constraints

4. **Adversarial Training**
   - Add discriminator for realistic texture
   - Helps with fine capillaries/trabecular bone

## Quick Comparison

| Metric | Stage 2 (20 epochs) | Expected (100 epochs) |
|--------|--------------------|-----------------------|
| PSNR   | 27.63 dB          | 30-35 dB             |
| SSIM   | 0.4975            | 0.6-0.7              |
| Bone Detail | Blurry | Sharp edges |
| Organ Detail | Smeared | Clear boundaries |
| Vessel Detail | Missing | Visible |
| Training Time | 32 min | ~3 hours |

## Files Modified

1. [train_progressive_1gpu.py](train_progressive_1gpu.py) - 100 epochs/stage
2. [loss_multiscale.py](loss_multiscale.py) - Added gradient loss to Stage 2
3. [train_stage2_extended.py](train_stage2_extended.py) - NEW dedicated training script

## Why 20 Epochs Wasn't Enough

Medical image reconstruction is harder than natural images because:
- **Sparse features**: Bones/vessels occupy small % of volume
- **High dynamic range**: -1000 to +3000 HU (vs 0-255 RGB)
- **3D spatial coherence**: Must maintain anatomy across all slices
- **Fine structures**: 1-2 voxel vessels require precise reconstruction

Most published medical DL papers use **100-200 epochs**. Your 20 epochs was just the warm-up phase!

## Recommended Action

**Start with Option A** (train_stage2_extended.py) since you already have stage1_best.pth. This will take ~3 hours and should show significant improvement.

If you're satisfied with the results, you can then proceed to Stage 3 training with the same extended approach.
