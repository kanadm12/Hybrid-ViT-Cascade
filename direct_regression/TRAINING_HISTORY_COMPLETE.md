# CT Reconstruction Training History & Results

## Project Overview
**Goal**: Achieve diagnostic-quality CT volume reconstruction (256³ resolution) from dual-view X-ray images using deep learning, targeting 20-25+ dB PSNR for clinical utility.

**Dataset**: 100 patients from `/workspace/drr_patient_data`, dual-view X-rays (frontal/lateral, 512×512) → CT volumes

**Infrastructure**: 4x NVIDIA A100 80GB GPUs, Distributed Data Parallel (DDP) training, mixed precision (FP16)

---

## Implementation Timeline & Results

### 1. Base Direct Regression Model (64³) - **SUCCESS** ✅

**Approach**: Direct regression from X-rays to 64³ CT volumes without diffusion
- **Architecture**: 
  - `XrayConditioningModule`: Sophisticated dual-view X-ray encoder
  - `HybridViT3D`: ViT backbone with cross-attention to X-ray features
  - Learnable initial 64³ volume embedding
  - ~15.3M parameters
- **Training Config**:
  - 100 patients, 80/20 train/val split
  - 100 epochs, batch_size=8 per GPU
  - Learning rate: Default (from original config)
  - Loss: L1 + SSIM (weights: 1.0 + 0.5)
  - Optimizer: AdamW with CosineAnnealingLR
- **Files**: 
  - Model: `model_direct.py`
  - Training: `train_direct_4gpu.py`
  - Config: `config_direct.json`
  - Checkpoint: `checkpoints_direct/best_model.pt`

**Results**:
- **Final PSNR**: **13.35 dB** after 100 epochs
- Training stable and consistent
- Model learns meaningful CT structure from X-rays
- Inference generates reasonable but blurry volumes at 64³ resolution

**Issues**:
- 64³ resolution too low for diagnostic quality
- Visible blurriness in reconstructed volumes
- Need higher resolution for clinical use

---

### 2. Enhanced Model with Complex Losses (64³) - **FAILED** ❌

**Approach**: Add perceptual, edge, and multi-scale losses to base model
- **Architecture**: Same as base DirectCTRegression but with enhanced loss functions
- **Training Config**:
  - Same 100 patients, 100 epochs
  - Added losses: Perceptual (VGG16), Edge (Sobel), Multi-scale
  - Batch size: 8 per GPU
  - Loss weights: L1(1.0) + Perceptual(0.2) + Edge(0.1) + Multi-scale(0.3)
- **Files**:
  - Model: `model_enhanced.py`
  - Training: `train_enhanced_4gpu.py`
  - Config: `config_enhanced.json`

**Results**:
- **Epoch 21 PSNR**: **5.29 dB** (vs base 13.35 dB)
- Training unstable, severe performance degradation
- Enhanced losses interfered with convergence
- Training abandoned at epoch 21

**Issues**:
- Perceptual loss overwhelmed other losses
- Edge loss added instability
- More complex ≠ better performance
- Base model with simple L1+SSIM worked better

**Lessons Learned**:
- Simple loss functions often outperform complex combinations
- VGG perceptual loss problematic for medical imaging (designed for natural images)
- Architecture matters more than loss complexity

---

### 3. Refinement Network (64³ → 256³ Upsampling) - **FAILED** ❌

**Approach**: Two-stage training
1. Use frozen base 64³ model (13.35 dB)
2. Train lightweight upsampling network to refine 64³ → 256³

#### 3.1. Initial Refinement Architecture

**Architecture**:
- Simple `RefinementNetwork` with progressive upsampling
- 64³ → 128³ → 256³ using trilinear interpolation
- Residual blocks at each stage
- ~2.1M parameters
- **Files**:
  - Model: `model_enhanced.py` (RefinementNetwork class)
  - Training: `train_refinement_4gpu.py`
  - Config: `config_refinement.json`

**Training Config**:
- Frozen base model from `checkpoints_direct/best_model.pt`
- 100 patients, 100 epochs, batch_size=2
- Learning rate: 5e-5
- Loss: L1 + Perceptual + Edge (weights: 1.0, 0.1, 0.1)
- No proper train/val split initially (data leakage)

**Results**:
- **Epoch 1**: 1.04 dB
- **Epoch 8**: 6.62 dB
- **Status**: Quality worse than base model, not improving

**Issues Identified**:
1. No train/val split (both using same 100 patients) ✗
2. Simple upsampling insufficient
3. Network learning from scratch, not building on base quality
4. Perceptual loss too high (5.8+), dominating training

#### 3.2. Improved Refinement Architecture

**Approach**: Complete redesign with multiple improvements
- **Architecture Changes** (`model_refinement_improved.py`):
  - Residual learning (predict correction, not full volume)
  - Sub-pixel convolution (PixelShuffle3D) instead of trilinear
  - Channel attention blocks
  - Multi-scale supervision (64³, 128³, 256³)
  - SSIM loss added (weight 2.0 for direct SSIM optimization)
  - ~721K parameters (0.72M)

**Training Config Updates**:
- Fixed: 80/20 train/val split ✓
- Loss weights: L1(1.0), SSIM(2.0→0.5→0.0), Perceptual(0.1→0.01→0.0), Edge(0.1→0.05), MultiScale(0.5→0.2)
- Multiple iterations with different weight combinations
- num_workers reduced: 4→2 (memory issues with 256³)

**Iteration History**:
1. **Initial improved architecture**: SSIM weight 2.0
   - Result: 6.55 dB at epoch 1, not improving
   - Issue: SSIM loss too aggressive
   
2. **Reduced perceptual weight**: 0.1→0.01
   - Result: Still ~6.5 dB
   - Issue: Perceptual loss still problematic (6.9+ values)
   
3. **Fixed VGG loading + ImageNet normalization**
   - Added proper mean/std normalization for VGG16
   - Result: No improvement
   
4. **Disabled perceptual loss entirely** (weight=0.0)
   - Result: Still 6-7 dB range
   - Issue: Not reaching base model quality
   
5. **Residual correction with 0.3 scale**
   - Removed Tanh, increased correction scale
   - Result: 6.63 dB at epoch 1
   
6. **Zero-initialized final layer**
   - Initialize to output ~0 corrections (start from base quality)
   - Reduced loss weights: SSIM(0.5), Edge(0.05), MS(0.2)
   - Result: 5.99 dB at epoch 1 with zero init

**Final Results**:
- **Best PSNR**: ~7 dB at epoch 7 (with zero init)
- Training abandoned - refinement actively degrading quality
- Never reached base model's 13.35 dB

**Root Cause Analysis**:
- Upsampling from 64³ requires learning high-frequency details that don't exist in base prediction
- Network can't "hallucinate" missing information effectively
- Skip connection from upsampled base not sufficient
- Random initialization of upsampling layers destroys base model quality initially

**Files**:
- Model: `model_refinement_improved.py`
- Training: `train_refinement_4gpu.py` (multiple versions)
- Config: `config_refinement.json`
- Checkpoint: `checkpoints_refinement/` (various epochs, none useful)

---

### 4. Direct 256³ Training - **IN PROGRESS** ⏳

**Approach**: Train model to directly output 256³ volumes (no upsampling stage)

#### 4.1. Initial Simplified Architecture - **FAILED** ❌

**Architecture** (`model_direct_256.py`):
- Simple patch-based ViT (16×16 patches)
- No XrayConditioningModule (basic patch embedding)
- No learned initial volume
- Progressive decoder: 32³ → 64³ → 128³ → 256³
- ~7.5M parameters

**Training Config**:
- 100 patients, 80/20 split
- 100 epochs, batch_size=1→2 (for A100 80GB)
- Learning rate: 1e-4
- Gradient accumulation: 4→2 steps
- OneCycleLR scheduler
- Loss: L1(1.0) + SSIM(0.1)
- Gradient checkpointing enabled

**Results**:
- **Epoch 1**: 2.75 dB
- **Epoch 3**: 2.92 dB
- Training extremely slow, poor quality

**Issues**:
- Architecture too simple (missing key components from successful 64³ model)
- No cross-attention with X-ray features
- SSIM loss causing instability (0.99+ values)
- Learning rate too high

#### 4.2. Proper Architecture Based on 64³ Model - **CURRENT** 🔄

**Approach**: Use proven 64³ architecture + progressive upsampling decoder

**Architecture** (`model_direct_256_v2.py`):
- Same as successful 64³ model:
  - `XrayConditioningModule` ✓
  - `HybridViT3D` with cross-attention ✓
  - Learnable initial 64³ volume ✓
- Added: Progressive upsampling decoder (64³ → 128³ → 256³)
- Gradient checkpointing for memory efficiency
- ~Same parameters as 64³ + decoder layers

**Training Config Iterations**:

1. **Initial config**:
   - LR: 1e-4, OneCycleLR
   - Loss: L1(1.0) + SSIM(0.1)
   - Result: 1.79 dB at epoch 1

2. **Disabled SSIM** (weight=0.0):
   - Pure L1 loss for stability
   - Result: Still ~2-3 dB range

3. **Reduced learning rate**: 1e-4 → 2e-5
   - Changed to CosineAnnealingLR (same as 64³)
   - Result: 3.90 dB at epoch 1, slow progress

4. **Increased learning rate**: 2e-5 → 5e-5
   - Trying to speed up convergence
   - Added resume capability
   - Current training in progress

**Current Results** (as of epoch 24):
- **Epoch 1**: 3.90 dB
- **Epoch 10**: 5.43 dB (+1.53 dB in 10 epochs)
- **Epoch 20**: 5.89 dB (+0.46 dB in next 10 epochs)
- **Epoch 24**: 5.97 dB
- **Progress rate**: ~0.5-1.0 dB per 10 epochs (slowing down)

**Projected Results**:
- Epoch 50: ~8-9 dB (estimated)
- Epoch 100: ~11-13 dB (estimated, may plateau)
- **Unlikely to significantly exceed 64³ model (13.35 dB)**

**Current Issues**:
- Training converging very slowly
- Upsampling decoder learning from scratch
- May plateau below base model quality
- Takes ~0.3-0.7 samples/sec (very slow)

**Files**:
- Model: `model_direct_256_v2.py`
- Training: `train_direct_256_4gpu.py` (with resume capability)
- Config: `config_direct_256.json`
- Checkpoint: `checkpoints_direct_256/best_model_256.pt` (ongoing)

---

## Summary of Key Findings

### ✅ What Worked:
1. **Simple Direct Regression (64³)**: Clean architecture, simple losses (L1+SSIM), achieved 13.35 dB
2. **80/20 Train/Val Split**: Proper data separation prevents overfitting
3. **Mixed Precision + DDP**: Enables efficient 4-GPU training
4. **Gradient Checkpointing**: Reduces memory for larger volumes
5. **CosineAnnealingLR**: Better than OneCycleLR for this task
6. **Resume Capability**: Allows training continuation with new hyperparameters

### ❌ What Failed:
1. **Complex Loss Functions**: Perceptual + Edge + Multi-scale degraded performance
2. **Two-Stage Refinement**: Upsampling network couldn't improve on base quality
3. **Direct 256³ Training**: Extremely slow, may not beat 64³ baseline
4. **High Learning Rates**: Caused instability (1e-4 too high for 256³)
5. **SSIM Loss at 256³ Scale**: Too memory intensive, unstable (0.99+ values)
6. **Residual Learning for Upsampling**: Couldn't effectively add missing details

### 🔍 Key Insights:

**Architecture**:
- Cross-attention to X-ray features is critical
- Learned initial volume embedding helps convergence
- Simple ViT backbone sufficient; complex attention mechanisms unnecessary
- Upsampling decoders struggle to add high-frequency details

**Training**:
- Lower learning rates (2e-5 to 5e-5) work better than high LR (1e-4+)
- Simple L1 loss often beats complex loss combinations
- Batch size limited by memory: 8 for 64³, 2 for 256³
- 100 epochs sufficient for 64³, may need 150-200 for 256³

**Resolution Strategy**:
- **64³ direct**: Fast training, good quality (13.35 dB) ✓
- **64³ + learned upsampling**: Failed, degraded quality ✗
- **256³ direct**: Very slow, uncertain if better than 64³ ❓
- **Best option**: Use 64³ model + simple interpolation to 256³

**Medical Imaging Specifics**:
- VGG perceptual loss (trained on ImageNet) not ideal for CT
- SSIM more meaningful than perceptual for medical images
- Edge preservation important but dedicated edge loss not helpful
- CT volumes have different statistics than natural images

---

## Performance Comparison

| Method | Resolution | PSNR (dB) | Training Time | Status | Quality |
|--------|-----------|-----------|---------------|---------|---------|
| **Base Direct (64³)** | 64³ | **13.35** | ~4-6 hours | ✅ Complete | Good |
| Enhanced (64³) | 64³ | 5.29 | Abandoned | ❌ Failed | Poor |
| Refinement v1 | 256³ | 6.62 | Abandoned | ❌ Failed | Poor |
| Refinement v2 (improved) | 256³ | ~7.0 | Abandoned | ❌ Failed | Poor |
| Direct 256³ v1 (simple) | 256³ | 2.92 | Abandoned | ❌ Failed | Very Poor |
| **Direct 256³ v2 (proper)** | 256³ | **5.97** | In Progress | ⏳ Ongoing | Improving Slowly |

---

## Recommended Next Steps

### Option 1: Use 64³ Model + Interpolation (FASTEST) ⚡
**Approach**: 
- Use trained 64³ model (13.35 dB)
- Apply trilinear interpolation to 256³ at inference
- Simple, immediate results

**Pros**:
- No additional training needed
- Proven 13.35 dB quality
- Fast inference
- Already works

**Cons**:
- Simple interpolation, no learned upsampling
- May have smoothing artifacts
- Limited detail at 256³

**Implementation**: Create inference script with F.interpolate(scale_factor=4)

---

### Option 2: Wait for 256³ Training (CURRENT) ⏳
**Approach**: Let current 256³ training complete (epoch 24/100)

**Projected Outcome**:
- Epoch 100: ~11-13 dB (may match 64³)
- Total time: ~8-12 more hours
- Uncertain if better than Option 1

**Pros**:
- Learned end-to-end 256³ generation
- No interpolation artifacts
- May capture some high-frequency details

**Cons**:
- Very slow convergence
- May plateau below 64³ baseline
- Resource intensive (8+ more hours)

---

### Option 3: Train 128³ Model (COMPROMISE) ⚖️
**Approach**: Train direct 128³ model (middle ground)

**Rationale**:
- 128³ = 8x smaller than 256³ (faster training)
- 128³ = 8x larger than 64³ (better resolution)
- Better balance of speed and quality

**Estimated**:
- Training time: ~4-6 hours
- Expected PSNR: 15-18 dB
- Interpolation to 256³ smoother than from 64³

**Implementation Needed**: Create config_direct_128.json

---

### Option 4: Medical-Grade U-Net (ADVANCED) 🏥
**Approach**: Use specialized medical imaging architecture

**Architecture**:
- 3D U-Net with attention
- Multi-scale feature extraction
- Skip connections throughout
- Designed for CT reconstruction

**Potential**:
- PSNR: 25-32 dB (clinical grade)
- Better for fine anatomical details
- State-of-the-art for medical imaging

**Requirements**:
- Significant architecture development
- Longer training (200+ epochs)
- More complex implementation

---

## File Organization

### Working Models:
- ✅ `direct_regression/model_direct.py` - Base 64³ model (13.35 dB)
- ⏳ `direct_regression/model_direct_256_v2.py` - Current 256³ model (5.97 dB, training)

### Training Scripts:
- ✅ `direct_regression/train_direct_4gpu.py` - Base 64³ training
- ⏳ `direct_regression/train_direct_256_4gpu.py` - Current 256³ training (with resume)

### Inference Scripts:
- ✅ `direct_regression/inference_direct.py` - For 64³ model
- 🆕 Need: Inference with 64³→256³ interpolation

### Checkpoints:
- ✅ `checkpoints_direct/best_model.pt` - Base 64³ (13.35 dB) - **BEST SO FAR**
- ⏳ `checkpoints_direct_256/best_model_256.pt` - 256³ ongoing (5.97 dB at epoch 24)

### Failed Experiments (Archived):
- ❌ `model_enhanced.py` - Enhanced losses (failed)
- ❌ `model_refinement_improved.py` - Refinement network (failed)
- ❌ `model_direct_256.py` - Simple 256³ (failed)
- ❌ `train_enhanced_4gpu.py` - Enhanced training (abandoned)
- ❌ `train_refinement_4gpu.py` - Refinement training (abandoned)

---

## Lessons for Future Work

1. **Start simple**: Complex architectures and losses don't guarantee better results
2. **Validate assumptions**: Test that upsampling can actually improve quality before committing
3. **Monitor closely**: Catch failing approaches early (epochs 10-20) rather than waiting
4. **Resolution matters**: Higher resolution doesn't always mean better reconstruction quality
5. **Domain-specific**: Medical imaging may need specialized approaches vs. natural images
6. **Baseline first**: Establish a strong simple baseline before trying advanced techniques
7. **Two-stage risky**: End-to-end training often better than staged approaches
8. **Learning rate critical**: Must tune carefully for each resolution/architecture

---

## Technical Specifications

### Hardware:
- **GPUs**: 4× NVIDIA A100 80GB
- **VRAM Usage**: 
  - 64³: ~8-10 GB per GPU
  - 256³: ~30-40 GB per GPU
- **Training Speed**:
  - 64³: ~2-3 samples/sec
  - 256³: ~0.3-0.7 samples/sec

### Software Stack:
- PyTorch 2.x with CUDA 12.8
- Distributed Data Parallel (DDP)
- Mixed Precision (FP16/FP32)
- Gradient Checkpointing (for 256³)

### Dataset:
- **Source**: `/workspace/drr_patient_data`
- **Patients**: 100 total
- **Split**: 80 train / 20 validation
- **Input**: Dual-view X-rays (512×512, frontal + lateral)
- **Output**: CT volumes (64³ or 256³)
- **Augmentation**: None (medical data)

---

## Conclusion

After extensive experimentation with multiple architectures and training strategies:

**Current Best Model**: **Base 64³ Direct Regression (13.35 dB)** ✅

**Recommended Path Forward**:
1. **Short-term**: Use 64³ model with interpolation to 256³
2. **Medium-term**: Consider 128³ direct training as compromise
3. **Long-term**: Explore medical-grade architectures (U-Net, nnU-Net)

**Key Takeaway**: Sometimes the simplest approach (direct 64³ regression with L1+SSIM loss) outperforms more complex alternatives (refinement networks, enhanced losses, direct 256³ training). Quality improvements require fundamental architectural innovations, not just resolution increases or loss function engineering.

---

*Document created: January 2-3, 2026*
*Last updated: Epoch 24/100 of 256³ training (5.97 dB)*
*Author: Training history compiled from session logs*
