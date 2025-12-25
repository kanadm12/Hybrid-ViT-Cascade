# Component-Specific Loss Monitoring Guide

## Why Component-Specific Losses?

**Standard approach:**
```python
total_loss = mse(predicted, target)  # One number → can't debug
```

**Diagnostic approach:**
```python
losses = {
    'diffusion': 0.045,      # ✓ Core working
    'projection': 0.132,     # ✗ Physics failing!
    'depth_consistency': 0.021,  # ✓ Priors helping
    'cross_attention': 0.089     # ⚠ Attention needs tuning
}
# Now you know EXACTLY what to fix!
```

---

## Loss Categories & Diagnosis

### 1. **Diffusion Loss** - Core Denoising

```python
diffusion_loss = F.mse_loss(predicted_noise, target_noise)
```

**What it measures:** How well the model predicts noise/velocity

**Healthy range:** 
- `< 0.01` → Excellent
- `0.01 - 0.05` → Good
- `0.05 - 0.1` → Warning
- `> 0.1` → Critical failure

**If failing:**
- ❌ Model capacity too small → Increase `voxel_dim`, `vit_depth`
- ❌ Learning rate too high → Reduce to `1e-5`
- ❌ Data normalization issue → Check X-ray/CT are in [-1, 1]

---

### 2. **Projection Loss** - Physics Constraint

```python
drr_pred = render_drr(predicted_ct)
projection_loss = F.mse_loss(drr_pred, input_xray)
```

**What it measures:** Does generated CT project back to input X-ray?

**Healthy range:**
- `< 0.005` → Excellent (clinically valid)
- `0.005 - 0.02` → Good
- `0.02 - 0.05` → Warning (visible artifacts)
- `> 0.05` → Critical (medically invalid)

**If failing:**
- ❌ Projection weight too low → Increase from 0.3 to 0.5
- ❌ DRR renderer incorrect → Check geometry/ray tracing
- ❌ Multi-view inconsistency → Check view angles (0°, 90°)

**Special check:**
```python
projection_gt_sanity = F.mse_loss(render_drr(ground_truth_ct), input_xray)
# Should be < 0.01. If not, your data has misalignment!
```

---

### 3. **Depth Consistency** - Is Depth Lifting Helping?

```python
correlation = cosine_similarity(predicted_ct, depth_prior)
depth_consistency = (correlation - 0.45)²  # Target: 0.45
```

**What it measures:** How much the model uses anatomical priors

**Interpretation:**
- Correlation `0.5-0.6` → **Excellent** - Using priors intelligently
- Correlation `0.3-0.5` → **Good** - Balanced use
- Correlation `0.1-0.3` → **Warning** - Barely using priors
- Correlation `< 0.1` → **Critical** - Ignoring priors completely
- Correlation `> 0.7` → **Over-reliance** - Just copying prior

**If correlation too low:**
```python
# Prior is being ignored → Increase depth lifting influence
noisy_volume = noisy_volume + 0.3 * depth_prior  # Was 0.1
```

**If correlation too high:**
```python
# Over-copying prior → Model not learning
# Reduce depth prior weight or increase dropout
```

**Prior Improvement Check:**
```python
prior_error = mse(depth_prior, ground_truth)
pred_error = mse(predicted_ct, ground_truth)
improvement = (prior_error - pred_error) / prior_error

# improvement > 0.3 → Good (30% better than prior)
# improvement < 0 → BAD (worse than prior!)
```

---

### 4. **Cross-Attention Alignment** - Is X-ray Conditioning Working?

```python
# Check attention entropy
attn_probs = softmax(cross_attention_scores)  # (B, voxels, xray_tokens)
entropy = -sum(attn_probs * log(attn_probs))

target_entropy = log(num_xray_tokens) * 0.6
attention_loss = (entropy - target_entropy)²
```

**What it measures:** 
- **Entropy:** How diverse is attention? (Low = collapsed, High = uniform/random)
- **Sparsity:** Does each voxel attend to few relevant X-ray regions?

**Healthy patterns:**
```
Entropy 3.5-4.5 (with 256 tokens) → Moderately diverse ✓
Sparsity 0.6-0.8 → Peaky attention ✓

Entropy < 2.0 → COLLAPSED - All voxels attend to same token ✗
Entropy > 6.0 → RANDOM - Attention not learning ✗
Sparsity < 0.3 → Too uniform ✗
```

**If attention collapsed:**
- ❌ Cross-attention scale too large → Add temperature scaling
- ❌ X-ray features not informative → Check encoder quality
- ❌ Attention dropout too high → Reduce from 0.1 to 0.05

**If attention random:**
- ❌ Learning rate too high
- ❌ Not enough training
- ❌ X-ray encoder frozen → Unfreeze and train jointly

---

### 5. **Stage Transition** - Cascade Coherence

```python
prev_upsampled = interpolate(prev_stage_output, size=current_size)

# Low-frequency should match
low_freq_loss = mse(avg_pool(pred), avg_pool(prev_upsampled))

# High-frequency should differ (adding details)
detail_difference = -mse(pred - low_freq_pred, prev - low_freq_prev)
```

**What it measures:** Are cascade stages coherent yet improving?

**Healthy behavior:**
- `stage_transition < 0.05` → Smooth transition ✓
- `detail_addition < 0` → Adding new high-freq details ✓

**If stage_transition > 0.1:**
- ❌ Stages disconnected → Not using `prev_stage_embed` properly
- ❌ Resolution jump too large → Add intermediate stage (96³ between 64³ and 128³)
- ❌ Previous stage undertrained → Train longer

**Visualization:**
```python
# Plot stage 1 vs stage 2 (upsampled)
stage1_up = interpolate(stage1_output, size=(128,128,128))
stage2_out = stage2_output

diff = stage2_out - stage1_up
# diff should show fine details, not large structural changes
```

---

### 6. **Frequency Analysis** - Structure vs Details

```python
# Low-frequency (structure)
pred_low = avg_pool(pred, kernel=8)
gt_low = avg_pool(gt, kernel=8)
loss_structure = mse(pred_low, gt_low)

# High-frequency (details)
pred_high = pred - pred_low
gt_high = gt - gt_low
loss_details = mse(pred_high, gt_high)
```

**What it measures:** Which aspects are failing?

**Diagnosis:**

| Scenario | Interpretation | Fix |
|----------|---------------|-----|
| `loss_structure > 2 × loss_details` | **Struggling with anatomy** | • Increase depth priors<br>• Add anatomical segmentation loss<br>• Check training data quality |
| `loss_details > 2 × loss_structure` | **Missing fine details** | • Increase model capacity<br>• Add perceptual loss<br>• Reduce smoothing regularization |
| `loss_structure ≈ loss_details` | **Balanced** ✓ | Continue training |

**Example:**
```
loss_structure = 0.082  }  Ratio = 4.1 → ANATOMY PROBLEM
loss_details = 0.020    }
→ Model can't capture organ shapes, focus on depth priors
```

---

### 7. **Multi-View & Multi-Scale Projection**

```python
# Single view (AP)
loss_ap = mse(render_drr(pred, angle=0°), xray_ap)

# Multi-view (Lateral)
loss_lateral = mse(render_drr(pred, angle=90°), xray_lateral)

# Multi-scale
loss_64 = mse(render_drr_64(pred), downsample(xray, 64))
loss_128 = mse(render_drr_128(pred), downsample(xray, 128))
loss_512 = mse(render_drr_512(pred), xray)
```

**What it measures:** Consistency across views and scales

**If multi-view failing:**
```
loss_ap = 0.015      }  Lateral view failing!
loss_lateral = 0.089 }
→ 3D geometry incorrect, check depth estimation
```

**If multi-scale failing:**
```
loss_64 = 0.012   }  Fine details failing
loss_128 = 0.018  }
loss_512 = 0.095  }
→ Model good at coarse structure, bad at high-res
→ Increase capacity at final stage
```

---

## Complete Monitoring Dashboard

### During Training - W&B Logging

```python
wandb.log({
    # Core losses
    'loss/total': total_loss,
    'loss/diffusion': diffusion_loss,
    
    # Physics
    'physics/projection_single': proj_single,
    'physics/projection_multi_view': proj_multi_view,
    'physics/projection_multi_scale': proj_multi_scale,
    'physics/gt_sanity_check': proj_gt_sanity,
    
    # Architecture components
    'component/depth_consistency': depth_consistency,
    'component/depth_correlation': depth_correlation,
    'component/prior_improvement': prior_improvement,
    'component/cross_attention_entropy': attn_entropy,
    'component/cross_attention_sparsity': attn_sparsity,
    
    # Cascade
    'cascade/stage_transition': stage_transition,
    'cascade/detail_addition': detail_addition,
    
    # Frequency
    'frequency/structure': loss_structure,
    'frequency/details': loss_details,
    'frequency/ratio': loss_structure / loss_details,
    
    # Perceptual
    'perceptual/features': perceptual_loss,
    
    # Health scores
    'health/denoising': health_scores['denoising'],
    'health/physics': health_scores['physics'],
    'health/depth_lifting': health_scores['depth_lifting'],
})
```

### Alert Triggers

```python
# Auto-alert when components fail
if projection_loss > 0.05:
    print("⚠️ ALERT: Physics constraint failing!")
    print(f"   Current: {projection_loss:.4f}, Target: < 0.02")
    print("   → Increase projection_weight or check DRR renderer")

if depth_correlation < 0.2:
    print("⚠️ ALERT: Depth priors being ignored!")
    print(f"   Correlation: {depth_correlation:.4f}, Target: 0.4-0.6")
    print("   → Increase depth_prior weight in input")

if prior_improvement < 0:
    print("🚨 CRITICAL: Model worse than depth prior!")
    print("   → Check model architecture or data quality")

if loss_structure / loss_details > 3.0:
    print("⚠️ ALERT: Struggling with anatomy!")
    print("   → Add anatomical constraints or improve depth lifting")
```

---

## Ablation Study Guide

### Test Each Component

```python
# Baseline
model = UnifiedModel(use_depth_lifting=False, use_physics_loss=False)
# Result: PSNR 34 dB

# + Depth Lifting
model = UnifiedModel(use_depth_lifting=True, use_physics_loss=False)
# Result: PSNR 36 dB (+2 dB) → Depth lifting helps!

# + Physics Loss
model = UnifiedModel(use_depth_lifting=True, use_physics_loss=True)
# Result: PSNR 38 dB (+2 dB) → Physics helps!

# + Cross-Attention
model = UnifiedModel(..., use_cross_attention=True)
# Result: PSNR 39 dB (+1 dB) → Cross-attention helps!
```

### Loss Weight Sensitivity

```python
projection_weights = [0.0, 0.1, 0.3, 0.5, 1.0]
results = {}

for weight in projection_weights:
    model.loss_weights['projection_single'] = weight
    psnr = train_and_evaluate(model)
    results[weight] = psnr

# Expected:
# 0.0 → 36 dB (no physics)
# 0.1 → 37 dB
# 0.3 → 38 dB ← Optimal
# 0.5 → 37.5 dB (over-constrained)
# 1.0 → 36 dB (physics dominates, diffusion suffers)
```

---

## Summary: Diagnostic Workflow

1. **Train model** with `DiagnosticLosses`
2. **Monitor W&B** dashboard for all component losses
3. **Identify bottleneck:**
   - Diffusion high → Model capacity
   - Projection high → Physics weight or DRR bug
   - Depth low correlation → Prior being ignored
   - Attention entropy extreme → Attention collapsed/random
   - Structure >> Details → Anatomy problem
   - Details >> Structure → Missing fine details
4. **Apply targeted fix** based on diagnosis
5. **Repeat** until all components healthy

**Result:** Systematic debugging instead of random hyperparameter tuning!
