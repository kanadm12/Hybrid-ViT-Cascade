# Quick Reference: Training Modes

## Single-View vs Multi-View DRR

### When to Use Each

| Mode | Use Case | Data Requirement | Expected Quality | Training Time |
|------|----------|------------------|------------------|---------------|
| **Single-View** | • Standard datasets<br>• Quick prototyping<br>• Limited data | 1 X-ray per CT | 36-38 dB PSNR | 1x (baseline) |
| **Dual-View** | • Clinical datasets<br>• Best quality/cost<br>• AP + Lateral | 2 X-rays per CT | **40-42 dB PSNR** | 1.2x |
| **Multi-View** | • Research datasets<br>• Maximum quality<br>• 360° acquisition | 3+ X-rays per CT | 42-44 dB PSNR | 1.5x+ |

**Recommendation:** Start with **dual-view** (AP + Lateral) for best balance of quality and practicality.

---

## Configuration Examples

### Config 1: Single-View Training
```json
{
  "stages": [
    {
      "name": "stage1_low",
      "volume_size": [64, 64, 64],
      "voxel_dim": 256,
      "vit_depth": 4,
      "num_heads": 4,
      "use_depth_lifting": true,
      "use_physics_loss": true
    }
  ],
  "num_views": 1,
  "xray_img_size": 512
}
```

**Training:**
```python
# Load single-view data
xrays = load_single_view()  # (B, 1, 1, 512, 512)
model = UnifiedHybridViTCascade(config)
loss_dict = model(ct_volume, xrays, 'stage1_low')
```

---

### Config 2: Dual-View Training (Recommended)
```json
{
  "stages": [
    {
      "name": "stage1_low",
      "volume_size": [64, 64, 64],
      "voxel_dim": 256,
      "vit_depth": 4,
      "num_heads": 4,
      "use_depth_lifting": true,
      "use_physics_loss": true
    }
  ],
  "num_views": 2,
  "xray_img_size": 512
}
```

**Training:**
```python
# Load dual-view data (AP + Lateral)
xrays_ap = load_frontal_xray()      # (B, 1, 512, 512)
xrays_lateral = load_lateral_xray() # (B, 1, 512, 512)
xrays = torch.stack([xrays_ap, xrays_lateral], dim=1)  # (B, 2, 1, 512, 512)

model = UnifiedHybridViTCascade(config)
loss_dict = model(ct_volume, xrays, 'stage1_low')
# Physics loss automatically averages over both views
```

---

### Config 3: Progressive Cascade with Dual-View
```json
{
  "stages": [
    {
      "name": "stage1_low",
      "volume_size": [64, 64, 64],
      "voxel_dim": 256,
      "vit_depth": 4,
      "num_heads": 4
    },
    {
      "name": "stage2_mid",
      "volume_size": [128, 128, 128],
      "voxel_dim": 384,
      "vit_depth": 6,
      "num_heads": 6
    },
    {
      "name": "stage3_high",
      "volume_size": [256, 256, 256],
      "voxel_dim": 512,
      "vit_depth": 8,
      "num_heads": 8
    }
  ],
  "num_views": 2,
  "training": {
    "batch_size": 4,
    "num_epochs": 50,
    "learning_rate": 1e-4
  }
}
```

**Progressive Training:**
```python
# Stage 1: Train on 64³
for epoch in range(50):
    loss = model(ct_64, xrays_dual, 'stage1_low')
    ...

# Stage 2: Use Stage 1 output as conditioning
prev_output = generate_stage1(xrays_dual)
for epoch in range(50):
    loss = model(ct_128, xrays_dual, 'stage2_mid', prev_stage_volume=prev_output)
    ...

# Stage 3: Use Stage 2 output
prev_output = generate_stage2(xrays_dual, stage1_output)
for epoch in range(50):
    loss = model(ct_256, xrays_dual, 'stage3_high', prev_stage_volume=prev_output)
    ...
```

---

## Loss Weight Tuning

### Default Weights
```python
physics_weight = 0.3  # Balance between diffusion and physics
```

### Stage-Dependent Weights (Recommended)
```python
physics_weights = {
    'stage1_low': 0.2,   # Lower res → less strict physics
    'stage2_mid': 0.3,   # Medium res → standard
    'stage3_high': 0.5   # High res → enforce strong physics
}
```

### Multi-View Weight Adjustment
```python
# Single-view
physics_weight = 0.3

# Dual-view (stronger constraint)
physics_weight = 0.4

# Multi-view (strongest constraint)
physics_weight = 0.5
```

---

## Monitoring Checklist

### Essential Metrics (Every Epoch)
- [ ] `diffusion_loss` → Should decrease from ~0.5 to <0.05
- [ ] `physics_loss` → Should decrease from ~0.2 to <0.02
- [ ] `total_loss` → Should steadily decrease
- [ ] Gradient norms → Should be stable (not exploding)

### Diagnostic Metrics (Every 10 Epochs)
- [ ] `depth_consistency` → Correlation 0.4-0.6 is healthy
- [ ] `frequency_low` vs `frequency_high` → Ratio should be ~1:1
- [ ] `projection_gt_sanity` → Should be <0.01 (validates data)
- [ ] `prior_improvement_ratio` → Should be >0.3 (30% better than prior)

### Advanced Monitoring (Optional)
- [ ] Attention entropy → 3.5-4.5 is good (not collapsed or random)
- [ ] Stage transition loss → <0.05 for smooth cascade
- [ ] Per-view physics losses → Should be similar across views

---

## Common Issues & Solutions

### Issue 1: Physics Loss Stuck High (>0.1)
**Symptoms:** `physics_loss` stays above 0.1 after 20+ epochs

**Diagnosis:**
```python
# Check if GT also has high projection error (data misalignment)
drr_gt = drr_renderer(ground_truth_ct)
gt_sanity = F.mse_loss(drr_gt, xray_target)
print(f"GT sanity check: {gt_sanity.item()}")

# If gt_sanity > 0.05 → DATA PROBLEM (misalignment)
# If gt_sanity < 0.01 → MODEL PROBLEM
```

**Solutions:**
- Data problem: Realign X-rays and CT volumes
- Model problem: Increase `physics_weight` to 0.5-0.7

---

### Issue 2: Dual-View Inconsistency
**Symptoms:** AP view loss low, lateral view loss high

**Diagnosis:**
```python
# Check individual view losses
for view_idx in range(num_views):
    drr = render_drr(pred_ct)
    view_loss = mse(drr, xrays[:, view_idx])
    print(f"View {view_idx}: {view_loss.item():.6f}")

# Large difference (>2x) indicates 3D geometry issues
```

**Solutions:**
- Check lateral X-ray orientation (should be 90° from AP)
- Verify CT volume axes match X-ray coordinate systems
- Add view-specific weight balancing

---

### Issue 3: Depth Priors Ignored
**Symptoms:** `depth_consistency` shows correlation <0.2

**Diagnosis:**
```python
# Check prior quality
prior_error = mse(depth_prior, ground_truth)
pred_error = mse(prediction, ground_truth)
improvement = (prior_error - pred_error) / prior_error

# If improvement < 0 → Model worse than prior!
```

**Solutions:**
- Increase depth prior weight in input: `0.1 → 0.3`
- Check anatomical priors are sensible
- Reduce model capacity (may be ignoring prior due to overfit)

---

## Quick Start Commands

### Test Installation
```bash
cd hybrid_vit_cascade
python test_all_fixes.py
# Should see: ✅ 6/6 tests passed
```

### Train Stage 1 (Single-View)
```bash
python training/train_progressive.py \
    --config config/quick_2stage.json \
    --checkpoint_dir checkpoints/single_view \
    --wandb
```

### Train Stage 1 (Dual-View, Recommended)
```bash
# Edit config: "num_views": 2
python training/train_progressive.py \
    --config config/quick_2stage_dualview.json \
    --checkpoint_dir checkpoints/dual_view \
    --wandb \
    --wandb_project xray2ct-dualview
```

### Resume Training
```bash
python training/train_progressive.py \
    --config config/quick_2stage.json \
    --checkpoint_dir checkpoints/dual_view \
    --resume_from checkpoints/dual_view/stage1_low_best.pt
```

---

## Expected Timings (A100 GPU)

| Stage | Resolution | Batch Size | Time/Epoch | Total Training |
|-------|-----------|------------|------------|----------------|
| Stage 1 | 64³ | 8 | 5 min | 4 hours (50 epochs) |
| Stage 2 | 128³ | 4 | 15 min | 12 hours (50 epochs) |
| Stage 3 | 256³ | 2 | 45 min | 36 hours (50 epochs) |

**Total for full 3-stage:** ~52 hours (~2 days)

With gradient checkpointing:
- Stage 3 batch size: 2 → 4 (double throughput)
- Total time: ~40 hours

---

## Memory Requirements

| Configuration | GPU Memory | Recommended GPU |
|--------------|-----------|-----------------|
| Stage 1 (batch=8) | 12 GB | RTX 3080, RTX 4070 |
| Stage 2 (batch=4) | 18 GB | RTX 3090, RTX 4080 |
| Stage 3 (batch=2) | 24 GB | RTX 4090, A5000 |
| Stage 3 (batch=4, checkpointing) | 24 GB | A100-40GB |

---

## Final Checklist Before Training

- [ ] Run `test_all_fixes.py` → All 6 tests pass
- [ ] Data loaded correctly → Check shapes match expected
- [ ] Config file validated → JSON syntax correct
- [ ] W&B initialized → Can see project dashboard
- [ ] Checkpoint directory created → Has write permissions
- [ ] GPU memory sufficient → Check with `nvidia-smi`
- [ ] Decide: single-view or dual-view? (dual recommended)

**Ready to train!** 🚀
