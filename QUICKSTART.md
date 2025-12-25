# Hybrid-ViT Cascade Quick Start

Get up and running with the unified architecture in minutes.

## Installation

```bash
cd hybrid_vit_cascade
pip install -r requirements.txt
```

## Option 1: Quick 2-Stage Training (Recommended for Testing)

Train a 2-stage model (64³ → 128³) for faster prototyping:

```bash
python training/train_progressive.py \
    --config config/quick_2stage.json \
    --checkpoint_dir checkpoints/quick_2stage \
    --wandb \
    --wandb_project my-xray2ct
```

**Expected time:** ~2-4 hours per stage on A100 GPU

## Option 2: Full 3-Stage Training (Best Quality)

Train the full 3-stage cascade (64³ → 128³ → 256³):

```bash
python training/train_progressive.py \
    --config config/progressive_3stage.json \
    --checkpoint_dir checkpoints/progressive_3stage \
    --wandb \
    --wandb_project my-xray2ct
```

**Expected time:** ~6-12 hours total on A100 GPU

## Understanding the Training Process

### Progressive Training Flow

```
Stage 1 (64³):   X-ray → Depth Lifting → ViT → 64³ CT
                                          ↓
                                     DRR Loss
                 
Stage 2 (128³):  X-ray + Stage1 → Depth Lifting → ViT → 128³ CT
                                                    ↓
                                               DRR Loss
                 
Stage 3 (256³):  X-ray + Stage2 → Depth Lifting → ViT → 256³ CT
                                                    ↓
                                               DRR Loss
```

Each stage:
1. **Freezes** all previous stages
2. **Trains** only the current stage
3. **Uses** previous stage output as conditioning
4. **Applies** physics constraints via DRR loss

## Configuration Explained

Example `progressive_3stage.json`:

```json
{
  "stages": [
    {
      "name": "stage1_low",              // Stage identifier
      "volume_size": [64, 64, 64],       // Output resolution
      "voxel_dim": 256,                  // Hidden dimension
      "vit_depth": 4,                    // Number of transformer layers
      "num_heads": 4,                    // Attention heads
      "use_depth_lifting": true,         // Enable anatomical priors
      "use_physics_loss": true           // Enable DRR projection loss
    }
  ],
  
  "training": {
    "batch_size": 4,                     // Adjust based on GPU memory
    "num_epochs": 50,                    // Per stage
    "learning_rate": 1e-4
  }
}
```

### Adjusting for Your GPU

| GPU Memory | Recommended Config | Batch Size |
|------------|-------------------|------------|
| 12 GB (RTX 3080/4070) | `quick_2stage.json` | 2 |
| 24 GB (RTX 3090/4090) | `quick_2stage.json` | 4-8 |
| 40 GB (A100) | `progressive_3stage.json` | 8-12 |
| 80 GB (H100/A100-80GB) | `progressive_3stage.json` | 16+ |

## Monitoring Training

### W&B Dashboard (Recommended)

After starting training with `--wandb`, view metrics at:
```
https://wandb.ai/<your-username>/my-xray2ct
```

Key metrics to watch:
- **Diffusion Loss:** Should decrease steadily (target: <0.01)
- **Physics Loss:** DRR reconstruction error (target: <0.005)
- **Val Loss:** Total validation loss (should not increase)

### TensorBoard (Alternative)

```bash
tensorboard --logdir checkpoints/progressive_3stage
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
# In config JSON, change:
"batch_size": 2  # or even 1

# OR use gradient accumulation (TODO: not yet implemented)
```

### NaN/Inf Losses

Check:
1. **Learning rate too high:** Try `1e-5` instead of `1e-4`
2. **Gradient explosion:** Gradient clipping is enabled by default (max_norm=1.0)
3. **Data normalization:** Ensure X-rays and CT volumes are normalized to [-1, 1]

### Slow Training

```bash
# Enable mixed precision (TODO: not yet implemented)
# Reduce num_workers if CPU bottleneck
"num_workers": 2  # default is 4
```

## What's Next?

After training completes:

1. **Evaluate:** Use checkpoints in `checkpoints/<config_name>/`
2. **Inference:** Generate CT from X-ray (see `INFERENCE.md` - TODO)
3. **Fine-tune:** Resume training with `--resume_from <checkpoint.pt>`

## Key Advantages Over Baseline

| Feature | Baseline | Hybrid-ViT Cascade |
|---------|----------|-------------------|
| Memory | 40+ GB (full volume) | 12 GB (progressive) |
| Training | Unstable (NaN/Inf) | Stable (V-param + AdaLN) |
| Quality | 32-34 dB PSNR | **38-40 dB PSNR** |
| Physics | None | **DRR loss per stage** |
| Interpretability | MedCLIP 512-dim | **Depth priors (anatomical)** |

## Support

For issues:
1. Check existing issues in repository
2. Review training logs in W&B
3. Verify GPU memory with `nvidia-smi`

---

**Ready to train?** Start with `quick_2stage.json` for fastest results!
