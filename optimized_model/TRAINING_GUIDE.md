# Quick Start Guide for Optimized Model Training

## What is this?

This trains the **optimized direct regression model** with all 6 improvements:

1. **Cascaded Group Attention (CGA)** - Efficient attention mechanism
2. **CNN + ViT Hybrid** - Best of both worlds
3. **Learnable Depth Priors** - Helps with 2D→3D reconstruction
4. **Sandwich Layout** - Better gradient flow
5. **Multi-Scale Attention** - Captures details at multiple resolutions
6. **Hierarchical Conditioning** - Stronger X-ray conditioning

## Model Stats

- **Parameters**: ~180M (vs 15M direct model, 353M original)
- **Training Time**: ~45 minutes for 15 epochs on A100
- **Target Performance**: 25-28 dB PSNR (vs 12.99 dB from direct model)

## How to Run

### On RunPod:

```bash
cd /workspace/Hybrid-ViT-Cascade/optimized_model

# Option 1: Use launch script (recommended)
chmod +x run_optimized_training.sh
./run_optimized_training.sh

# Option 2: Run directly
python3 train_optimized.py --config config_optimized.json
```

### On Local Machine:

```bash
cd optimized_model
python train_optimized.py --config config_optimized.json
```

## Configuration

Edit `config_optimized.json` to adjust:

```json
{
  "training": {
    "num_epochs": 15,           // Increase for better results
    "batch_size": 8,            // Reduce if OOM
    "learning_rate": 1e-4
  },
  "model": {
    "voxel_dim": 256,           // Model capacity
    "num_attn_blocks": 2,       // Number of attention blocks
    "num_ffn_blocks": 4         // Number of FFN blocks
  }
}
```

## Outputs

Training produces:

1. **checkpoints_optimized/best_model.pt** - Best model checkpoint
2. **checkpoints_optimized/training_log.json** - Metrics history
3. **feature_visualizations/** - Feature maps every epoch (15 subplots)

## Monitoring

Watch training progress:

```bash
# In another terminal
tail -f checkpoints_optimized/training_log.json

# Or watch GPU usage
watch -n 1 nvidia-smi
```

## Expected Results

Based on optimizations:

- **Epoch 1**: ~15-18 dB PSNR
- **Epoch 5**: ~20-22 dB PSNR
- **Epoch 10**: ~23-25 dB PSNR
- **Epoch 15**: ~25-28 dB PSNR (target)

## Troubleshooting

### Out of Memory?

Reduce batch size in `config_optimized.json`:
```json
"batch_size": 4  // or 2 if still OOM
```

### Training too slow?

Enable model compilation (PyTorch 2.0+):
```json
"optimization": {
  "compile_model": true
}
```

### Low PSNR?

- Check if direct model reached >15 dB (prerequisite)
- Increase `num_epochs` to 20-25
- Verify data loading is correct

## Next Steps

After training completes:

1. **Run inference** to visualize results
2. **Compare to direct model** (12.99 dB baseline)
3. **If >25 dB**: Add diffusion for final push to 30-35 dB
4. **If <20 dB**: May need architectural changes

## Memory Requirements

- **GPU Memory**: ~20-24 GB (fits on A100 40GB)
- **RAM**: ~16 GB
- **Disk**: ~5 GB for checkpoints + visualizations
