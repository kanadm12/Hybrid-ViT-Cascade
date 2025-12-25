# Training Monitoring Summary

## ✅ Implemented Features

### 1. TensorBoard Logging
**Status**: ✅ Fully Implemented

**Logged Metrics**:
- Per-step losses (every 50 steps): train loss, diffusion loss, physics loss
- Per-epoch metrics: train/val losses, loss components, learning rate, epoch time
- Separate tracking for each stage (stage_64, stage_128, stage_256)

**Usage**:
```bash
# Start TensorBoard
tensorboard --logdir checkpoints/tensorboard --port 6006 --bind_all

# Training automatically logs to: checkpoints/tensorboard/
```

See [TENSORBOARD_GUIDE.md](TENSORBOARD_GUIDE.md) for details.

---

### 2. Weights & Biases (W&B) Logging
**Status**: ✅ Fully Implemented

**Logged Metrics**:
- All TensorBoard metrics plus W&B-specific features
- Automatic config tracking
- Run comparison and hyperparameter sweeps

**Usage**:
```bash
# Enable W&B logging
python training/train_runpod.py --config config/runpod_config.json --wandb
```

---

### 3. Model Checkpointing
**Status**: ✅ Fully Implemented

**Checkpoint Types**:
1. **Best Validation Checkpoint**: `{stage_name}_best.pt`
   - Saved when validation loss improves
   - Contains: model, optimizer, scaler, epoch, val_loss, config

2. **Periodic Checkpoints**: `{stage_name}_epoch_{N}.pt`
   - Saved every 5 epochs (configurable via `save_every_n_epochs`)
   - Same contents as best checkpoint

3. **Final Stage Checkpoint**: `{stage_name}_final.pt`
   - Saved at end of each stage
   - Model state dict only

**Checkpoint Contents**:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scaler_state_dict': dict,  # Mixed precision
    'val_loss': float,
    'stage_name': str,
    'config': dict  # Full training config
}
```

**Location**: `checkpoints/` (default) or custom via `--checkpoint_dir`

---

### 4. Console Logging
**Status**: ✅ Fully Implemented

**Output**:
- Real-time progress bars with loss values
- Per-epoch summaries (train/val losses, timing)
- Checkpoint save notifications
- Learning rate adjustments

---

## Training Configuration

**Current Settings** (config/runpod_config.json):
- **Epochs per stage**: 30
- **Batch size**: 2
- **Mixed precision**: Enabled (FP16)
- **Gradient accumulation**: 4 steps
- **Periodic checkpoint frequency**: Every 5 epochs
- **Log step frequency**: Every 50 steps
- **Data validation**: Alignment checking enabled

---

## Quick Start

```bash
# 1. Start training (on RunPod)
python training/train_runpod.py --config config/runpod_config.json --wandb

# 2. Monitor with TensorBoard (separate terminal)
tensorboard --logdir checkpoints/tensorboard --port 6006 --bind_all

# 3. Access TensorBoard via RunPod exposed port
# Check RunPod dashboard for the URL (port 6006)
```

---

## Verification Checklist

Before starting training:
- ✅ Data verified: 100 patients with correct file naming
- ✅ Alignment validated: 0.09 average error (excellent)
- ✅ TensorBoard logging implemented
- ✅ W&B logging available
- ✅ Comprehensive checkpointing (best + periodic + final)
- ✅ Config set to 30 epochs per stage
- ✅ All code syntax verified

**Status**: Ready to train! 🚀

---

## Expected Output Structure

```
checkpoints/
├── tensorboard/              # TensorBoard logs
│   └── events.out.tfevents.*
├── stage_64_best.pt         # Best checkpoint for stage 1
├── stage_64_epoch_5.pt      # Periodic checkpoint
├── stage_64_epoch_10.pt
├── ...
├── stage_64_final.pt        # Final checkpoint for stage 1
├── stage_128_best.pt        # Stage 2 checkpoints...
├── stage_128_final.pt
├── stage_256_best.pt        # Stage 3 checkpoints...
└── stage_256_final.pt
```

---

## Monitoring Best Practices

1. **Watch Training Loss**: Should decrease steadily
2. **Check Val/Train Gap**: Large gap indicates overfitting
3. **Monitor Learning Rate**: Should decrease when validation plateaus
4. **Track Loss Components**: Diffusion vs Physics balance
5. **Verify Checkpoints**: Check file sizes and save times
6. **Compare Stages**: Each stage should improve on previous

---

## Resuming Training

If training is interrupted:

```python
# Load from best checkpoint
checkpoint = torch.load('checkpoints/stage_128_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scaler.load_state_dict(checkpoint['scaler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

Currently, automatic resume is not implemented. You would need to modify the training script or start the interrupted stage again (it will reuse the previous stages' weights).

---

## Estimated Training Time (RTX 5090)

- **Stage 1 (64³)**: ~30-45 minutes (30 epochs)
- **Stage 2 (128³)**: ~1.5-2 hours (30 epochs)
- **Stage 3 (256³)**: ~3-4 hours (30 epochs)

**Total**: ~5-7 hours for full progressive training on 100 patients

---

## Troubleshooting

### TensorBoard not showing data
```bash
# Check log directory
ls checkpoints/tensorboard/

# Restart TensorBoard with correct path
tensorboard --logdir checkpoints/tensorboard --port 6006 --bind_all
```

### Checkpoints not saving
- Check disk space: `df -h`
- Verify write permissions: `ls -la checkpoints/`
- Check console output for save notifications

### W&B not logging
```bash
# Login to W&B
wandb login

# Verify W&B is enabled
python training/train_runpod.py --config config/runpod_config.json --wandb
```
