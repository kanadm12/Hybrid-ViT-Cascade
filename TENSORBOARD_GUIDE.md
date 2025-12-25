# TensorBoard Logging Guide

## Overview
The training script now includes comprehensive TensorBoard logging alongside W&B logging.

## Starting TensorBoard

### On RunPod
```bash
# In a separate terminal/tmux session
tensorboard --logdir /workspace/hybrid_vit_cascade/checkpoints/tensorboard --port 6006 --bind_all
```

Then access via your RunPod's exposed ports (see RunPod dashboard for the URL).

### Locally
```bash
tensorboard --logdir checkpoints/tensorboard --port 6006
```

Then open http://localhost:6006 in your browser.

## What is Logged

### Per-Step Metrics (every 50 steps by default)
- `{stage_name}/train_loss_step` - Training loss per batch
- `{stage_name}/train_diffusion_step` - Diffusion loss component per batch
- `{stage_name}/train_physics_step` - Physics loss component per batch

### Per-Epoch Metrics
- `{stage_name}/train_loss` - Average training loss for epoch
- `{stage_name}/train_diffusion` - Average diffusion loss for epoch
- `{stage_name}/train_physics` - Average physics loss for epoch
- `{stage_name}/val_loss` - Validation loss
- `{stage_name}/val_diffusion` - Validation diffusion loss
- `{stage_name}/val_physics` - Validation physics loss
- `{stage_name}/epoch_time` - Time taken for epoch
- `{stage_name}/learning_rate` - Current learning rate

## Log Location
TensorBoard logs are saved to: `<checkpoint_dir>/tensorboard/`

Default: `checkpoints/tensorboard/`

## Viewing Multiple Runs
You can compare multiple training runs by keeping them in separate subdirectories:
```bash
tensorboard --logdir checkpoints/ --port 6006
```

## Tips
1. **Monitor Training Progress**: Check loss curves to ensure convergence
2. **Learning Rate**: Track learning rate adjustments from the scheduler
3. **Compare Stages**: Compare `stage_64`, `stage_128`, `stage_256` metrics
4. **Epoch Time**: Monitor training speed and detect slowdowns
5. **Loss Components**: Track diffusion vs physics loss balance

## Model Checkpointing
Checkpoints are saved with complete training state:
- **Best checkpoint**: `{stage_name}_best.pt` (lowest validation loss)
- **Periodic checkpoints**: `{stage_name}_epoch_{N}.pt` (every 5 epochs by default)
- **Final checkpoint**: `{stage_name}_final.pt` (end of stage)

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Mixed precision scaler state
- Epoch number
- Validation loss
- Full config

## Resume Training
```python
# Load checkpoint
checkpoint = torch.load('checkpoints/stage_64_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scaler.load_state_dict(checkpoint['scaler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```
