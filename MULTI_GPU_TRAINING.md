# Multi-GPU Training Guide for 4x A100 (80GB)

## Quick Start

### Option 1: Accelerate (Recommended - Easiest)

```bash
# 1. Install accelerate
pip install accelerate

# 2. Configure (optional - script has defaults)
accelerate config

# 3. Train
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=fp16 training/train_accelerate.py --config config/multi_view_config.json --wandb
```

**Or use the convenience script:**

Windows:
```cmd
train_4gpu.bat --config config/multi_view_config.json --method accelerate --wandb
```

Linux/Mac:
```bash
bash train_4gpu.sh --config config/multi_view_config.json --method accelerate --wandb
```

### Option 2: PyTorch DDP with torchrun

```bash
torchrun --nproc_per_node=4 training/train_distributed.py --config config/multi_view_config.json --wandb
```

**Or use the convenience script:**
```cmd
train_4gpu.bat --config config/multi_view_config.json --method torchrun --wandb
```

## Performance Optimization for A100s

### 1. Batch Size Configuration

With 4x A100 (80GB each), you can use larger batches:

```json
{
  "training": {
    "batch_size": 8,  // Per GPU (effective batch: 8x4=32)
    "num_timesteps": 1000,
    "v_parameterization": true,
    "learning_rate": 1e-4,
    "num_epochs_per_stage": 100
  }
}
```

### 2. Recommended Batch Sizes

| Stage | Volume Size | Batch/GPU | Total Batch | Memory/GPU |
|-------|-------------|-----------|-------------|------------|
| Stage 1 | 64³ | 16 | 64 | ~15GB |
| Stage 2 | 128³ | 8 | 32 | ~40GB |
| Stage 3 | 256³ | 4 | 16 | ~70GB |

### 3. Memory-Saving Techniques

If you hit OOM errors, enable these in your config:

```json
{
  "training": {
    "gradient_checkpointing": true,
    "mixed_precision": "fp16",
    "accumulation_steps": 2
  }
}
```

## Training Scripts Comparison

### train_distributed.py (torchrun)
- ✅ Native PyTorch DDP
- ✅ More control over distributed setup
- ✅ Works out-of-the-box
- ⚠️ More boilerplate code

### train_accelerate.py (Accelerate)
- ✅ Simpler code
- ✅ Automatic mixed precision
- ✅ Easier to extend
- ✅ Better for multi-node setups
- ⚠️ Requires accelerate library

## Multi-Node Training (8 GPUs across 2 nodes)

### Using torchrun:

**Node 0 (master):**
```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    training/train_distributed.py --config config/multi_view_config.json
```

**Node 1:**
```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    training/train_distributed.py --config config/multi_view_config.json
```

### Using Accelerate:

1. Configure on both nodes:
```bash
accelerate config
```

2. Launch on both nodes with same config:
```bash
accelerate launch training/train_accelerate.py --config config/multi_view_config.json
```

## Monitoring

### 1. GPU Utilization
```bash
watch -n 1 nvidia-smi
```

### 2. Weights & Biases
```bash
# Add --wandb flag to any training command
accelerate launch --multi_gpu --num_processes=4 training/train_accelerate.py --config config/multi_view_config.json --wandb
```

### 3. TensorBoard (alternative)
```bash
tensorboard --logdir=checkpoints_4gpu/logs
```

## Expected Training Times (4x A100)

| Stage | Epochs | Time per Epoch | Total Time |
|-------|--------|---------------|------------|
| Stage 1 (64³) | 100 | ~2 min | ~3.5 hours |
| Stage 2 (128³) | 100 | ~8 min | ~13 hours |
| Stage 3 (256³) | 100 | ~25 min | ~42 hours |
| **Total** | 300 | - | **~58 hours (~2.4 days)** |

*Assuming 1000 samples, effective batch size of 32*

## Troubleshooting

### OOM Errors
1. Reduce `batch_size` in config
2. Enable gradient checkpointing
3. Use `--mixed_precision=fp16`

### Slow Training
1. Check `nvidia-smi` for GPU utilization (<90% means bottleneck)
2. Increase `num_workers` in DataLoader (4-8 per GPU)
3. Enable `pin_memory=True` (already default)

### Distributed Sync Issues
1. Ensure all GPUs can communicate
2. Check firewall settings for multi-node
3. Verify same PyTorch version on all nodes

### NaN Losses
1. Reduce learning rate
2. Enable gradient clipping (already at 1.0)
3. Check data normalization

## Best Practices

1. **Start Small**: Test with 1 GPU first, then scale
2. **Save Checkpoints**: Training saves best model per stage
3. **Monitor Memory**: Use `torch.cuda.max_memory_allocated()`
4. **Use Mixed Precision**: Speeds up training by ~2x on A100
5. **Gradient Accumulation**: If batch too small, accumulate over steps

## Example Commands

### Development (single GPU, fast iteration)
```bash
python training/train_progressive.py --config config/single_view_config.json
```

### Production (4 GPUs, full training)
```bash
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=fp16 \
    training/train_accelerate.py \
    --config config/multi_view_config.json \
    --checkpoint_dir checkpoints_production \
    --wandb \
    --wandb_project hybrid-vit-cascade-production
```

### Resume from Checkpoint
```bash
accelerate launch --multi_gpu --num_processes=4 \
    training/train_accelerate.py \
    --config config/multi_view_config.json \
    --resume_from checkpoints_4gpu/stage2_best.pt
```

## Configuration Tips for 4x A100

Edit your config file:

```json
{
  "xray_config": {
    "img_size": 512,
    "num_views": 2
  },
  "stage_configs": [
    {
      "name": "stage1",
      "volume_size": [64, 64, 64],
      "voxel_dim": 384,      // Can increase with more memory
      "vit_depth": 6,         // Can increase to 8
      "num_heads": 6
    }
  ],
  "training": {
    "batch_size": 8,          // 8 per GPU = 32 total
    "learning_rate": 1e-4,
    "num_epochs_per_stage": 100,
    "num_timesteps": 1000,
    "v_parameterization": true
  },
  "data": {
    "dataset_path": "./data/drr_patient_data",
    "augmentation": true,      // Enable for better generalization
    "cache_in_memory": false,  // True if dataset is small (<10GB)
    "num_workers": 4           // 4 per GPU
  }
}
```

## Performance Metrics to Track

- **Training Loss**: Should decrease smoothly
- **Validation Loss**: Should track training loss
- **Physics Loss**: Should be <0.1 by end of training
- **GPU Memory**: Should be 70-80% utilized
- **Throughput**: Samples/second (aim for >10/sec per GPU)
- **Learning Rate**: Monitor for proper scheduling

Happy Training! 🚀
