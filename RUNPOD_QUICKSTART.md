# Quick Start: RunPod Training

This README provides quick commands to get started with training on RunPod.

## One-Line Setup and Train

```bash
cd /workspace/x2ctpa/hybrid_vit_cascade && chmod +x setup_and_train_runpod.sh && ./setup_and_train_runpod.sh
```

This script will:
1. ✓ Verify GPU and environment
2. ✓ Install all dependencies
3. ✓ Locate your data directory
4. ✓ Configure paths automatically
5. ✓ Optionally setup W&B logging
6. ✓ Start training

## Manual Commands

If you prefer manual setup:

```bash
# 1. Navigate to project
cd /workspace/x2ctpa/hybrid_vit_cascade

# 2. Install dependencies
pip install -r requirements.txt nibabel scipy Pillow wandb

# 3. Verify data (should be at /workspace/drr_patient_data)
ls -la /workspace/drr_patient_data

# 4. Start training
python training/train_runpod.py --config config/runpod_config.json --wandb
```

## Monitor Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# View logs (if using tmux/screen)
tail -f training.log

# Check checkpoints
ls -lh /workspace/checkpoints/hybrid_vit_cascade/
```

## Data Requirements

Your data should be organized as:

```
/workspace/drr_patient_data/
├── patient_001/
│   ├── drr_frontal.png      # Frontal X-ray/DRR
│   ├── drr_lateral.png      # Lateral X-ray/DRR
│   └── ct_volume.nii.gz     # CT volume
├── patient_002/
│   ├── drr_frontal.png
│   ├── drr_lateral.png
│   └── ct_volume.nii.gz
└── ...
```

Supported formats:
- DRRs: `.png`, `.jpg`, `.npy`
- CT volumes: `.nii.gz`, `.nii`, `.npy`

## Key Features

### 1. Automatic DRR-CT Alignment Validation
The training script validates that your DRR images are properly aligned with the CT volumes by:
- Generating synthetic DRRs from the CT volume
- Comparing with input DRRs using MSE
- Reporting alignment statistics before training

### 2. Multi-Resolution Progressive Training
- Stage 1: 64³ resolution (fast, ~2 hours)
- Stage 2: 128³ resolution (medium, ~3 hours)
- Stage 3: 256³ resolution (full detail, ~6 hours)

### 3. Memory-Efficient Training
- Mixed precision (FP16)
- Gradient accumulation
- Optimized batch sizes per GPU

## Configuration

Edit `config/runpod_config.json` to adjust:

```json
{
  "training": {
    "batch_size": 2,           // Reduce if OOM
    "num_epochs_per_stage": 50,
    "mixed_precision": true,
    "gradient_accumulation_steps": 4
  },
  "data": {
    "validate_alignment": true,  // Alignment validation
    "augmentation": true         // Data augmentation
  }
}
```

## Troubleshooting

### Out of Memory (OOM)
```json
// Reduce batch size
"batch_size": 1,
"gradient_accumulation_steps": 8
```

### Data Not Found
```bash
# Check actual data location
find /workspace -name "drr_patient_data" -type d

# Update config manually
nano config/runpod_config.json
# Change "dataset_path" to your actual path
```

### Slow Training
```json
// Disable alignment validation if dataset is pre-validated
"validate_alignment": false,

// Reduce workers if I/O bound
"num_workers": 2
```

## Expected Training Time

On NVIDIA A100 40GB:
- **Stage 1 (64³)**: ~1-2 hours
- **Stage 2 (128³)**: ~2-4 hours  
- **Stage 3 (256³)**: ~4-8 hours
- **Total**: ~7-14 hours

On RTX 3090/4090 24GB:
- Add 30-50% more time

## Output

After training, you'll have:
```
/workspace/checkpoints/hybrid_vit_cascade/
├── stage1_best.pt    # Best Stage 1 model
├── stage1_final.pt   # Final Stage 1 model
├── stage2_best.pt
├── stage2_final.pt
├── stage3_best.pt
└── stage3_final.pt
```

## Need Help?

See full documentation:
- [RUNPOD_TRAINING_GUIDE.md](RUNPOD_TRAINING_GUIDE.md) - Complete training guide
- [QUICKSTART.md](QUICKSTART.md) - General quickstart
- [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md) - Model architecture

## Support

For issues:
1. Check [RUNPOD_TRAINING_GUIDE.md](RUNPOD_TRAINING_GUIDE.md) troubleshooting section
2. Verify data structure matches expected format
3. Check GPU memory with `nvidia-smi`
4. Review training logs for specific errors
