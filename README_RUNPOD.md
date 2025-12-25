# Hybrid-ViT Cascade: Complete RunPod Training Setup

## 📋 Overview

This setup enables training the Hybrid-ViT Cascade architecture on RunPod with your patient DRR data. Key features include:

- ✅ **Automatic DRR-CT alignment validation** before training
- ✅ **Multi-resolution progressive training** (64³ → 128³ → 256³)
- ✅ **Memory-efficient** with mixed precision and gradient accumulation
- ✅ **Flexible data loading** supporting multiple file formats
- ✅ **Complete automation** with setup scripts and verification tools

---

## 🚀 Quick Start (3 Steps)

### On RunPod:

```bash
# 1. Navigate to project
cd /workspace/x2ctpa/hybrid_vit_cascade

# 2. Run automated setup (installs deps, verifies data, starts training)
chmod +x setup_and_train_runpod.sh
./setup_and_train_runpod.sh

# 3. Monitor in another terminal
watch -n 1 nvidia-smi
```

That's it! The script will guide you through the rest.

---

## 📁 Data Requirements

Your data should be organized as:

```
/workspace/drr_patient_data/
├── patient_001/
│   ├── drr_frontal.png      # or .npy, .jpg
│   ├── drr_lateral.png      # or .npy, .jpg
│   └── ct_volume.nii.gz     # or .nii, .npy
├── patient_002/
│   ├── drr_frontal.png
│   ├── drr_lateral.png
│   └── ct_volume.nii.gz
└── ...
```

**Supported formats:**
- DRRs: `.png`, `.jpg`, `.jpeg`, `.npy`
- CT volumes: `.nii.gz`, `.nii`, `.npy`

---

## 🔍 Before RunPod: Test Locally

Before uploading to RunPod, verify your setup:

```bash
# On your local machine
cd hybrid_vit_cascade

# Install dependencies
pip install -r requirements.txt

# Test setup
python test_setup.py
```

This checks:
- All dependencies are installed
- Project files are present
- Modules can be imported
- Model can be created

---

## 📤 Uploading to RunPod

### Option 1: Via SCP

```bash
# From your local machine
scp -r -P <port> hybrid_vit_cascade root@<pod-ip>:/workspace/x2ctpa/
scp -r -P <port> drr_patient_data root@<pod-ip>:/workspace/
```

### Option 2: Via Git

```bash
# On RunPod
cd /workspace
git clone <your-repo-url> x2ctpa
```

### Option 3: Via RunPod Web Interface

1. Open RunPod file browser
2. Upload `hybrid_vit_cascade` folder
3. Upload `drr_patient_data` folder

---

## 🎯 Training Workflow

### Step 1: Verify Data

```bash
cd /workspace/x2ctpa/hybrid_vit_cascade
python verify_data.py --data_path /workspace/drr_patient_data
```

**Expected output:**
```
✓ Data path exists
✓ Found 150 patient folders
✓ Data loading: OK

DRR-CT Alignment Validation Report
Total validated: 120
Passed: 115
Failed: 5
Pass rate: 95.83%
Average error: 0.0234

✓ All checks passed! Ready for training.
```

### Step 2: Start Training

```bash
# Basic training
python training/train_runpod.py --config config/runpod_config.json

# With W&B logging
python training/train_runpod.py \
    --config config/runpod_config.json \
    --wandb \
    --wandb_project my-xray2ct \
    --wandb_run_name runpod_training_v1
```

### Step 3: Monitor Progress

```bash
# GPU usage
watch -n 1 nvidia-smi

# Check checkpoints
ls -lh /workspace/checkpoints/hybrid_vit_cascade/

# View logs (if redirected)
tail -f training.log
```

---

## ⚙️ Configuration

Edit [config/runpod_config.json](config/runpod_config.json) to customize:

### Adjust for Your GPU

| GPU | Batch Size | Gradient Accum | Total Effective Batch |
|-----|-----------|----------------|---------------------|
| RTX 3090 24GB | 1 | 8 | 8 |
| RTX 4090 24GB | 2 | 4 | 8 |
| A100 40GB | 2-4 | 4 | 8-16 |
| A100 80GB | 4-8 | 2 | 8-16 |

```json
{
  "training": {
    "batch_size": 2,                    // ← Adjust here
    "gradient_accumulation_steps": 4,   // ← And here
    "mixed_precision": true,
    "num_epochs_per_stage": 50
  }
}
```

### Data Settings

```json
{
  "data": {
    "dataset_path": "/workspace/drr_patient_data",
    "validate_alignment": true,  // DRR-CT alignment check
    "augmentation": true,        // Random flips, intensity
    "target_xray_size": 512,
    "target_volume_size": [256, 256, 256]
  }
}
```

---

## 🎓 Understanding the Training

### Progressive Training Stages

1. **Stage 1 (64³)**: Learn basic structure
   - Fast: ~1-2 hours on A100
   - Establishes coarse CT volume

2. **Stage 2 (128³)**: Refine details
   - Medium: ~2-4 hours on A100
   - Uses Stage 1 output as prior

3. **Stage 3 (256³)**: Full resolution
   - Detailed: ~4-8 hours on A100
   - Final high-quality output

**Total: ~7-14 hours on A100**

### DRR-CT Alignment Validation

For each patient, the system:

1. Generates synthetic DRRs from the CT volume
   - Frontal: Max intensity projection along depth
   - Lateral: Max intensity projection along width

2. Compares synthetic DRRs with input DRRs using MSE

3. Reports alignment quality:
   - ✓ Pass: Error < threshold (default 0.5)
   - ✗ Fail: Error ≥ threshold

**Benefits:**
- Catches preprocessing errors early
- Ensures data quality before long training runs
- Provides quantitative metrics

---

## 📊 What You'll See During Training

### Initial Output
```
================================================================================
RUNPOD ENVIRONMENT VERIFICATION
================================================================================
PyTorch version: 2.0.1
CUDA available: True
GPU count: 1
  GPU 0: NVIDIA A100-SXM4-40GB

✓ Data directory found: /workspace/drr_patient_data
  Found 150 patient folders

DRR-CT Alignment Validation Report
Total validated: 120
Passed: 115
Failed: 5
Pass rate: 95.83%
```

### Training Progress
```
================================================================================
TRAINING STAGE 1/3: stage1
Volume size: [64, 64, 64]
================================================================================

Epoch 1/50 [Train]: 100%|██████| 60/60 [02:15<00:00]
  loss: 0.234  diff: 0.187  phys: 0.047

Epoch 1/50 [Val]: 100%|██████| 8/8 [00:15<00:00]

Epoch 1/50 (time: 150.23s):
  Train - Total: 0.234567, Diff: 0.187234, Phys: 0.047333
  Val   - Total: 0.198765, Diff: 0.154321, Phys: 0.044444
  ✓ Saved best checkpoint: stage1_best.pt
```

---

## 💾 Checkpoints

Saved to `/workspace/checkpoints/hybrid_vit_cascade/`:

```
stage1_best.pt      # Best validation loss
stage1_final.pt     # Final Stage 1
stage1_epoch_5.pt   # Periodic checkpoint
stage2_best.pt
stage2_final.pt
stage3_best.pt
stage3_final.pt
```

Each checkpoint contains:
```python
{
    'epoch': epoch_number,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'val_loss': validation_loss,
    'stage_name': 'stage1',
    'config': full_config_dict
}
```

---

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Edit config/runpod_config.json
"batch_size": 1,
"gradient_accumulation_steps": 8
```

### Issue: Data Not Found

```bash
# Find your data
find /workspace -name "drr_patient_data" -type d

# Update config
nano config/runpod_config.json
# Change "dataset_path" to actual location
```

### Issue: Low Alignment Pass Rate

```bash
# Option 1: Check a few patients manually
python verify_data.py --num_samples 10

# Option 2: Disable validation temporarily
# Edit config/runpod_config.json
"validate_alignment": false

# Option 3: Adjust threshold in utils/dataset.py (line ~230)
alignment_threshold = 1.0  # More lenient
```

### Issue: Slow Data Loading

```bash
# Reduce workers
# Edit config/runpod_config.json
"num_workers": 2  # or 0
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| [RUNPOD_QUICKSTART.md](RUNPOD_QUICKSTART.md) | Quick reference and commands |
| [RUNPOD_TRAINING_GUIDE.md](RUNPOD_TRAINING_GUIDE.md) | Complete step-by-step guide |
| [RUNPOD_SETUP_SUMMARY.md](RUNPOD_SETUP_SUMMARY.md) | What was set up and why |
| [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md) | Model architecture details |
| [QUICKSTART.md](QUICKSTART.md) | General quickstart |

---

## 🛠️ Key Files Created

### Dataset and Loading
- `utils/dataset.py`: PatientDRRDataset with alignment validation
- `utils/__init__.py`: Utils module

### Training
- `training/train_runpod.py`: RunPod-optimized training script
- `training/train_progressive.py`: Updated with new dataset

### Configuration
- `config/runpod_config.json`: RunPod-specific config

### Tools
- `verify_data.py`: Data verification tool
- `test_setup.py`: Local setup testing
- `setup_and_train_runpod.sh`: Automated setup script

---

## 🎯 Expected Results

After successful training:

✅ 3 trained stage models (64³, 128³, 256³)  
✅ Checkpoints at each stage  
✅ Training curves (W&B or local logs)  
✅ Alignment validation report  
✅ Ready for inference on new data  

---

## 🔄 Resume Training

If training is interrupted:

```bash
python training/train_runpod.py \
    --config config/runpod_config.json \
    --resume_from /workspace/checkpoints/hybrid_vit_cascade/stage2_best.pt \
    --wandb
```

---

## 📈 Next Steps After Training

1. **Evaluate**: Test on held-out test set
2. **Inference**: Generate CT volumes from new DRR pairs
3. **Visualize**: Create 3D renderings of outputs
4. **Fine-tune**: Adjust hyperparameters if needed
5. **Deploy**: Use for clinical applications

---

## 💡 Tips for Best Results

1. **Data Quality**: Ensure DRRs are properly aligned before training
2. **GPU Memory**: Start with small batch size, increase if stable
3. **W&B Logging**: Highly recommended for tracking experiments
4. **Checkpointing**: Keep multiple checkpoints for comparison
5. **Validation**: Monitor validation loss to detect overfitting

---

## 🆘 Getting Help

1. Check [RUNPOD_TRAINING_GUIDE.md](RUNPOD_TRAINING_GUIDE.md) troubleshooting section
2. Run `python verify_data.py` to diagnose data issues
3. Check GPU memory: `nvidia-smi`
4. Review error messages in terminal output
5. Examine alignment report for data quality issues

---

## ✅ Pre-Flight Checklist

Before starting training on RunPod:

- [ ] RunPod GPU pod launched (A100 recommended)
- [ ] Data uploaded to `/workspace/drr_patient_data`
- [ ] Code uploaded to `/workspace/x2ctpa/hybrid_vit_cascade`
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data verified (`python verify_data.py`)
- [ ] Config adjusted for GPU memory
- [ ] W&B account ready (optional)
- [ ] Sufficient time allocated for training

---

## 🚀 Ready to Train!

Everything is set up for training on RunPod. Start with:

```bash
cd /workspace/x2ctpa/hybrid_vit_cascade
./setup_and_train_runpod.sh
```

Good luck with your training! 🎉
