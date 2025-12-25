# RunPod Training Guide for Hybrid-ViT Cascade

Complete guide for training the Hybrid-ViT Cascade architecture on RunPod with your DRR patient data.

## Prerequisites

1. RunPod account with GPU pod (recommended: A100 40GB or better)
2. Patient data organized in `/workspace/drr_patient_data` directory
3. Each patient folder should contain:
   - `drr_frontal.png` (or .npy)
   - `drr_lateral.png` (or .npy)
   - `ct_volume.nii.gz` (or .npy)

## Step 1: Pod Setup

### Launch RunPod Pod

1. Go to RunPod and create a new GPU pod
2. Recommended template: `PyTorch 2.0` or `RunPod PyTorch`
3. Minimum GPU: RTX 3090 or better
4. Recommended: A100 40GB or 80GB

### Connect via SSH or Web Terminal

```bash
# If using SSH
ssh root@<your-pod-ip> -p <port>
```

## Step 2: Data Setup

### Option A: Upload Data to RunPod

If your data is local, upload it to RunPod:

```bash
# On your local machine
scp -r -P <port> /path/to/drr_patient_data root@<your-pod-ip>:/workspace/
```

### Option B: Download from Cloud Storage

If your data is on cloud storage (Google Drive, Azure, AWS S3):

```bash
# For Google Drive (if using gdown)
pip install gdown
gdown --folder <google-drive-folder-id> -O /workspace/drr_patient_data

# For Azure Blob Storage
az storage blob download-batch --source <container> --destination /workspace/drr_patient_data

# For AWS S3
aws s3 sync s3://your-bucket/drr_patient_data /workspace/drr_patient_data
```

### Verify Data Structure

```bash
cd /workspace/drr_patient_data
ls -la

# Should see:
# patient_001/
# patient_002/
# ...

# Check one patient folder
ls -la patient_001/
# Should contain: drr_frontal.png, drr_lateral.png, ct_volume.nii.gz
```

## Step 3: Install Dependencies

```bash
# Navigate to project directory
cd /workspace/x2ctpa/hybrid_vit_cascade

# Install requirements
pip install -r requirements.txt

# Additional dependencies for RunPod
pip install nibabel scipy Pillow wandb
```

## Step 4: Configure Training

The default RunPod config is already set up at `config/runpod_config.json`.

### Key Configuration Parameters

```json
{
  "data": {
    "dataset_path": "/workspace/drr_patient_data",
    "train_split": 0.8,
    "val_split": 0.1,
    "validate_alignment": true,  // Validates DRR-CT alignment
    "augmentation": true,
    "target_xray_size": 512,
    "target_volume_size": [256, 256, 256]
  },
  "training": {
    "batch_size": 2,  // Adjust based on GPU memory
    "num_epochs_per_stage": 50,
    "mixed_precision": true,  // Enable for faster training
    "gradient_accumulation_steps": 4
  }
}
```

### Adjust Batch Size Based on GPU

| GPU Model | Recommended Batch Size |
|-----------|----------------------|
| RTX 3090 (24GB) | 1-2 |
| RTX 4090 (24GB) | 2 |
| A100 (40GB) | 2-4 |
| A100 (80GB) | 4-8 |

## Step 5: Start Training

### Basic Training (No W&B)

```bash
cd /workspace/x2ctpa/hybrid_vit_cascade

python training/train_runpod.py \
    --config config/runpod_config.json
```

### With Weights & Biases Logging

```bash
# First, login to W&B
wandb login <your-api-key>

# Start training with W&B
python training/train_runpod.py \
    --config config/runpod_config.json \
    --wandb \
    --wandb_project xray2ct-runpod \
    --wandb_run_name my_training_run
```

### Resume from Checkpoint

```bash
python training/train_runpod.py \
    --config config/runpod_config.json \
    --resume_from /workspace/checkpoints/hybrid_vit_cascade/stage1_best.pt \
    --wandb
```

## Step 6: Monitor Training

### View Training Progress

The script will display:
- Environment verification
- Data loading and alignment validation report
- Training/validation losses per epoch
- Checkpoint saving notifications

### Example Output

```
================================================================================
RUNPOD ENVIRONMENT VERIFICATION
================================================================================
PyTorch version: 2.0.1
CUDA available: True
CUDA version: 11.8
GPU count: 1
  GPU 0: NVIDIA A100-SXM4-40GB
    Memory: 40.00 GB

✓ Data directory found: /workspace/drr_patient_data
  Found 150 patient folders
  Sample patients: ['patient_001', 'patient_002', 'patient_003', 'patient_004', 'patient_005']
================================================================================

Found 150 valid patient datasets in /workspace/drr_patient_data

Dataset splits:
  Train: 120 samples
  Val:   15 samples
  Test:  15 samples

================================================================================
DRR-CT Alignment Validation Report
================================================================================
Total validated: 120
Passed: 115
Failed: 5
Pass rate: 95.83%
Average error: 0.0234
================================================================================
```

### Check GPU Usage

In another terminal:

```bash
watch -n 1 nvidia-smi
```

## Step 7: Understanding the Training Stages

The model trains progressively through 3 stages:

### Stage 1: 64³ Resolution
- Fast training (~1-2 hours on A100)
- Learns basic CT structure from DRRs
- Validates DRR-CT alignment

### Stage 2: 128³ Resolution
- Medium detail (~2-4 hours on A100)
- Refines predictions from Stage 1
- Uses Stage 1 output as prior

### Stage 3: 256³ Resolution
- Full detail (~4-8 hours on A100)
- Final high-resolution output
- Uses Stage 2 output as prior

**Total training time: ~7-14 hours on A100**

## Step 8: Checkpoints and Output

### Checkpoint Location

```bash
/workspace/checkpoints/hybrid_vit_cascade/
├── stage1_best.pt          # Best validation loss for Stage 1
├── stage1_final.pt         # Final Stage 1 checkpoint
├── stage1_epoch_5.pt       # Periodic checkpoint
├── stage2_best.pt
├── stage2_final.pt
└── stage3_best.pt
```

### What's Saved in Each Checkpoint

```python
checkpoint = {
    'epoch': epoch_number,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),  # For mixed precision
    'val_loss': validation_loss,
    'stage_name': 'stage1',
    'config': full_config_dict
}
```

## Troubleshooting

### 1. CUDA Out of Memory

```bash
# Reduce batch size in config
# Edit config/runpod_config.json
"batch_size": 1,
"gradient_accumulation_steps": 8  # Increase to maintain effective batch size
```

### 2. Data Not Found

```bash
# Check data path
ls -la /workspace/drr_patient_data

# Update config if data is elsewhere
# Edit config/runpod_config.json
"dataset_path": "/your/actual/path/to/drr_patient_data"
```

### 3. Alignment Validation Failures

If many patients fail alignment validation:

```bash
# Disable strict validation temporarily
# Edit config/runpod_config.json
"validate_alignment": false

# Or check individual patient data
python -c "
from utils.dataset import PatientDRRDataset
dataset = PatientDRRDataset('/workspace/drr_patient_data', validate_alignment=True)
data = dataset[0]
print(f'Patient: {data[\"patient_id\"]}, Aligned: {data[\"aligned\"]}')
"
```

### 4. Slow Data Loading

```bash
# Reduce num_workers if causing issues
# Edit config/runpod_config.json
"num_workers": 2  # or 0 for debugging

# Or enable caching for small datasets
"cache_in_memory": true  # Only if dataset fits in RAM
```

## Advanced Options

### Custom Data Preprocessing

Edit `utils/dataset.py` to customize:
- Image normalization ranges
- Alignment validation thresholds
- Augmentation strategies

### Multi-GPU Training

For multi-GPU pods:

```bash
# Use PyTorch DDP (Distributed Data Parallel)
python -m torch.distributed.launch \
    --nproc_per_node=<num_gpus> \
    training/train_runpod.py \
    --config config/runpod_config.json
```

## Quick Start Commands

### Complete Setup and Training

```bash
# 1. Navigate to project
cd /workspace/x2ctpa/hybrid_vit_cascade

# 2. Install dependencies
pip install -r requirements.txt nibabel scipy Pillow wandb

# 3. Verify data
ls -la /workspace/drr_patient_data

# 4. Start training
python training/train_runpod.py \
    --config config/runpod_config.json \
    --wandb \
    --wandb_project my-xray2ct

# 5. Monitor in another terminal
watch -n 1 nvidia-smi
```

## Expected Results

After successful training, you should have:
- ✓ 3 stage models trained progressively
- ✓ Checkpoints saved at each stage
- ✓ Validation loss decreasing over epochs
- ✓ DRR-CT alignment validation report
- ✓ W&B training curves (if enabled)

## Next Steps

After training:
1. Evaluate model on test set
2. Generate predictions for new DRR pairs
3. Visualize 3D CT reconstructions
4. Fine-tune hyperparameters if needed

## Support

For issues specific to:
- **Data loading**: Check `utils/dataset.py`
- **Model architecture**: Check `models/unified_model.py`
- **Training loop**: Check `training/train_runpod.py`
- **Configuration**: Check `config/runpod_config.json`
