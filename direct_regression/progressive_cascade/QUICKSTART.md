# Progressive Multi-Scale CT Reconstruction - Quick Start Guide

## 🚀 Quick Start (5 Minutes)

### 1. Navigate to Directory
```bash
cd progressive_cascade
```

### 2. Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. Test Model Architecture
```bash
python model_progressive.py
```
Expected output: Model shapes for each stage (64³, 128³, 256³)

### 4. Test Loss Functions
```bash
python loss_multiscale.py
```
Expected output: Loss values for all three stages

### 5. Configure Training
Edit `config_progressive.json`:
- Set `data.dataset_path` to your dataset location
- Adjust `data.max_patients` based on available data
- Modify batch sizes if you have memory constraints

### 6. Start Training

**Windows:**
```bash
launch_progressive.bat
```

**Linux/Mac:**
```bash
python train_progressive_4gpu.py
```

## 📊 Training Progress

The training will proceed in 3 stages:

### Stage 1 (64³) - ~6 hours on 4×A100
```
Training 64³ base reconstruction...
Epoch 1/50 | Loss: 0.234 | PSNR: 28.5 dB | SSIM: 0.87
...
✓ Stage 1 complete: checkpoints_progressive/stage1_best.pth
```

### Stage 2 (128³) - ~12 hours on 4×A100
```
Loading Stage 1 checkpoint...
Stage 1 frozen ✓
Training 128³ refinement...
Epoch 1/30 | Loss: 0.156 | PSNR: 32.3 dB | SSIM: 0.93
...
✓ Stage 2 complete: checkpoints_progressive/stage2_best.pth
```

### Stage 3 (256³) - ~24 hours on 4×A100
```
Loading Stages 1+2 checkpoint...
Stages 1+2 frozen ✓
Training 256³ refinement with gradient checkpointing...
Epoch 1/20 | Loss: 0.089 | PSNR: 35.8 dB | SSIM: 0.96
...
✓ Stage 3 complete: checkpoints_progressive/stage3_best.pth
```

## 🔍 Running Inference

### Single Sample
```bash
python inference_progressive.py \
  --checkpoint checkpoints_progressive/stage3_best.pth \
  --mode single \
  --sample-idx 0 \
  --save-nifti
```

Output:
- `outputs_progressive/comparison_sample0.png` - Visual comparison
- `outputs_progressive/sample0_stage1_64.nii.gz` - Stage 1 output
- `outputs_progressive/sample0_stage2_128.nii.gz` - Stage 2 output
- `outputs_progressive/sample0_stage3_256.nii.gz` - Stage 3 output

### Full Evaluation
```bash
python inference_progressive.py \
  --checkpoint checkpoints_progressive/stage3_best.pth \
  --mode evaluate \
  --num-samples 100
```

Output:
```
======================================================================
EVALUATION RESULTS
======================================================================

Stage           PSNR (dB)            SSIM                 L1 Error       
----------------------------------------------------------------------
Stage 1 (64³)   28.45 ± 2.31         0.8756 ± 0.0345      0.0521 ± 0.0089
Stage 2 (128³)  32.87 ± 2.15         0.9312 ± 0.0267      0.0342 ± 0.0067
Stage 3 (256³)  35.92 ± 1.98         0.9587 ± 0.0198      0.0219 ± 0.0045
======================================================================
```

## 🎛️ Configuration Presets

### Fast Prototyping (Half training time)
```json
{
  "training": {
    "stage1": {"num_epochs": 20},
    "stage2": {"num_epochs": 15},
    "stage3": {"num_epochs": 10}
  },
  "data": {"max_patients": 50}
}
```

### Memory Constrained (<32GB GPU)
```json
{
  "training": {
    "stage3": {
      "batch_size": 1,
      "voxel_dim": 192
    }
  }
}
```

### Maximum Quality (Longer training)
```json
{
  "training": {
    "stage1": {"num_epochs": 100},
    "stage2": {"num_epochs": 50},
    "stage3": {"num_epochs": 30}
  },
  "model": {"voxel_dim": 320}
}
```

## 📈 Expected Results

| Metric | Stage 1 | Stage 2 | Stage 3 | Target |
|--------|---------|---------|---------|--------|
| Resolution | 64³ | 128³ | 256³ | 256³+ |
| PSNR | 28-30 dB | 32-35 dB | 35-38 dB | - |
| SSIM | 0.85-0.90 | 0.92-0.95 | 0.95-0.97 | - |
| Training Time | ~6h | ~12h | ~24h | - |
| Memory/GPU | 10GB | 16GB | 35GB | - |

## 🐛 Troubleshooting

### Out of Memory Error
```python
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in `config_progressive.json`
- Stage 1: 8 → 4
- Stage 2: 4 → 2
- Stage 3: 2 → 1

### Import Error
```python
ModuleNotFoundError: No module named 'models.diagnostic_losses'
```
**Solution**: The import paths assume parent directory structure. Run from `progressive_cascade/`:
```bash
cd progressive_cascade
python train_progressive_4gpu.py
```

### Low PSNR on Stage 1
**Solution**: 
1. Train longer: increase `stage1.num_epochs` from 50 to 100
2. Increase model capacity: `voxel_dim` from 256 to 320
3. Adjust loss weights: increase `loss.stage1.ssim` from 0.5 to 0.8

### Training Unstable (Loss NaN)
**Solution**:
1. Reduce learning rates by 50%
2. Increase `gradient_clip` from 1.0 to 2.0
3. Check data normalization (should be [-1, 1])

## 📁 Output Files

After training, you'll have:

```
checkpoints_progressive/
├── stage1_best.pth          # Best stage 1 model (by val loss)
├── stage1_epoch50.pth       # Final stage 1 checkpoint
├── stage2_best.pth          # Best stage 2 model
├── stage2_epoch30.pth       # Final stage 2 checkpoint
├── stage3_best.pth          # Best stage 3 model (USE THIS)
└── stage3_epoch20.pth       # Final stage 3 checkpoint

outputs_progressive/
├── comparison_sample0.png   # Visual comparisons
├── evaluation_metrics.json  # Quantitative results
└── *.nii.gz                # NIfTI volumes (if --save-nifti)
```

## 🔬 Advanced Usage

### Resume Training from Checkpoint
Modify `train_progressive_4gpu.py` to load checkpoint:
```python
checkpoint = torch.load('checkpoints_progressive/stage2_epoch15.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Fine-tune Specific Stage
Unfreeze and train only Stage 2:
```python
model.unfreeze_stage(2)
model.freeze_stage(1)
model.freeze_stage(3)
```

### Custom Loss Weights
Adjust in `config_progressive.json`:
```json
{
  "loss": {
    "stage3": {
      "l1": 1.0,
      "ssim": 0.5,
      "vgg": 0.2,      // Increase for better texture
      "gradient": 0.3, // Increase for sharper edges
      "drr": 0.5       // Increase for better X-ray consistency
    }
  }
}
```

## 📞 Support

For issues or questions:
1. Check this guide's Troubleshooting section
2. Review [README.md](README.md) for detailed documentation
3. Examine configuration in `config_progressive.json`
4. Check GPU memory with `python utils.py`

## ✅ Next Steps

1. ✓ Train your model: `launch_progressive.bat`
2. ✓ Evaluate results: `python inference_progressive.py --mode evaluate`
3. ✓ Visualize outputs: Check `outputs_progressive/`
4. ✓ Fine-tune if needed: Adjust config and retrain specific stages
5. ✓ Deploy: Use `stage3_best.pth` for production inference

**Happy Training! 🎉**
