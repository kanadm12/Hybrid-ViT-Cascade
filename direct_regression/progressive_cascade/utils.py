"""
Utility functions for progressive cascade training and inference
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def count_parameters(model):
    """Count trainable parameters in model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def print_model_summary(model):
    """Print detailed model summary"""
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    params = count_parameters(model)
    print(f"\nTotal parameters:     {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters:    {params['frozen']:,}")
    
    # Stage-wise breakdown
    if hasattr(model, 'stage1'):
        stage1_params = count_parameters(model.stage1)
        print(f"\nStage 1 parameters:   {stage1_params['total']:,}")
    
    if hasattr(model, 'stage2'):
        stage2_params = count_parameters(model.stage2)
        print(f"Stage 2 parameters:   {stage2_params['total']:,}")
    
    if hasattr(model, 'stage3'):
        stage3_params = count_parameters(model.stage3)
        print(f"Stage 3 parameters:   {stage3_params['total']:,}")
    
    print("="*60 + "\n")


def visualize_training_curves(history, save_path=None):
    """
    Visualize training curves
    Args:
        history: dict with 'train_loss', 'val_loss', 'val_psnr', 'val_ssim'
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PSNR curve
    axes[1].plot(epochs, history['val_psnr'], label='Val PSNR', 
                color='green', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('Validation PSNR')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # SSIM curve
    axes[2].plot(epochs, history['val_ssim'], label='Val SSIM',
                color='orange', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('SSIM')
    axes[2].set_title('Validation SSIM')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_stage_outputs(stage1_vol, stage2_vol, stage3_vol, target_vol, 
                          save_path=None):
    """
    Create side-by-side comparison of all stages
    Args:
        stage1_vol: (D, H, W) numpy array or torch tensor
        stage2_vol: (D, H, W)
        stage3_vol: (D, H, W)
        target_vol: (D, H, W)
        save_path: Path to save figure
    """
    # Convert to numpy if needed
    def to_numpy(vol):
        if torch.is_tensor(vol):
            vol = vol.cpu().numpy()
        if vol.ndim == 5:
            vol = vol[0, 0]  # Remove batch and channel dims
        elif vol.ndim == 4:
            vol = vol[0]
        return vol
    
    stage1 = to_numpy(stage1_vol)
    stage2 = to_numpy(stage2_vol)
    stage3 = to_numpy(stage3_vol)
    target = to_numpy(target_vol)
    
    # Resize all to same size for comparison
    def resize_to_match(vol, target_shape):
        if vol.shape != target_shape:
            vol_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
            vol_resized = F.interpolate(vol_tensor, size=target_shape,
                                       mode='trilinear', align_corners=False)
            return vol_resized[0, 0].numpy()
        return vol
    
    target_shape = target.shape
    stage1 = resize_to_match(stage1, target_shape)
    stage2 = resize_to_match(stage2, target_shape)
    stage3 = resize_to_match(stage3, target_shape)
    
    # Plot middle slices
    mid_slice = target_shape[0] // 2
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    volumes = [
        ('Stage 1 (64³)', stage1),
        ('Stage 2 (128³)', stage2),
        ('Stage 3 (256³)', stage3),
        ('Ground Truth', target)
    ]
    
    for ax, (title, vol) in zip(axes, volumes):
        ax.imshow(vol[mid_slice], cmap='gray', vmin=-1, vmax=1)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved stage comparison: {save_path}")
    else:
        plt.show()
    
    plt.close()


def check_gpu_memory():
    """Check available GPU memory"""
    if not torch.cuda.is_available():
        print("No GPU available")
        return
    
    print("\n" + "="*60)
    print("GPU MEMORY STATUS")
    print("="*60)
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3
        
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total memory:     {total:.2f} GB")
        print(f"  Allocated:        {allocated:.2f} GB ({allocated/total*100:.1f}%)")
        print(f"  Reserved:         {reserved:.2f} GB ({reserved/total*100:.1f}%)")
        print(f"  Available:        {total-reserved:.2f} GB")
    
    print("="*60 + "\n")


def estimate_memory_usage(batch_size, resolution, voxel_dim=256):
    """
    Estimate memory usage for training
    Args:
        batch_size: Batch size
        resolution: (D, H, W) tuple
        voxel_dim: Voxel embedding dimension
    Returns:
        Estimated memory in GB
    """
    D, H, W = resolution
    
    # Input volume
    input_mem = batch_size * 1 * D * H * W * 4  # float32
    
    # Feature maps (rough estimate)
    feature_mem = batch_size * voxel_dim * D * H * W * 4
    
    # Gradients (roughly 2x model size)
    gradient_mem = feature_mem * 2
    
    # Optimizer states (roughly 2x model size for AdamW)
    optimizer_mem = feature_mem * 2
    
    # Total
    total_mem = input_mem + feature_mem + gradient_mem + optimizer_mem
    
    # Convert to GB
    total_gb = total_mem / (1024**3)
    
    print(f"\nEstimated memory usage for {resolution}:")
    print(f"  Batch size: {batch_size}")
    print(f"  Voxel dim:  {voxel_dim}")
    print(f"  Total:      {total_gb:.2f} GB")
    
    return total_gb


def validate_config(config):
    """Validate configuration file"""
    required_keys = ['model', 'training', 'loss', 'data', 'checkpoints']
    
    print("\nValidating configuration...")
    
    for key in required_keys:
        if key not in config:
            print(f"✗ Missing required key: {key}")
            return False
        else:
            print(f"✓ Found key: {key}")
    
    # Check stage configs
    for stage in [1, 2, 3]:
        stage_key = f'stage{stage}'
        if stage_key not in config['training']:
            print(f"✗ Missing training config for {stage_key}")
            return False
        if stage_key not in config['loss']:
            print(f"✗ Missing loss config for {stage_key}")
            return False
    
    print("✓ Configuration valid!")
    return True


def create_launch_script():
    """Create batch launch script for Windows"""
    script = """@echo off
REM Progressive Cascade Training Launcher
REM Automatically uses all available GPUs

echo ========================================
echo Progressive Multi-Scale CT Reconstruction
echo ========================================
echo.

REM Activate conda environment (if using conda)
REM call conda activate your_env_name

REM Set CUDA visible devices (optional - comment out to use all GPUs)
REM set CUDA_VISIBLE_DEVICES=0,1,2,3

echo Starting training...
echo.

REM Run training
python train_progressive_4gpu.py

echo.
echo ========================================
echo Training Complete!
echo ========================================
pause
"""
    
    script_path = Path(__file__).parent / "launch_progressive.bat"
    with open(script_path, 'w') as f:
        f.write(script)
    
    print(f"Created launch script: {script_path}")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Check GPU memory
    check_gpu_memory()
    
    # Estimate memory for different stages
    estimate_memory_usage(batch_size=8, resolution=(64, 64, 64))
    estimate_memory_usage(batch_size=4, resolution=(128, 128, 128))
    estimate_memory_usage(batch_size=2, resolution=(256, 256, 256))
    
    # Create launch script
    create_launch_script()
