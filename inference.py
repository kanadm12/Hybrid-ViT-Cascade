"""
Inference Script for Hybrid-ViT Cascade
Reconstructs 3D CT volumes from dual-view X-ray images
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import json
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

# Optional metrics - graceful fallback if not installed
try:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: torchmetrics not installed. Install with: pip install torchmetrics")
    METRICS_AVAILABLE = False
    PeakSignalNoiseRatio = None
    StructuralSimilarityIndexMeasure = None

from models.unified_model import UnifiedHybridViTCascade
from utils.dataset import PatientDRRDataset


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model
    model = UnifiedHybridViTCascade(
        stage_configs=config['stage_configs'],
        xray_img_size=config['xray_config']['img_size'],
        xray_channels=1,
        xray_embed_dim=512,
        num_views=config['xray_config']['num_views'],
        share_view_weights=config['xray_config'].get('share_view_weights', False),
        num_timesteps=config['training']['num_timesteps']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', None))
        if val_loss is not None:
            print(f"  Best val loss: {val_loss:.6f}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def reconstruct_volume(model, xrays, stage_name='stage3', num_steps=50, device='cuda'):
    """
    Reconstruct 3D CT volume from X-ray images using DDIM sampling
    
    Args:
        model: Trained UnifiedHybridViTCascade model
        xrays: X-ray images (B, num_views, 1, H, W)
        stage_name: Which cascade stage to use
        num_steps: Number of diffusion sampling steps
        device: Device to run on
    
    Returns:
        Reconstructed CT volume (B, 1, D, H, W)
    """
    batch_size = xrays.shape[0]
    xrays = xrays.to(device)
    
    # Get target volume size
    volume_size = model.stage_sizes[stage_name]
    D, H, W = volume_size
    
    print(f"\nReconstructing {stage_name} volume: {volume_size}")
    print(f"Using {num_steps} sampling steps")
    
    # Encode X-rays
    print("Encoding X-ray features...")
    timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
    timestep_embed = model.time_embed(timesteps.float().unsqueeze(-1))
    
    xray_context, time_xray_cond, features = model.xray_encoder(xrays, timestep_embed)
    
    # Use features (4D) for stage input - stage will reshape internally
    # features shape: (B, C, H, W)
    xray_features_4d = features
    
    # Start from random noise
    print("Initializing from random noise...")
    x = torch.randn(batch_size, 1, D, H, W, device=device)
    
    # DDIM sampling schedule
    timesteps = torch.linspace(model.num_timesteps - 1, 0, num_steps, device=device).long()
    
    stage = model.stages[stage_name]
    
    # Diffusion sampling loop
    print("Sampling...")
    for i, t in enumerate(tqdm(timesteps, desc="Diffusion steps")):
        t_batch = t.unsqueeze(0).expand(batch_size)
        
        # Get timestep embeddings (FIXED: normalize to [0,1] to match training)
        t_normalized = t_batch.float() / model.num_timesteps
        timestep_embed = model.time_embed(t_normalized.unsqueeze(-1))
        _, time_xray_cond_step, _ = model.xray_encoder(xrays, timestep_embed)
        
        # Predict noise/velocity
        predicted = stage(
            noisy_volume=x,
            xray_features=xray_features_4d,
            xray_context=xray_features_4d,
            time_xray_cond=time_xray_cond_step,
            prev_stage_volume=None,
            prev_stage_embed=None
        )
        
        # DDIM update step
        sqrt_alphas_t = model.sqrt_alphas_cumprod[t_batch].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_t = model.sqrt_one_minus_alphas_cumprod[t_batch].view(-1, 1, 1, 1, 1)
        
        if model.v_parameterization:
            # v-parameterization: predict x_0 from v
            pred_x0 = sqrt_alphas_t * x - sqrt_one_minus_alphas_t * predicted
        else:
            # epsilon-parameterization: predict x_0 from noise
            pred_x0 = (x - sqrt_one_minus_alphas_t * predicted) / torch.clamp(sqrt_alphas_t, min=1e-8)
        
        # Clamp to training range [-1, 1]
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # DDIM step (deterministic)
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            sqrt_alphas_next = model.sqrt_alphas_cumprod[t_next].view(-1, 1, 1, 1, 1)
            sqrt_one_minus_alphas_next = model.sqrt_one_minus_alphas_cumprod[t_next].view(-1, 1, 1, 1, 1)
            
            # Recompute noise from pred_x0 and current x
            if model.v_parameterization:
                # FIXED: Correct formula for noise recovery from v-prediction
                # pred_noise = (x_t - sqrt(α̅_t) * x_0) / sqrt(1 - α̅_t)
                pred_noise = (x - sqrt_alphas_t * pred_x0) / torch.clamp(sqrt_one_minus_alphas_t, min=1e-8)
            else:
                pred_noise = predicted
            
            x = sqrt_alphas_next * pred_x0 + sqrt_one_minus_alphas_next * pred_noise
        else:
            # Final step - use predicted x_0
            x = pred_x0
    
    # Final clamp to ensure output is in [-1, 1] range
    x = torch.clamp(x, -1.0, 1.0)
    
    print("Reconstruction complete!")
    print(f"  Output range: [{x.min().item():.3f}, {x.max().item():.3f}]")
    return x


def save_volume(volume, output_path, spacing=(1.0, 1.0, 1.0)):
    """Save volume as NIfTI file"""
    volume_np = volume.cpu().numpy().squeeze()
    
    # Create NIfTI image with proper spacing
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    
    nii = nib.Nifti1Image(volume_np, affine)
    nib.save(nii, output_path)
    print(f"Saved volume to {output_path}")


def visualize_slices(volume, xrays, output_path):
    """Create visualization of reconstructed volume and input X-rays"""
    volume_np = volume.cpu().numpy().squeeze()
    xrays_np = xrays.cpu().numpy().squeeze()
    
    D, H, W = volume_np.shape
    
    # Create figure with multiple views
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Input X-rays
    if xrays_np.ndim == 3:  # Multi-view
        axes[0, 0].imshow(xrays_np[0], cmap='gray')
        axes[0, 0].set_title('Input X-ray (View 1)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(xrays_np[1], cmap='gray')
        axes[0, 1].set_title('Input X-ray (View 2)')
        axes[0, 1].axis('off')
    else:  # Single view
        axes[0, 0].imshow(xrays_np, cmap='gray')
        axes[0, 0].set_title('Input X-ray')
        axes[0, 0].axis('off')
        axes[0, 1].axis('off')
    
    axes[0, 2].axis('off')
    
    # Reconstructed volume slices
    axes[1, 0].imshow(volume_np[D//2, :, :], cmap='bone', vmin=-1, vmax=1)
    axes[1, 0].set_title(f'Axial Slice (z={D//2})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(volume_np[:, H//2, :], cmap='bone', vmin=-1, vmax=1)
    axes[1, 1].set_title(f'Coronal Slice (y={H//2})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(volume_np[:, :, W//2], cmap='bone', vmin=-1, vmax=1)
    axes[1, 2].set_title(f'Sagittal Slice (x={W//2})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Inference with Hybrid-ViT Cascade')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (same as training)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Directory to save results')
    parser.add_argument('--stage', type=str, default='stage3',
                       choices=['stage1', 'stage2', 'stage3'],
                       help='Which cascade stage to use')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of test samples to process')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of diffusion sampling steps (more=slower but better)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    parser.add_argument('--flip_drrs_vertical', action='store_true',
                       help='Vertically flip DRRs (overrides config setting)')
    parser.add_argument('--no_flip_drrs_vertical', action='store_true',
                       help='Do not flip DRRs (overrides config setting)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    print("Loading configuration...")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Initialize metrics with data_range=1.0 for [0,1] normalized data
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Load test dataset
    print(f"\nLoading data from {args.data_dir}")
    
    # Get flip setting from config (default to False for backward compatibility)
    flip_drrs = config.get('data', {}).get('flip_drrs_vertical', False)
    
    # Command-line arguments override config
    if args.flip_drrs_vertical:
        flip_drrs = True
    elif args.no_flip_drrs_vertical:
        flip_drrs = False
    
    if flip_drrs:
        print("✓ DRRs will be vertically flipped to match CT orientation")
    else:
        print("✗ DRRs will NOT be flipped (using as-is)")
    
    test_dataset = PatientDRRDataset(
        data_path=args.data_dir,
        target_volume_size=model.stage_sizes[args.stage],
        target_xray_size=config['xray_config']['img_size'],
        normalize_range=tuple(config['data']['normalize_range']),
        validate_alignment=False,
        augmentation=False,
        flip_drrs_vertical=flip_drrs  # Apply same flip as training
    )
    
    print(f"Found {len(test_dataset)} test samples")
    num_samples = min(args.num_samples, len(test_dataset))
    
    # Run inference on test samples
    print(f"\n{'='*70}")
    print(f"Running inference on {num_samples} samples")
    print(f"{'='*70}\n")
    
    for idx in range(num_samples):
        print(f"\n--- Sample {idx+1}/{num_samples} ---")
        
        # Get data
        sample = test_dataset[idx]
        gt_volume = sample['ct_volume'].unsqueeze(0)  # (1, 1, D, H, W)
        xrays = sample['drr_stacked'].unsqueeze(0)     # (1, num_views, 1, H, W)
        patient_id = sample.get('patient_id', f'patient_{idx}')
        
        print(f"Patient ID: {patient_id}")
        print(f"Ground truth volume shape: {gt_volume.shape}")
        print(f"X-ray input shape: {xrays.shape}")
        
        # Reconstruct volume
        pred_volume = reconstruct_volume(
            model, xrays,
            stage_name=args.stage,
            num_steps=args.num_steps,
            device=device
        )
        
        # Compute metrics
        gt_on_device = gt_volume.to(device)
        
        # Basic metrics
        mse = torch.nn.functional.mse_loss(pred_volume, gt_on_device)
        mae = torch.nn.functional.l1_loss(pred_volume, gt_on_device)
        
        # Normalize to [0, 1] for PSNR/SSIM
        # Find min/max across both volumes for consistent normalization
        min_val = min(pred_volume.min().item(), gt_on_device.min().item())
        max_val = max(pred_volume.max().item(), gt_on_device.max().item())
        
        pred_normalized = (pred_volume - min_val) / (max_val - min_val + 1e-8)
        gt_normalized = (gt_on_device - min_val) / (max_val - min_val + 1e-8)
        
        # Image quality metrics (data_range=1.0 for [0,1] normalized data)
        psnr = psnr_metric(pred_normalized, gt_normalized)
        ssim = ssim_metric(pred_normalized, gt_normalized)
        
        print(f"\nMetrics:")
        print(f"  MSE:  {mse.item():.6f}")
        print(f"  MAE:  {mae.item():.6f}")
        print(f"  PSNR: {psnr.item():.2f} dB")
        print(f"  SSIM: {ssim.item():.4f}")
        
        # Save results
        sample_dir = output_dir / f"sample_{idx:03d}_{patient_id}"
        sample_dir.mkdir(exist_ok=True)
        
        # Save reconstructed volume
        save_volume(
            pred_volume[0],
            sample_dir / 'reconstructed.nii.gz',
            spacing=(1.0, 1.0, 1.0)
        )
        
        # Save ground truth for comparison
        save_volume(
            gt_volume[0],
            sample_dir / 'ground_truth.nii.gz',
            spacing=(1.0, 1.0, 1.0)
        )
        
        # Create visualization
        visualize_slices(
            pred_volume[0],
            xrays[0],
            sample_dir / 'visualization.png'
        )
        
        print(f"Results saved to {sample_dir}")
    
    print(f"\n{'='*70}")
    print("Inference completed successfully!")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
