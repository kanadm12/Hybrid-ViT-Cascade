"""
Progressive Cascade Inference and Evaluation
Supports:
- Full cascade inference (64³→128³→256³)
- 3D visualization comparing outputs at each stage
- PSNR/SSIM metrics per stage
- NIfTI export for medical imaging software
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, '../..')

from progressive_cascade.model_progressive import ProgressiveCascadeModel
from progressive_cascade.loss_multiscale import compute_psnr, compute_ssim_metric
from utils.dataset import PatientDRRDataset


def load_model(checkpoint_path, config, device='cuda'):
    """Load trained progressive cascade model"""
    model = ProgressiveCascadeModel(
        xray_img_size=config['model']['xray_img_size'],
        xray_feature_dim=config['model']['xray_feature_dim'],
        voxel_dim=config['model']['voxel_dim'],
        use_gradient_checkpointing=False  # Not needed for inference
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'val_psnr' in checkpoint:
        print(f"Val PSNR: {checkpoint['val_psnr']:.2f} dB")
    if 'val_ssim' in checkpoint:
        print(f"Val SSIM: {checkpoint['val_ssim']:.4f}")
    
    return model


def inference_single(model, xrays, device='cuda'):
    """
    Run inference on a single X-ray pair
    Returns outputs from all stages
    """
    with torch.no_grad():
        xrays = xrays.to(device)
        outputs = model(xrays, return_intermediate=True, max_stage=3)
    
    return {
        'stage1_64': outputs['stage1'].cpu(),
        'stage2_128': outputs['stage2'].cpu(),
        'stage3_256': outputs['stage3'].cpu()
    }


def evaluate_stage(pred, target, stage_name):
    """
    Evaluate a single stage output
    Args:
        pred: (B, 1, D, H, W)
        target: (B, 1, D, H, W)
        stage_name: str
    Returns:
        metrics: dict
    """
    # Resize target to match prediction
    if pred.shape != target.shape:
        target = F.interpolate(target, size=pred.shape[2:], 
                              mode='trilinear', align_corners=False)
    
    psnr = compute_psnr(pred, target)
    ssim = compute_ssim_metric(pred, target)
    
    # L1 error
    l1_error = torch.mean(torch.abs(pred - target)).item()
    
    return {
        f'{stage_name}_psnr': psnr,
        f'{stage_name}_ssim': ssim,
        f'{stage_name}_l1': l1_error
    }


def visualize_comparison(outputs, target, save_path=None):
    """
    Create visualization comparing all stages
    Shows axial, sagittal, and coronal slices
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    
    # Resize target to different resolutions
    target_64 = F.interpolate(target, size=(64, 64, 64), 
                              mode='trilinear', align_corners=False)[0, 0].cpu().numpy()
    target_128 = F.interpolate(target, size=(128, 128, 128),
                               mode='trilinear', align_corners=False)[0, 0].cpu().numpy()
    target_256 = F.interpolate(target, size=(256, 256, 256),
                               mode='trilinear', align_corners=False)[0, 0].cpu().numpy()
    
    # Get predictions
    pred_64 = outputs['stage1_64'][0, 0].numpy()
    pred_128 = outputs['stage2_128'][0, 0].numpy()
    pred_256 = outputs['stage3_256'][0, 0].numpy()
    
    # Define slice indices (middle slices)
    slices = {
        '64': (32, 32, 32),
        '128': (64, 64, 64),
        '256': (128, 128, 128)
    }
    
    volumes = [
        ('Stage 1 (64³)', pred_64, target_64, '64'),
        ('Stage 2 (128³)', pred_128, target_128, '128'),
        ('Stage 3 (256³)', pred_256, target_256, '256'),
        ('Ground Truth (256³)', target_256, target_256, '256')
    ]
    
    for row, (title, pred_vol, tgt_vol, size_key) in enumerate(volumes):
        d_slice, h_slice, w_slice = slices[size_key]
        
        # Axial slice (D)
        axes[row, 0].imshow(pred_vol[d_slice, :, :], cmap='gray', vmin=-1, vmax=1)
        axes[row, 0].set_title(f'{title} - Axial')
        axes[row, 0].axis('off')
        
        # Sagittal slice (H)
        axes[row, 1].imshow(pred_vol[:, h_slice, :], cmap='gray', vmin=-1, vmax=1)
        axes[row, 1].set_title(f'{title} - Sagittal')
        axes[row, 1].axis('off')
        
        # Coronal slice (W)
        axes[row, 2].imshow(pred_vol[:, :, w_slice], cmap='gray', vmin=-1, vmax=1)
        axes[row, 2].set_title(f'{title} - Coronal')
        axes[row, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_nifti(volume, output_path, affine=None):
    """
    Save volume as NIfTI file
    Requires nibabel: pip install nibabel
    """
    try:
        import nibabel as nib
        
        # Convert to numpy
        if torch.is_tensor(volume):
            volume = volume.cpu().numpy()
        
        # Remove batch and channel dimensions
        if volume.ndim == 5:
            volume = volume[0, 0]
        elif volume.ndim == 4:
            volume = volume[0]
        
        # Create affine matrix if not provided
        if affine is None:
            affine = np.eye(4)
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(volume, affine)
        
        # Save
        nib.save(nifti_img, output_path)
        print(f"Saved NIfTI: {output_path}")
        
    except ImportError:
        print("Warning: nibabel not installed. Cannot save NIfTI files.")
        print("Install with: pip install nibabel")


def evaluate_dataset(model, dataset, config, device='cuda', num_samples=None):
    """
    Evaluate model on entire dataset
    Args:
        model: ProgressiveCascadeModel
        dataset: PatientDRRDataset
        num_samples: Limit number of samples (None = all)
    """
    model.eval()
    
    metrics_stage1 = []
    metrics_stage2 = []
    metrics_stage3 = []
    
    num_samples = num_samples or len(dataset)
    
    print(f"\nEvaluating on {num_samples} samples...")
    
    for idx in tqdm(range(min(num_samples, len(dataset)))):
        sample = dataset[idx]
        xrays = sample['drr_stacked'].unsqueeze(0).to(device)
        target = sample['ct_volume'].unsqueeze(0).to(device)
        
        # Inference
        outputs = inference_single(model, xrays, device)
        
        # Evaluate each stage
        metrics_stage1.append(evaluate_stage(outputs['stage1_64'], target, 'stage1'))
        metrics_stage2.append(evaluate_stage(outputs['stage2_128'], target, 'stage2'))
        metrics_stage3.append(evaluate_stage(outputs['stage3_256'], target, 'stage3'))
    
    # Average metrics
    avg_metrics = {}
    
    for stage_metrics in [metrics_stage1, metrics_stage2, metrics_stage3]:
        for key in stage_metrics[0].keys():
            values = [m[key] for m in stage_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
    
    return avg_metrics


def print_metrics_table(metrics):
    """Print metrics in a nice table format"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    stages = ['stage1', 'stage2', 'stage3']
    
    print(f"\n{'Stage':<15} {'PSNR (dB)':<20} {'SSIM':<20} {'L1 Error':<15}")
    print("-"*70)
    
    for stage in stages:
        psnr = metrics.get(f'{stage}_psnr', 0)
        psnr_std = metrics.get(f'{stage}_psnr_std', 0)
        
        ssim = metrics.get(f'{stage}_ssim', 0)
        ssim_std = metrics.get(f'{stage}_ssim_std', 0)
        
        l1 = metrics.get(f'{stage}_l1', 0)
        l1_std = metrics.get(f'{stage}_l1_std', 0)
        
        stage_name = f"{stage.replace('stage', 'Stage ')} ({['64³', '128³', '256³'][int(stage[-1])-1]})"
        
        print(f"{stage_name:<15} {psnr:>6.2f} ± {psnr_std:<6.2f}  "
              f"{ssim:>6.4f} ± {ssim_std:<6.4f}  "
              f"{l1:>6.4f} ± {l1_std:<6.4f}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Progressive Cascade Inference')
    parser.add_argument('--config', type=str, default='config_progressive.json',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (stage3_best.pth)')
    parser.add_argument('--mode', type=str, choices=['single', 'evaluate'], default='single',
                       help='Inference mode: single sample or full evaluation')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Sample index for single inference')
    parser.add_argument('--output-dir', type=str, default='outputs_progressive',
                       help='Output directory for results')
    parser.add_argument('--save-nifti', action='store_true',
                       help='Save outputs as NIfTI files')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / args.config
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, config, device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset_path = config['data']['dataset_path']
    dataset = PatientDRRDataset(
        root_dir=dataset_path,
        max_patients=config['data']['max_patients'],
        split='test',
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split']
    )
    
    print(f"Dataset: {len(dataset)} samples")
    
    if args.mode == 'single':
        # Single sample inference
        print(f"\nRunning inference on sample {args.sample_idx}...")
        
        sample = dataset[args.sample_idx]
        xrays = sample['drr_stacked'].unsqueeze(0)
        target = sample['ct_volume'].unsqueeze(0)
        
        # Run inference
        outputs = inference_single(model, xrays, device)
        
        # Evaluate each stage
        metrics = {}
        metrics.update(evaluate_stage(outputs['stage1_64'], target, 'stage1'))
        metrics.update(evaluate_stage(outputs['stage2_128'], target, 'stage2'))
        metrics.update(evaluate_stage(outputs['stage3_256'], target, 'stage3'))
        
        # Print metrics
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Visualize
        viz_path = output_dir / f"comparison_sample{args.sample_idx}.png"
        visualize_comparison(outputs, target, save_path=viz_path)
        
        # Save NIfTI if requested
        if args.save_nifti:
            save_nifti(outputs['stage1_64'], 
                      output_dir / f"sample{args.sample_idx}_stage1_64.nii.gz")
            save_nifti(outputs['stage2_128'],
                      output_dir / f"sample{args.sample_idx}_stage2_128.nii.gz")
            save_nifti(outputs['stage3_256'],
                      output_dir / f"sample{args.sample_idx}_stage3_256.nii.gz")
            save_nifti(target,
                      output_dir / f"sample{args.sample_idx}_target.nii.gz")
    
    elif args.mode == 'evaluate':
        # Full dataset evaluation
        metrics = evaluate_dataset(model, dataset, config, device, args.num_samples)
        
        # Print results
        print_metrics_table(metrics)
        
        # Save metrics to JSON
        metrics_path = output_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to: {metrics_path}")
        
        # Visualize a few samples
        print("\nGenerating visualizations for first 5 samples...")
        for idx in range(min(5, len(dataset))):
            sample = dataset[idx]
            xrays = sample['drr_stacked'].unsqueeze(0)
            target = sample['ct_volume'].unsqueeze(0)
            
            outputs = inference_single(model, xrays, device)
            viz_path = output_dir / f"comparison_sample{idx}.png"
            visualize_comparison(outputs, target, save_path=viz_path)


if __name__ == "__main__":
    main()
