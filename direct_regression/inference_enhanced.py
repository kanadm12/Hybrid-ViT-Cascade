"""
Inference Script for Enhanced Direct Regression Model
Reconstructs 3D CT volumes from dual-view X-ray images using:
- Stage 1: EnhancedDirectModel (64³)
- Stage 2: RefinementNetwork (64³→256³)
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
import json
import nibabel as nib
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '..')

from model_enhanced import EnhancedDirectModel, RefinementNetwork
from utils.dataset import PatientDRRDataset


def compute_metrics(pred, target):
    """Compute PSNR and SSIM"""
    # PSNR
    mse = F.mse_loss(pred, target)
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # Range [-1, 1]
    
    # Simple SSIM approximation
    mu_pred = pred.mean()
    mu_target = target.mean()
    sigma_pred = pred.std()
    sigma_target = target.std()
    sigma_pt = ((pred - mu_pred) * (target - mu_target)).mean()
    
    c1 = (0.01 * 2) ** 2
    c2 = (0.03 * 2) ** 2
    
    ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pt + c2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred ** 2 + sigma_target ** 2 + c2))
    
    return psnr.item(), ssim.item()


def load_models(base_checkpoint, refinement_checkpoint, device):
    """Load base and refinement models"""
    print("="*60)
    print("Loading Models")
    print("="*60)
    
    # Load base model (64³)
    print(f"\n1. Base Model (64³):")
    print(f"   Loading from: {base_checkpoint}")
    base_model = EnhancedDirectModel(
        volume_size=(64, 64, 64),
        base_channels=64
    )
    checkpoint = torch.load(base_checkpoint, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model = base_model.to(device)
    base_model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    psnr = checkpoint.get('best_psnr', checkpoint.get('val_psnr', 'unknown'))
    print(f"   ✓ Loaded from epoch {epoch}")
    print(f"   ✓ Training PSNR: {psnr} dB")
    
    # Load refinement model (64³→256³)
    refinement_model = None
    if refinement_checkpoint:
        print(f"\n2. Refinement Model (64³→256³):")
        print(f"   Loading from: {refinement_checkpoint}")
        refinement_model = RefinementNetwork()
        checkpoint = torch.load(refinement_checkpoint, map_location=device)
        refinement_model.load_state_dict(checkpoint['model_state_dict'])
        refinement_model = refinement_model.to(device)
        refinement_model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        psnr = checkpoint.get('best_psnr', 'unknown')
        print(f"   ✓ Loaded from epoch {epoch}")
        print(f"   ✓ Training PSNR: {psnr} dB")
    else:
        print(f"\n2. Refinement Model: Not provided (will output 64³ only)")
    
    return base_model, refinement_model


@torch.no_grad()
def run_inference(base_model, refinement_model, xrays, device):
    """
    Run inference through base model and optional refinement
    
    Args:
        base_model: EnhancedDirectModel
        refinement_model: RefinementNetwork or None
        xrays: (B, 2, 1, H, W) dual-view X-rays
        device: cuda/cpu
    
    Returns:
        pred_64: (B, 1, 64, 64, 64) base prediction
        pred_256: (B, 1, 256, 256, 256) refined prediction or None
    """
    xrays = xrays.to(device)
    
    # Stage 1: Base model → 64³
    print("Running base model (64³)...")
    pred_64, aux_outputs = base_model(xrays)
    
    # Stage 2: Refinement → 256³
    pred_256 = None
    if refinement_model is not None:
        print("Running refinement network (64³→256³)...")
        pred_256 = refinement_model(pred_64)
    
    return pred_64, pred_256


def save_volume(volume, output_path, affine=None):
    """Save CT volume as NIfTI"""
    if affine is None:
        affine = np.eye(4)
    
    # Convert to numpy
    volume_np = volume.cpu().numpy().squeeze()  # (D, H, W)
    
    # Denormalize from [-1, 1] to HU
    volume_np = (volume_np + 1) / 2 * 400 - 200  # → [-200, 200] HU
    
    # Save
    nii = nib.Nifti1Image(volume_np, affine)
    nib.save(nii, output_path)
    print(f"   Saved: {output_path}")


def visualize_slices(pred_64, pred_256, target, output_dir, patient_id):
    """Create visualization of axial slices"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Take middle slice
    slice_idx_64 = pred_64.shape[2] // 2
    slice_idx_256 = pred_256.shape[2] // 2 if pred_256 is not None else target.shape[2] // 2
    
    fig, axes = plt.subplots(1, 4 if pred_256 is not None else 3, figsize=(20, 5))
    
    # Convert to numpy
    pred_64_slice = pred_64[0, 0, slice_idx_64].cpu().numpy()
    target_slice = target[0, 0, slice_idx_256].cpu().numpy()
    
    axes[0].imshow(pred_64_slice, cmap='gray', vmin=-1, vmax=1)
    axes[0].set_title(f'Prediction 64³ (slice {slice_idx_64})')
    axes[0].axis('off')
    
    if pred_256 is not None:
        pred_256_slice = pred_256[0, 0, slice_idx_256].cpu().numpy()
        axes[1].imshow(pred_256_slice, cmap='gray', vmin=-1, vmax=1)
        axes[1].set_title(f'Prediction 256³ (slice {slice_idx_256})')
        axes[1].axis('off')
        
        axes[2].imshow(target_slice, cmap='gray', vmin=-1, vmax=1)
        axes[2].set_title(f'Ground Truth 256³ (slice {slice_idx_256})')
        axes[2].axis('off')
        
        # Difference map
        diff = np.abs(pred_256_slice - target_slice)
        axes[3].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[3].set_title('Absolute Difference')
        axes[3].axis('off')
    else:
        # Upsample pred_64 to compare with target
        pred_64_upsampled = F.interpolate(
            pred_64,
            size=(256, 256, 256),
            mode='trilinear',
            align_corners=True
        )
        pred_64_up_slice = pred_64_upsampled[0, 0, slice_idx_256].cpu().numpy()
        
        axes[1].imshow(pred_64_up_slice, cmap='gray', vmin=-1, vmax=1)
        axes[1].set_title(f'Prediction 64³↑256³ (slice {slice_idx_256})')
        axes[1].axis('off')
        
        axes[2].imshow(target_slice, cmap='gray', vmin=-1, vmax=1)
        axes[2].set_title(f'Ground Truth 256³ (slice {slice_idx_256})')
        axes[2].axis('off')
    
    plt.tight_layout()
    viz_path = output_dir / f'{patient_id}_comparison.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {viz_path}")


def main():
    parser = argparse.ArgumentParser(description='Run inference with enhanced direct regression model')
    parser.add_argument('--base_checkpoint', type=str, required=True,
                       help='Path to base model checkpoint (64³)')
    parser.add_argument('--refinement_checkpoint', type=str, default=None,
                       help='Path to refinement model checkpoint (optional, for 256³)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to patient data directory')
    parser.add_argument('--patient_id', type=str, default=None,
                       help='Specific patient ID to process (optional, processes all if not specified)')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--num_patients', type=int, default=10,
                       help='Number of patients to process if patient_id not specified')
    parser.add_argument('--save_volumes', action='store_true',
                       help='Save reconstructed volumes as NIfTI files')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Enhanced Direct Regression - Inference")
    print("="*60)
    
    # Load models
    base_model, refinement_model = load_models(
        args.base_checkpoint,
        args.refinement_checkpoint,
        device
    )
    
    # Load dataset
    print(f"\n{'='*60}")
    print("Loading Dataset")
    print("="*60)
    print(f"Data path: {args.data_path}")
    
    dataset = PatientDRRDataset(
        data_path=args.data_path,
        target_xray_size=512,
        target_volume_size=(256, 256, 256),
        max_patients=args.num_patients if args.patient_id is None else None,
        validate_alignment=False
    )
    
    # Find patient(s) to process
    if args.patient_id:
        patient_indices = [i for i, folder in enumerate(dataset.patient_folders) 
                          if folder.name == args.patient_id]
        if not patient_indices:
            print(f"Error: Patient {args.patient_id} not found!")
            return
        print(f"Processing patient: {args.patient_id}")
    else:
        patient_indices = list(range(min(args.num_patients, len(dataset))))
        print(f"Processing {len(patient_indices)} patients")
    
    # Run inference
    print(f"\n{'='*60}")
    print("Running Inference")
    print("="*60)
    
    all_metrics = {'psnr_64': [], 'psnr_256': []}
    
    for idx in patient_indices:
        data = dataset[idx]
        patient_id = data['patient_id']
        
        print(f"\n--- Patient: {patient_id} ---")
        
        # Prepare inputs
        xrays = data['drr_stacked'].unsqueeze(0)  # (1, 2, 1, H, W)
        target_256 = data['ct_volume'].unsqueeze(0)  # (1, 1, 256, 256, 256)
        
        # Run inference
        pred_64, pred_256 = run_inference(base_model, refinement_model, xrays, device)
        
        # Compute metrics
        # 64³: Compare against downsampled target
        target_64 = F.interpolate(target_256, size=(64, 64, 64), mode='trilinear', align_corners=True)
        psnr_64, ssim_64 = compute_metrics(pred_64, target_64)
        all_metrics['psnr_64'].append(psnr_64)
        
        print(f"   64³  → PSNR: {psnr_64:.2f} dB | SSIM: {ssim_64:.4f}")
        
        if pred_256 is not None:
            # 256³: Compare against original target
            psnr_256, ssim_256 = compute_metrics(pred_256, target_256)
            all_metrics['psnr_256'].append(psnr_256)
            print(f"   256³ → PSNR: {psnr_256:.2f} dB | SSIM: {ssim_256:.4f}")
        
        # Save volumes
        if args.save_volumes:
            patient_dir = output_dir / patient_id
            patient_dir.mkdir(exist_ok=True)
            
            save_volume(pred_64, patient_dir / f'{patient_id}_pred_64.nii.gz')
            if pred_256 is not None:
                save_volume(pred_256, patient_dir / f'{patient_id}_pred_256.nii.gz')
            save_volume(target_256, patient_dir / f'{patient_id}_ground_truth.nii.gz')
        
        # Visualize
        visualize_slices(pred_64, pred_256, target_256, output_dir, patient_id)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print("="*60)
    print(f"Processed {len(patient_indices)} patients")
    print(f"\nAverage Metrics:")
    print(f"   64³  → PSNR: {np.mean(all_metrics['psnr_64']):.2f} ± {np.std(all_metrics['psnr_64']):.2f} dB")
    if all_metrics['psnr_256']:
        print(f"   256³ → PSNR: {np.mean(all_metrics['psnr_256']):.2f} ± {np.std(all_metrics['psnr_256']):.2f} dB")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
