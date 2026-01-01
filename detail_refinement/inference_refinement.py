"""
Inference script for detail refinement network
Compares coarse base model vs. refined output
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import PatientDRRDataset
from direct_regression.model_enhanced import EnhancedDirectModel
from model_refinement import DetailRefinementNetwork


def calculate_psnr(pred, target, max_val=1.0):
    """Calculate PSNR"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(pred, target, window_size=11):
    """Calculate SSIM (simplified)"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool3d(pred, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool3d(target, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool3d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool3d(target ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool3d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def load_models(base_checkpoint, refinement_checkpoint, device='cuda'):
    """Load base model and refinement model"""
    print("\nLoading Models")
    print("=" * 60)
    
    # Load base model
    print(f"Base Model: {base_checkpoint}")
    base_model = EnhancedDirectModel(
        volume_size=(64, 64, 64),
        base_channels=128
    ).to(device)
    
    base_ckpt = torch.load(base_checkpoint, map_location=device)
    base_model.load_state_dict(base_ckpt['model_state_dict'])
    base_model.eval()
    
    print(f"  Epoch: {base_ckpt.get('epoch', 'unknown')}")
    print(f"  PSNR: {base_ckpt.get('val_psnr', 'unknown')} dB")
    
    # Load refinement model
    print(f"\nRefinement Model: {refinement_checkpoint}")
    refinement_model = DetailRefinementNetwork(
        volume_size=(64, 64, 64),
        xray_size=512,
        hidden_channels=64
    ).to(device)
    
    refine_ckpt = torch.load(refinement_checkpoint, map_location=device)
    refinement_model.load_state_dict(refine_ckpt['model_state_dict'])
    refinement_model.eval()
    
    print(f"  Epoch: {refine_ckpt.get('epoch', 'unknown')}")
    print(f"  PSNR Coarse: {refine_ckpt.get('val_psnr_coarse', 'unknown')} dB")
    print(f"  PSNR Refined: {refine_ckpt.get('val_psnr_refined', 'unknown')} dB")
    
    return base_model, refinement_model


def save_nifti(volume, affine, output_path):
    """Save 3D volume as NIfTI"""
    if volume.dim() == 5:
        volume = volume[0, 0].cpu().numpy()
    elif volume.dim() == 4:
        volume = volume[0].cpu().numpy()
    else:
        volume = volume.cpu().numpy()
    
    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, output_path)


def create_comparison_plot(xrays, coarse, refined, target, patient_idx, output_dir):
    """Create 3-way comparison plot: Coarse vs. Refined vs. Ground Truth"""
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    # Convert to numpy
    coarse = coarse[0, 0].cpu().numpy()
    refined = refined[0, 0].cpu().numpy()
    target = target[0, 0].cpu().numpy()
    
    xray_frontal = xrays[0, 0].cpu().numpy()
    xray_lateral = xrays[0, 1].cpu().numpy() if xrays.shape[1] > 1 else xray_frontal
    
    # Row 0: X-rays
    axes[0, 0].imshow(xray_frontal, cmap='gray')
    axes[0, 0].set_title('Frontal X-ray', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(xray_lateral, cmap='gray')
    axes[0, 1].set_title('Lateral X-ray', fontsize=12)
    axes[0, 1].axis('off')
    
    for i in range(2, 5):
        axes[0, i].axis('off')
    
    # Axial slice
    slice_idx = coarse.shape[0] // 2
    
    axes[1, 0].imshow(coarse[slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Coarse (Axial)', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(refined[slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Refined (Axial)', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(target[slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('Ground Truth (Axial)', fontsize=12)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(np.abs(coarse[slice_idx] - target[slice_idx]), cmap='hot', vmin=0, vmax=0.5)
    axes[1, 3].set_title('Coarse Error', fontsize=12)
    axes[1, 3].axis('off')
    
    axes[1, 4].imshow(np.abs(refined[slice_idx] - target[slice_idx]), cmap='hot', vmin=0, vmax=0.5)
    axes[1, 4].set_title('Refined Error', fontsize=12)
    axes[1, 4].axis('off')
    
    # Coronal slice
    axes[2, 0].imshow(coarse[:, slice_idx, :], cmap='gray', vmin=0, vmax=1)
    axes[2, 0].set_title('Coarse (Coronal)', fontsize=12)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(refined[:, slice_idx, :], cmap='gray', vmin=0, vmax=1)
    axes[2, 1].set_title('Refined (Coronal)', fontsize=12)
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(target[:, slice_idx, :], cmap='gray', vmin=0, vmax=1)
    axes[2, 2].set_title('Ground Truth (Coronal)', fontsize=12)
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(np.abs(coarse[:, slice_idx, :] - target[:, slice_idx, :]), cmap='hot', vmin=0, vmax=0.5)
    axes[2, 3].set_title('Coarse Error', fontsize=12)
    axes[2, 3].axis('off')
    
    axes[2, 4].imshow(np.abs(refined[:, slice_idx, :] - target[:, slice_idx, :]), cmap='hot', vmin=0, vmax=0.5)
    axes[2, 4].set_title('Refined Error', fontsize=12)
    axes[2, 4].axis('off')
    
    # Sagittal slice
    axes[3, 0].imshow(coarse[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[3, 0].set_title('Coarse (Sagittal)', fontsize=12)
    axes[3, 0].axis('off')
    
    axes[3, 1].imshow(refined[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[3, 1].set_title('Refined (Sagittal)', fontsize=12)
    axes[3, 1].axis('off')
    
    axes[3, 2].imshow(target[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[3, 2].set_title('Ground Truth (Sagittal)', fontsize=12)
    axes[3, 2].axis('off')
    
    axes[3, 3].imshow(np.abs(coarse[:, :, slice_idx] - target[:, :, slice_idx]), cmap='hot', vmin=0, vmax=0.5)
    axes[3, 3].set_title('Coarse Error', fontsize=12)
    axes[3, 3].axis('off')
    
    axes[3, 4].imshow(np.abs(refined[:, :, slice_idx] - target[:, :, slice_idx]), cmap='hot', vmin=0, vmax=0.5)
    axes[3, 4].set_title('Refined Error', fontsize=12)
    axes[3, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'patient_{patient_idx:03d}_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def run_inference(base_model, refinement_model, dataloader, output_dir, save_volumes=False, device='cuda'):
    """Run inference"""
    base_model.eval()
    refinement_model.eval()
    
    coarse_psnr = []
    refined_psnr = []
    coarse_ssim = []
    refined_ssim = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running Inference")):
            xrays = batch['drr_stacked'].to(device)
            ct_volume = batch['ct_volume'].to(device)
            
            # Stage 1: Coarse prediction
            coarse_pred, _ = base_model(xrays)
            
            # Stage 2: Refined prediction
            refined_pred, _ = refinement_model(coarse_pred, xrays)
            
            # Metrics
            c_psnr = calculate_psnr(coarse_pred, ct_volume)
            r_psnr = calculate_psnr(refined_pred, ct_volume)
            c_ssim = calculate_ssim(coarse_pred, ct_volume)
            r_ssim = calculate_ssim(refined_pred, ct_volume)
            
            coarse_psnr.append(c_psnr)
            refined_psnr.append(r_psnr)
            coarse_ssim.append(c_ssim)
            refined_ssim.append(r_ssim)
            
            # Save
            if save_volumes:
                patient_dir = os.path.join(output_dir, f'patient_{batch_idx:03d}')
                os.makedirs(patient_dir, exist_ok=True)
                
                affine = np.eye(4)
                
                save_nifti(coarse_pred, affine, os.path.join(patient_dir, 'coarse.nii.gz'))
                save_nifti(refined_pred, affine, os.path.join(patient_dir, 'refined.nii.gz'))
                save_nifti(ct_volume, affine, os.path.join(patient_dir, 'ground_truth.nii.gz'))
                
                create_comparison_plot(xrays, coarse_pred, refined_pred, ct_volume, batch_idx, patient_dir)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"\n[{batch_idx + 1}/{len(dataloader)}]")
                print(f"  Coarse  PSNR: {np.mean(coarse_psnr):.2f} dB | SSIM: {np.mean(coarse_ssim):.4f}")
                print(f"  Refined PSNR: {np.mean(refined_psnr):.2f} dB | SSIM: {np.mean(refined_ssim):.4f}")
                print(f"  Gain: +{np.mean(refined_psnr) - np.mean(coarse_psnr):.2f} dB")
    
    return coarse_psnr, refined_psnr, coarse_ssim, refined_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_checkpoint', type=str, required=True)
    parser.add_argument('--refinement_checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='inference_results_refinement')
    parser.add_argument('--num_patients', type=int, default=20)
    parser.add_argument('--save_volumes', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Detail Refinement Inference")
    print("=" * 60)
    
    # Load models
    base_model, refinement_model = load_models(
        args.base_checkpoint,
        args.refinement_checkpoint,
        args.device
    )
    
    # Dataset
    print(f"\nLoading dataset: {args.data_path}")
    dataset = PatientDRRDataset(
        data_path=args.data_path,
        target_xray_size=512,
        target_volume_size=(64, 64, 64),
        max_patients=args.num_patients
    )
    print(f"Found {len(dataset)} patients")
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run inference
    print("\n" + "=" * 60)
    print("Running Inference")
    print("=" * 60)
    
    coarse_psnr, refined_psnr, coarse_ssim, refined_ssim = run_inference(
        base_model, refinement_model, dataloader,
        args.output_dir, args.save_volumes, args.device
    )
    
    # Results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Coarse  PSNR: {np.mean(coarse_psnr):.2f} ± {np.std(coarse_psnr):.2f} dB")
    print(f"Refined PSNR: {np.mean(refined_psnr):.2f} ± {np.std(refined_psnr):.2f} dB")
    print(f"PSNR Gain: +{np.mean(refined_psnr) - np.mean(coarse_psnr):.2f} dB")
    print()
    print(f"Coarse  SSIM: {np.mean(coarse_ssim):.4f} ± {np.std(coarse_ssim):.4f}")
    print(f"Refined SSIM: {np.mean(refined_ssim):.4f} ± {np.std(refined_ssim):.4f}")
    print(f"SSIM Gain: +{np.mean(refined_ssim) - np.mean(coarse_ssim):.4f}")
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write("Detail Refinement Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Coarse  PSNR: {np.mean(coarse_psnr):.2f} ± {np.std(coarse_psnr):.2f} dB\n")
        f.write(f"Refined PSNR: {np.mean(refined_psnr):.2f} ± {np.std(refined_psnr):.2f} dB\n")
        f.write(f"PSNR Gain: +{np.mean(refined_psnr) - np.mean(coarse_psnr):.2f} dB\n\n")
        f.write(f"Coarse  SSIM: {np.mean(coarse_ssim):.4f} ± {np.std(coarse_ssim):.4f}\n")
        f.write(f"Refined SSIM: {np.mean(refined_ssim):.4f} ± {np.std(refined_ssim):.4f}\n")
        f.write(f"SSIM Gain: +{np.mean(refined_ssim) - np.mean(coarse_ssim):.4f}\n\n")
        f.write("Per-patient results:\n")
        for i in range(len(coarse_psnr)):
            f.write(f"Patient {i:03d}: Coarse={coarse_psnr[i]:.2f} dB, Refined={refined_psnr[i]:.2f} dB, Gain=+{refined_psnr[i]-coarse_psnr[i]:.2f} dB\n")


if __name__ == '__main__':
    main()
