"""
Inference script for Medical-Grade Hybrid CNN-ViT model.

Compares:
  - Baseline (13.77 dB)
  - Medical-Grade Hybrid CNN-ViT (target 18-20 dB)

Usage:
    python inference_unet3d.py --checkpoint checkpoints/best_model.pth --output results/
"""

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset import PatientDRRDataset
from medical_grade.model_unet3d import HybridCNNViTUNet3D


def compute_metrics(pred, target):
    """Compute PSNR and SSIM"""
    # PSNR
    mse = np.mean((pred - target) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))
    
    # SSIM (simplified)
    mu_x = np.mean(pred)
    mu_y = np.mean(target)
    sigma_x = np.std(pred)
    sigma_y = np.std(target)
    sigma_xy = np.mean((pred - mu_x) * (target - mu_y))
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))
    
    return psnr, ssim


def create_comparison_plot(xrays, pred, target, patient_id, metrics, save_path):
    """
    Create 4-row comparison visualization:
      Row 1: Frontal + Lateral X-rays
      Row 2: Predicted CT slices (axial, coronal, sagittal)
      Row 3: Target CT slices (axial, coronal, sagittal)
      Row 4: Error maps (|pred - target|)
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    
    # Row 1: X-rays
    axes[0, 0].imshow(xrays[0, 0], cmap='gray')
    axes[0, 0].set_title('Frontal X-ray', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(xrays[0, 1], cmap='gray')
    axes[0, 1].set_title('Lateral X-ray', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].text(0.5, 0.5, f'PSNR: {metrics["psnr"]:.2f} dB\nSSIM: {metrics["ssim"]:.4f}',
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    transform=axes[0, 2].transAxes)
    axes[0, 2].axis('off')
    
    # Get mid slices
    D, H, W = pred.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2
    
    # Row 2: Predicted CT
    axes[1, 0].imshow(pred[mid_d], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Predicted - Axial', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred[:, mid_h, :], cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Predicted - Coronal', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(pred[:, :, mid_w], cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('Predicted - Sagittal', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Row 3: Target CT
    axes[2, 0].imshow(target[mid_d], cmap='gray', vmin=0, vmax=1)
    axes[2, 0].set_title('Target - Axial', fontsize=11, fontweight='bold')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(target[:, mid_h, :], cmap='gray', vmin=0, vmax=1)
    axes[2, 1].set_title('Target - Coronal', fontsize=11, fontweight='bold')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(target[:, :, mid_w], cmap='gray', vmin=0, vmax=1)
    axes[2, 2].set_title('Target - Sagittal', fontsize=11, fontweight='bold')
    axes[2, 2].axis('off')
    
    # Row 4: Error maps
    error = np.abs(pred - target)
    
    im0 = axes[3, 0].imshow(error[mid_d], cmap='hot', vmin=0, vmax=0.3)
    axes[3, 0].set_title('Error - Axial', fontsize=11, fontweight='bold')
    axes[3, 0].axis('off')
    
    im1 = axes[3, 1].imshow(error[:, mid_h, :], cmap='hot', vmin=0, vmax=0.3)
    axes[3, 1].set_title('Error - Coronal', fontsize=11, fontweight='bold')
    axes[3, 1].axis('off')
    
    im2 = axes[3, 2].imshow(error[:, :, mid_w], cmap='hot', vmin=0, vmax=0.3)
    axes[3, 2].set_title('Error - Sagittal', fontsize=11, fontweight='bold')
    axes[3, 2].axis('off')
    
    # Colorbar for error
    fig.colorbar(im0, ax=axes[3, :], orientation='horizontal', fraction=0.05, pad=0.05)
    
    plt.suptitle(f'Patient: {patient_id} | Medical-Grade Hybrid CNN-ViT', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data_dir', type=str, default='../data/patients_64', help='Data directory')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    parser.add_argument('--volume_size', type=int, nargs=3, default=[64, 64, 64], help='Volume size')
    parser.add_argument('--base_channels', type=int, default=32, help='Base channels')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples (-1 for all)')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('\n=== Medical-Grade Hybrid CNN-ViT Inference ===')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Output: {args.output}\n')
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')
    
    model = HybridCNNViTUNet3D(
        volume_size=tuple(args.volume_size),
        xray_size=512,
        base_channels=args.base_channels
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    print(f'Checkpoint metrics:')
    for k, v in checkpoint['metrics'].items():
        print(f'  {k}: {v:.4f}')
    print()
    
    # Load dataset
    dataset = PatientDRRDataset(
        data_path=args.data_dir,
        target_xray_size=512,
        target_volume_size=tuple(args.volume_size)
    )
    
    num_samples = len(dataset) if args.num_samples == -1 else min(args.num_samples, len(dataset))
    print(f'Processing {num_samples} test samples...\n')
    
    # Inference
    all_psnrs = []
    all_ssims = []
    
    with torch.no_grad():
        for idx in tqdm(range(num_samples)):
            batch = dataset[idx]
            
            xrays = batch['drr_stacked'].unsqueeze(0).to(device)  # (1, 2, 1, H, W)
            target_ct = batch['ct_volume'].cpu().numpy()[0]  # (D, H, W)
            patient_id = batch['patient_id']
            
            # Predict
            pred_ct, _ = model(xrays)
            pred_ct = pred_ct.cpu().numpy()[0, 0]  # (D, H, W)
            
            # Metrics
            psnr, ssim = compute_metrics(pred_ct, target_ct)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)
            
            metrics = {'psnr': psnr, 'ssim': ssim}
            
            # Save NIfTI
            nifti = nib.Nifti1Image(pred_ct, affine=np.eye(4))
            nib.save(nifti, output_dir / f'{patient_id}_pred.nii.gz')
            
            # Save visualization
            xrays_np = batch['drr_stacked'].cpu().numpy()
            create_comparison_plot(
                xrays_np, pred_ct, target_ct, patient_id, metrics,
                output_dir / f'{patient_id}_comparison.png'
            )
    
    # Summary
    mean_psnr = np.mean(all_psnrs)
    std_psnr = np.std(all_psnrs)
    mean_ssim = np.mean(all_ssims)
    std_ssim = np.std(all_ssims)
    
    print(f'\n=== Results Summary ===')
    print(f'Samples: {num_samples}')
    print(f'PSNR: {mean_psnr:.2f} ± {std_psnr:.2f} dB')
    print(f'SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}')
    print(f'\nTarget (Milestone 1): 18-20 dB PSNR')
    
    if mean_psnr >= 18:
        print('✓ Milestone 1 achieved!')
    elif mean_psnr >= 16:
        print('⚠ Close to target, consider more training epochs')
    else:
        print('✗ Below target, debugging needed')
    
    print(f'\nResults saved to: {args.output}')
    
    # Save summary
    summary = {
        'num_samples': num_samples,
        'mean_psnr': float(mean_psnr),
        'std_psnr': float(std_psnr),
        'mean_ssim': float(mean_ssim),
        'std_ssim': float(std_ssim),
        'per_sample': [
            {'patient_id': dataset[i]['patient_id'], 'psnr': float(all_psnrs[i]), 'ssim': float(all_ssims[i])}
            for i in range(num_samples)
        ]
    }
    
    import json
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'Summary saved to: {output_dir / "summary.json"}')


if __name__ == '__main__':
    main()
