"""
Inference script for attention-enhanced model
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
from model_attention import AttentionEnhancedModel


def calculate_psnr(pred, target, max_val=1.0):
    """Calculate PSNR between prediction and target."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(pred, target, window_size=11):
    """Calculate SSIM between prediction and target (simplified 3D version)."""
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


def load_model(checkpoint_path, base_channels=96, device='cuda'):
    """Load attention-enhanced model from checkpoint."""
    print(f"\nLoading Attention Model:")
    print(f"   Checkpoint: {checkpoint_path}")
    
    # Create model
    model = AttentionEnhancedModel(
        volume_size=(64, 64, 64),
        base_channels=base_channels
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 'unknown')
    psnr = checkpoint.get('val_psnr', 'unknown')
    
    print(f"   Epoch: {epoch}")
    print(f"   Validation PSNR: {psnr} dB")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    model.eval()
    return model


def save_nifti(volume, affine, output_path):
    """Save 3D volume as NIfTI file."""
    # Remove channel and batch dimensions
    if volume.dim() == 5:
        volume = volume[0, 0].cpu().numpy()
    elif volume.dim() == 4:
        volume = volume[0].cpu().numpy()
    else:
        volume = volume.cpu().numpy()
    
    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, output_path)


def create_comparison_plot(xrays, prediction, target, patient_idx, output_dir):
    """Create comparison plot with X-rays, prediction, and ground truth."""
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    # Remove channel and batch dimensions
    if prediction.dim() == 5:
        prediction = prediction[0, 0].cpu().numpy()
    else:
        prediction = prediction.cpu().numpy()
    
    if target.dim() == 5:
        target = target[0, 0].cpu().numpy()
    else:
        target = target.cpu().numpy()
    
    # X-rays (frontal and lateral)
    xray_frontal = xrays[0, 0].cpu().numpy() if xrays.dim() == 4 else xrays[0].cpu().numpy()
    xray_lateral = xrays[0, 1].cpu().numpy() if xrays.shape[1] > 1 else xray_frontal
    
    axes[0, 0].imshow(xray_frontal, cmap='gray')
    axes[0, 0].set_title('Frontal X-ray')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(xray_lateral, cmap='gray')
    axes[0, 1].set_title('Lateral X-ray')
    axes[0, 1].axis('off')
    
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')
    axes[0, 4].axis('off')
    
    # CT slices - axial views
    slice_idx = prediction.shape[0] // 2
    
    axes[1, 0].imshow(prediction[slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Prediction (Axial)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(target[slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Ground Truth (Axial)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.abs(prediction[slice_idx] - target[slice_idx]), cmap='hot', vmin=0, vmax=0.5)
    axes[1, 2].set_title('Error (Axial)')
    axes[1, 2].axis('off')
    
    # CT slices - coronal views
    axes[1, 3].imshow(prediction[:, slice_idx, :], cmap='gray', vmin=0, vmax=1)
    axes[1, 3].set_title('Prediction (Coronal)')
    axes[1, 3].axis('off')
    
    axes[1, 4].imshow(target[:, slice_idx, :], cmap='gray', vmin=0, vmax=1)
    axes[1, 4].set_title('Ground Truth (Coronal)')
    axes[1, 4].axis('off')
    
    # CT slices - sagittal views
    axes[2, 0].imshow(prediction[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[2, 0].set_title('Prediction (Sagittal)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(target[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[2, 1].set_title('Ground Truth (Sagittal)')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(np.abs(prediction[:, :, slice_idx] - target[:, :, slice_idx]), cmap='hot', vmin=0, vmax=0.5)
    axes[2, 2].set_title('Error (Sagittal)')
    axes[2, 2].axis('off')
    
    axes[2, 3].axis('off')
    axes[2, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'patient_{patient_idx:03d}_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def run_inference(model, dataloader, output_dir, save_volumes=False, device='cuda'):
    """Run inference on dataset."""
    model.eval()
    
    all_psnr = []
    all_ssim = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running Inference")):
            xrays = batch['drr_stacked'].to(device)
            ct_volume = batch['ct_volume'].to(device)
            
            # Forward pass
            prediction, aux_outputs = model(xrays)
            
            # Calculate metrics at 64³ resolution
            psnr = calculate_psnr(prediction, ct_volume)
            ssim = calculate_ssim(prediction, ct_volume)
            
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            
            # Save volumes if requested
            if save_volumes:
                patient_dir = os.path.join(output_dir, f'patient_{batch_idx:03d}')
                os.makedirs(patient_dir, exist_ok=True)
                
                # Use identity affine (can be replaced with actual affine from dataset)
                affine = np.eye(4)
                
                # Save prediction
                save_nifti(prediction, affine, os.path.join(patient_dir, 'prediction.nii.gz'))
                
                # Save ground truth
                save_nifti(ct_volume, affine, os.path.join(patient_dir, 'ground_truth.nii.gz'))
                
                # Create comparison plot
                create_comparison_plot(xrays, prediction, ct_volume, batch_idx, patient_dir)
            
            # Print progress
            if (batch_idx + 1) % 5 == 0:
                avg_psnr = np.mean(all_psnr)
                avg_ssim = np.mean(all_ssim)
                print(f"\n[{batch_idx + 1}/{len(dataloader)}] "
                      f"PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
    
    return all_psnr, all_ssim


def main():
    parser = argparse.ArgumentParser(description='Inference for Attention-Enhanced Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to attention model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to patient data directory')
    parser.add_argument('--output_dir', type=str, default='inference_results_attention',
                        help='Output directory for results')
    parser.add_argument('--num_patients', type=int, default=None,
                        help='Number of patients to evaluate (default: all)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--base_channels', type=int, default=96,
                        help='Base channels in model (default: 96)')
    parser.add_argument('--save_volumes', action='store_true',
                        help='Save reconstructed volumes as NIfTI files')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Attention-Enhanced Model Inference")
    print("=" * 60)
    
    # Load model
    model = load_model(args.checkpoint, args.base_channels, args.device)
    
    # Create dataset
    print(f"\nLoading dataset from: {args.data_path}")
    dataset = PatientDRRDataset(
        data_path=args.data_path,
        target_xray_size=512,
        target_volume_size=(64, 64, 64),
        max_patients=args.num_patients
    )
    
    print(f"Found {len(dataset)} patients")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run inference
    print("\n" + "=" * 60)
    print("Running Inference")
    print("=" * 60)
    
    all_psnr, all_ssim = run_inference(
        model, dataloader, args.output_dir,
        save_volumes=args.save_volumes,
        device=args.device
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Average PSNR: {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
    print(f"Average SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    print(f"Median PSNR: {np.median(all_psnr):.2f} dB")
    print(f"Median SSIM: {np.median(all_ssim):.4f}")
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Attention-Enhanced Model Inference Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Number of patients: {len(dataset)}\n\n")
        f.write(f"Average PSNR: {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB\n")
        f.write(f"Average SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}\n")
        f.write(f"Median PSNR: {np.median(all_psnr):.2f} dB\n")
        f.write(f"Median SSIM: {np.median(all_ssim):.4f}\n\n")
        f.write("Per-patient results:\n")
        for i, (psnr, ssim) in enumerate(zip(all_psnr, all_ssim)):
            f.write(f"Patient {i:03d}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}\n")
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Metrics saved to: {metrics_file}")


if __name__ == '__main__':
    main()
