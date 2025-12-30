"""
Inference Script for Direct Regression CT Reconstruction
Loads trained model and generates CT volumes from X-ray pairs
"""
import torch
import torch.nn.functional as F
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, '..')

from model_direct import DirectCTRegression
from utils.dataset import PatientDRRDataset


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', None)
    if config is None:
        # Default config if not saved in checkpoint
        config = {
            'volume_size': (64, 64, 64),
            'xray_img_size': 512,
            'voxel_dim': 256,
            'vit_depth': 4,
            'num_heads': 4,
            'xray_feature_dim': 512
        }
    
    # Create model
    model = DirectCTRegression(
        volume_size=tuple(config['volume_size']),
        xray_img_size=config['xray_img_size'],
        voxel_dim=config['voxel_dim'],
        vit_depth=config['vit_depth'],
        num_heads=config['num_heads'],
        xray_feature_dim=config['xray_feature_dim']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'psnr' in checkpoint:
        print(f"Best PSNR: {checkpoint['psnr']:.2f} dB")
    
    return model, config


def compute_metrics(pred, target):
    """Compute evaluation metrics"""
    # PSNR
    mse = F.mse_loss(pred, target)
    data_range = target.max() - target.min()
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    
    # MAE
    mae = F.l1_loss(pred, target)
    
    # SSIM (simplified)
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_std = pred.std()
    target_std = target.std()
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim = ((2 * pred_mean * target_mean + C1) * (2 * covariance + C2)) / \
           ((pred_mean**2 + target_mean**2 + C1) * (pred_std**2 + target_std**2 + C2))
    
    return {
        'psnr': psnr.item(),
        'mae': mae.item(),
        'ssim': ssim.item()
    }


def visualize_results(xrays, predicted, target, metrics, save_path):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(20, 10))
    
    B, num_views, C, H, W = xrays.shape
    D, H_vol, W_vol = predicted.shape[2:]
    
    # Input X-rays
    ax1 = plt.subplot(3, 6, 1)
    ax1.imshow(xrays[0, 0, 0].cpu().numpy(), cmap='gray')
    ax1.set_title('Input X-ray (AP)')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 6, 2)
    ax2.imshow(xrays[0, 1, 0].cpu().numpy(), cmap='gray')
    ax2.set_title('Input X-ray (Lateral)')
    ax2.axis('off')
    
    # Predicted CT - Axial slices at different depths
    ax3 = plt.subplot(3, 6, 3)
    pred_axial_1 = predicted[0, 0, D//4].cpu().numpy()
    im3 = ax3.imshow(pred_axial_1, cmap='gray', vmin=-1, vmax=1)
    ax3.set_title(f'Predicted (Axial D={D//4})')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = plt.subplot(3, 6, 4)
    pred_axial_2 = predicted[0, 0, D//2].cpu().numpy()
    im4 = ax4.imshow(pred_axial_2, cmap='gray', vmin=-1, vmax=1)
    ax4.set_title(f'Predicted (Axial D={D//2})')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    ax5 = plt.subplot(3, 6, 5)
    pred_axial_3 = predicted[0, 0, 3*D//4].cpu().numpy()
    im5 = ax5.imshow(pred_axial_3, cmap='gray', vmin=-1, vmax=1)
    ax5.set_title(f'Predicted (Axial D={3*D//4})')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # Predicted CT - Other views
    ax6 = plt.subplot(3, 6, 6)
    pred_sagittal = predicted[0, 0, :, H_vol//2, :].cpu().numpy()
    im6 = ax6.imshow(pred_sagittal, cmap='gray', vmin=-1, vmax=1)
    ax6.set_title('Predicted (Sagittal)')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Target CT - Axial slices
    if target is not None:
        ax7 = plt.subplot(3, 6, 9)
        target_axial_1 = target[0, 0, D//4].cpu().numpy()
        im7 = ax7.imshow(target_axial_1, cmap='gray', vmin=-1, vmax=1)
        ax7.set_title(f'Target (Axial D={D//4})')
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046)
        
        ax8 = plt.subplot(3, 6, 10)
        target_axial_2 = target[0, 0, D//2].cpu().numpy()
        im8 = ax8.imshow(target_axial_2, cmap='gray', vmin=-1, vmax=1)
        ax8.set_title(f'Target (Axial D={D//2})')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046)
        
        ax9 = plt.subplot(3, 6, 11)
        target_axial_3 = target[0, 0, 3*D//4].cpu().numpy()
        im9 = ax9.imshow(target_axial_3, cmap='gray', vmin=-1, vmax=1)
        ax9.set_title(f'Target (Axial D={3*D//4})')
        ax9.axis('off')
        plt.colorbar(im9, ax=ax9, fraction=0.046)
        
        ax10 = plt.subplot(3, 6, 12)
        target_sagittal = target[0, 0, :, H_vol//2, :].cpu().numpy()
        im10 = ax10.imshow(target_sagittal, cmap='gray', vmin=-1, vmax=1)
        ax10.set_title('Target (Sagittal)')
        ax10.axis('off')
        plt.colorbar(im10, ax=ax10, fraction=0.046)
        
        # Error maps
        error = torch.abs(predicted - target)
        
        ax11 = plt.subplot(3, 6, 15)
        error_axial_1 = error[0, 0, D//4].cpu().numpy()
        im11 = ax11.imshow(error_axial_1, cmap='hot', vmin=0, vmax=0.5)
        ax11.set_title(f'Error (Axial D={D//4})')
        ax11.axis('off')
        plt.colorbar(im11, ax=ax11, fraction=0.046)
        
        ax12 = plt.subplot(3, 6, 16)
        error_axial_2 = error[0, 0, D//2].cpu().numpy()
        im12 = ax12.imshow(error_axial_2, cmap='hot', vmin=0, vmax=0.5)
        ax12.set_title(f'Error (Axial D={D//2})')
        ax12.axis('off')
        plt.colorbar(im12, ax=ax12, fraction=0.046)
        
        ax13 = plt.subplot(3, 6, 17)
        error_axial_3 = error[0, 0, 3*D//4].cpu().numpy()
        im13 = ax13.imshow(error_axial_3, cmap='hot', vmin=0, vmax=0.5)
        ax13.set_title(f'Error (Axial D={3*D//4})')
        ax13.axis('off')
        plt.colorbar(im13, ax=ax13, fraction=0.046)
        
        ax14 = plt.subplot(3, 6, 18)
        error_sagittal = error[0, 0, :, H_vol//2, :].cpu().numpy()
        im14 = ax14.imshow(error_sagittal, cmap='hot', vmin=0, vmax=0.5)
        ax14.set_title('Error (Sagittal)')
        ax14.axis('off')
        plt.colorbar(im14, ax=ax14, fraction=0.046)
    
    # Predicted coronal
    ax7_alt = plt.subplot(3, 6, 7)
    pred_coronal = predicted[0, 0, :, :, W_vol//2].cpu().numpy()
    im7_alt = ax7_alt.imshow(pred_coronal, cmap='gray', vmin=-1, vmax=1)
    ax7_alt.set_title('Predicted (Coronal)')
    ax7_alt.axis('off')
    plt.colorbar(im7_alt, ax=ax7_alt, fraction=0.046)
    
    # 3D projection (MIP - Maximum Intensity Projection)
    ax8_alt = plt.subplot(3, 6, 8)
    mip_frontal = predicted[0, 0].max(dim=0)[0].cpu().numpy()
    im8_alt = ax8_alt.imshow(mip_frontal, cmap='gray')
    ax8_alt.set_title('MIP (Frontal)')
    ax8_alt.axis('off')
    plt.colorbar(im8_alt, ax=ax8_alt, fraction=0.046)
    
    # Title with metrics
    if metrics:
        title = f"Direct Regression Inference - PSNR: {metrics['psnr']:.2f} dB | " \
                f"MAE: {metrics['mae']:.4f} | SSIM: {metrics['ssim']:.3f}"
    else:
        title = "Direct Regression Inference"
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved visualization to {save_path}")


def save_volume_nifti(volume, save_path):
    """Save volume as NIfTI file (requires nibabel)"""
    try:
        import nibabel as nib
        volume_np = volume[0, 0].cpu().numpy()  # (D, H, W)
        nifti_img = nib.Nifti1Image(volume_np, affine=np.eye(4))
        nib.save(nifti_img, save_path)
        print(f"  ✓ Saved NIfTI volume to {save_path}")
    except ImportError:
        print("  ! nibabel not installed, skipping NIfTI save")
        print("    Install with: pip install nibabel")


def run_inference(model, dataloader, device, output_dir, max_samples=None):
    """Run inference on dataset"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    all_metrics = []
    
    print("\n=== Running Inference ===")
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if max_samples and idx >= max_samples:
                break
            
            # Get data
            if isinstance(batch, dict):
                xrays = batch['drr_stacked'].to(device)
                target = batch['ct_volume'].to(device) if 'ct_volume' in batch else None
            else:
                target, xrays = batch
                xrays = xrays.to(device)
                target = target.to(device) if target is not None else None
            
            # Forward pass
            predicted = model(xrays)
            
            # Compute metrics if target available
            metrics = None
            if target is not None:
                metrics = compute_metrics(predicted, target)
                all_metrics.append(metrics)
                print(f"\nSample {idx + 1}:")
                print(f"  PSNR: {metrics['psnr']:.2f} dB")
                print(f"  MAE: {metrics['mae']:.4f}")
                print(f"  SSIM: {metrics['ssim']:.3f}")
            else:
                print(f"\nSample {idx + 1}: No ground truth")
            
            # Save visualizations
            vis_path = output_dir / f'sample_{idx+1:03d}_visualization.png'
            visualize_results(xrays, predicted, target, metrics, vis_path)
            
            # Save predicted volume as numpy
            np_path = output_dir / f'sample_{idx+1:03d}_predicted.npy'
            np.save(np_path, predicted[0, 0].cpu().numpy())
            print(f"  ✓ Saved numpy array to {np_path}")
            
            # Save as NIfTI if available
            nifti_path = output_dir / f'sample_{idx+1:03d}_predicted.nii.gz'
            save_volume_nifti(predicted, nifti_path)
    
    # Print average metrics
    if all_metrics:
        avg_metrics = {
            'psnr': np.mean([m['psnr'] for m in all_metrics]),
            'mae': np.mean([m['mae'] for m in all_metrics]),
            'ssim': np.mean([m['ssim'] for m in all_metrics])
        }
        
        print("\n" + "="*60)
        print("Average Metrics:")
        print(f"  PSNR: {avg_metrics['psnr']:.2f} dB")
        print(f"  MAE: {avg_metrics['mae']:.4f}")
        print(f"  SSIM: {avg_metrics['ssim']:.3f}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Direct Regression CT Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='/workspace/drr_patient_data',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (recommended: 1 for inference)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # Create dataset
    print(f"\nLoading {args.split} dataset from {args.data_dir}")
    dataset = PatientDRRDataset(
        data_path=args.data_dir,
        target_xray_size=512,
        target_volume_size=(64, 64, 64),
        max_patients=None  # Load all available
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Found {len(dataset)} samples")
    
    # Run inference
    run_inference(model, dataloader, device, args.output_dir, args.max_samples)
    
    print(f"\n✓ Inference complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
