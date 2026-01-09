"""
Inference script for Coordinate-Based Transformer
Supports querying at arbitrary resolutions and custom point sets
"""

import torch
import numpy as np
import nibabel as nib
import json
import argparse
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coord_transformer.model_coord_transformer import CoordinateTransformer
from utils.dataset import PatientDRRDataset


def normalize_volume(volume):
    """Normalize volume to [0, 1]"""
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        return (volume - vmin) / (vmax - vmin)
    return volume


def compute_metrics(pred, target):
    """Compute reconstruction metrics"""
    mse = np.mean((pred - target) ** 2)
    
    if mse > 0:
        psnr = 20 * np.log10(2.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    mae = np.mean(np.abs(pred - target))
    
    return {'mse': mse, 'psnr': psnr, 'mae': mae}


def visualize_slices(volume, save_path, title=""):
    """Save middle slices as image"""
    import matplotlib.pyplot as plt
    
    D, H, W = volume.shape
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(volume[D//2, :, :], cmap='gray')
    axes[0].set_title(f'{title} - Axial')
    axes[0].axis('off')
    
    axes[1].imshow(volume[:, H//2, :], cmap='gray')
    axes[1].set_title(f'{title} - Coronal')
    axes[1].axis('off')
    
    axes[2].imshow(volume[:, :, W//2], cmap='gray')
    axes[2].set_title(f'{title} - Sagittal')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def inference_single_patient(model, xrays, device, output_resolution=None):
    """Run inference on a single patient"""
    model.eval()
    
    xrays = xrays.unsqueeze(0).to(device)
    
    with torch.no_grad():
        volume = model(xrays, query_resolution=output_resolution)
    
    volume = volume.squeeze(0).squeeze(0).cpu().numpy()
    
    return volume


def batch_inference(model, dataset, device, output_dir,
                   base_resolution=(64, 64, 64),
                   output_resolution=None,
                   num_samples=None,
                   save_nifti=True,
                   save_visualizations=True):
    """Run inference on multiple patients"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if save_nifti:
        nifti_dir = output_dir / "nifti"
        nifti_dir.mkdir(exist_ok=True)
    
    if save_visualizations:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
    
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    print(f"\nRunning inference on {num_samples} patients...")
    print(f"Base resolution (training): {base_resolution}")
    print(f"Output resolution: {output_resolution if output_resolution else base_resolution}")
    
    metrics_list = []
    
    for idx in tqdm(range(num_samples), desc="Inference"):
        batch_data = dataset[idx]
        
        if isinstance(batch_data, dict):
            xrays = batch_data['drr_stacked']
            target = batch_data['ct_volume'].squeeze(0).numpy()
        else:
            target, xrays = batch_data
            target = target.squeeze(0).numpy()
        
        pred_volume = inference_single_patient(
            model, xrays, device, output_resolution=output_resolution
        )
        
        # Resize target if needed
        if output_resolution and output_resolution != base_resolution:
            import torch.nn.functional as F
            target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float()
            target_resized = F.interpolate(
                target_tensor, size=output_resolution,
                mode='trilinear', align_corners=False
            )
            target = target_resized.squeeze(0).squeeze(0).numpy()
        
        # Normalize
        pred_volume = normalize_volume(pred_volume)
        target = normalize_volume(target)
        
        # Compute metrics
        metrics = compute_metrics(pred_volume, target)
        metrics['patient_id'] = idx
        metrics_list.append(metrics)
        
        # Save NIfTI
        if save_nifti:
            pred_nifti = nib.Nifti1Image(pred_volume, affine=np.eye(4))
            target_nifti = nib.Nifti1Image(target, affine=np.eye(4))
            nib.save(pred_nifti, nifti_dir / f"patient_{idx:03d}_pred.nii.gz")
            nib.save(target_nifti, nifti_dir / f"patient_{idx:03d}_target.nii.gz")
        
        # Save visualizations
        if save_visualizations:
            visualize_slices(pred_volume, vis_dir / f"patient_{idx:03d}_pred.png",
                           title=f"Patient {idx} - Predicted")
            visualize_slices(target, vis_dir / f"patient_{idx:03d}_target.png",
                           title=f"Patient {idx} - Ground Truth")
    
    # Average metrics
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in metrics_list]),
        'psnr': np.mean([m['psnr'] for m in metrics_list if m['psnr'] != float('inf')]),
        'mae': np.mean([m['mae'] for m in metrics_list])
    }
    
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)
    print(f"Average MSE:  {avg_metrics['mse']:.6f}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"Average MAE:  {avg_metrics['mae']:.6f}")
    print("=" * 80)
    
    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump({'average': avg_metrics, 'per_patient': metrics_list}, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    return avg_metrics, metrics_list


def main():
    parser = argparse.ArgumentParser(description='Coordinate Transformer Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, 
                       default='coord_transformer/config_coord_transformer.json',
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, 
                       default='inference_results_coord_transformer',
                       help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to process')
    parser.add_argument('--output_resolution', type=str, default=None,
                       help='Output resolution as D,H,W (e.g., 96,96,96)')
    parser.add_argument('--no_nifti', action='store_true',
                       help='Skip saving NIfTI files')
    parser.add_argument('--no_vis', action='store_true',
                       help='Skip saving visualizations')
    
    args = parser.parse_args()
    
    if args.output_resolution:
        output_resolution = tuple(map(int, args.output_resolution.split(',')))
    else:
        output_resolution = None
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("\nLoading model...")
    model = CoordinateTransformer(
        volume_size=tuple(config['model']['volume_size']),
        xray_img_size=config['model']['xray_img_size'],
        xray_patch_size=config['model']['xray_patch_size'],
        xray_embed_dim=config['model']['xray_embed_dim'],
        xray_depth=config['model']['xray_depth'],
        coord_embed_dim=config['model']['coord_embed_dim'],
        num_transformer_blocks=config['model']['num_transformer_blocks'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout']
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_volume_size=tuple(config['model']['volume_size']),
        max_patients=config['data']['max_patients'],
        validate_alignment=False
    )
    
    # Run inference
    avg_metrics, metrics_list = batch_inference(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=args.output_dir,
        base_resolution=tuple(config['model']['volume_size']),
        output_resolution=output_resolution,
        num_samples=args.num_samples,
        save_nifti=not args.no_nifti,
        save_visualizations=not args.no_vis
    )
    
    print("\n✓ Inference complete!")
    
    if output_resolution and output_resolution != tuple(config['model']['volume_size']):
        print("\n" + "=" * 80)
        print("🎯 MULTI-RESOLUTION INFERENCE with Coordinate-Based Transformer!")
        print(f"   Trained at: {tuple(config['model']['volume_size'])}")
        print(f"   Inferred at: {output_resolution}")
        print("=" * 80)


if __name__ == "__main__":
    main()
