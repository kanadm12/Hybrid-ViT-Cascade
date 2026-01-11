"""
Inference Script for Progressive Cascade CT Reconstruction
- Load trained model checkpoint
- Process input X-ray images (PA + Lateral)
- Generate full 256³ CT volume
- Visualize 3-view (Axial, Sagittal, Coronal)
- Save as .nii.gz file
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import nibabel as nib
from PIL import Image

from model_progressive import ProgressiveCascadeModel


def load_checkpoint(checkpoint_path, config, device='cuda'):
    """Load trained model from checkpoint"""
    model = ProgressiveCascadeModel(
        xray_img_size=config['model']['xray_img_size'],
        xray_feature_dim=config['model']['xray_feature_dim'],
        voxel_dim=config['model']['voxel_dim'],
        use_gradient_checkpointing=False  # Inference mode
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"✓ Loaded checkpoint: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'best_psnr' in checkpoint:
        print(f"  Best PSNR: {checkpoint['best_psnr']:.2f} dB")
    
    return model


def load_xray_images(pa_path, lat_path, target_size=512):
    """
    Load and preprocess X-ray images
    Args:
        pa_path: Path to PA (frontal) X-ray
        lat_path: Path to lateral X-ray
        target_size: Resize to this size
    Returns:
        xrays: Tensor (1, 2, 1, H, W) ready for model input
    """
    # Load images
    pa_img = Image.open(pa_path).convert('L')
    lat_img = Image.open(lat_path).convert('L')
    
    # Resize
    pa_img = pa_img.resize((target_size, target_size), Image.BILINEAR)
    lat_img = lat_img.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to numpy and normalize to [0, 1]
    pa_array = np.array(pa_img, dtype=np.float32) / 255.0
    lat_array = np.array(lat_img, dtype=np.float32) / 255.0
    
    # Stack as (num_views, 1, H, W)
    xrays = np.stack([pa_array, lat_array], axis=0)  # (2, H, W)
    xrays = xrays[:, np.newaxis, :, :]  # (2, 1, H, W)
    
    # Add batch dimension and convert to tensor
    xrays = torch.from_numpy(xrays).unsqueeze(0)  # (1, 2, 1, H, W)
    
    return xrays


def run_inference(model, xrays, stage=3, device='cuda'):
    """
    Run inference to generate CT volume
    Args:
        model: Trained ProgressiveCascadeModel
        xrays: Input X-rays (1, 2, 1, H, W)
        stage: Maximum stage to compute (1=64³, 2=128³, 3=256³)
        device: cuda or cpu
    Returns:
        ct_volume: Generated CT volume (1, 1, D, H, W)
    """
    xrays = xrays.to(device)
    
    with torch.no_grad():
        ct_volume = model(xrays, max_stage=stage)
    
    return ct_volume


def visualize_3view(ct_volume, save_path=None, title="CT Volume - 3 Views"):
    """
    Visualize CT volume in 3 orthogonal views
    Args:
        ct_volume: Numpy array (D, H, W) or torch tensor (1, 1, D, H, W)
        save_path: If provided, save figure to this path
        title: Figure title
    """
    # Convert to numpy if tensor
    if torch.is_tensor(ct_volume):
        ct_volume = ct_volume.squeeze().cpu().numpy()
    
    D, H, W = ct_volume.shape
    
    # Get middle slices
    mid_d = D // 2
    mid_h = H // 2
    mid_w = W // 2
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Axial view (top-down)
    axes[0].imshow(ct_volume[mid_d, :, :], cmap='gray')
    axes[0].set_title(f'Axial (Slice {mid_d}/{D})')
    axes[0].axis('off')
    
    # Sagittal view (side)
    axes[1].imshow(ct_volume[:, mid_h, :], cmap='gray')
    axes[1].set_title(f'Sagittal (Slice {mid_h}/{H})')
    axes[1].axis('off')
    
    # Coronal view (front)
    axes[2].imshow(ct_volume[:, :, mid_w], cmap='gray')
    axes[2].set_title(f'Coronal (Slice {mid_w}/{W})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: {save_path}")
    
    plt.show()


def save_as_nifti(ct_volume, save_path, affine=None):
    """
    Save CT volume as NIfTI file (.nii.gz)
    Args:
        ct_volume: Numpy array (D, H, W) or torch tensor (1, 1, D, H, W)
        save_path: Output path (should end with .nii.gz)
        affine: 4x4 affine matrix (if None, uses identity)
    """
    # Convert to numpy if tensor
    if torch.is_tensor(ct_volume):
        ct_volume = ct_volume.squeeze().cpu().numpy()
    
    # Ensure correct dtype
    ct_volume = ct_volume.astype(np.float32)
    
    # Create affine matrix if not provided
    if affine is None:
        affine = np.eye(4)
        # Set voxel spacing (e.g., 1mm isotropic)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(ct_volume, affine)
    
    # Save
    nib.save(nifti_img, save_path)
    print(f"✓ Saved NIfTI: {save_path}")
    print(f"  Shape: {ct_volume.shape}")
    print(f"  Data type: {ct_volume.dtype}")
    print(f"  Value range: [{ct_volume.min():.3f}, {ct_volume.max():.3f}]")


def denormalize_ct(ct_volume, hu_min=-1024, hu_max=3071):
    """
    Convert normalized CT values back to HU
    Args:
        ct_volume: Normalized volume (typically [0, 1] or [-1, 1])
        hu_min: Minimum HU value
        hu_max: Maximum HU value
    Returns:
        ct_hu: CT volume in Hounsfield Units
    """
    if torch.is_tensor(ct_volume):
        ct_volume = ct_volume.cpu().numpy()
    
    # Assume input is normalized to [0, 1]
    ct_hu = ct_volume * (hu_max - hu_min) + hu_min
    
    return ct_hu


def main():
    parser = argparse.ArgumentParser(description='Progressive Cascade CT Reconstruction Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--pa_xray', type=str, required=True,
                        help='Path to PA (frontal) X-ray image')
    parser.add_argument('--lat_xray', type=str, required=True,
                        help='Path to lateral X-ray image')
    parser.add_argument('--output_dir', type=str, default='inference_output',
                        help='Directory to save outputs')
    parser.add_argument('--config', type=str, default='config_progressive.json',
                        help='Path to model config JSON')
    parser.add_argument('--stage', type=int, default=3, choices=[1, 2, 3],
                        help='Maximum stage to run (1=64³, 2=128³, 3=256³)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output_name', type=str, default='reconstructed_ct',
                        help='Output filename (without extension)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Progressive Cascade CT Reconstruction - Inference")
    print("="*60)
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    print(f"✓ Loaded config: {args.config}")
    
    # Load model
    model = load_checkpoint(args.checkpoint, config, device=args.device)
    
    # Load X-ray images
    print(f"\nLoading X-ray images...")
    print(f"  PA: {args.pa_xray}")
    print(f"  Lateral: {args.lat_xray}")
    xrays = load_xray_images(args.pa_xray, args.lat_xray, 
                             target_size=config['model']['xray_img_size'])
    print(f"✓ X-rays loaded: {xrays.shape}")
    
    # Run inference
    print(f"\nRunning inference (Stage {args.stage})...")
    ct_volume = run_inference(model, xrays, stage=args.stage, device=args.device)
    print(f"✓ Generated CT volume: {ct_volume.shape}")
    
    # Resolution info
    resolutions = {1: '64³', 2: '128³', 3: '256³'}
    print(f"  Resolution: {resolutions[args.stage]}")
    
    # Visualize
    print(f"\nGenerating 3-view visualization...")
    vis_path = output_dir / f"{args.output_name}_3view.png"
    visualize_3view(ct_volume, save_path=vis_path, 
                    title=f"Reconstructed CT - Stage {args.stage} ({resolutions[args.stage]})")
    
    # Save as NIfTI
    print(f"\nSaving as NIfTI...")
    nifti_path = output_dir / f"{args.output_name}.nii.gz"
    
    # Denormalize to HU values (optional)
    ct_hu = denormalize_ct(ct_volume.squeeze())
    save_as_nifti(ct_hu, nifti_path)
    
    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"  - Visualization: {vis_path.name}")
    print(f"  - NIfTI file: {nifti_path.name}")
    print("\nYou can view the .nii.gz file in:")
    print("  - 3D Slicer: https://www.slicer.org/")
    print("  - ITK-SNAP: http://www.itksnap.org/")
    print("  - MITK Workbench: https://www.mitk.org/")


if __name__ == '__main__':
    main()
