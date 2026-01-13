"""
Inference script for Direct128 H200 Model
Loads trained checkpoint and generates 512³ CT volume from DRR pair
Saves volume + 3 orthogonal views (axial, sagittal, coronal)
"""

import argparse
import time
from pathlib import Path
import random

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from PIL import Image

from model_direct128_h200 import Direct128Model_H200
from dataset_simple import PatientDRRDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Direct128 H200 model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to DRR patient dataset root")
    parser.add_argument("--patient_id", type=str, default=None,
                        help="Specific patient ID to run inference on (e.g., 'patient_001'). If None, uses random patient")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="Directory to save inference results")
    parser.add_argument("--output_size", type=int, default=512,
                        help="Output CT volume size (default 512³)")
    return parser.parse_args()


def load_checkpoint(checkpoint_path, device):
    """Load trained model checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = Direct128Model_H200(
        xray_img_size=512,
        xray_feature_dim=512,
        num_rdb=5,
        use_checkpoint=False  # Disable checkpointing for inference
    )
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        val_psnr = checkpoint.get('val_psnr', 'N/A')
        val_ssim = checkpoint.get('val_ssim', 'N/A')
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        epoch = checkpoint.get('epoch', 'N/A')
        val_psnr = checkpoint.get('val_psnr', 'N/A')
        val_ssim = checkpoint.get('val_ssim', 'N/A')
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.get('epoch', 'N/A')
        val_psnr = checkpoint.get('psnr', checkpoint.get('val_psnr', 'N/A'))
        val_ssim = checkpoint.get('ssim', checkpoint.get('val_ssim', 'N/A'))
    else:
        # Assume checkpoint is just the state dict
        model.load_state_dict(checkpoint)
        epoch = 'N/A'
        val_psnr = 'N/A'
        val_ssim = 'N/A'
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Checkpoint loaded successfully")
    print(f"  - Epoch: {epoch}")
    if val_psnr != 'N/A':
        print(f"  - PSNR: {val_psnr:.2f} dB")
    if val_ssim != 'N/A':
        print(f"  - SSIM: {val_ssim:.4f}")
    
    return model


def load_patient_data(dataset_path, patient_id, split="val"):
    """Load DRRs and ground truth CT for a specific patient"""
    dataset = PatientDRRDataset(
        dataset_path=dataset_path,
        max_patients=None,
        split=split,
        ct_size=128,
        drr_size=512,
        vertical_flip=True
    )
    
    if patient_id is None:
        # Random patient
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        patient_id = dataset.patients[idx]
        print(f"Random patient selected: {patient_id}")
    else:
        # Specific patient
        if patient_id not in dataset.patients:
            raise ValueError(f"Patient {patient_id} not found in dataset")
        idx = dataset.patients.index(patient_id)
        sample = dataset[idx]
        print(f"Using specified patient: {patient_id}")
    
    return sample, patient_id


def upscale_volume(volume_128, target_size=512):
    """
    Upscale 128³ volume to target_size³ (default 512³)
    Uses trilinear interpolation for smooth upscaling
    
    Args:
        volume_128: (1, 1, 128, 128, 128) tensor
        target_size: target resolution (default 512)
    Returns:
        volume_512: (1, 1, target_size, target_size, target_size) tensor
    """
    print(f"Upscaling volume from 128³ to {target_size}³...")
    volume_upscaled = F.interpolate(
        volume_128,
        size=(target_size, target_size, target_size),
        mode='trilinear',
        align_corners=False
    )
    return volume_upscaled


def save_volume_nifti(volume, save_path):
    """Save volume as NIfTI file"""
    volume_np = volume.squeeze().cpu().numpy()  # (D, H, W)
    nifti_img = nib.Nifti1Image(volume_np, affine=np.eye(4))
    nib.save(nifti_img, save_path)
    print(f"✓ Saved volume: {save_path}")


def save_orthogonal_views(volume, output_dir, prefix=""):
    """
    Save 3 orthogonal views (axial, sagittal, coronal) as PNG images
    Takes middle slices from each view
    
    Args:
        volume: (D, H, W) numpy array
        output_dir: directory to save images
        prefix: filename prefix (e.g., "pred_" or "gt_")
    """
    D, H, W = volume.shape
    
    # Normalize to [0, 255] for visualization
    vmin, vmax = volume.min(), volume.max()
    volume_norm = ((volume - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
    
    # Axial view (horizontal slice, looking down from top)
    axial_slice = volume_norm[D // 2, :, :]
    axial_img = Image.fromarray(axial_slice, mode='L')
    axial_path = output_dir / f"{prefix}axial.png"
    axial_img.save(axial_path)
    print(f"✓ Saved axial view: {axial_path}")
    
    # Sagittal view (side view, looking from right)
    sagittal_slice = volume_norm[:, :, W // 2]
    sagittal_img = Image.fromarray(sagittal_slice, mode='L')
    sagittal_path = output_dir / f"{prefix}sagittal.png"
    sagittal_img.save(sagittal_path)
    print(f"✓ Saved sagittal view: {sagittal_path}")
    
    # Coronal view (front view, looking from front)
    coronal_slice = volume_norm[:, H // 2, :]
    coronal_img = Image.fromarray(coronal_slice, mode='L')
    coronal_path = output_dir / f"{prefix}coronal.png"
    coronal_img.save(coronal_path)
    print(f"✓ Saved coronal view: {coronal_path}")


def save_drr_images(drr_stacked, output_dir):
    """Save input DRR images"""
    drr_np = drr_stacked.squeeze().cpu().numpy()  # (2, 512, 512)
    
    for i, view_name in enumerate(['drr_ap', 'drr_lateral']):
        drr_slice = drr_np[i]
        vmin, vmax = drr_slice.min(), drr_slice.max()
        drr_norm = ((drr_slice - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
        drr_img = Image.fromarray(drr_norm, mode='L')
        drr_path = output_dir / f"{view_name}.png"
        drr_img.save(drr_path)
        print(f"✓ Saved DRR: {drr_path}")


def run_inference(args):
    """Main inference pipeline"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_checkpoint(args.checkpoint, device)
    
    # Load patient data
    sample, patient_id = load_patient_data(
        args.dataset_path,
        args.patient_id,
        split="val"
    )
    
    # Create patient-specific output directory
    patient_output_dir = output_dir / patient_id
    patient_output_dir.mkdir(exist_ok=True)
    
    # Prepare inputs
    drr_stacked = sample["drr_stacked"].unsqueeze(0).to(device)  # (1, 2, 1, 512, 512)
    ct_gt_128 = sample["ct_volume"]  # (1, 128, 128, 128)
    
    print(f"\n{'='*60}")
    print(f"Running inference on {patient_id}")
    print(f"{'='*60}")
    
    # Save input DRRs
    print("\n[1/5] Saving input DRR images...")
    save_drr_images(drr_stacked, patient_output_dir)
    
    # Run inference at 128³
    print("\n[2/5] Running model inference (128³)...")
    start_time = time.time()
    with torch.no_grad():
        pred_128 = model(drr_stacked)  # (1, 1, 128, 128, 128)
    inference_time = time.time() - start_time
    print(f"✓ Inference completed in {inference_time:.2f}s")
    
    # Upscale to target resolution
    print(f"\n[3/5] Upscaling to {args.output_size}³...")
    pred_upscaled = upscale_volume(pred_128, target_size=args.output_size)
    
    # Save predicted volume
    print(f"\n[4/5] Saving predicted volume...")
    pred_nifti_path = patient_output_dir / f"pred_{args.output_size}cubed.nii.gz"
    save_volume_nifti(pred_upscaled, pred_nifti_path)
    
    # Save orthogonal views
    print(f"\n[5/5] Generating orthogonal views...")
    pred_np = pred_upscaled.squeeze().cpu().numpy()
    save_orthogonal_views(pred_np, patient_output_dir, prefix="pred_")
    
    # Also save ground truth views (at 128³, then upscaled for fair comparison)
    print("\n[Bonus] Saving ground truth comparison...")
    ct_gt_128_tensor = ct_gt_128.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 128, 128, 128)
    ct_gt_upscaled = upscale_volume(ct_gt_128_tensor, target_size=args.output_size)
    
    gt_nifti_path = patient_output_dir / f"gt_{args.output_size}cubed.nii.gz"
    save_volume_nifti(ct_gt_upscaled, gt_nifti_path)
    
    gt_np = ct_gt_upscaled.squeeze().cpu().numpy()
    save_orthogonal_views(gt_np, patient_output_dir, prefix="gt_")
    
    # Compute metrics
    print(f"\n{'='*60}")
    print("Inference Summary")
    print(f"{'='*60}")
    print(f"Patient ID: {patient_id}")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Output resolution: {args.output_size}³")
    print(f"Output directory: {patient_output_dir}")
    print(f"\nGenerated files:")
    print(f"  - Predicted volume: pred_{args.output_size}cubed.nii.gz")
    print(f"  - Ground truth volume: gt_{args.output_size}cubed.nii.gz")
    print(f"  - Orthogonal views: pred_[axial|sagittal|coronal].png")
    print(f"  - Ground truth views: gt_[axial|sagittal|coronal].png")
    print(f"  - Input DRRs: drr_ap.png, drr_lateral.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
