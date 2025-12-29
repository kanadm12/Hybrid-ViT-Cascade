"""
Simple inference script to generate CT from X-rays using stage3 checkpoint
"""

import torch
import json
from pathlib import Path
from inference import load_model, reconstruct_volume
from utils.dataset import PatientDRRDataset
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def save_orthogonal_views(volume, output_path, title="CT Volume"):
    """
    Save orthogonal views (axial, coronal, sagittal) of a 3D volume
    
    Args:
        volume: 3D numpy array (D, H, W)
        output_path: Path to save the figure
        title: Title for the figure
    """
    D, H, W = volume.shape
    
    # Get middle slices
    axial_slice = volume[D//2, :, :]        # XY plane (top view)
    coronal_slice = volume[:, H//2, :]      # XZ plane (front view)
    sagittal_slice = volume[:, :, W//2]     # YZ plane (side view)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial view (XY plane)
    axes[0].imshow(axial_slice, cmap='gray', origin='lower', vmin=-1, vmax=1)
    axes[0].set_title(f'{title}\nAxial View (XY plane)')
    axes[0].axis('off')
    
    # Coronal view (XZ plane)
    axes[1].imshow(coronal_slice, cmap='gray', origin='lower', vmin=-1, vmax=1)
    axes[1].set_title(f'{title}\nCoronal View (XZ plane)')
    axes[1].axis('off')
    
    # Sagittal view (YZ plane)
    axes[2].imshow(sagittal_slice, cmap='gray', origin='lower', vmin=-1, vmax=1)
    axes[2].set_title(f'{title}\nSagittal View (YZ plane)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved orthogonal views: {output_path}")

def main():
    # Configuration
    config_path = "config/runpod_config.json"
    checkpoint_path = "checkpoints/stage3_best.pt"  # Change to stage2_best.pt if stage3 isn't done
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path, config, device)
    
    # Load test dataset
    print("\nLoading test dataset...")
    dataset = PatientDRRDataset(
        data_path="/workspace/drr_patient_data",  # Change if running locally
        target_xray_size=config['xray_config']['img_size'],
        target_volume_size=tuple(config['stage_configs'][-1]['volume_size']),  # Stage 3 size
        normalize_range=(-1, 1)
    )
    
    print(f"Found {len(dataset)} test samples")
    
    # Select a sample (change index to test different samples)
    sample_idx = 0
    sample = dataset[sample_idx]
    
    print(f"\nGenerating CT for sample {sample_idx}")
    print(f"Patient ID: {sample['patient_id']}")
    
    # Prepare input
    xrays = sample['drr_stacked'].unsqueeze(0)  # Add batch dimension
    gt_volume = sample['ct_volume'].unsqueeze(0)
    
    # Run inference through all stages
    print("\n" + "="*60)
    print("STAGE 1: Coarse Reconstruction (64³)")
    print("="*60)
    volume_stage1 = reconstruct_volume(
        model, xrays, 
        stage_name='stage1', 
        num_steps=50,  # Increase for better quality
        device=device
    )
    
    print("\n" + "="*60)
    print("STAGE 2: Medium Resolution (128³)")
    print("="*60)
    volume_stage2 = reconstruct_volume(
        model, xrays,
        stage_name='stage2',
        num_steps=50,
        device=device
    )
    
    print("\n" + "="*60)
    print("STAGE 3: High Resolution (256³)")
    print("="*60)
    volume_stage3 = reconstruct_volume(
        model, xrays,
        stage_name='stage3',
        num_steps=50,
        device=device
    )
    
    # Clamp to valid range [-1, 1]
    volume_stage1 = torch.clamp(volume_stage1, -1, 1)
    volume_stage2 = torch.clamp(volume_stage2, -1, 1)
    volume_stage3 = torch.clamp(volume_stage3, -1, 1)
    
    # Save results
    output_dir = Path("inference_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving results to {output_dir}/")
    
    # Convert to numpy and save as NIfTI
    stage1_np = volume_stage1[0, 0].cpu().numpy()
    stage2_np = volume_stage2[0, 0].cpu().numpy()
    stage3_np = volume_stage3[0, 0].cpu().numpy()
    gt_np = gt_volume[0, 0].cpu().numpy()
    
    # Save as NIfTI files
    nib.save(nib.Nifti1Image(stage1_np, np.eye(4)), output_dir / f"sample_{sample_idx}_stage1.nii.gz")
    nib.save(nib.Nifti1Image(stage2_np, np.eye(4)), output_dir / f"sample_{sample_idx}_stage2.nii.gz")
    nib.save(nib.Nifti1Image(stage3_np, np.eye(4)), output_dir / f"sample_{sample_idx}_stage3.nii.gz")
    nib.save(nib.Nifti1Image(gt_np, np.eye(4)), output_dir / f"sample_{sample_idx}_ground_truth.nii.gz")
    
    # Also save as numpy for quick loading
    np.save(output_dir / f"sample_{sample_idx}_stage1.npy", stage1_np)
    np.save(output_dir / f"sample_{sample_idx}_stage2.npy", stage2_np)
    np.save(output_dir / f"sample_{sample_idx}_stage3.npy", stage3_np)
    np.save(output_dir / f"sample_{sample_idx}_ground_truth.npy", gt_np)
    
    # Save orthogonal views (axial, coronal, sagittal)
    print("\nGenerating orthogonal views...")
    
    # Print value statistics for debugging
    print(f"\nValue ranges:")
    print(f"  Stage 1: [{stage1_np.min():.3f}, {stage1_np.max():.3f}], mean={stage1_np.mean():.3f}")
    print(f"  Stage 2: [{stage2_np.min():.3f}, {stage2_np.max():.3f}], mean={stage2_np.mean():.3f}")
    print(f"  Stage 3: [{stage3_np.min():.3f}, {stage3_np.max():.3f}], mean={stage3_np.mean():.3f}")
    print(f"  Ground Truth: [{gt_np.min():.3f}, {gt_np.max():.3f}], mean={gt_np.mean():.3f}")
    
    save_orthogonal_views(stage1_np, output_dir / f"sample_{sample_idx}_stage1_views.png", "Stage 1 (64³)")
    save_orthogonal_views(stage2_np, output_dir / f"sample_{sample_idx}_stage2_views.png", "Stage 2 (128³)")
    save_orthogonal_views(stage3_np, output_dir / f"sample_{sample_idx}_stage3_views.png", "Stage 3 (256³)")
    save_orthogonal_views(gt_np, output_dir / f"sample_{sample_idx}_ground_truth_views.png", "Ground Truth")
    
    # Save input X-rays
    xray_frontal = xrays[0, 0, 0].cpu().numpy()
    xray_lateral = xrays[0, 1, 0].cpu().numpy()
    np.save(output_dir / f"sample_{sample_idx}_xray_frontal.npy", xray_frontal)
    np.save(output_dir / f"sample_{sample_idx}_xray_lateral.npy", xray_lateral)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("METRICS")
    print("="*60)
    
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)  # Range is [-1, 1] = 2.0
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    
    # Stage 1 metrics
    psnr1 = psnr_metric(volume_stage1, gt_volume)
    ssim1 = ssim_metric(volume_stage1.unsqueeze(1), gt_volume.unsqueeze(1))
    print(f"Stage 1: PSNR = {psnr1:.2f} dB, SSIM = {ssim1:.4f}")
    
    # Stage 2 metrics
    psnr2 = psnr_metric(volume_stage2, gt_volume)
    ssim2 = ssim_metric(volume_stage2.unsqueeze(1), gt_volume.unsqueeze(1))
    print(f"Stage 2: PSNR = {psnr2:.2f} dB, SSIM = {ssim2:.4f}")
    
    # Stage 3 metrics
    psnr3 = psnr_metric(volume_stage3, gt_volume)
    ssim3 = ssim_metric(volume_stage3.unsqueeze(1), gt_volume.unsqueeze(1))
    print(f"Stage 3: PSNR = {psnr3:.2f} dB, SSIM = {ssim3:.4f}")
    
    print(f"\n✅ Inference complete! Results saved to {output_dir}/")
    print(f"\nNIfTI Volumes (3D):")
    print(f"   - Stage 1 (64³): {output_dir}/sample_{sample_idx}_stage1.nii.gz")
    print(f"   - Stage 2 (128³): {output_dir}/sample_{sample_idx}_stage2.nii.gz")
    print(f"   - Stage 3 (256³): {output_dir}/sample_{sample_idx}_stage3.nii.gz")
    print(f"   - Ground Truth: {output_dir}/sample_{sample_idx}_ground_truth.nii.gz")
    print(f"\nOrthogonal Views (Axial/Coronal/Sagittal):")
    print(f"   - Stage 1 views: {output_dir}/sample_{sample_idx}_stage1_views.png")
    print(f"   - Stage 2 views: {output_dir}/sample_{sample_idx}_stage2_views.png")
    print(f"   - Stage 3 views: {output_dir}/sample_{sample_idx}_stage3_views.png")
    print(f"   - Ground truth views: {output_dir}/sample_{sample_idx}_ground_truth_views.png")

if __name__ == "__main__":
    main()
