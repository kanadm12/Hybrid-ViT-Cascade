"""
Visual DRR-CT Alignment Checker
Helps verify if DRRs are properly oriented relative to CT volumes
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from utils.dataset import PatientDRRDataset


def visualize_alignment(data: dict, save_path: str = None):
    """
    Visualize DRR-CT alignment
    Shows:
    - Input DRRs (frontal and lateral)
    - Synthetic DRRs generated from CT
    - Difference maps
    """
    drr_frontal = data['drr_frontal'].squeeze().cpu().numpy()
    drr_lateral = data['drr_lateral'].squeeze().cpu().numpy()
    ct_volume = data['ct_volume'].squeeze().cpu().numpy()
    patient_id = data['patient_id']
    
    # Generate synthetic DRRs from CT
    synth_frontal = np.max(ct_volume, axis=0)  # Max projection along depth
    synth_lateral = np.max(ct_volume, axis=2)  # Max projection along width
    
    # Normalize for visualization
    def normalize(img):
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        return img
    
    drr_frontal = normalize(drr_frontal)
    drr_lateral = normalize(drr_lateral)
    synth_frontal = normalize(synth_frontal)
    synth_lateral = normalize(synth_lateral)
    
    # Resize synthetic to match DRR size if needed
    if synth_frontal.shape != drr_frontal.shape:
        from scipy.ndimage import zoom
        zoom_factor = drr_frontal.shape[0] / synth_frontal.shape[0]
        synth_frontal = zoom(synth_frontal, zoom_factor, order=1)
        synth_lateral = zoom(synth_lateral, zoom_factor, order=1)
    
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    fig.suptitle(f'DRR-CT Alignment Check: {patient_id}', fontsize=16, fontweight='bold')
    
    # Row 1: Input DRRs
    axes[0, 0].imshow(drr_frontal, cmap='gray')
    axes[0, 0].set_title('Input Frontal/PA DRR', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(drr_lateral, cmap='gray')
    axes[0, 1].set_title('Input Lateral DRR', fontsize=12)
    axes[0, 1].axis('off')
    
    # Row 2: Synthetic DRRs from CT
    axes[1, 0].imshow(synth_frontal, cmap='gray')
    axes[1, 0].set_title('Synthetic Frontal from CT', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(synth_lateral, cmap='gray')
    axes[1, 1].set_title('Synthetic Lateral from CT', fontsize=12)
    axes[1, 1].axis('off')
    
    # Row 3: Difference maps
    diff_frontal = np.abs(drr_frontal - synth_frontal)
    diff_lateral = np.abs(drr_lateral - synth_lateral)
    
    im1 = axes[2, 0].imshow(diff_frontal, cmap='hot', vmin=0, vmax=1)
    axes[2, 0].set_title(f'Frontal Difference (MSE: {np.mean(diff_frontal**2):.4f})', fontsize=12)
    axes[2, 0].axis('off')
    plt.colorbar(im1, ax=axes[2, 0], fraction=0.046)
    
    im2 = axes[2, 1].imshow(diff_lateral, cmap='hot', vmin=0, vmax=1)
    axes[2, 1].set_title(f'Lateral Difference (MSE: {np.mean(diff_lateral**2):.4f})', fontsize=12)
    axes[2, 1].axis('off')
    plt.colorbar(im2, ax=axes[2, 1], fraction=0.046)
    
    # Add alignment status
    avg_error = (np.mean(diff_frontal**2) + np.mean(diff_lateral**2)) / 2
    status = "✓ ALIGNED" if avg_error < 0.5 else "✗ MISALIGNED"
    color = 'green' if avg_error < 0.5 else 'red'
    
    fig.text(0.5, 0.02, f'{status} (Average Error: {avg_error:.4f})', 
             ha='center', fontsize=14, fontweight='bold', color=color)
    
    # Add note about vertical flip
    note = ("Note: If images look similar but difference is high, they might be flipped.\n"
            "Try setting 'flip_drrs_vertical: true' in config if DRRs appear upside down.")
    fig.text(0.5, 0.005, note, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return avg_error


def main():
    parser = argparse.ArgumentParser(description='Visualize DRR-CT alignment')
    parser.add_argument('--data_path', type=str, default='/workspace/drr_patient_data',
                       help='Path to patient data directory')
    parser.add_argument('--patient_idx', type=int, default=0,
                       help='Index of patient to visualize (0-based)')
    parser.add_argument('--num_patients', type=int, default=5,
                       help='Number of patients to check')
    parser.add_argument('--save_dir', type=str, default='alignment_checks',
                       help='Directory to save visualizations')
    parser.add_argument('--flip_vertical', action='store_true',
                       help='Try flipping DRRs vertically to check alignment')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DRR-CT ALIGNMENT VISUAL CHECKER")
    print("="*70)
    
    # Create dataset
    print(f"\nLoading dataset from: {args.data_path}")
    dataset = PatientDRRDataset(
        data_path=args.data_path,
        target_xray_size=512,
        target_volume_size=(64, 64, 64),
        validate_alignment=True,
        augmentation=False
    )
    
    print(f"Found {len(dataset)} patients")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Check multiple patients
    errors = []
    for i in range(args.patient_idx, min(args.patient_idx + args.num_patients, len(dataset))):
        print(f"\n{'='*70}")
        print(f"Checking patient {i+1}/{len(dataset)}")
        print(f"{'='*70}")
        
        data = dataset[i]
        patient_id = data['patient_id']
        
        # Optionally flip DRRs
        if args.flip_vertical:
            print("Flipping DRRs vertically...")
            data['drr_frontal'] = torch.flip(data['drr_frontal'], dims=[-2])
            data['drr_lateral'] = torch.flip(data['drr_lateral'], dims=[-2])
        
        save_path = save_dir / f"alignment_{patient_id}.png"
        error = visualize_alignment(data, save_path=str(save_path))
        errors.append(error)
        
        print(f"Patient ID: {patient_id}")
        print(f"Alignment error: {error:.4f}")
        print(f"Status: {'✓ ALIGNED' if error < 0.5 else '✗ MISALIGNED'}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Patients checked: {len(errors)}")
    print(f"Average alignment error: {np.mean(errors):.4f}")
    print(f"Aligned (error < 0.5): {sum(1 for e in errors if e < 0.5)}/{len(errors)}")
    print(f"Misaligned (error >= 0.5): {sum(1 for e in errors if e >= 0.5)}/{len(errors)}")
    
    if np.mean(errors) >= 0.5:
        print(f"\n⚠ WARNING: High average alignment error!")
        print("Possible causes:")
        print("  1. DRRs are vertically flipped - try running with --flip_vertical")
        print("  2. DRRs don't match CT volume")
        print("  3. Wrong patient matching between DRRs and CT")
        print(f"\nTo fix vertical flip, add to config:")
        print('  "flip_drrs_vertical": true')
    else:
        print(f"\n✓ Alignment looks good!")
    
    print(f"\nVisualizations saved to: {save_dir}/")


if __name__ == "__main__":
    main()
