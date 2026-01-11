"""
Simple Dataset for Progressive Cascade Training
Loads CT volumes and DRR X-ray images from patient folders
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import nibabel as nib
from PIL import Image
import os


class PatientDRRDataset(Dataset):
    """
    Dataset for CT reconstruction from DRR X-rays
    
    Expected structure:
    root_dir/
        patient_id_1/
            patient_id_1.nii.gz
            patient_id_1_pa_drr.png
            patient_id_1_lat_drr.png
        patient_id_2/
            ...
    """
    
    def __init__(self, dataset_path, max_patients=None, split='train', 
                 train_split=0.8, val_split=0.1):
        """
        Args:
            dataset_path: Path to root directory containing patient folders
            max_patients: Maximum number of patients to load
            split: 'train', 'val', or 'test'
            train_split: Fraction for training
            val_split: Fraction for validation
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        
        # Find all patient directories
        patient_dirs = sorted([d for d in self.dataset_path.iterdir() if d.is_dir()])
        
        if max_patients is not None:
            patient_dirs = patient_dirs[:max_patients]
        
        # Split dataset
        n_total = len(patient_dirs)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        if split == 'train':
            self.patient_dirs = patient_dirs[:n_train]
        elif split == 'val':
            self.patient_dirs = patient_dirs[n_train:n_train + n_val]
        else:  # test
            self.patient_dirs = patient_dirs[n_train + n_val:]
        
        print(f"[{split}] Found {len(self.patient_dirs)} patients")
    
    def __len__(self):
        return len(self.patient_dirs)
    
    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_id = patient_dir.name
        
        # Load CT volume
        ct_path = patient_dir / f"{patient_id}.nii.gz"
        ct_nifti = nib.load(str(ct_path))
        ct_volume = ct_nifti.get_fdata().astype(np.float32)
        
        # Normalize CT to [0, 1] range (assuming HU values -1024 to 3071)
        ct_volume = np.clip(ct_volume, -1024, 3071)
        ct_volume = (ct_volume + 1024) / 4095.0
        
        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        ct_volume = torch.from_numpy(ct_volume).unsqueeze(0)
        
        # Load DRR images (PA and Lateral)
        pa_drr_path = patient_dir / f"{patient_id}_pa_drr.png"
        lat_drr_path = patient_dir / f"{patient_id}_lat_drr.png"
        
        pa_drr = Image.open(pa_drr_path).convert('L')  # Convert to grayscale
        lat_drr = Image.open(lat_drr_path).convert('L')
        
        # Convert to numpy and normalize to [0, 1]
        pa_drr = np.array(pa_drr, dtype=np.float32) / 255.0
        lat_drr = np.array(lat_drr, dtype=np.float32) / 255.0
        
        # Vertically flip DRRs (important for proper orientation)
        pa_drr = np.flipud(pa_drr)
        lat_drr = np.flipud(lat_drr)
        
        # Convert to torch tensors: (H, W) -> (1, H, W)
        pa_drr = torch.from_numpy(pa_drr).unsqueeze(0)
        lat_drr = torch.from_numpy(lat_drr).unsqueeze(0)
        
        # Stack PA and Lateral views: (2, 1, H, W)
        drr_stacked = torch.stack([pa_drr, lat_drr], dim=0)
        
        return {
            'ct_volume': ct_volume,
            'drr_stacked': drr_stacked,
            'patient_id': patient_id
        }


def test_dataset():
    """Test the dataset loading"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_simple.py <dataset_path>")
        print("Example: python dataset_simple.py /workspace/drr_patient_data")
        return
    
    dataset_path = sys.argv[1]
    
    print(f"Testing dataset from: {dataset_path}")
    print("="*60)
    
    # Create dataset
    dataset = PatientDRRDataset(
        dataset_path=dataset_path,
        max_patients=10,
        split='train'
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Load first sample
    print("\nLoading first sample...")
    sample = dataset[0]
    
    print(f"Patient ID: {sample['patient_id']}")
    print(f"CT Volume shape: {sample['ct_volume'].shape}")
    print(f"CT Volume range: [{sample['ct_volume'].min():.3f}, {sample['ct_volume'].max():.3f}]")
    print(f"DRR Stacked shape: {sample['drr_stacked'].shape}")
    print(f"DRR range: [{sample['drr_stacked'].min():.3f}, {sample['drr_stacked'].max():.3f}]")
    
    print("\n✓ Dataset loading successful!")


if __name__ == "__main__":
    test_dataset()
