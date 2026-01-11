"""
Dataset loader for Progressive Cascade Training
Loads CT volumes and DRR X-ray images from patient folders
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import nibabel as nib
from PIL import Image


class PatientDRRDataset(Dataset):
    """
    Dataset for DRR patient data
    Loads CT volumes (.nii.gz) and corresponding DRR images (.png)
    Applies vertical flip to DRR images during training
    
    Expected structure:
        dataset_path/
        ├── patient_id_1/
        │   ├── patient_id_1.nii.gz
        │   ├── patient_id_1_pa_drr.png
        │   └── patient_id_1_lat_drr.png
        └── patient_id_2/
            └── ...
    """
    def __init__(
        self,
        dataset_path,
        max_patients=None,
        split='train',
        train_split=0.8,
        val_split=0.1,
        ct_size=256,
        drr_size=512,
        vertical_flip=True,
    ):
        """
        Args:
            dataset_path: Path to root directory containing patient folders
            max_patients: Maximum number of patients to load
            split: 'train', 'val', or 'test'
            train_split: Fraction for training
            val_split: Fraction for validation
            ct_size: Target size for CT volumes (D, H, W) - default 256
            drr_size: Target size for DRR images (H, W) - default 512
            vertical_flip: Whether to vertically flip DRR images (default: True)
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.ct_size = ct_size if isinstance(ct_size, tuple) else (ct_size, ct_size, ct_size)
        self.drr_size = drr_size
        self.vertical_flip = vertical_flip
        
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
        """
        Args:
            idx: Index of sample
            
        Returns:
            dict with:
                'ct_volume': CT volume (1, D, H, W)
                'drr_stacked': Stacked DRR images (2, 1, H, W) [PA, Lateral]
                'patient_id': Patient ID string
        """
        patient_dir = self.patient_dirs[idx]
        patient_id = patient_dir.name
        
        # Load CT volume from .nii.gz with error handling
        ct_path = patient_dir / f"{patient_id}.nii.gz"
        try:
            ct_nifti = nib.load(str(ct_path))
            ct_volume = ct_nifti.get_fdata().astype(np.float32)
        except Exception as e:
            print(f"Error loading CT file {ct_path}: {e}")
            raise
        
        # Normalize CT to [0, 1] using HU values
        ct_volume = np.clip(ct_volume, -1024, 3071)
        ct_volume = (ct_volume + 1024) / 4095.0
        
        # Resize CT volume to standard size if needed
        if ct_volume.shape != self.ct_size:
            ct_volume = self._resize_3d(ct_volume, self.ct_size)
        
        # Convert to torch: (D, H, W) -> (1, D, H, W)
        ct_volume = torch.from_numpy(ct_volume).unsqueeze(0)
        
        # Load DRR images from .png
        pa_drr_path = patient_dir / f"{patient_id}_pa_drr.png"
        lat_drr_path = patient_dir / f"{patient_id}_lat_drr.png"
        
        pa_drr = np.array(Image.open(pa_drr_path).convert('L')).astype(np.float32)
        lat_drr = np.array(Image.open(lat_drr_path).convert('L')).astype(np.float32)
        
        # Apply vertical flip if enabled (must use .copy() to avoid negative strides)
        if self.vertical_flip:
            pa_drr = np.flipud(pa_drr).copy()
            lat_drr = np.flipud(lat_drr).copy()
        
        # Resize DRRs if needed
        if isinstance(self.drr_size, int):
            target_size = (self.drr_size, self.drr_size)
        else:
            target_size = self.drr_size
        
        if pa_drr.shape != target_size:
            pa_drr = self._resize_2d(pa_drr, target_size)
            lat_drr = self._resize_2d(lat_drr, target_size)
        
        # Normalize to [0, 1]
        pa_drr = pa_drr / 255.0
        lat_drr = lat_drr / 255.0
        
        # Convert to tensors: (H, W) -> (1, H, W)
        pa_drr = torch.from_numpy(pa_drr).unsqueeze(0)
        lat_drr = torch.from_numpy(lat_drr).unsqueeze(0)
        
        # Stack PA and Lateral views: (2, 1, H, W)
        drr_stacked = torch.stack([pa_drr, lat_drr], dim=0)
        
        return {
            'ct_volume': ct_volume,
            'drr_stacked': drr_stacked,
            'patient_id': patient_id
        }
    
    @staticmethod
    def _resize_2d(img, target_size):
        """Resize 2D image to target size using scipy"""
        from scipy import ndimage
        
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        
        if img.shape != target_size:
            zoom_factors = (target_size[0] / img.shape[0], target_size[1] / img.shape[1])
            img = ndimage.zoom(img, zoom_factors, order=1)
        
        return img
    
    @staticmethod
    def _resize_3d(volume, target_size):
        """Resize 3D volume to target size using scipy"""
        from scipy import ndimage
        
        if isinstance(target_size, int):
            target_size = (target_size, target_size, target_size)
        
        if volume.shape != target_size:
            zoom_factors = tuple(target_size[i] / volume.shape[i] for i in range(3))
            volume = ndimage.zoom(volume, zoom_factors, order=1)
        
        return volume


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
