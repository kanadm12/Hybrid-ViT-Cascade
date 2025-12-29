"""
Dataset for Hybrid-ViT Cascade Training
Loads patient DRR images (frontal and lateral) with CT volumes
Includes alignment validation and preprocessing
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class PatientDRRDataset(Dataset):
    """
    Dataset for loading patient DRR data with CT volumes
    
    Expected directory structure:
    data_path/
        patient_001/
            drr_frontal.png (or .npy)
            drr_lateral.png (or .npy)
            ct_volume.nii.gz (or .npy)
            metadata.json (optional - contains alignment info)
        patient_002/
            ...
    """
    
    def __init__(
        self,
        data_path: str,
        target_xray_size: int = 512,
        target_volume_size: Tuple[int, int, int] = (256, 256, 256),
        normalize_range: Tuple[float, float] = (-1, 1),
        validate_alignment: bool = True,
        augmentation: bool = False,
        cache_in_memory: bool = False,
        flip_drrs_vertical: bool = False
    ):
        """
        Args:
            data_path: Root directory containing patient folders
            target_xray_size: Target size for DRR images (square)
            target_volume_size: Target size for CT volumes (D, H, W)
            normalize_range: Normalization range for images and volumes
            validate_alignment: Whether to validate DRR-CT alignment
            augmentation: Whether to apply data augmentation
            cache_in_memory: Cache loaded data in memory (for small datasets)
        """
        self.data_path = Path(data_path)
        self.target_xray_size = target_xray_size
        self.target_volume_size = target_volume_size
        self.normalize_range = normalize_range
        self.validate_alignment = validate_alignment
        self.augmentation = augmentation
        self.cache_in_memory = cache_in_memory
        self.flip_drrs_vertical = flip_drrs_vertical
        
        # Find all patient folders (any directory with required files)
        self.patient_folders = []
        if self.data_path.exists():
            for folder in sorted(self.data_path.iterdir()):
                if folder.is_dir() and not folder.name.startswith('.'):
                    # Verify required files exist
                    if self._validate_patient_folder(folder):
                        self.patient_folders.append(folder)
        
        if len(self.patient_folders) == 0:
            raise ValueError(f"No valid patient folders found in {data_path}")
        
        print(f"Found {len(self.patient_folders)} valid patient datasets in {data_path}")
        
        # Cache for in-memory loading
        self.cache = {} if cache_in_memory else None
        
        # Alignment statistics
        self.alignment_stats = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'avg_error': 0.0
        }
    
    def _validate_patient_folder(self, folder: Path) -> bool:
        """Check if patient folder contains required files"""
        patient_id = folder.name
        
        # Check for frontal/PA image (patient_id_pa_drr.* or patient_id_pa.* or patient_id_frontal.*)
        frontal_found = False
        for pattern in [f"{patient_id}_pa_drr.*", f"{patient_id}_pa.*", f"{patient_id}_frontal.*"]:
            matches = list(folder.glob(pattern))
            if matches:
                frontal_found = True
                break
        
        # Check for lateral image (patient_id_lat_drr.* or patient_id_lat.* or patient_id_lateral.*)
        lateral_found = False
        for pattern in [f"{patient_id}_lat_drr.*", f"{patient_id}_lat.*", f"{patient_id}_lateral.*"]:
            matches = list(folder.glob(pattern))
            if matches:
                lateral_found = True
                break
        
        # Check for CT volume (patient_id.nii.gz, patient_id.nii, or patient_id.npy)
        ct_found = False
        for ext in ['.nii.gz', '.nii', '.npy']:
            if (folder / f"{patient_id}{ext}").exists():
                ct_found = True
                break
        
        if not (frontal_found and lateral_found and ct_found):
            if not frontal_found:
                print(f"Warning: Missing frontal/PA image in {folder.name}")
            if not lateral_found:
                print(f"Warning: Missing lateral image in {folder.name}")
            if not ct_found:
                print(f"Warning: Missing CT volume in {folder.name}")
            return False
        return True
    
    def __len__(self) -> int:
        return len(self.patient_folders)
    
    def _find_file(self, folder: Path, file_type: str) -> Optional[Path]:
        """Find file with patient ID naming convention"""
        patient_id = folder.name
        
        if file_type == 'drr_frontal':
            # Look for patient_id_pa_drr.*, patient_id_pa.*, or patient_id_frontal.*
            for pattern in [f"{patient_id}_pa_drr.*", f"{patient_id}_pa.*", f"{patient_id}_frontal.*"]:
                matches = list(folder.glob(pattern))
                if matches:
                    return matches[0]
        
        elif file_type == 'drr_lateral':
            # Look for patient_id_lat_drr.*, patient_id_lat.*, or patient_id_lateral.*
            for pattern in [f"{patient_id}_lat_drr.*", f"{patient_id}_lat.*", f"{patient_id}_lateral.*"]:
                matches = list(folder.glob(pattern))
                if matches:
                    return matches[0]
        
        elif file_type == 'ct_volume':
            # Look for patient_id.nii.gz, patient_id.nii, or patient_id.npy
            for ext in ['.nii.gz', '.nii', '.npy']:
                filepath = folder / f"{patient_id}{ext}"
                if filepath.exists():
                    return filepath
        
        return None
    
    def _load_image(self, filepath: Path, target_size: int) -> torch.Tensor:
        """Load and preprocess DRR image"""
        if filepath.suffix == '.npy':
            img = np.load(filepath).astype(np.float32)
            if img.ndim == 2:
                img = img[np.newaxis, :, :]  # Add channel dimension
        else:
            # Load as PIL Image
            img = Image.open(filepath).convert('L')
            img = np.array(img, dtype=np.float32)[np.newaxis, :, :]
        
        # Convert to tensor and resize
        img_tensor = torch.from_numpy(img)  # (1, H, W)
        
        # Resize to target size
        if img_tensor.shape[1] != target_size or img_tensor.shape[2] != target_size:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Normalize to [0, 1] if needed
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0
        
        # Normalize to target range
        min_val, max_val = self.normalize_range
        img_tensor = img_tensor * (max_val - min_val) + min_val
        
        return img_tensor
    
    def _load_volume(self, filepath: Path, target_size: Tuple[int, int, int]) -> torch.Tensor:
        """Load and preprocess CT volume"""
        if filepath.suffix == '.npy':
            volume = np.load(filepath).astype(np.float32)
        else:
            # Load NIfTI
            nii = nib.load(filepath)
            volume = nii.get_fdata().astype(np.float32)
        
        # Ensure correct shape
        if volume.ndim == 3:
            volume = volume[np.newaxis, :, :, :]  # Add channel dimension
        
        # Convert to tensor
        volume_tensor = torch.from_numpy(volume)  # (1, D, H, W)
        
        # Resize to target size
        if volume_tensor.shape[1:] != target_size:
            volume_tensor = F.interpolate(
                volume_tensor.unsqueeze(0),
                size=target_size,
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
        
        # FIXED: Use soft tissue window [-200, 200 HU] for better contrast
        # This maps: Air(-200)→-1, Soft tissue(0-100)→[-0.5,0], Bone(200)→+1
        # Previous [-1000,3000] compressed all soft tissue into narrow range
        volume_tensor = torch.clamp(volume_tensor, -200, 200)
        
        # Normalize to [0, 1]
        volume_tensor = (volume_tensor + 200) / 400
        
        # Normalize to target range [-1, 1]
        min_val, max_val = self.normalize_range
        volume_tensor = volume_tensor * (max_val - min_val) + min_val
        
        return volume_tensor
    
    def _validate_drr_ct_alignment(
        self,
        drr_frontal: torch.Tensor,
        drr_lateral: torch.Tensor,
        ct_volume: torch.Tensor,
        patient_id: str
    ) -> Tuple[bool, float]:
        """
        Validate alignment between DRRs and CT volume
        
        Returns:
            (is_aligned, alignment_error)
        """
        # Generate synthetic DRRs from CT volume for comparison
        # Frontal view: max projection along depth (axis 1)
        synth_frontal = torch.max(ct_volume, dim=1)[0]  # (1, H, W)
        
        # Lateral view: max projection along width (axis 3)
        synth_lateral = torch.max(ct_volume, dim=3)[0]  # (1, D, H)
        
        # Resize synthetic DRRs to match input DRR size
        synth_frontal = F.interpolate(
            synth_frontal.unsqueeze(0),
            size=(self.target_xray_size, self.target_xray_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        synth_lateral = F.interpolate(
            synth_lateral.unsqueeze(0),
            size=(self.target_xray_size, self.target_xray_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Compute alignment error (normalized L2 distance)
        frontal_error = F.mse_loss(drr_frontal, synth_frontal).item()
        lateral_error = F.mse_loss(drr_lateral, synth_lateral).item()
        
        avg_error = (frontal_error + lateral_error) / 2
        
        # Threshold for alignment (adjust based on your data)
        alignment_threshold = 0.5  # Can be adjusted
        
        is_aligned = avg_error < alignment_threshold
        
        if not is_aligned:
            print(f"Warning: Alignment issue for {patient_id}")
            print(f"  Frontal error: {frontal_error:.4f}, Lateral error: {lateral_error:.4f}")
        
        return is_aligned, avg_error
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict containing:
                'drr_frontal': (1, H, W) tensor
                'drr_lateral': (1, H, W) tensor
                'ct_volume': (1, D, H, W) tensor
                'patient_id': string
                'aligned': bool
        """
        # Check cache
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
        
        patient_folder = self.patient_folders[idx]
        patient_id = patient_folder.name
        
        # Load DRRs
        frontal_path = self._find_file(patient_folder, 'drr_frontal')
        lateral_path = self._find_file(patient_folder, 'drr_lateral')
        ct_path = self._find_file(patient_folder, 'ct_volume')
        
        drr_frontal = self._load_image(frontal_path, self.target_xray_size)
        drr_lateral = self._load_image(lateral_path, self.target_xray_size)
        ct_volume = self._load_volume(ct_path, self.target_volume_size)
        
        # Apply vertical flip if needed (for misaligned DRRs)
        if self.flip_drrs_vertical:
            drr_frontal = torch.flip(drr_frontal, dims=[-2])
            drr_lateral = torch.flip(drr_lateral, dims=[-2])
        
        # Validate alignment
        aligned = True
        if self.validate_alignment:
            aligned, error = self._validate_drr_ct_alignment(
                drr_frontal, drr_lateral, ct_volume, patient_id
            )
            self.alignment_stats['total'] += 1
            if aligned:
                self.alignment_stats['passed'] += 1
            else:
                self.alignment_stats['failed'] += 1
            self.alignment_stats['avg_error'] += error
        
        # Stack DRRs for dual-view input (2, 1, H, W)
        drr_stacked = torch.stack([drr_frontal, drr_lateral], dim=0)
        
        # Apply augmentation if enabled
        if self.augmentation:
            drr_stacked, ct_volume = self._apply_augmentation(drr_stacked, ct_volume)
        
        data = {
            'drr_frontal': drr_frontal,
            'drr_lateral': drr_lateral,
            'drr_stacked': drr_stacked,  # (2, 1, H, W) for model input
            'ct_volume': ct_volume,
            'patient_id': patient_id,
            'aligned': aligned
        }
        
        # Cache if enabled
        if self.cache is not None:
            self.cache[idx] = data
        
        return data
    
    def _apply_augmentation(
        self,
        drr_stacked: torch.Tensor,
        ct_volume: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations"""
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            drr_stacked = torch.flip(drr_stacked, [-1])  # Flip width
            ct_volume = torch.flip(ct_volume, [-1])  # Flip width
        
        # Random intensity scaling
        if torch.rand(1).item() > 0.5:
            scale = 0.9 + 0.2 * torch.rand(1).item()  # [0.9, 1.1]
            drr_stacked = drr_stacked * scale
            ct_volume = ct_volume * scale
        
        # Clamp to valid range
        min_val, max_val = self.normalize_range
        drr_stacked = torch.clamp(drr_stacked, min_val, max_val)
        ct_volume = torch.clamp(ct_volume, min_val, max_val)
        
        return drr_stacked, ct_volume
    
    def get_alignment_report(self) -> Dict:
        """Get alignment validation statistics"""
        if self.alignment_stats['total'] > 0:
            avg_error = self.alignment_stats['avg_error'] / self.alignment_stats['total']
            pass_rate = self.alignment_stats['passed'] / self.alignment_stats['total']
        else:
            avg_error = 0.0
            pass_rate = 0.0
        
        return {
            'total_validated': self.alignment_stats['total'],
            'passed': self.alignment_stats['passed'],
            'failed': self.alignment_stats['failed'],
            'pass_rate': pass_rate,
            'average_error': avg_error
        }


def create_train_val_datasets(
    data_path: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    **dataset_kwargs
) -> Tuple[PatientDRRDataset, PatientDRRDataset, PatientDRRDataset]:
    """
    Create train, validation, and test datasets with proper splits
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    from torch.utils.data import random_split
    
    # Create full dataset
    full_dataset = PatientDRRDataset(data_path, **dataset_kwargs)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Random split
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset
