"""
Data Verification Script for Hybrid-ViT Cascade
Run this before training to verify your data is properly formatted and aligned
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import argparse

# Add parent to path
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

from utils.dataset import PatientDRRDataset


def verify_data_structure(data_path: str) -> dict:
    """Verify basic data structure"""
    print("\n" + "="*70)
    print("DATA STRUCTURE VERIFICATION")
    print("="*70)
    
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"✗ Data path does not exist: {data_path}")
        return {'valid': False}
    
    print(f"✓ Data path exists: {data_path}")
    
    # Find patient folders (any directory, excluding hidden folders)
    patient_folders = [f for f in data_path.iterdir() 
                      if f.is_dir() and not f.name.startswith('.')]
    
    print(f"✓ Found {len(patient_folders)} patient folders")
    
    if len(patient_folders) == 0:
        print("✗ No patient folders found")
        return {'valid': False}
    
    # Check first few patient folders
    print(f"\nChecking first 5 patient folders for required files...")
    
    valid_count = 0
    invalid_folders = []
    
    for patient_folder in patient_folders[:5]:
        patient_id = patient_folder.name
        print(f"\n  {patient_id}:")
        folder_valid = True
        
        # Check for frontal/PA image
        frontal_found = False
        for pattern in [f"{patient_id}_pa_drr.*", f"{patient_id}_pa.*", f"{patient_id}_frontal.*"]:
            matches = list(patient_folder.glob(pattern))
            if matches:
                print(f"    ✓ {matches[0].name} (frontal)")
                frontal_found = True
                break
        if not frontal_found:
            print(f"    ✗ frontal/PA image not found (expected: {patient_id}_pa_drr.* or {patient_id}_pa.*)")
            folder_valid = False
        
        # Check for lateral image
        lateral_found = False
        for pattern in [f"{patient_id}_lat_drr.*", f"{patient_id}_lat.*", f"{patient_id}_lateral.*"]:
            matches = list(patient_folder.glob(pattern))
            if matches:
                print(f"    ✓ {matches[0].name} (lateral)")
                lateral_found = True
                break
        if not lateral_found:
            print(f"    ✗ lateral image not found (expected: {patient_id}_lat_drr.* or {patient_id}_lat.*)")
            folder_valid = False
        
        # Check for CT volume
        ct_found = False
        for ext in ['.nii.gz', '.nii', '.npy']:
            filepath = patient_folder / f"{patient_id}{ext}"
            if filepath.exists():
                print(f"    ✓ {patient_id}{ext} (CT volume)")
                ct_found = True
                break
        if not ct_found:
            print(f"    ✗ CT volume not found (expected: {patient_id}.nii.gz, {patient_id}.nii, or {patient_id}.npy)")
            folder_valid = False
        
        if folder_valid:
            valid_count += 1
        else:
            invalid_folders.append(patient_id)
    
    print(f"\nValid folders checked: {valid_count}/5")
    if invalid_folders:
        print(f"Invalid folders: {', '.join(invalid_folders)}")
    
    return {
        'valid': len(invalid_folders) == 0,
        'total_patients': len(patient_folders),
        'checked_valid': valid_count,
        'invalid_folders': invalid_folders
    }


def test_data_loading(data_path: str, num_samples: int = 5):
    """Test loading a few samples"""
    print("\n" + "="*70)
    print("DATA LOADING TEST")
    print("="*70)
    
    try:
        # Create dataset
        dataset = PatientDRRDataset(
            data_path=data_path,
            target_xray_size=512,
            target_volume_size=(64, 64, 64),  # Small size for quick testing
            validate_alignment=True,
            augmentation=False
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")
        
        # Test loading samples
        print(f"\nLoading {min(num_samples, len(dataset))} sample(s)...")
        
        for i in range(min(num_samples, len(dataset))):
            print(f"\n  Sample {i+1}:")
            data = dataset[i]
            
            print(f"    Patient ID: {data['patient_id']}")
            print(f"    DRR frontal shape: {data['drr_frontal'].shape}")
            print(f"    DRR lateral shape: {data['drr_lateral'].shape}")
            print(f"    DRR stacked shape: {data['drr_stacked'].shape}")
            print(f"    CT volume shape: {data['ct_volume'].shape}")
            print(f"    Aligned: {data['aligned']}")
            
            # Check value ranges
            print(f"    DRR frontal range: [{data['drr_frontal'].min():.3f}, {data['drr_frontal'].max():.3f}]")
            print(f"    CT volume range: [{data['ct_volume'].min():.3f}, {data['ct_volume'].max():.3f}]")
        
        # Get alignment report
        print("\n" + "-"*70)
        alignment_report = dataset.get_alignment_report()
        print("Alignment Report:")
        print(f"  Total validated: {alignment_report['total_validated']}")
        print(f"  Passed: {alignment_report['passed']}")
        print(f"  Failed: {alignment_report['failed']}")
        print(f"  Pass rate: {alignment_report['pass_rate']*100:.2f}%")
        print(f"  Average error: {alignment_report['average_error']:.6f}")
        
        return {
            'success': True,
            'dataset_size': len(dataset),
            'alignment_report': alignment_report
        }
        
    except Exception as e:
        print(f"\n✗ Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def test_dataloader(data_path: str, batch_size: int = 2):
    """Test DataLoader with batching"""
    print("\n" + "="*70)
    print("DATALOADER TEST")
    print("="*70)
    
    try:
        from torch.utils.data import DataLoader
        
        # Create dataset
        dataset = PatientDRRDataset(
            data_path=data_path,
            target_xray_size=512,
            target_volume_size=(64, 64, 64),
            validate_alignment=False,  # Skip for speed
            augmentation=False
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Single worker for testing
            pin_memory=False
        )
        
        print(f"✓ DataLoader created")
        print(f"  Batch size: {batch_size}")
        print(f"  Total batches: {len(dataloader)}")
        
        # Test one batch
        print(f"\nTesting one batch...")
        batch = next(iter(dataloader))
        
        print(f"  DRR stacked batch shape: {batch['drr_stacked'].shape}")
        print(f"  CT volume batch shape: {batch['ct_volume'].shape}")
        print(f"  Patient IDs: {batch['patient_id']}")
        
        return {'success': True}
        
    except Exception as e:
        print(f"\n✗ Error with DataLoader: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def verify_gpu():
    """Verify GPU availability"""
    print("\n" + "="*70)
    print("GPU VERIFICATION")
    print("="*70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {props.name}")
            print(f"    Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Multi-processor count: {props.multi_processor_count}")
    else:
        print("⚠ No GPU detected. Training will be very slow on CPU.")


def main():
    parser = argparse.ArgumentParser(description='Verify data for Hybrid-ViT Cascade training')
    parser.add_argument('--data_path', type=str, default='/workspace/drr_patient_data',
                       help='Path to patient data directory')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to test')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for DataLoader test')
    parser.add_argument('--skip_loading', action='store_true',
                       help='Skip data loading test (only check structure)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("HYBRID-VIT CASCADE DATA VERIFICATION")
    print("="*70)
    
    # Verify GPU
    verify_gpu()
    
    # Verify data structure
    structure_result = verify_data_structure(args.data_path)
    
    if not structure_result['valid']:
        print("\n✗ Data structure verification failed!")
        print("  Please ensure your data is organized correctly.")
        print("\nExpected structure:")
        print("  /workspace/drr_patient_data/")
        print("    00ba4c616c15/  (patient ID)")
        print("      00ba4c616c15_pa_drr.png (frontal)")
        print("      00ba4c616c15_lat_drr.png (lateral)")
        print("      00ba4c616c15.nii.gz (CT volume)")
        print("    <another_patient_id>/")
        print("      ...")
        return
    
    if args.skip_loading:
        print("\n✓ Structure verification passed!")
        print("  Skipping loading tests (--skip_loading flag set)")
        return
    
    # Test data loading
    loading_result = test_data_loading(args.data_path, args.num_samples)
    
    if not loading_result['success']:
        print("\n✗ Data loading test failed!")
        return
    
    # Test DataLoader
    dataloader_result = test_dataloader(args.data_path, args.batch_size)
    
    if not dataloader_result['success']:
        print("\n✗ DataLoader test failed!")
        return
    
    # Final summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print("✓ Data structure: OK")
    print("✓ Data loading: OK")
    print("✓ DataLoader: OK")
    
    if loading_result.get('alignment_report'):
        report = loading_result['alignment_report']
        if report['pass_rate'] < 0.9:
            print(f"⚠ Warning: Alignment pass rate is low ({report['pass_rate']*100:.1f}%)")
            print("  You may want to investigate alignment issues before training")
        else:
            print(f"✓ Alignment: {report['pass_rate']*100:.1f}% pass rate")
    
    print("\n✓ All checks passed! Ready for training.")
    print("\nTo start training, run:")
    print(f"  python training/train_runpod.py --config config/runpod_config.json")
    print("\nOr use the automated setup script:")
    print(f"  ./setup_and_train_runpod.sh")


if __name__ == "__main__":
    main()
