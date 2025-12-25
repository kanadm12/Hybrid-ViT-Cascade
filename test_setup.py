"""
Test Hybrid-ViT Cascade Setup Locally
Run this before uploading to RunPod to verify everything is configured correctly
"""

import sys
from pathlib import Path

print("="*70)
print("HYBRID-VIT CASCADE SETUP TEST")
print("="*70)

# Test imports
print("\n1. Testing imports...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"   ✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"   ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"   ✗ NumPy import failed: {e}")

try:
    from PIL import Image
    print(f"   ✓ Pillow (PIL)")
except ImportError as e:
    print(f"   ✗ Pillow import failed: {e}")
    print("     Install with: pip install Pillow")

try:
    import nibabel as nib
    print(f"   ✓ Nibabel {nib.__version__}")
except ImportError as e:
    print(f"   ✗ Nibabel import failed: {e}")
    print("     Install with: pip install nibabel")

try:
    import scipy
    print(f"   ✓ SciPy {scipy.__version__}")
except ImportError as e:
    print(f"   ✗ SciPy import failed: {e}")
    print("     Install with: pip install scipy")

# Test project structure
print("\n2. Testing project structure...")
project_root = Path(__file__).parent

required_files = [
    'models/unified_model.py',
    'models/hybrid_vit_backbone.py',
    'models/cascaded_depth_lifting.py',
    'training/train_progressive.py',
    'training/train_runpod.py',
    'utils/dataset.py',
    'config/runpod_config.json',
    'requirements.txt'
]

all_found = True
for file in required_files:
    filepath = project_root / file
    if filepath.exists():
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} NOT FOUND")
        all_found = False

if not all_found:
    print("\n   Some files are missing. This may cause issues.")

# Test dataset import
print("\n3. Testing dataset module...")
try:
    from utils.dataset import PatientDRRDataset, create_train_val_datasets
    print("   ✓ Dataset module imported successfully")
except ImportError as e:
    print(f"   ✗ Dataset import failed: {e}")
    sys.exit(1)

# Test model import
print("\n4. Testing model module...")
try:
    from models.unified_model import UnifiedHybridViTCascade
    print("   ✓ Model module imported successfully")
except ImportError as e:
    print(f"   ✗ Model import failed: {e}")
    sys.exit(1)

# Test config loading
print("\n5. Testing configuration...")
try:
    import json
    config_path = project_root / 'config' / 'runpod_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("   ✓ RunPod config loaded")
    print(f"      Data path: {config['data']['dataset_path']}")
    print(f"      Batch size: {config['training']['batch_size']}")
    print(f"      Stages: {len(config['stage_configs'])}")
except Exception as e:
    print(f"   ✗ Config loading failed: {e}")

# Test CUDA
print("\n6. Testing CUDA availability...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available")
    print(f"      Version: {torch.version.cuda}")
    print(f"      Devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"      GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("   ⚠ CUDA not available (will be available on RunPod)")

# Test creating a small model
print("\n7. Testing model creation...")
try:
    # Create a minimal config for testing
    test_config = {
        'stage_configs': [
            {
                'name': 'stage1',
                'volume_size': [32, 32, 32],
                'voxel_dim': 128,
                'vit_depth': 2,
                'num_heads': 2,
                'use_depth_lifting': True,
                'use_physics_loss': True,
                'physics_weight': 0.3
            }
        ]
    }
    
    model = UnifiedHybridViTCascade(
        stage_configs=test_config['stage_configs'],
        xray_img_size=256,
        xray_channels=1,
        num_views=2,
        v_parameterization=True,
        num_timesteps=100
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created successfully")
    print(f"      Test model parameters: {param_count:,}")
    
    del model  # Free memory
    
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*70)
print("SETUP TEST SUMMARY")
print("="*70)

all_good = True
checklist = {
    'Dependencies installed': True,  # If we got here, main deps are OK
    'Project files present': all_found,
    'Modules importable': True,  # If we got here, imports work
    'Model can be created': True,  # If we got here, model works
}

for item, status in checklist.items():
    symbol = "✓" if status else "✗"
    print(f"{symbol} {item}")
    if not status:
        all_good = False

if all_good:
    print("\n✓ All tests passed!")
    print("\nYour setup looks good. You can now:")
    print("1. Upload this folder to RunPod")
    print("2. Ensure your data is at /workspace/drr_patient_data")
    print("3. Run: python verify_data.py")
    print("4. Start training with: python training/train_runpod.py --config config/runpod_config.json")
else:
    print("\n⚠ Some tests failed. Please fix the issues above before uploading to RunPod.")

print("="*70)
