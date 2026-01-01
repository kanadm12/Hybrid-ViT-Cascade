"""
Quick test to verify Hybrid CNN-ViT model can be instantiated.
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from medical_grade.model_unet3d import HybridCNNViTUNet3D, MedicalGradeLoss

def test_model():
    print('\n=== Testing Hybrid CNN-ViT Model ===\n')
    
    # Create model
    print('Creating model...')
    model = HybridCNNViTUNet3D(
        volume_size=(64, 64, 64),
        xray_size=512,
        base_channels=32
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'✓ Model created successfully')
    print(f'  Total parameters: {total_params / 1e6:.2f}M')
    print(f'  Trainable parameters: {trainable_params / 1e6:.2f}M\n')
    
    # Test forward pass
    print('Testing forward pass...')
    batch_size = 2
    xrays = torch.randn(batch_size, 2, 1, 512, 512)  # (B, num_views, 1, H, W)
    
    model.eval()
    with torch.no_grad():
        pred_ct, aux_outputs = model(xrays)
    
    print(f'✓ Forward pass successful')
    print(f'  Input shape: {xrays.shape}')
    print(f'  Output shape: {pred_ct.shape}')
    print(f'  Aux outputs: {list(aux_outputs.keys())}\n')
    
    # Test loss
    print('Testing loss function...')
    criterion = MedicalGradeLoss()
    target_ct = torch.randn(batch_size, 1, 64, 64, 64)
    
    loss, loss_dict = criterion(pred_ct, target_ct, aux_outputs)
    print(f'✓ Loss computed successfully')
    print(f'  Total loss: {loss.item():.4f}')
    print(f'  Loss components:')
    for k, v in loss_dict.items():
        print(f'    {k}: {v:.4f}')
    print()
    
    print('=== All tests passed! ===')
    print('\nReady to start training:')
    print('  cd medical_grade')
    print('  python train_unet3d_4gpu.py --config config_unet3d.json --world_size 4')


if __name__ == '__main__':
    test_model()
