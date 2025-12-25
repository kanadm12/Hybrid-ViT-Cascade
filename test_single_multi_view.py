"""
Test single-view and multi-view X-ray input handling
"""

import torch
import sys
from pathlib import Path

# Add paths
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "vit_cascading_diffusion" / "models"))
sys.path.insert(0, str(parent_dir / "hybrid_vit_cascade" / "models"))

from xray_encoder import XrayConditioningModule, MultiViewXrayEncoder
from unified_model import UnifiedHybridViTCascade


def test_single_view_encoder():
    """Test single-view X-ray encoding"""
    print("\n" + "="*60)
    print("TEST 1: Single-View X-ray Encoder")
    print("="*60)
    
    encoder = MultiViewXrayEncoder(
        img_size=512,
        in_channels=1,
        embed_dim=256,
        num_views=2,  # Max 2 views, but can handle 1
        share_weights=False
    )
    
    # Single view input (4D)
    xrays_single = torch.randn(2, 1, 512, 512)
    print(f"Input shape (single-view): {xrays_single.shape}")
    
    features, pooled = encoder(xrays_single)
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Pooled shape: {pooled.shape}")
    assert features.dim() == 3  # (B, seq_len, embed_dim)
    assert pooled.dim() == 2    # (B, embed_dim)
    print("✓ Single-view encoding works!")
    

def test_multi_view_encoder():
    """Test multi-view X-ray encoding"""
    print("\n" + "="*60)
    print("TEST 2: Multi-View X-ray Encoder")
    print("="*60)
    
    encoder = MultiViewXrayEncoder(
        img_size=512,
        in_channels=1,
        embed_dim=256,
        num_views=2,
        share_weights=False
    )
    
    # Multi-view input (5D)
    xrays_multi = torch.randn(2, 2, 1, 512, 512)
    print(f"Input shape (multi-view): {xrays_multi.shape}")
    
    features, pooled = encoder(xrays_multi)
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Pooled shape: {pooled.shape}")
    assert features.dim() == 3
    assert pooled.dim() == 2
    print("✓ Multi-view encoding works!")


def test_conditioning_module_single_view():
    """Test full conditioning module with single view"""
    print("\n" + "="*60)
    print("TEST 3: Conditioning Module (Single-View)")
    print("="*60)
    
    module = XrayConditioningModule(
        img_size=512,
        in_channels=1,
        embed_dim=256,
        num_views=2,
        share_view_weights=False
    )
    
    xrays = torch.randn(2, 1, 512, 512)  # Single view
    timesteps = torch.randint(0, 1000, (2,))
    
    print(f"Input X-ray shape: {xrays.shape}")
    print(f"Timesteps: {timesteps.shape}")
    
    context, cond, features_2d = module(xrays, timesteps)
    
    print(f"✓ Context shape: {context.shape}")
    print(f"✓ Conditioning shape: {cond.shape}")
    print(f"✓ 2D Features shape: {features_2d.shape}")
    print("✓ Single-view conditioning works!")


def test_conditioning_module_multi_view():
    """Test full conditioning module with multi-view"""
    print("\n" + "="*60)
    print("TEST 4: Conditioning Module (Multi-View)")
    print("="*60)
    
    module = XrayConditioningModule(
        img_size=512,
        in_channels=1,
        embed_dim=256,
        num_views=2,
        share_view_weights=False
    )
    
    xrays = torch.randn(2, 2, 1, 512, 512)  # Multi-view
    timesteps = torch.randint(0, 1000, (2,))
    
    print(f"Input X-ray shape: {xrays.shape}")
    
    context, cond, features_2d = module(xrays, timesteps)
    
    print(f"✓ Context shape: {context.shape}")
    print(f"✓ Conditioning shape: {cond.shape}")
    print(f"✓ 2D Features shape: {features_2d.shape}")
    print("✓ Multi-view conditioning works!")


def test_full_model_single_view():
    """Test complete model with single-view input"""
    print("\n" + "="*60)
    print("TEST 5: Full Model (Single-View)")
    print("="*60)
    
    stage_configs = [
        {
            'name': 'stage1',
            'volume_size': [64, 64, 64],
            'voxel_dim': 128,
            'vit_depth': 2,
            'num_heads': 4,
            'use_depth_lifting': True,
            'use_physics_loss': False  # Skip for test
        }
    ]
    
    model = UnifiedHybridViTCascade(
        stage_configs=stage_configs,
        xray_img_size=512,
        num_views=1,  # Single view
        xray_embed_dim=256,
        share_view_weights=False
    )
    
    # Single-view input
    xrays = torch.randn(1, 1, 512, 512)
    ct_volumes = torch.randn(1, 1, 64, 64, 64)
    
    print(f"Input X-ray shape: {xrays.shape}")
    print(f"Input CT shape: {ct_volumes.shape}")
    
    output = model(ct_volumes, xrays, stage_name='stage1', return_loss_dict=True)
    
    print(f"✓ Loss: {output['loss'].item():.4f}")
    print(f"✓ V-loss: {output['v_loss'].item():.4f}")
    print("✓ Single-view full model works!")


def test_full_model_multi_view():
    """Test complete model with multi-view input"""
    print("\n" + "="*60)
    print("TEST 6: Full Model (Multi-View)")
    print("="*60)
    
    stage_configs = [
        {
            'name': 'stage1',
            'volume_size': [64, 64, 64],
            'voxel_dim': 128,
            'vit_depth': 2,
            'num_heads': 4,
            'use_depth_lifting': True,
            'use_physics_loss': False
        }
    ]
    
    model = UnifiedHybridViTCascade(
        stage_configs=stage_configs,
        xray_img_size=512,
        num_views=2,  # Multi-view
        xray_embed_dim=256,
        share_view_weights=False
    )
    
    # Multi-view input
    xrays = torch.randn(1, 2, 1, 512, 512)
    ct_volumes = torch.randn(1, 1, 64, 64, 64)
    
    print(f"Input X-ray shape: {xrays.shape}")
    print(f"Input CT shape: {ct_volumes.shape}")
    
    output = model(ct_volumes, xrays, stage_name='stage1', return_loss_dict=True)
    
    print(f"✓ Loss: {output['loss'].item():.4f}")
    print(f"✓ V-loss: {output['v_loss'].item():.4f}")
    print("✓ Multi-view full model works!")


def test_view_weight_sharing():
    """Test shared weights option"""
    print("\n" + "="*60)
    print("TEST 7: Shared View Weights")
    print("="*60)
    
    encoder_shared = MultiViewXrayEncoder(
        img_size=512,
        embed_dim=256,
        num_views=2,
        share_weights=True  # Shared
    )
    
    encoder_separate = MultiViewXrayEncoder(
        img_size=512,
        embed_dim=256,
        num_views=2,
        share_weights=False  # Separate
    )
    
    # Count parameters
    params_shared = sum(p.numel() for p in encoder_shared.parameters())
    params_separate = sum(p.numel() for p in encoder_separate.parameters())
    
    print(f"Shared weights params: {params_shared:,}")
    print(f"Separate weights params: {params_separate:,}")
    print(f"✓ Shared saves {params_separate - params_shared:,} parameters!")
    assert params_shared < params_separate


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING SINGLE-VIEW AND MULTI-VIEW SUPPORT")
    print("="*60)
    
    tests = [
        ("Single-View Encoder", test_single_view_encoder),
        ("Multi-View Encoder", test_multi_view_encoder),
        ("Single-View Conditioning", test_conditioning_module_single_view),
        ("Multi-View Conditioning", test_conditioning_module_multi_view),
        ("Single-View Full Model", test_full_model_single_view),
        ("Multi-View Full Model", test_full_model_multi_view),
        ("View Weight Sharing", test_view_weight_sharing),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED:")
            print(f"  Error: {str(e)}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✓ ALL TESTS PASSED!")
    else:
        print(f"✗ {failed} tests failed")
    print("="*60)
