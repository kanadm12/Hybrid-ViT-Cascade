"""
Comprehensive Dimension Validation Test
Tests all dimension flows through the optimized CT regression model
"""
import torch
import sys
sys.path.insert(0, '..')

from model_optimized import OptimizedCTRegression


def test_dimension_flow():
    """Test complete dimension flow through the model"""
    print("="*80)
    print("DIMENSION VALIDATION TEST")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Test configuration
    batch_size = 2
    num_views = 2
    img_size = 512
    volume_size = (64, 64, 64)
    xray_feature_dim = 512
    voxel_dim = 256
    
    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input X-rays: ({batch_size}, {num_views}, 1, {img_size}, {img_size})")
    print(f"  Target volume: ({batch_size}, 1, {volume_size[0]}, {volume_size[1]}, {volume_size[2]})")
    print(f"  X-ray feature dim: {xray_feature_dim}")
    print(f"  Voxel dim: {voxel_dim}")
    
    # Test with all optimizations enabled
    print("\n" + "-"*80)
    print("TEST 1: Full Model (CNN + Learnable Priors)")
    print("-"*80)
    
    model = OptimizedCTRegression(
        volume_size=volume_size,
        xray_img_size=img_size,
        voxel_dim=voxel_dim,
        num_attn_blocks=2,
        num_ffn_blocks=4,
        num_heads=8,
        xray_feature_dim=xray_feature_dim,
        use_cnn_branch=True,
        use_learnable_priors=True
    ).to(device)
    
    # Create input
    xrays = torch.randn(batch_size, num_views, 1, img_size, img_size).to(device)
    print(f"\nInput shape: {xrays.shape}")
    
    try:
        # Forward pass
        with torch.no_grad():
            output, aux_info = model(xrays)
        
        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: ({batch_size}, 1, {volume_size[0]}, {volume_size[1]}, {volume_size[2]})")
        
        # Validate output shape
        assert output.shape == (batch_size, 1, *volume_size), \
            f"Output shape mismatch: {output.shape} vs ({batch_size}, 1, {volume_size[0]}, {volume_size[1]}, {volume_size[2]})"
        print(f"✓ Output shape validated!")
        
        # Check auxiliary info
        print(f"\n  Auxiliary info:")
        for key, value in aux_info.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape}")
            else:
                print(f"    {key}: {type(value)}")
        
        # Validate aux info shapes
        if 'boundaries' in aux_info:
            assert aux_info['boundaries'].shape[0] == batch_size
            print(f"  ✓ Boundaries shape correct: {aux_info['boundaries'].shape}")
        
        if 'uncertainties' in aux_info:
            assert aux_info['uncertainties'].shape[0] == batch_size
            print(f"  ✓ Uncertainties shape correct: {aux_info['uncertainties'].shape}")
        
        print(f"\n✅ TEST 1 PASSED")
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test without CNN branch
    print("\n" + "-"*80)
    print("TEST 2: Without CNN Branch")
    print("-"*80)
    
    model_no_cnn = OptimizedCTRegression(
        volume_size=volume_size,
        xray_img_size=img_size,
        voxel_dim=voxel_dim,
        num_attn_blocks=2,
        num_ffn_blocks=4,
        num_heads=8,
        xray_feature_dim=xray_feature_dim,
        use_cnn_branch=False,
        use_learnable_priors=True
    ).to(device)
    
    try:
        with torch.no_grad():
            output, aux_info = model_no_cnn(xrays)
        
        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {output.shape}")
        assert output.shape == (batch_size, 1, *volume_size)
        print(f"✓ Output shape validated!")
        print(f"\n✅ TEST 2 PASSED")
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test without learnable priors
    print("\n" + "-"*80)
    print("TEST 3: Without Learnable Priors")
    print("-"*80)
    
    model_no_priors = OptimizedCTRegression(
        volume_size=volume_size,
        xray_img_size=img_size,
        voxel_dim=voxel_dim,
        num_attn_blocks=2,
        num_ffn_blocks=4,
        num_heads=8,
        xray_feature_dim=xray_feature_dim,
        use_cnn_branch=True,
        use_learnable_priors=False
    ).to(device)
    
    try:
        with torch.no_grad():
            output, aux_info = model_no_priors(xrays)
        
        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {output.shape}")
        assert output.shape == (batch_size, 1, *volume_size)
        print(f"✓ Output shape validated!")
        print(f"\n✅ TEST 3 PASSED")
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with different input sizes
    print("\n" + "-"*80)
    print("TEST 4: Different Batch Sizes")
    print("-"*80)
    
    for bs in [1, 4, 8]:
        print(f"\n  Testing batch size: {bs}")
        xrays_test = torch.randn(bs, num_views, 1, img_size, img_size).to(device)
        
        try:
            with torch.no_grad():
                output, _ = model(xrays_test)
            
            assert output.shape == (bs, 1, *volume_size)
            print(f"    ✓ Batch size {bs} successful: {output.shape}")
            
        except Exception as e:
            print(f"    ❌ Batch size {bs} failed: {e}")
            return False
    
    print(f"\n✅ TEST 4 PASSED")
    
    # Test gradient flow
    print("\n" + "-"*80)
    print("TEST 5: Gradient Flow")
    print("-"*80)
    
    model.train()
    xrays_grad = torch.randn(batch_size, num_views, 1, img_size, img_size, requires_grad=True).to(device)
    target = torch.randn(batch_size, 1, *volume_size).to(device)
    
    try:
        output, aux_info = model(xrays_grad)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        print(f"✓ Backward pass successful!")
        print(f"  Loss: {loss.item():.6f}")
        
        # Check if gradients exist
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grads, "No gradients computed!"
        print(f"✓ Gradients computed!")
        
        print(f"\n✅ TEST 5 PASSED")
        
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Parameter count
    print("\n" + "-"*80)
    print("MODEL STATISTICS")
    print("-"*80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Memory estimate
    param_size_mb = total_params * 4 / (1024**2)  # Assuming float32
    print(f"  Estimated memory (params only): {param_size_mb:.2f} MB")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    
    return True


if __name__ == "__main__":
    success = test_dimension_flow()
    sys.exit(0 if success else 1)
