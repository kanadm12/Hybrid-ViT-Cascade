"""
Quick test script for Spatial Clustering CT Generator
Tests architecture components and validates dimensions
"""

import torch
import torch.nn as nn
from spatial_cluster_architecture import (
    PositionEncodingModule,
    VoxelClusteringModule,
    ClusterAwareAttention,
    PositionIntensityTracker,
    SpatialClusteringCTGenerator,
    ClusterTrackingLoss
)

# Import PSNR and SSIM from training script
import sys
sys.path.append('.')
from train_spatial_clustering import compute_psnr, compute_ssim


def test_position_encoding():
    """Test position encoding module"""
    print("\n" + "="*80)
    print("Testing Position Encoding Module")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pos_encoder = PositionEncodingModule(num_freq_bands=10, d_model=128).to(device)
    volume_size = (64, 64, 64)
    
    pos_features = pos_encoder(volume_size, device)
    
    print(f"Volume size: {volume_size}")
    print(f"Position features shape: {pos_features.shape}")
    print(f"Expected: ({64*64*64}, 128) = {(64*64*64, 128)}")
    
    assert pos_features.shape == (64*64*64, 128), "Position encoding shape mismatch!"
    print("✓ Position encoding test passed!")


def test_voxel_clustering():
    """Test voxel clustering module"""
    print("\n" + "="*80)
    print("Testing Voxel Clustering Module")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 2
    num_voxels = 64 * 64 * 64
    voxel_dim = 256
    num_clusters = 64
    
    clustering = VoxelClusteringModule(
        voxel_dim=voxel_dim,
        num_clusters=num_clusters,
        use_position=True,
        use_intensity=True
    ).to(device)
    
    # Create dummy inputs
    voxel_features = torch.randn(batch_size, num_voxels, voxel_dim).to(device)
    position_features = torch.randn(num_voxels, 128).to(device)
    intensities = torch.randn(batch_size, num_voxels, 1).to(device)
    
    cluster_assignments, fused_features = clustering(voxel_features, position_features, intensities)
    
    print(f"Voxel features: {voxel_features.shape}")
    print(f"Cluster assignments: {cluster_assignments.shape}")
    print(f"Fused features: {fused_features.shape}")
    
    assert cluster_assignments.shape == (batch_size, num_voxels, num_clusters), "Cluster assignments shape mismatch!"
    assert fused_features.shape == (batch_size, num_voxels, voxel_dim), "Fused features shape mismatch!"
    
    # Check that assignments sum to 1 (soft assignment)
    assignment_sums = cluster_assignments.sum(dim=-1)
    print(f"Assignment sums (should be ~1.0): min={assignment_sums.min():.4f}, max={assignment_sums.max():.4f}")
    
    assert torch.allclose(assignment_sums, torch.ones_like(assignment_sums), atol=1e-5), "Assignments don't sum to 1!"
    print("✓ Voxel clustering test passed!")


def test_cluster_aware_attention():
    """Test cluster-aware attention"""
    print("\n" + "="*80)
    print("Testing Cluster-Aware Attention")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 2
    num_voxels = 1024  # Smaller for attention test
    voxel_dim = 256
    num_clusters = 64
    num_heads = 8
    
    attention = ClusterAwareAttention(
        voxel_dim=voxel_dim,
        num_heads=num_heads,
        num_clusters=num_clusters
    ).to(device)
    
    voxel_features = torch.randn(batch_size, num_voxels, voxel_dim).to(device)
    cluster_assignments = torch.softmax(torch.randn(batch_size, num_voxels, num_clusters), dim=-1).to(device)
    
    attended_features = attention(voxel_features, cluster_assignments)
    
    print(f"Input features: {voxel_features.shape}")
    print(f"Attended features: {attended_features.shape}")
    
    assert attended_features.shape == voxel_features.shape, "Attention output shape mismatch!"
    print("✓ Cluster-aware attention test passed!")


def test_position_intensity_tracker():
    """Test position-intensity tracker"""
    print("\n" + "="*80)
    print("Testing Position-Intensity Tracker")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tracker = PositionIntensityTracker().to(device)
    
    batch_size = 2
    volume_size = (64, 64, 64)
    
    pred_volume = torch.randn(batch_size, 1, *volume_size).to(device)
    gt_volume = torch.randn(batch_size, 1, *volume_size).to(device)
    
    # Position accuracy
    position_accuracy = tracker.compute_position_accuracy(pred_volume, gt_volume, volume_size)
    
    print(f"Predicted volume: {pred_volume.shape}")
    print(f"Position accuracy: {position_accuracy.shape}")
    print(f"Position accuracy range: [{position_accuracy.min():.4f}, {position_accuracy.max():.4f}]")
    
    assert position_accuracy.shape == (batch_size, *volume_size), "Position accuracy shape mismatch!"
    
    # Intensity accuracy
    intensity_metrics = tracker.compute_intensity_accuracy(pred_volume, gt_volume)
    
    print(f"Intensity MAE: {intensity_metrics['intensity_mae']:.4f}")
    print(f"Contrast error: {intensity_metrics['contrast_error']:.4f}")
    print(f"Voxel-wise MAE shape: {intensity_metrics['voxel_wise_mae'].shape}")
    
    print("✓ Position-intensity tracker test passed!")


def test_full_model():
    """Test full spatial clustering CT generator"""
    print("\n" + "="*80)
    print("Testing Full Spatial Clustering CT Generator")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SpatialClusteringCTGenerator(
        volume_size=(64, 64, 64),
        voxel_dim=256,
        num_clusters=64,
        num_heads=8,
        num_blocks=6
    ).to(device)
    
    batch_size = 2
    
    # Create dummy inputs
    frontal_xray = torch.randn(batch_size, 1, 512, 512).to(device)
    lateral_xray = torch.randn(batch_size, 1, 512, 512).to(device)
    gt_volume = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    
    print(f"Frontal X-ray: {frontal_xray.shape}")
    print(f"Lateral X-ray: {lateral_xray.shape}")
    print(f"Ground truth: {gt_volume.shape}")
    
    # Forward pass
    output = model(frontal_xray, lateral_xray, gt_volume)
    
    print("\nOutput keys:", list(output.keys()))
    print(f"Predicted volume: {output['pred_volume'].shape}")
    print(f"Cluster assignments: {output['cluster_assignments'].shape}")
    print(f"Voxel features: {output['voxel_features'].shape}")
    print(f"Position accuracy: {output['position_accuracy'].shape}")
    
    # Validate shapes
    assert output['pred_volume'].shape == (batch_size, 1, 64, 64, 64), "Predicted volume shape mismatch!"
    assert output['cluster_assignments'].shape == (batch_size, 64*64*64, 64), "Cluster assignments shape mismatch!"
    assert output['position_accuracy'].shape == (batch_size, 64, 64, 64), "Position accuracy shape mismatch!"
    
    print("\n✓ Full model test passed!")
    
    # Test loss
    print("\nTesting Loss Function...")
    loss_fn = ClusterTrackingLoss()
    
    loss_dict = loss_fn(
        output['pred_volume'],
        gt_volume,
        output['position_accuracy'],
        output['intensity_metrics'],
        output['cluster_assignments']
    )
    
    print("\nLoss breakdown:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.6f}")
    
    # Check backward pass
    loss_dict['total_loss'].backward()
    
    print("\n✓ Loss and backward pass test passed!")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params / 1e6:.2f}M")


def test_memory_usage():
    """Test memory usage"""
    print("\n" + "="*80)
    print("Testing Memory Usage")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    device = torch.device('cuda')
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = SpatialClusteringCTGenerator(
        volume_size=(64, 64, 64),
        voxel_dim=256,
        num_clusters=64,
        num_heads=8,
        num_blocks=6
    ).to(device)
    
    batch_size = 4  # Test with larger batch
    
    frontal = torch.randn(batch_size, 1, 512, 512).to(device)
    lateral = torch.randn(batch_size, 1, 512, 512).to(device)
    gt = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    
    # Forward pass
    output = model(frontal, lateral, gt)
    
    # Loss
    loss_fn = ClusterTrackingLoss()
    loss_dict = loss_fn(
        output['pred_volume'],
        gt,
        output['position_accuracy'],
        output['intensity_metrics'],
        output['cluster_assignments']
    )
    
    # Backward
    loss_dict['total_loss'].backward()
    
    # Check memory
    max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    
    print(f"Batch size: {batch_size}")
    print(f"Current memory: {current_memory:.2f} GB")
    print(f"Peak memory: {max_memory:.2f} GB")
    print(f"Memory per sample: {max_memory / batch_size:.2f} GB")
    
    print("\n✓ Memory usage test passed!")


def test_psnr_ssim():
    """Test PSNR and SSIM calculations"""
    print("\n" + "="*80)
    print("Testing PSNR and SSIM Metrics")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test volumes
    batch_size = 2
    pred = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    
    # Test 1: Perfect reconstruction (PSNR should be very high)
    target_perfect = pred.clone()
    psnr_perfect = compute_psnr(pred, target_perfect)
    ssim_perfect = compute_ssim(pred, target_perfect)
    
    print(f"Perfect reconstruction:")
    print(f"  PSNR: {psnr_perfect:.2f} dB (expected: very high)")
    print(f"  SSIM: {ssim_perfect:.4f} (expected: ~1.0)")
    
    # Test 2: Noisy reconstruction
    noise = torch.randn_like(pred) * 0.1
    target_noisy = pred + noise
    psnr_noisy = compute_psnr(pred, target_noisy)
    ssim_noisy = compute_ssim(pred, target_noisy)
    
    print(f"\nNoisy reconstruction (noise=0.1):")
    print(f"  PSNR: {psnr_noisy:.2f} dB")
    print(f"  SSIM: {ssim_noisy:.4f}")
    
    # Test 3: Very noisy reconstruction
    noise_large = torch.randn_like(pred) * 0.5
    target_very_noisy = pred + noise_large
    psnr_very_noisy = compute_psnr(pred, target_very_noisy)
    ssim_very_noisy = compute_ssim(pred, target_very_noisy)
    
    print(f"\nVery noisy reconstruction (noise=0.5):")
    print(f"  PSNR: {psnr_very_noisy:.2f} dB")
    print(f"  SSIM: {ssim_very_noisy:.4f}")
    
    # Validate that metrics make sense
    assert psnr_perfect > psnr_noisy > psnr_very_noisy, "PSNR should decrease with more noise!"
    assert ssim_perfect > ssim_noisy > ssim_very_noisy, "SSIM should decrease with more noise!"
    assert ssim_perfect > 0.99, "Perfect reconstruction SSIM should be close to 1.0!"
    
    print("\n✓ PSNR and SSIM test passed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("SPATIAL CLUSTERING CT GENERATOR - COMPONENT TESTS")
    print("="*80)
    
    try:
        test_position_encoding()
        test_voxel_clustering()
        test_cluster_aware_attention()
        test_position_intensity_tracker()
        test_full_model()
        test_memory_usage()
        test_psnr_ssim()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
