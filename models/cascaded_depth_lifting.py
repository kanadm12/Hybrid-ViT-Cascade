"""
Cascaded Depth Lifting Module
Multi-resolution depth priors for progressive generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


class ResolutionDepthPriors:
    """
    Anatomical depth priors at different resolutions
    """
    
    PRIORS = {
        64: {
            'anterior': (0, 16),      # Ribs, sternum (0-25% depth)
            'mid': (16, 48),          # Heart, vessels (25-75% depth)
            'posterior': (48, 64)     # Spine, esophagus (75-100% depth)
        },
        128: {
            'anterior': (0, 32),
            'mid': (32, 96),
            'posterior': (96, 128)
        },
        256: {
            'anterior': (0, 64),
            'mid': (64, 192),
            'posterior': (192, 256)
        },
        512: {
            'anterior': (0, 128),
            'mid': (128, 384),
            'posterior': (384, 512)
        },
        604: {
            'anterior': (0, 151),
            'mid': (151, 453),
            'posterior': (453, 604)
        }
    }
    
    @staticmethod
    def get_priors(depth_size: int) -> Dict[str, Tuple[int, int]]:
        """Get anatomical priors for given depth resolution"""
        if depth_size in ResolutionDepthPriors.PRIORS:
            return ResolutionDepthPriors.PRIORS[depth_size]
        
        # Interpolate for custom sizes
        ratio = depth_size / 604.0
        return {
            'anterior': (0, int(151 * ratio)),
            'mid': (int(151 * ratio), int(453 * ratio)),
            'posterior': (int(453 * ratio), depth_size)
        }


class CascadedDepthWeightNetwork(nn.Module):
    """
    Resolution-adaptive depth weight network
    """
    
    def __init__(self, feature_dim: int, max_depth: int, resolution_level: int = 1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_depth = max_depth
        self.resolution_level = resolution_level
        
        # Get anatomical priors for this resolution
        self.priors = ResolutionDepthPriors.get_priors(max_depth)
        
        # Depth prediction network
        self.depth_net = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.GroupNorm(8, feature_dim // 2),
            nn.SiLU(),
            nn.Conv2d(feature_dim // 2, feature_dim // 4, 3, padding=1),
            nn.GroupNorm(8, feature_dim // 4),
            nn.SiLU(),
            nn.Conv2d(feature_dim // 4, max_depth, 1)  # Predict depth weights
        )
        
        # Prior injection network
        self.prior_modulation = nn.Sequential(
            nn.Conv2d(feature_dim, max_depth, 1),
            nn.Sigmoid()
        )
    
    def forward(self, xray_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xray_features: (B, C, H, W) X-ray features
        Returns:
            depth_weights: (B, H, W, D) depth distribution per pixel
        """
        batch_size, _, height, width = xray_features.shape
        
        # Predict depth weights
        depth_logits = self.depth_net(xray_features)  # (B, D, H, W)
        
        # Get prior modulation
        prior_mask = self.prior_modulation(xray_features)  # (B, D, H, W)
        
        # Apply anatomical priors as soft constraints
        depth_weights = torch.softmax(depth_logits, dim=1)  # (B, D, H, W)
        depth_weights = depth_weights * prior_mask
        
        # Re-normalize
        depth_weights = depth_weights / (depth_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Transpose to (B, H, W, D)
        depth_weights = depth_weights.permute(0, 2, 3, 1)
        
        return depth_weights


class CascadedDepthLifting(nn.Module):
    """
    Multi-resolution depth lifting module
    Progressively refines 3D structure from coarse to fine
    """
    
    def __init__(self, 
                 feature_dim: int = 512,
                 depth_sizes: List[int] = [64, 128, 256],
                 use_prev_stage: bool = True):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.depth_sizes = depth_sizes
        self.use_prev_stage = use_prev_stage
        
        # Create depth weight network for each resolution
        self.depth_networks = nn.ModuleDict({
            f'depth_{d}': CascadedDepthWeightNetwork(
                feature_dim=feature_dim,
                max_depth=d,
                resolution_level=i
            )
            for i, d in enumerate(depth_sizes)
        })
        
        # Previous stage integration (if using cascading)
        if use_prev_stage:
            self.prev_stage_fusion = nn.ModuleDict({
                f'fusion_{d}': nn.Sequential(
                    nn.Conv3d(feature_dim * 2, feature_dim, 3, padding=1),
                    nn.GroupNorm(8, feature_dim),
                    nn.SiLU(),
                    nn.Conv3d(feature_dim, feature_dim, 3, padding=1)
                )
                for d in depth_sizes[1:]  # Skip first stage
            })
    
    def lift_to_3d(self, xray_features: torch.Tensor, depth_size: int,
                   prev_stage_volume: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Lift 2D X-ray features to 3D volume at specific resolution
        
        Args:
            xray_features: (B, C, H, W)
            depth_size: Target depth dimension
            prev_stage_volume: (B, C, D_prev, H_prev, W_prev) - from previous stage
        Returns:
            volume_3d: (B, C, D, H, W)
        """
        batch_size, channels, height, width = xray_features.shape
        
        # Get depth weights for this resolution
        depth_weights = self.depth_networks[f'depth_{depth_size}'](xray_features)  # (B, H, W, D)
        
        # Broadcast 2D features to 3D using depth weights
        # Expand features: (B, C, H, W) -> (B, C, H, W, 1)
        xray_features_expanded = xray_features.unsqueeze(-1)  # (B, C, H, W, 1)
        
        # Expand weights: (B, H, W, D) -> (B, 1, H, W, D)
        depth_weights_expanded = depth_weights.unsqueeze(1)  # (B, 1, H, W, D)
        
        # Multiply and sum: (B, C, H, W, D)
        volume_3d = xray_features_expanded * depth_weights_expanded
        
        # Permute to (B, C, D, H, W)
        volume_3d = volume_3d.permute(0, 1, 4, 2, 3)
        
        # Integrate previous stage if provided
        if prev_stage_volume is not None and self.use_prev_stage and depth_size > self.depth_sizes[0]:
            # Upsample previous stage to current resolution
            prev_upsampled = F.interpolate(
                prev_stage_volume,
                size=(depth_size, height, width),
                mode='trilinear',
                align_corners=True
            )
            
            # Concatenate and fuse
            combined = torch.cat([volume_3d, prev_upsampled], dim=1)  # (B, 2C, D, H, W)
            volume_3d = self.prev_stage_fusion[f'fusion_{depth_size}'](combined)
        
        return volume_3d
    
    def forward(self, xray_features: torch.Tensor, 
                target_depth: int,
                prev_stage_volume: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for specific resolution
        
        Args:
            xray_features: (B, C, H, W)
            target_depth: Target depth dimension (64, 128, 256, etc.)
            prev_stage_volume: Previous stage output (if cascading)
        Returns:
            volume_3d: (B, C, D, H, W)
        """
        return self.lift_to_3d(xray_features, target_depth, prev_stage_volume)


# Test code
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Cascaded Depth Lifting...")
    
    # Create module
    depth_lifter = CascadedDepthLifting(
        feature_dim=512,
        depth_sizes=[64, 128, 256],
        use_prev_stage=True
    ).to(device)
    
    # Test Stage 1
    batch_size = 2
    xray_features = torch.randn(batch_size, 512, 64, 64).to(device)
    
    print("\n=== Stage 1 (64³) ===")
    volume_64 = depth_lifter(xray_features, target_depth=64)
    print(f"Output shape: {volume_64.shape}")
    print(f"Memory: {volume_64.element_size() * volume_64.nelement() / 1024**2:.2f} MB")
    
    # Test Stage 2 (with previous stage)
    print("\n=== Stage 2 (128³) with Stage 1 conditioning ===")
    xray_features_128 = torch.randn(batch_size, 512, 128, 128).to(device)
    volume_128 = depth_lifter(xray_features_128, target_depth=128, prev_stage_volume=volume_64)
    print(f"Output shape: {volume_128.shape}")
    print(f"Memory: {volume_128.element_size() * volume_128.nelement() / 1024**2:.2f} MB")
    
    # Test Stage 3 (with previous stage)
    print("\n=== Stage 3 (256³) with Stage 2 conditioning ===")
    xray_features_256 = torch.randn(batch_size, 512, 256, 256).to(device)
    volume_256 = depth_lifter(xray_features_256, target_depth=256, prev_stage_volume=volume_128)
    print(f"Output shape: {volume_256.shape}")
    print(f"Memory: {volume_256.element_size() * volume_256.nelement() / 1024**2:.2f} MB")
    
    # Check anatomical priors
    print("\n=== Anatomical Priors ===")
    for depth_size in [64, 128, 256, 512, 604]:
        priors = ResolutionDepthPriors.get_priors(depth_size)
        print(f"\nDepth {depth_size}:")
        for region, (start, end) in priors.items():
            print(f"  {region}: {start}-{end} ({100*(end-start)/depth_size:.1f}% of volume)")
    
    # Parameter count
    total_params = sum(p.numel() for p in depth_lifter.parameters())
    print(f"\n=== Model Stats ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    
    print("\nCascaded depth lifting test completed!")
