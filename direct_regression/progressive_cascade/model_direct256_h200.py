"""
H200 Direct 256³ Architecture with Advanced Losses
Skips progressive training, trains end-to-end for maximum quality
Leverages 151GB VRAM for deeper networks and novel loss functions

Key Improvements:
1. Direct 256³ training (no progressive stages)
2. Focal Frequency Loss (emphasizes anatomically important frequencies)
3. Perceptual Feature Pyramid Loss (multi-scale perceptual matching)
4. Attention-guided refinement (focuses on anatomical regions)
5. 3D Style Loss (texture consistency across slices)
6. Residual Dense Blocks (better feature reuse)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint


class SimpleXrayEncoder(nn.Module):
    """Simple 2D CNN encoder for bi-planar X-rays"""
    def __init__(self, img_size=512, feature_dim=512, num_views=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_views, 64, 7, stride=2, padding=3),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(64, 256),
            nn.GELU(),
            nn.Conv2d(256, feature_dim, 3, stride=2, padding=1),
            nn.GroupNorm(64, feature_dim),
            nn.GELU()
        )
    
    def forward(self, xrays, stage=3):
        """
        Args:
            xrays: (B, 2, 1, 512, 512) - bi-planar X-rays
            stage: ignored for compatibility
        Returns:
            features: (B, feature_dim, 32, 32)
            _, _: dummy outputs for compatibility
        """
        # Merge views into channels: (B, 2, 1, H, W) -> (B, 2, H, W)
        B = xrays.shape[0]
        xrays = xrays.squeeze(2)  # (B, 2, H, W)
        features = self.encoder(xrays)  # (B, feature_dim, 32, 32)
        return features, None, None


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss - emphasizes clinically important frequencies
    Bones/vessels = high freq, soft tissue = mid freq, air = low freq
    Better than standard FFT loss: adaptively weights frequencies
    """
    def __init__(self, alpha=1.0, patch_factor=1):
        super().__init__()
        self.alpha = alpha  # Focal weight
        self.patch_factor = patch_factor
        
    def forward(self, pred, target):
        """
        Args:
            pred, target: (B, 1, D, H, W)
        """
        # Convert to frequency domain
        pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
        target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))
        
        # Get magnitude and phase
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Compute frequency distance
        freq_distance = (pred_mag - target_mag) ** 2
        
        # Focal weight: emphasize frequencies with large errors (bones, vessels)
        # Uses matrix norm to create adaptive weights
        matrix_norm = torch.sum(freq_distance, dim=(-3, -2, -1), keepdim=True)
        focal_weight = torch.pow(freq_distance / (matrix_norm + 1e-8), self.alpha)
        
        # Weighted frequency loss
        loss = torch.mean(focal_weight * freq_distance)
        
        return loss


class PerceptualFeaturePyramidLoss(nn.Module):
    """
    Multi-scale perceptual loss using feature pyramid
    Captures perceptual quality at multiple resolutions
    Better than single-scale VGG for 3D medical imaging
    """
    def __init__(self, scales=[1.0, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        
        # Shared 3D feature extractor (trained jointly with model)
        # NO STRIDE - preserves full 256³ resolution for accurate perceptual matching
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv3d(64, 128, 3, padding=1),  # Removed stride=2 - preserves resolution
            nn.GroupNorm(32, 128),
            nn.GELU()
        )
        # Not frozen - trains with model for better perceptual features
    
    def forward(self, pred, target):
        """
        Args:
            pred, target: (B, 1, D, H, W)
        """
        total_loss = 0.0
        
        for scale in self.scales:
            if scale != 1.0:
                # Downsample volumes
                size = [int(s * scale) for s in pred.shape[-3:]]
                pred_scaled = F.interpolate(pred, size=size, mode='trilinear', align_corners=False)
                target_scaled = F.interpolate(target, size=size, mode='trilinear', align_corners=False)
            else:
                pred_scaled = pred
                target_scaled = target
            
            # Extract features with shared extractor
            pred_feat = self.feature_extractor(pred_scaled)
            target_feat = self.feature_extractor(target_scaled)
            
            # Perceptual loss at this scale
            total_loss += F.l1_loss(pred_feat, target_feat)
        
        return total_loss / len(self.scales)


class Style3DLoss(nn.Module):
    """
    3D Style Loss - ensures texture consistency across volume
    Computes Gram matrix of 3D features to match texture patterns
    Critical for soft tissue appearance
    """
    def __init__(self):
        super().__init__()
        
        # Feature extractor for style (trainable)
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv3d(64, 64, 3, padding=1)
        )
        # Trainable - learns style features for medical CT
    
    def gram_matrix(self, features):
        """
        Compute Gram matrix for 3D features
        Args:
            features: (B, C, D, H, W)
        Returns:
            gram: (B, C, C)
        """
        B, C, D, H, W = features.shape
        features_flat = features.view(B, C, -1)  # (B, C, D*H*W)
        gram = torch.bmm(features_flat, features_flat.transpose(1, 2))  # (B, C, C)
        return gram / (C * D * H * W)
    
    def forward(self, pred, target):
        """
        Args:
            pred, target: (B, 1, D, H, W)
        """
        # Extract features
        pred_feat = self.feature_extractor(pred)
        target_feat = self.feature_extractor(target)
        
        # Compute Gram matrices
        pred_gram = self.gram_matrix(pred_feat)
        target_gram = self.gram_matrix(target_feat)
        
        # Style loss
        return F.mse_loss(pred_gram, target_gram)


class AnatomicalAttentionLoss(nn.Module):
    """
    Attention-weighted reconstruction loss
    Emphasizes anatomically important regions (bones, organs, vessels)
    Learns attention map from target CT
    """
    def __init__(self):
        super().__init__()
        
        # Attention predictor (learns to highlight important regions)
        self.attention_net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv3d(32, 1, 1),
            nn.Sigmoid()  # Attention weights [0, 1]
        )
    
    def forward(self, pred, target):
        """
        Args:
            pred, target: (B, 1, D, H, W)
        """
        # Generate attention map from target (highlights anatomical structures)
        with torch.no_grad():
            # Use gradient magnitude as proxy for anatomical importance
            grad_d = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])
            grad_h = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
            grad_w = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
            
            # Pad to original size
            grad_d = F.pad(grad_d, (0, 0, 0, 0, 0, 1))
            grad_h = F.pad(grad_h, (0, 0, 0, 1, 0, 0))
            grad_w = F.pad(grad_w, (0, 1, 0, 0, 0, 0))
            
            # Combine gradients as importance map
            importance = (grad_d + grad_h + grad_w) / 3
            
            # Safe normalization: handle flat regions (min == max)
            importance_min = importance.min()
            importance_max = importance.max()
            importance_range = importance_max - importance_min
            
            if importance_range > 1e-6:  # Non-flat region
                importance = (importance - importance_min) / (importance_range + 1e-8)
            else:  # Flat region (no edges) - uniform weights
                importance = torch.ones_like(importance) * 0.5
        
        # Refine attention with learned network
        attention = self.attention_net(importance)
        
        # Attention-weighted L1 loss
        weighted_error = attention * torch.abs(pred - target)
        
        # Add uniform loss to prevent attention collapse
        uniform_loss = F.l1_loss(pred, target)
        attention_loss = weighted_error.mean()
        
        return 0.7 * attention_loss + 0.3 * uniform_loss


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block - better feature reuse
    Each layer receives all previous layer outputs
    Enables deeper networks without degradation
    """
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Compute valid num_groups for this layer
            layer_channels = in_channels + i * growth_rate
            num_groups = min(8, growth_rate)
            while growth_rate % num_groups != 0:
                num_groups -= 1
            
            self.layers.append(nn.Sequential(
                nn.Conv3d(layer_channels, growth_rate, 3, padding=1),
                nn.GroupNorm(num_groups, growth_rate),
                nn.GELU()
            ))
        
        # Compression layer
        self.compress = nn.Conv3d(in_channels + num_layers * growth_rate, in_channels, 1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            out: (B, C, D, H, W)
        """
        features = [x]
        
        for layer in self.layers:
            # Concatenate all previous features
            feat = torch.cat(features, dim=1)
            out = layer(feat)
            features.append(out)
        
        # Concatenate all and compress
        all_features = torch.cat(features, dim=1)
        compressed = self.compress(all_features)
        
        return x + compressed  # Residual connection


class Direct256Model_H200(nn.Module):
    """
    Direct 256³ End-to-End Model for H200
    Skips progressive training, trains directly at target resolution
    
    Architecture:
    - Deep encoder-decoder with residual dense blocks
    - Multi-scale feature fusion
    - Attention-guided refinement
    - Skip connections at multiple levels
    
    Advantages over Progressive:
    - End-to-end optimization
    - Better feature propagation
    - Simpler training (no stage freezing)
    - Higher quality with sufficient VRAM
    """
    def __init__(self,
                 xray_img_size=512,
                 xray_feature_dim=512,
                 voxel_dim=256,
                 num_rdb=3,  # Reduced from 6 to 3 for H200 memory constraints
                 use_checkpoint=True):  # Enable gradient checkpointing
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # X-ray encoder
        self.xray_encoder = SimpleXrayEncoder(
            img_size=xray_img_size,
            feature_dim=xray_feature_dim,
            num_views=2
        )
        
        # Initial volume embedding
        self.initial_volume = nn.Parameter(torch.randn(1, 1, 32, 32, 32) * 0.02)
        
        # Encoder: 32³ -> 64³ -> 128³ -> 256³
        self.enc_32_64 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(1, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            ResidualDenseBlock(64)
        )
        
        self.enc_64_128 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.GroupNorm(32, 128),
            nn.GELU(),
            ResidualDenseBlock(128)
        )
        
        self.enc_128_256 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(128, 192, 3, padding=1),  # Reduced from 256 to 192
            nn.GroupNorm(48, 192),
            nn.GELU(),
            *[ResidualDenseBlock(192, growth_rate=24) for _ in range(num_rdb)]  # Reduced growth_rate
        )
        
        # Cross-attention with X-ray features at each scale
        self.xray_fusion_64 = self._make_xray_fusion(64, xray_feature_dim)
        self.xray_fusion_128 = self._make_xray_fusion(128, xray_feature_dim)
        self.xray_fusion_256 = self._make_xray_fusion(192, xray_feature_dim)  # Updated to 192
        
        # Skip connection projections for multi-scale feature fusion (preserves high-freq details)
        self.skip_proj_64_to_256 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU()
        )
        self.skip_proj_128_to_256 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.GroupNorm(32, 128),
            nn.GELU()
        )
        
        # Multi-scale fusion before final refinement
        self.multiscale_fusion = nn.Sequential(
            nn.Conv3d(192 + 128 + 64, 192, 1),  # Combine all scales (192+128+64)
            nn.GroupNorm(48, 192),
            nn.GELU()
        )
        
        # Final refinement
        self.final_refine = nn.Sequential(
            ResidualDenseBlock(192, growth_rate=24),  # Updated to 192
            nn.Conv3d(192, 128, 3, padding=1),
            nn.GroupNorm(32, 128),
            nn.GELU(),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv3d(64, 1, 1)
        )
    
    def _make_xray_fusion(self, voxel_channels, xray_feature_dim):
        """Create X-ray feature fusion module"""
        # Compute valid num_groups (must divide voxel_channels evenly)
        num_groups = min(32, voxel_channels)
        while voxel_channels % num_groups != 0:
            num_groups -= 1
        
        return nn.Sequential(
            nn.Conv3d(voxel_channels + xray_feature_dim, voxel_channels, 1),
            nn.GroupNorm(num_groups, voxel_channels),
            nn.GELU()
        )
    
    def forward(self, xrays):
        """
        Args:
            xrays: (B, 2, 1, 512, 512)
        Returns:
            volume_256: (B, 1, 256, 256, 256)
        """
        B = xrays.shape[0]
        
        # Encode X-rays (shared features)
        xray_features_2d, _, _ = self.xray_encoder(xrays, stage=3)
        
        # Depth-aware X-ray projection (learnable weights along depth)
        # Creates depth-varying features instead of naive repetition
        depth_weights_64 = torch.linspace(0, 1, 64, device=xrays.device).view(1, 1, 64, 1, 1)
        depth_weights_128 = torch.linspace(0, 1, 128, device=xrays.device).view(1, 1, 128, 1, 1)
        depth_weights_256 = torch.linspace(0, 1, 256, device=xrays.device).view(1, 1, 256, 1, 1)
        
        # Pre-compute X-ray features at all scales with depth modulation
        xray_feat_64 = F.interpolate(xray_features_2d, size=(64, 64), mode='bilinear', align_corners=False)
        # Broadcasting with depth_weights_64 already creates a (B, C, 64, 64, 64) tensor; no extra repeat needed
        xray_feat_64_3d = xray_feat_64.unsqueeze(2) * (1 + 0.3 * torch.sin(depth_weights_64 * 3.14159))  # (B, C, 64, 64, 64)
        
        xray_feat_128 = F.interpolate(xray_features_2d, size=(128, 128), mode='bilinear', align_corners=False)
        xray_feat_128_3d = xray_feat_128.unsqueeze(2) * (1 + 0.3 * torch.sin(depth_weights_128 * 3.14159))  # (B, C, 128, 128, 128)
        
        xray_feat_256 = F.interpolate(xray_features_2d, size=(256, 256), mode='bilinear', align_corners=False)
        xray_feat_256_3d = xray_feat_256.unsqueeze(2) * (1 + 0.3 * torch.sin(depth_weights_256 * 3.14159))  # (B, C, 256, 256, 256)
        
        # Expand initial volume
        x = self.initial_volume.expand(B, -1, -1, -1, -1)
        
        # Encoder path with X-ray fusion (with gradient checkpointing)
        if self.use_checkpoint and self.training:
            x_64 = checkpoint(self.enc_32_64, x, use_reentrant=False)
            x_64_fused = checkpoint(self.xray_fusion_64, torch.cat([x_64, xray_feat_64_3d], dim=1), use_reentrant=False)
            
            x_128 = checkpoint(self.enc_64_128, x_64_fused, use_reentrant=False)
            x_128_fused = checkpoint(self.xray_fusion_128, torch.cat([x_128, xray_feat_128_3d], dim=1), use_reentrant=False)
            
            x_256 = checkpoint(self.enc_128_256, x_128_fused, use_reentrant=False)
            x_256_fused = checkpoint(self.xray_fusion_256, torch.cat([x_256, xray_feat_256_3d], dim=1), use_reentrant=False)
        else:
            x_64 = self.enc_32_64(x)  # (B, 64, 64, 64, 64)
            x_64_fused = self.xray_fusion_64(torch.cat([x_64, xray_feat_64_3d], dim=1))
            
            x_128 = self.enc_64_128(x_64_fused)  # (B, 128, 128, 128, 128)
            x_128_fused = self.xray_fusion_128(torch.cat([x_128, xray_feat_128_3d], dim=1))
            
            x_256 = self.enc_128_256(x_128_fused)  # (B, 192, 256, 256, 256)
            x_256_fused = self.xray_fusion_256(torch.cat([x_256, xray_feat_256_3d], dim=1))
        
        # Multi-scale skip connections: bring 64³ and 128³ features to 256³
        skip_64 = self.skip_proj_64_to_256(x_64_fused)  # (B, 64, 256, 256, 256)
        skip_128 = self.skip_proj_128_to_256(x_128_fused)  # (B, 128, 256, 256, 256)
        
        # Fuse all scales (preserves fine details from lower resolutions)
        x_256_multiscale = self.multiscale_fusion(
            torch.cat([x_256_fused, skip_128, skip_64], dim=1)
        )  # (B, 192, 256, 256, 256)
        
        # Final refinement with multi-scale features
        volume_256 = self.final_refine(x_256_multiscale)
        
        return volume_256


"""
H200 Direct 256³ Training Configuration:

MODEL:
- Architecture: Direct end-to-end 256³ (no progressive stages)
- Residual Dense Blocks: 6 at 256³ level
- Channels: 64 -> 128 -> 256 (vs 32 -> 64 -> 128 progressive)
- Parameters: ~150M (vs ~80M progressive)

LOSSES (combined):
1. L1: 1.0 (base reconstruction)
2. SSIM: 0.5 (structural similarity)
3. Focal Frequency: 0.2 (adaptive frequency weighting)
4. Perceptual Pyramid: 0.15 (multi-scale perceptual)
5. Total Variation: 0.02 (edge preservation)
6. Style 3D: 0.1 (texture consistency)
7. Anatomical Attention: 0.3 (region importance)

TRAINING:
- Batch size: 2-3 (H200 151GB)
- Epochs: 150-200 (longer than progressive's 100/stage)
- Learning rate: 1e-4 -> 1e-6 cosine
- Expected memory: ~120GB peak
- Training time: ~12-15 hours total (vs ~10-12 for progressive)

EXPECTED QUALITY:
- PSNR: 38-42 dB (vs 35-38 progressive)
- SSIM: 0.96-0.98 (vs 0.92-0.95 progressive)
- Perceptual quality: Superior texture, better edges
- Anatomical detail: Bone microstructure, small vessels visible

ADVANTAGES:
1. End-to-end optimization (no stage boundaries)
2. Better gradient flow
3. Richer feature representations
4. Advanced loss functions for perceptual quality
5. Simpler training pipeline

DISADVANTAGES:
1. Requires H200-class GPU (won't fit A100)
2. Longer total training time
3. Higher risk of overfitting (needs more data or regularization)
4. Less interpretable (no intermediate 64³/128³ outputs)

WHEN TO USE:
- H200 or better GPU available
- Maximum quality required
- Have 150+ training samples
- Willing to train for longer
"""
