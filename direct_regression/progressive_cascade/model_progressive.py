"""
Progressive Multi-Scale CT Reconstruction: 64³→128³→256³ Cascade
Combines proven baseline with multi-resolution refinement for high PSNR/SSIM
while capturing fine details and remaining memory-efficient.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '../..')

from models.diagnostic_losses import XrayConditioningModule
from models.hybrid_vit_backbone import HybridViT3D


class MultiScaleXrayEncoder(nn.Module):
    """
    Progressive X-ray encoder: 512→256→128 downsampling
    Provides multi-scale features for cross-attention at each stage
    """
    def __init__(self, img_size=512, in_channels=1, base_dim=512, num_views=2):
        super().__init__()
        
        # Main X-ray encoder at full resolution
        self.xray_encoder = XrayConditioningModule(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=base_dim,
            num_views=num_views,
            time_embed_dim=256,
            cond_dim=1024,
            share_view_weights=False
        )
        
        # Progressive downsampling branches for multi-scale features
        # Stage 1: 64³ - use 128x128 X-ray features
        self.to_stage1 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, 3, stride=2, padding=1),  # 512->256
            nn.GroupNorm(32, base_dim),
            nn.GELU(),
            nn.Conv2d(base_dim, base_dim, 3, stride=2, padding=1),  # 256->128
            nn.GroupNorm(32, base_dim),
            nn.GELU()
        )
        
        # Stage 2: 128³ - use 256x256 X-ray features
        self.to_stage2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, 3, stride=2, padding=1),  # 512->256
            nn.GroupNorm(32, base_dim),
            nn.GELU()
        )
        
        # Stage 3: 256³ - use full 512x512 X-ray features
        # No downsampling needed, use features directly
        
    def forward(self, xrays, stage=1):
        """
        Args:
            xrays: (B, num_views, 1, H, W) - input X-ray images
            stage: 1, 2, or 3 - which stage to extract features for
        Returns:
            xray_features_2d: Multi-scale X-ray features
            time_xray_cond: Conditioning vector
            xray_context: Global context
        """
        batch_size = xrays.shape[0]
        
        # Dummy timestep (not used in direct regression)
        dummy_t = torch.zeros(batch_size, 256, device=xrays.device)
        
        # Encode X-rays at full resolution
        xray_context, time_xray_cond, xray_features_2d = self.xray_encoder(xrays, dummy_t)
        
        # Progressive downsampling based on stage
        if stage == 1:
            # 64³ volume → 128x128 X-ray features
            xray_features_2d = self.to_stage1(xray_features_2d)
        elif stage == 2:
            # 128³ volume → 256x256 X-ray features
            xray_features_2d = self.to_stage2(xray_features_2d)
        # stage == 3: use full 512x512 features
        
        return xray_features_2d, time_xray_cond, xray_context


class Stage1Base64(nn.Module):
    """
    Stage 1: Base 64³ reconstruction
    Focus: Coarse structure and overall anatomy
    """
    def __init__(self, 
                 volume_size=(64, 64, 64),
                 xray_img_size=512,
                 voxel_dim=256,
                 vit_depth=4,
                 num_heads=4,
                 xray_feature_dim=512):
        super().__init__()
        
        self.volume_size = volume_size
        
        # Shared multi-scale X-ray encoder
        self.xray_encoder = MultiScaleXrayEncoder(
            img_size=xray_img_size,
            in_channels=1,
            base_dim=xray_feature_dim,
            num_views=2
        )
        
        # ViT backbone for 64³ volume
        self.vit_backbone = HybridViT3D(
            volume_size=volume_size,
            in_channels=1,
            voxel_dim=voxel_dim,
            depth=vit_depth,
            num_heads=num_heads,
            context_dim=xray_feature_dim,
            cond_dim=1024,
            use_prev_stage=False
        )
        
        # Learnable initial volume embedding
        D, H, W = volume_size
        self.initial_volume = nn.Parameter(torch.randn(1, 1, D, H, W) * 0.01)
        
    def forward(self, xrays):
        """
        Args:
            xrays: (B, 2, 1, 512, 512)
        Returns:
            volume_64: (B, 1, 64, 64, 64)
        """
        batch_size = xrays.shape[0]
        
        # Extract stage 1 features (128x128)
        xray_features_2d, time_xray_cond, _ = self.xray_encoder(xrays, stage=1)
        
        # Expand initial volume
        x = self.initial_volume.expand(batch_size, -1, -1, -1, -1)
        
        # Generate 64³ volume
        volume_64 = self.vit_backbone(
            x=x,
            context=xray_features_2d.flatten(2).transpose(1, 2),
            cond=time_xray_cond,
            prev_stage_embed=None
        )
        
        return volume_64


class Stage2Refiner128(nn.Module):
    """
    Stage 2: Refine to 128³
    Focus: Add texture and medium-frequency details
    Uses stage 1 output as prior
    """
    def __init__(self,
                 volume_size=(128, 128, 128),
                 voxel_dim=256,
                 vit_depth=6,
                 num_heads=8,
                 xray_feature_dim=512):
        super().__init__()
        
        self.volume_size = volume_size
        
        # Upsampling from 64³ to 128³
        self.upsample_from_64 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )
        
        # Refinement ViT for 128³
        self.vit_refiner = HybridViT3D(
            volume_size=volume_size,
            in_channels=32,
            voxel_dim=voxel_dim,
            depth=vit_depth,
            num_heads=num_heads,
            context_dim=xray_feature_dim,
            cond_dim=1024,
            use_prev_stage=False
        )
        
        # Residual connection
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, volume_64, xray_features_2d, time_xray_cond):
        """
        Args:
            volume_64: (B, 1, 64, 64, 64) - stage 1 output
            xray_features_2d: (B, C, 256, 256) - stage 2 X-ray features
            time_xray_cond: (B, 1024) - conditioning vector
        Returns:
            volume_128: (B, 1, 128, 128, 128)
        """
        # Upsample 64³ to 128³
        x = self.upsample_from_64(volume_64)
        
        # Refine with ViT
        refinement = self.vit_refiner(
            x=x,
            context=xray_features_2d.flatten(2).transpose(1, 2),
            cond=time_xray_cond,
            prev_stage_embed=None
        )
        
        # Ensure refinement is 1 channel (should already be from HybridViT3D)
        if refinement.shape[1] != 1:
            raise ValueError(f"Expected refinement to have 1 channel, got {refinement.shape[1]} channels. Shape: {refinement.shape}")
        
        # Residual connection with upsampled base
        volume_64_upsampled = F.interpolate(volume_64, size=self.volume_size, 
                                            mode='trilinear', align_corners=False)
        volume_128 = volume_64_upsampled + self.residual_weight * refinement
        
        return volume_128


class Stage3Refiner256(nn.Module):
    """
    Stage 3: Refine to 256³
    Focus: Capture fine details, edges, and high-frequency content
    Uses stage 2 output as prior
    """
    def __init__(self,
                 volume_size=(256, 256, 256),
                 voxel_dim=256,
                 vit_depth=8,
                 num_heads=8,
                 xray_feature_dim=512,
                 use_gradient_checkpointing=True):
        super().__init__()
        
        self.volume_size = volume_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Upsampling from 128³ to 256³
        self.upsample_from_128 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )
        
        # Refinement ViT for 256³ (with gradient checkpointing option)
        self.vit_refiner = HybridViT3D(
            volume_size=volume_size,
            in_channels=32,
            voxel_dim=voxel_dim,
            depth=vit_depth,
            num_heads=num_heads,
            context_dim=xray_feature_dim,
            cond_dim=1024,
            use_prev_stage=False
        )
        
        # High-frequency enhancement
        self.detail_enhancer = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv3d(32, 1, 1)
        )
        
        # Residual and detail blending weights
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.detail_weight = nn.Parameter(torch.ones(1) * 0.3)
        
    def forward(self, volume_128, xray_features_2d, time_xray_cond):
        """
        Args:
            volume_128: (B, 1, 128, 128, 128) - stage 2 output
            xray_features_2d: (B, C, 512, 512) - stage 3 X-ray features
            time_xray_cond: (B, 1024) - conditioning vector
        Returns:
            volume_256: (B, 1, 256, 256, 256)
        """
        # Upsample 128³ to 256³
        x = self.upsample_from_128(volume_128)
        
        # Refine with ViT (with optional gradient checkpointing)
        if self.use_gradient_checkpointing and self.training:
            refinement = torch.utils.checkpoint.checkpoint(
                self._vit_forward,
                x, xray_features_2d, time_xray_cond,
                use_reentrant=False
            )
        else:
            refinement = self._vit_forward(x, xray_features_2d, time_xray_cond)
        
        # Upsample base volume
        volume_128_upsampled = F.interpolate(volume_128, size=self.volume_size,
                                             mode='trilinear', align_corners=False)
        
        # Extract high-frequency details
        details = self.detail_enhancer(volume_128_upsampled)
        
        # Combine: base + refinement + details
        volume_256 = (volume_128_upsampled + 
                     self.residual_weight * refinement +
                     self.detail_weight * details)
        
        return volume_256
    
    def _vit_forward(self, x, xray_features_2d, time_xray_cond):
        """Helper for gradient checkpointing"""
        return self.vit_refiner(
            x=x,
            context=xray_features_2d.flatten(2).transpose(1, 2),
            cond=time_xray_cond,
            prev_stage_embed=None
        )


class ProgressiveCascadeModel(nn.Module):
    """
    Full progressive cascade: 64³→128³→256³
    Can be trained stage-by-stage or end-to-end
    """
    def __init__(self,
                 xray_img_size=512,
                 xray_feature_dim=512,
                 voxel_dim=256,
                 use_gradient_checkpointing=True):
        super().__init__()
        
        # Shared X-ray encoder
        self.xray_encoder = MultiScaleXrayEncoder(
            img_size=xray_img_size,
            in_channels=1,
            base_dim=xray_feature_dim,
            num_views=2
        )
        
        # Stage 1: 64³ base
        self.stage1 = Stage1Base64(
            volume_size=(64, 64, 64),
            xray_img_size=xray_img_size,
            voxel_dim=voxel_dim,
            vit_depth=4,
            num_heads=4,
            xray_feature_dim=xray_feature_dim
        )
        
        # Stage 2: 128³ refiner
        self.stage2 = Stage2Refiner128(
            volume_size=(128, 128, 128),
            voxel_dim=voxel_dim,
            vit_depth=6,
            num_heads=8,
            xray_feature_dim=xray_feature_dim
        )
        
        # Stage 3: 256³ refiner
        self.stage3 = Stage3Refiner256(
            volume_size=(256, 256, 256),
            voxel_dim=voxel_dim,
            vit_depth=8,
            num_heads=8,
            xray_feature_dim=xray_feature_dim,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
    def forward(self, xrays, return_intermediate=False, max_stage=3):
        """
        Args:
            xrays: (B, 2, 1, 512, 512)
            return_intermediate: Return all intermediate outputs
            max_stage: Maximum stage to compute (1, 2, or 3)
        Returns:
            If return_intermediate:
                dict with 'stage1', 'stage2', 'stage3' keys
            Else:
                final volume from max_stage
        """
        outputs = {}
        
        # Stage 1: Generate 64³
        volume_64 = self.stage1(xrays)
        outputs['stage1'] = volume_64
        
        if max_stage == 1:
            return outputs if return_intermediate else volume_64
        
        # Stage 2: Refine to 128³
        xray_features_2d_stage2, time_xray_cond, _ = self.xray_encoder(xrays, stage=2)
        volume_128 = self.stage2(volume_64, xray_features_2d_stage2, time_xray_cond)
        outputs['stage2'] = volume_128
        
        if max_stage == 2:
            return outputs if return_intermediate else volume_128
        
        # Stage 3: Refine to 256³
        xray_features_2d_stage3, time_xray_cond, _ = self.xray_encoder(xrays, stage=3)
        volume_256 = self.stage3(volume_128, xray_features_2d_stage3, time_xray_cond)
        outputs['stage3'] = volume_256
        
        return outputs if return_intermediate else volume_256
    
    def freeze_stage(self, stage):
        """Freeze parameters of a specific stage"""
        if stage == 1:
            for param in self.stage1.parameters():
                param.requires_grad = False
            print("Stage 1 (64³) frozen")
        elif stage == 2:
            for param in self.stage2.parameters():
                param.requires_grad = False
            print("Stage 2 (128³) frozen")
        elif stage == 3:
            for param in self.stage3.parameters():
                param.requires_grad = False
            print("Stage 3 (256³) frozen")
    
    def unfreeze_stage(self, stage):
        """Unfreeze parameters of a specific stage"""
        if stage == 1:
            for param in self.stage1.parameters():
                param.requires_grad = True
            print("Stage 1 (64³) unfrozen")
        elif stage == 2:
            for param in self.stage2.parameters():
                param.requires_grad = True
            print("Stage 2 (128³) unfrozen")
        elif stage == 3:
            for param in self.stage3.parameters():
                param.requires_grad = True
            print("Stage 3 (256³) unfrozen")


if __name__ == "__main__":
    # Test the model
    print("Testing Progressive Cascade Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProgressiveCascadeModel().to(device)
    
    # Dummy input
    xrays = torch.randn(2, 2, 1, 512, 512).to(device)
    
    # Test each stage
    print("\n=== Stage 1 Only ===")
    with torch.no_grad():
        out1 = model(xrays, max_stage=1)
        print(f"Stage 1 output shape: {out1.shape}")
    
    print("\n=== Stage 1+2 ===")
    with torch.no_grad():
        out2 = model(xrays, max_stage=2)
        print(f"Stage 2 output shape: {out2.shape}")
    
    print("\n=== Full Cascade (1+2+3) ===")
    with torch.no_grad():
        out3 = model(xrays, max_stage=3)
        print(f"Stage 3 output shape: {out3.shape}")
    
    print("\n=== Return All Intermediate ===")
    with torch.no_grad():
        outputs = model(xrays, return_intermediate=True, max_stage=3)
        for stage, vol in outputs.items():
            print(f"{stage}: {vol.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n Total parameters: {total_params:,}")
