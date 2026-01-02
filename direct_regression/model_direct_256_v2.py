"""
Direct CT Regression Model for 256³ Resolution
Based on the successful 64³ architecture with memory optimizations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '..')

from models.diagnostic_losses import XrayConditioningModule
from models.hybrid_vit_backbone import HybridViT3D


class DirectCTRegression256(nn.Module):
    """
    Direct regression: X-rays → 256³ CT volume
    Same architecture as 64³ model but with progressive upsampling
    """
    
    def __init__(self,
                 volume_size=(256, 256, 256),
                 xray_img_size=512,
                 voxel_dim=256,
                 vit_depth=4,
                 num_heads=4,
                 xray_feature_dim=512,
                 use_checkpointing=True):
        super().__init__()
        
        self.volume_size = volume_size
        self.use_checkpointing = use_checkpointing
        
        # X-ray encoder (same as 64³ model)
        self.xray_encoder = XrayConditioningModule(
            img_size=xray_img_size,
            in_channels=1,
            embed_dim=xray_feature_dim,
            num_views=2,
            time_embed_dim=256,
            cond_dim=1024,
            share_view_weights=False
        )
        
        # ViT backbone at 64³ resolution (same as successful 64³ model)
        self.vit_backbone = HybridViT3D(
            volume_size=(64, 64, 64),  # Process at 64³ first
            in_channels=1,
            voxel_dim=voxel_dim,
            depth=vit_depth,
            num_heads=num_heads,
            context_dim=xray_feature_dim,
            cond_dim=1024,
            use_prev_stage=False
        )
        
        # Learnable initial volume embedding at 64³
        self.initial_volume = nn.Parameter(torch.randn(1, 1, 64, 64, 64) * 0.01)
        
        # Progressive upsampling decoder: 64³ → 128³ → 256³
        self.upsampler = nn.ModuleList([
            # 64³ → 128³
            nn.Sequential(
                nn.Conv3d(1, 64, 3, padding=1),
                nn.InstanceNorm3d(64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 64, 3, padding=1),
                nn.InstanceNorm3d(64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 8, 1),  # Reduce channels before upsampling
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ),
            # 128³ → 256³
            nn.Sequential(
                nn.Conv3d(8, 32, 3, padding=1),
                nn.InstanceNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 32, 3, padding=1),
                nn.InstanceNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 8, 1),  # Reduce channels before upsampling
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ),
        ])
        
        # Final refinement at 256³
        self.final_refine = nn.Sequential(
            nn.Conv3d(8, 16, 3, padding=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, 3, padding=1)
        )
        
    def forward(self, xrays):
        """
        Args:
            xrays: (B, num_views, 1, H, W) - input X-ray images
        Returns:
            predicted_volume: (B, 1, 256, 256, 256) - predicted CT volume
        """
        batch_size = xrays.shape[0]
        
        # Create dummy timestep (not used, but encoder expects it)
        dummy_t = torch.zeros(batch_size, 256, device=xrays.device)
        
        # Encode X-rays (same as 64³ model)
        xray_context, time_xray_cond, xray_features_2d = self.xray_encoder(xrays, dummy_t)
        
        # Expand initial volume to batch
        x = self.initial_volume.expand(batch_size, -1, -1, -1, -1)
        
        # ViT processes the 64³ volume conditioned on X-ray features
        volume_64 = self.vit_backbone(
            x=x,
            context=xray_features_2d.flatten(2).transpose(1, 2),  # (B, H*W, C)
            cond=time_xray_cond,
            prev_stage_embed=None
        )
        
        # Progressive upsampling with gradient checkpointing
        x = volume_64
        for i, upsampler_block in enumerate(self.upsampler):
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(upsampler_block, x, use_reentrant=False)
            else:
                x = upsampler_block(x)
        
        # Final refinement
        if self.use_checkpointing and self.training:
            predicted_volume = torch.utils.checkpoint.checkpoint(self.final_refine, x, use_reentrant=False)
        else:
            predicted_volume = self.final_refine(x)
        
        return predicted_volume


def compute_ssim_loss(pred, target, window_size=11):
    """Compute SSIM loss for 3D volumes"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Use stride for memory efficiency on 256³
    stride = window_size // 2
    mu_pred = F.avg_pool3d(pred, window_size, stride=stride, padding=window_size//2)
    mu_target = F.avg_pool3d(target, window_size, stride=stride, padding=window_size//2)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = F.avg_pool3d(pred ** 2, window_size, stride=stride, padding=window_size//2) - mu_pred_sq
    sigma_target_sq = F.avg_pool3d(target ** 2, window_size, stride=stride, padding=window_size//2) - mu_target_sq
    sigma_pred_target = F.avg_pool3d(pred * target, window_size, stride=stride, padding=window_size//2) - mu_pred_target
    
    ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
           ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    
    return 1 - ssim.mean()


if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DirectCTRegression256(volume_size=(256, 256, 256)).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Test forward pass
    xrays = torch.randn(1, 2, 1, 512, 512).to(device)
    with torch.no_grad():
        output = model(xrays)
    print(f"Input X-rays: {xrays.shape}")
    print(f"Output volume: {output.shape}")
