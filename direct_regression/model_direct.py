"""
Direct Regression Model (No Diffusion)
Tests if architecture can learn X-ray → CT without diffusion complexity
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '..')

from models.diagnostic_losses import XrayConditioningModule
from models.hybrid_vit_backbone import HybridViT3D


class DirectCTRegression(nn.Module):
    """
    Simple direct regression: X-rays → CT volume
    No diffusion, no timesteps, no noise
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
        
        # X-ray encoder (with BatchNorm fix)
        self.xray_encoder = XrayConditioningModule(
            img_size=xray_img_size,
            in_channels=1,
            embed_dim=xray_feature_dim,
            num_views=2,
            time_embed_dim=256,
            cond_dim=1024,
            share_view_weights=False
        )
        
        # ViT backbone (no time conditioning needed for direct regression)
        self.vit_backbone = HybridViT3D(
            volume_size=volume_size,
            in_channels=1,  # Just a learned embedding, no noisy input
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
            xrays: (B, num_views, 1, H, W) - input X-ray images
        Returns:
            predicted_volume: (B, 1, D, H, W) - predicted CT volume
        """
        batch_size = xrays.shape[0]
        
        # Create dummy timestep (not used, but encoder expects it)
        dummy_t = torch.zeros(batch_size, 256, device=xrays.device)
        
        # Encode X-rays
        xray_context, time_xray_cond, xray_features_2d = self.xray_encoder(xrays, dummy_t)
        
        # Expand initial volume to batch
        x = self.initial_volume.expand(batch_size, -1, -1, -1, -1)
        
        # ViT processes the volume conditioned on X-ray features
        predicted_volume = self.vit_backbone(
            x=x,
            context=xray_features_2d.flatten(2).transpose(1, 2),  # (B, H*W, C)
            cond=time_xray_cond,
            prev_stage_embed=None
        )
        
        return predicted_volume


def compute_ssim_loss(pred, target, window_size=11):
    """Compute SSIM loss for 3D volumes"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_pred = F.avg_pool3d(pred, window_size, stride=1, padding=window_size//2)
    mu_target = F.avg_pool3d(target, window_size, stride=1, padding=window_size//2)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = F.avg_pool3d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
    sigma_target_sq = F.avg_pool3d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
    sigma_pred_target = F.avg_pool3d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target
    
    ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
           ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    
    return 1 - ssim.mean()


class DirectRegressionLoss(nn.Module):
    """Combined loss for direct regression"""
    
    def __init__(self, l1_weight=1.0, ssim_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        
    def forward(self, pred, target):
        # L1 loss (primary)
        l1_loss = F.l1_loss(pred, target)
        
        # SSIM loss (structure)
        ssim_loss = compute_ssim_loss(pred, target)
        
        total_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'ssim_loss': ssim_loss
        }


if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DirectCTRegression(volume_size=(64, 64, 64)).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    xrays = torch.randn(2, 2, 1, 512, 512).to(device)
    output = model(xrays)
    print(f"Input X-rays: {xrays.shape}")
    print(f"Output volume: {output.shape}")
    
    # Test loss
    target = torch.randn_like(output)
    loss_fn = DirectRegressionLoss()
    loss_dict = loss_fn(output, target)
    print(f"Losses: {loss_dict}")
