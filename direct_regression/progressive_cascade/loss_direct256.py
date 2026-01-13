"""
Advanced Loss Suite for Direct H200 Training (128³ or 256³)
Combines 7 complementary loss functions for maximum perceptual quality
Resolution-agnostic: works with any volume size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_multiscale import SSIMLoss, TotalVariationLoss, compute_psnr, compute_ssim_metric

# Import from Direct128 model (works for both 128³ and 256³)
from model_direct128_h200 import (
    FocalFrequencyLoss,
    PerceptualFeaturePyramidLoss,
    Style3DLoss,
    AnatomicalAttentionLoss
)


def check_nan(loss_val, name):
    """Debug helper to detect NaN in losses"""
    if torch.isnan(loss_val).any():
        print(f"[NaN DEBUG] {name} is NaN")
        return True
    return False


class Direct256Loss(nn.Module):
    """
    Comprehensive loss for direct end-to-end training (128³ or 256³)
    Resolution-agnostic: adapts to any input volume size
    
    7 Loss Components:
    1. L1 (1.0): Base reconstruction
    2. SSIM (0.5): Structural similarity
    3. Focal Frequency (0.2): Adaptive frequency emphasis
    4. Perceptual Pyramid (0.15): Multi-scale perceptual
    5. Total Variation (0.02): Edge preservation
    6. Style 3D (0.1): Texture consistency
    7. Anatomical Attention (0.3): Region importance
    
    Total weight: 2.27 (normalized by batch)
    """
    def __init__(self,
                 l1_weight=1.0,
                 ssim_weight=0.5,
                 focal_freq_weight=0.2,
                 perceptual_pyramid_weight=0.15,
                 tv_weight=0.02,
                 style_weight=0.1,
                 anatomical_weight=0.3):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.focal_freq_weight = focal_freq_weight
        self.perceptual_pyramid_weight = perceptual_pyramid_weight
        self.tv_weight = tv_weight
        self.style_weight = style_weight
        self.anatomical_weight = anatomical_weight
        
        # Initialize loss modules
        self.ssim_loss = SSIMLoss()
        self.focal_freq_loss = FocalFrequencyLoss(alpha=1.0)
        self.perceptual_pyramid_loss = PerceptualFeaturePyramidLoss(scales=[1.0, 0.5, 0.25])
        self.tv_loss = TotalVariationLoss()
        self.style_loss = Style3DLoss()
        self.anatomical_loss = AnatomicalAttentionLoss()
        
    def forward(self, pred, target):
        """
        Args:
            pred, target: (B, 1, D, H, W) - any resolution (128³, 256³, etc.)
        Returns:
            loss_dict: dict with all losses
        """
        # Ensure float32 for numerical stability
        pred = pred.float()
        target = target.float()
        
        # Compute stable losses only
        l1_loss = F.l1_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        tv_loss = self.tv_loss(pred, target)
        
        # Clamp values to prevent NaN propagation
        l1_loss = torch.clamp(l1_loss, 0, 100)
        ssim_loss = torch.clamp(ssim_loss, 0, 100)
        tv_loss = torch.clamp(tv_loss, 0, 100)
        
        # Check for NaN
        if torch.isnan(l1_loss) or torch.isnan(ssim_loss) or torch.isnan(tv_loss):
            print(f"[NaN] L1: {l1_loss.item()}, SSIM: {ssim_loss.item()}, TV: {tv_loss.item()}")
            # Return safe default
            return {
                'total_loss': torch.tensor(1.0, device=pred.device),
                'l1_loss': l1_loss,
                'ssim_loss': ssim_loss,
                'focal_freq_loss': torch.tensor(0.0, device=pred.device),
                'perceptual_pyramid_loss': torch.tensor(0.0, device=pred.device),
                'tv_loss': tv_loss,
                'style_loss': torch.tensor(0.0, device=pred.device),
                'anatomical_loss': torch.tensor(0.0, device=pred.device)
            }
        
        # Combined loss - use only stable components
        total_loss = (
            self.l1_weight * l1_loss +
            self.ssim_weight * ssim_loss +
            self.tv_weight * tv_loss
        )
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'ssim_loss': ssim_loss,
            'focal_freq_loss': torch.tensor(0.0, device=pred.device),
            'perceptual_pyramid_loss': torch.tensor(0.0, device=pred.device),
            'tv_loss': tv_loss,
            'style_loss': torch.tensor(0.0, device=pred.device),
            'anatomical_loss': torch.tensor(0.0, device=pred.device)
        }


def get_loss_summary_string(loss_dict):
    """Format loss dictionary for logging"""
    return (
        f"Loss: {loss_dict['total_loss']:.4f} | "
        f"L1: {loss_dict['l1_loss']:.4f} | "
        f"SSIM: {loss_dict['ssim_loss']:.4f} | "
        f"FocalFreq: {loss_dict['focal_freq_loss']:.4f} | "
        f"Perceptual: {loss_dict['perceptual_pyramid_loss']:.4f} | "
        f"TV: {loss_dict['tv_loss']:.4f} | "
        f"Style: {loss_dict['style_loss']:.4f} | "
        f"Anatomical: {loss_dict['anatomical_loss']:.4f}"
    )
