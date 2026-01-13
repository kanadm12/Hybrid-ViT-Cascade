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
        # Compute all losses
        l1_loss = F.l1_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        focal_freq_loss = self.focal_freq_loss(pred, target)
        perceptual_pyramid_loss = self.perceptual_pyramid_loss(pred, target)
        tv_loss = self.tv_loss(pred, target)
        style_loss = self.style_loss(pred, target)
        anatomical_loss = self.anatomical_loss(pred, target)
        
        # Combined loss
        total_loss = (
            self.l1_weight * l1_loss +
            self.ssim_weight * ssim_loss +
            self.focal_freq_weight * focal_freq_loss +
            self.perceptual_pyramid_weight * perceptual_pyramid_loss +
            self.tv_weight * tv_loss +
            self.style_weight * style_loss +
            self.anatomical_weight * anatomical_loss
        )
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'ssim_loss': ssim_loss,
            'focal_freq_loss': focal_freq_loss,
            'perceptual_pyramid_loss': perceptual_pyramid_loss,
            'tv_loss': tv_loss,
            'style_loss': style_loss,
            'anatomical_loss': anatomical_loss
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
