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
        
        # Compute all 7 losses with safety checks (all have built-in NaN protection)
        l1_loss = torch.clamp(F.l1_loss(pred, target), 0, 100)
        ssim_loss = torch.clamp(self.ssim_loss(pred, target), 0, 100)
        focal_freq_loss = self.focal_freq_loss(pred, target)  # Has try-except internally
        perceptual_pyramid_loss = self.perceptual_pyramid_loss(pred, target)  # Has try-except internally
        tv_loss = torch.clamp(self.tv_loss(pred, target), 0, 100)
        style_loss = self.style_loss(pred, target)  # Has try-except internally
        anatomical_loss = self.anatomical_loss(pred, target)  # Has try-except internally
        
        # Final NaN check - if any loss is NaN, skip that component
        if torch.isnan(focal_freq_loss) or torch.isinf(focal_freq_loss):
            print(f"[WARNING] FocalFreq returned NaN, setting to 0")
            focal_freq_loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        
        if torch.isnan(perceptual_pyramid_loss) or torch.isinf(perceptual_pyramid_loss):
            print(f"[WARNING] PerceptualPyramid returned NaN, setting to 0")
            perceptual_pyramid_loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        
        if torch.isnan(style_loss) or torch.isinf(style_loss):
            print(f"[WARNING] Style3D returned NaN, setting to 0")
            style_loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        
        if torch.isnan(anatomical_loss) or torch.isinf(anatomical_loss):
            print(f"[WARNING] AnatomicalAttention returned NaN, setting to 0")
            anatomical_loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        
        # Combined loss with all 7 components
        total_loss = (
            self.l1_weight * l1_loss +
            self.ssim_weight * ssim_loss +
            self.focal_freq_weight * focal_freq_loss +
            self.perceptual_pyramid_weight * perceptual_pyramid_loss +
            self.tv_weight * tv_loss +
            self.style_weight * style_loss +
            self.anatomical_weight * anatomical_loss
        )
        
        # Final safety check on total loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[ERROR] Total loss is NaN/Inf! Returning fallback loss")
            total_loss = l1_loss + ssim_loss + tv_loss  # Fallback to stable losses only
        
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
