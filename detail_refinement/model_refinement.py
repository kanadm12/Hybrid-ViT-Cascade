"""
Detail Refinement Network for CT Reconstruction
Two-stage approach:
  Stage 1: Base model produces coarse 64³ structure
  Stage 2: This network adds high-frequency details for clinical sharpness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np


class FrequencyAttention3D(nn.Module):
    """Attention module that operates in frequency domain"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Learnable frequency weights
        self.freq_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # FFT to frequency domain
        x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
        x_abs = torch.abs(x_fft)  # Magnitude
        x_phase = torch.angle(x_fft)  # Phase
        
        # Learn attention weights in frequency domain
        freq_features = torch.cat([x_abs.real, x_abs.imag], dim=1)
        freq_weights = self.freq_conv(freq_features)
        
        # Apply attention
        x_fft_weighted = x_fft * freq_weights
        
        # IFFT back to spatial domain
        x_refined = torch.fft.ifftn(x_fft_weighted, dim=(-3, -2, -1)).real
        
        return x_refined


class HighPassFilter3D(nn.Module):
    """3D High-pass filter for extracting details"""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Create 3D Laplacian kernel
        kernel = torch.zeros((1, 1, kernel_size, kernel_size, kernel_size))
        center = kernel_size // 2
        
        # Laplacian: center = sum of neighbors
        kernel[0, 0, center, center, center] = -26
        kernel[0, 0, center, center, center-1] = 1
        kernel[0, 0, center, center, center+1] = 1
        kernel[0, 0, center, center-1, center] = 1
        kernel[0, 0, center, center+1, center] = 1
        kernel[0, 0, center-1, center, center] = 1
        kernel[0, 0, center+1, center, center] = 1
        
        # Face diagonals
        for i in [-1, 1]:
            for j in [-1, 1]:
                kernel[0, 0, center, center+i, center+j] = 1
                kernel[0, 0, center+i, center, center+j] = 1
                kernel[0, 0, center+i, center+j, center] = 1
        
        # Corners
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    kernel[0, 0, center+i, center+j, center+k] = 1
        
        self.register_buffer('kernel', kernel)
    
    def forward(self, x):
        # Apply Laplacian filter
        padding = self.kernel_size // 2
        x_padded = F.pad(x, [padding]*6, mode='replicate')
        
        high_freq = F.conv3d(x_padded, self.kernel.repeat(x.shape[1], 1, 1, 1, 1), 
                             groups=x.shape[1])
        return high_freq


class ResidualBlock3D(nn.Module):
    """Residual block with instance normalization"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class DetailRefinementNetwork(nn.Module):
    """
    Network that refines coarse CT predictions by adding high-frequency details.
    
    Inputs:
      - coarse_ct: (B, 1, D, H, W) - coarse prediction from base model
      - xrays: (B, 2, 1, H_xray, W_xray) - dual-view X-rays for detail guidance
    
    Output:
      - refined_ct: (B, 1, D, H, W) - sharp, detailed CT volume
    """
    def __init__(
        self,
        volume_size=(64, 64, 64),
        xray_size=512,
        hidden_channels=64
    ):
        super().__init__()
        self.volume_size = volume_size
        self.xray_size = xray_size
        
        # X-ray encoder for detail extraction
        self.xray_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # X-ray detail fusion
        self.xray_fusion = nn.Sequential(
            nn.Conv2d(512, 256, 1),  # Concatenated frontal + lateral
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 2D to 3D projection (with detail guidance)
        D, H, W = volume_size
        self.detail_lifter = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, D * 4, 1),  # Lift to 3D
        )
        
        # Extract high-frequency from coarse prediction
        self.high_pass = HighPassFilter3D(kernel_size=3)
        
        # Coarse CT encoder (extract features)
        self.coarse_encoder = nn.Sequential(
            nn.Conv3d(2, 32, 3, padding=1),  # Input: coarse + high-freq
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            ResidualBlock3D(32),
            
            nn.Conv3d(32, 64, 3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64),
        )
        
        # Detail refinement decoder (combines 3D features + X-ray guidance)
        self.detail_decoder = nn.Sequential(
            nn.Conv3d(64 + 4, hidden_channels, 3, padding=1),
            nn.InstanceNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            ResidualBlock3D(hidden_channels),
            
            FrequencyAttention3D(hidden_channels),
            
            ResidualBlock3D(hidden_channels),
            ResidualBlock3D(hidden_channels),
            
            nn.Conv3d(hidden_channels, 32, 3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 1, 1),  # Output: high-freq details to add
            nn.Tanh()  # Details can be positive or negative
        )
        
    def forward(self, coarse_ct: torch.Tensor, xrays: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            coarse_ct: (B, 1, D, H, W) - coarse prediction from base model
            xrays: (B, 2, 1, H_xray, W_xray) - dual-view X-rays
        
        Returns:
            refined_ct: (B, 1, D, H, W) - sharp prediction
            aux_outputs: Dict with intermediate outputs
        """
        B = coarse_ct.shape[0]
        
        # Extract detail guidance from X-rays
        xray_features = []
        for v in range(xrays.shape[1]):  # Frontal and lateral
            feat = self.xray_encoder(xrays[:, v])
            xray_features.append(feat)
        
        # Fuse frontal + lateral detail cues
        xray_fused = torch.cat(xray_features, dim=1)  # (B, 512, H', W')
        xray_detail = self.xray_fusion(xray_fused)  # (B, 128, H', W')
        
        # Lift X-ray details to 3D
        detail_3d = self.detail_lifter(xray_detail)  # (B, D*4, H', W')
        B, C_lifted, H_feat, W_feat = detail_3d.shape
        D = self.volume_size[0]
        detail_3d = detail_3d.view(B, 4, D, H_feat, W_feat)
        
        # Interpolate to match volume size
        if (H_feat, W_feat) != (self.volume_size[1], self.volume_size[2]):
            detail_3d = F.interpolate(
                detail_3d, size=self.volume_size,
                mode='trilinear', align_corners=True
            )
        
        # Extract high-frequency components from coarse prediction
        high_freq = self.high_pass(coarse_ct)
        
        # Encode coarse prediction + high-freq
        coarse_features = torch.cat([coarse_ct, high_freq], dim=1)
        coarse_encoded = self.coarse_encoder(coarse_features)
        
        # Combine 3D features + X-ray detail guidance
        combined = torch.cat([coarse_encoded, detail_3d], dim=1)
        
        # Decode to high-frequency details
        detail_residual = self.detail_decoder(combined)
        
        # Add details to coarse prediction
        refined_ct = coarse_ct + 0.2 * detail_residual  # Scale details
        
        # Clamp to valid range
        refined_ct = torch.clamp(refined_ct, 0, 1)
        
        aux_outputs = {
            'detail_residual': detail_residual,
            'high_freq': high_freq,
            'coarse_ct': coarse_ct
        }
        
        return refined_ct, aux_outputs


class DetailRefinementLoss(nn.Module):
    """
    Loss function emphasizing high-frequency detail preservation.
    
    Components:
      1. L1 Loss: Basic reconstruction
      2. Frequency Domain Loss: Match high-frequency components
      3. Gradient Loss: Sharp edges
      4. Perceptual Loss: Texture/detail similarity
      5. Consistency Loss: Refined should be close to coarse structure
    """
    def __init__(
        self,
        l1_weight=1.0,
        frequency_weight=0.5,
        gradient_weight=0.3,
        perceptual_weight=0.2,
        consistency_weight=0.1
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.frequency_weight = frequency_weight
        self.gradient_weight = gradient_weight
        self.perceptual_weight = perceptual_weight
        self.consistency_weight = consistency_weight
        
        # For gradient computation
        self.high_pass = HighPassFilter3D(kernel_size=3)
    
    def frequency_loss(self, pred, target):
        """Loss in frequency domain with emphasis on high frequencies"""
        # FFT
        pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
        target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))
        
        # Create high-pass filter (emphasize high frequencies)
        D, H, W = pred.shape[-3:]
        d_freq = torch.fft.fftfreq(D, device=pred.device)
        h_freq = torch.fft.fftfreq(H, device=pred.device)
        w_freq = torch.fft.fftfreq(W, device=pred.device)
        
        # 3D frequency grid
        D_grid, H_grid, W_grid = torch.meshgrid(d_freq, h_freq, w_freq, indexing='ij')
        freq_magnitude = torch.sqrt(D_grid**2 + H_grid**2 + W_grid**2)
        
        # High-pass emphasis (higher weight for high frequencies)
        high_pass_weight = torch.sigmoid(10 * (freq_magnitude - 0.1))
        high_pass_weight = high_pass_weight.unsqueeze(0).unsqueeze(0)
        
        # Weighted frequency loss
        freq_diff = torch.abs(pred_fft - target_fft) * high_pass_weight
        return freq_diff.mean()
    
    def gradient_loss(self, pred, target):
        """Loss on gradients (edges)"""
        pred_grad = self.high_pass(pred)
        target_grad = self.high_pass(target)
        return F.l1_loss(pred_grad, target_grad)
    
    def perceptual_loss_3d(self, pred, target):
        """Simplified perceptual loss using multi-scale features"""
        loss = 0
        
        # Multi-scale comparison
        for scale in [1, 2, 4]:
            if scale > 1:
                pred_scaled = F.avg_pool3d(pred, scale, stride=scale)
                target_scaled = F.avg_pool3d(target, scale, stride=scale)
            else:
                pred_scaled = pred
                target_scaled = target
            
            # Compare local statistics (mean, std)
            pred_mean = F.avg_pool3d(pred_scaled, 3, stride=1, padding=1)
            target_mean = F.avg_pool3d(target_scaled, 3, stride=1, padding=1)
            
            pred_var = F.avg_pool3d((pred_scaled - pred_mean)**2, 3, stride=1, padding=1)
            target_var = F.avg_pool3d((target_scaled - target_mean)**2, 3, stride=1, padding=1)
            
            loss += F.l1_loss(pred_mean, target_mean)
            loss += F.l1_loss(torch.sqrt(pred_var + 1e-8), torch.sqrt(target_var + 1e-8))
        
        return loss / 3
    
    def forward(self, refined, target, coarse, aux_outputs):
        """
        Args:
            refined: (B, 1, D, H, W) - refined prediction
            target: (B, 1, D, H, W) - ground truth
            coarse: (B, 1, D, H, W) - coarse prediction from base model
            aux_outputs: Dict with intermediate outputs
        
        Returns:
            total_loss: scalar
            loss_dict: Dict with individual losses
        """
        # 1. L1 Loss
        l1_loss = F.l1_loss(refined, target)
        
        # 2. Frequency Domain Loss
        freq_loss = self.frequency_loss(refined, target)
        
        # 3. Gradient Loss
        grad_loss = self.gradient_loss(refined, target)
        
        # 4. Perceptual Loss
        perc_loss = self.perceptual_loss_3d(refined, target)
        
        # 5. Consistency Loss (refined shouldn't deviate too much from coarse)
        consistency_loss = F.mse_loss(refined, coarse)
        
        # Total loss
        total_loss = (
            self.l1_weight * l1_loss +
            self.frequency_weight * freq_loss +
            self.gradient_weight * grad_loss +
            self.perceptual_weight * perc_loss +
            self.consistency_weight * consistency_loss
        )
        
        loss_dict = {
            'l1': l1_loss.item(),
            'frequency': freq_loss.item(),
            'gradient': grad_loss.item(),
            'perceptual': perc_loss.item(),
            'consistency': consistency_loss.item()
        }
        
        return total_loss, loss_dict
