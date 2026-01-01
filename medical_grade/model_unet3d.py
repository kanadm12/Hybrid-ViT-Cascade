"""
Medical-Grade 3D U-Net for CT Reconstruction
Target: 18-20 dB PSNR (first milestone toward 30+ dB)

Key improvements over base model:
1. U-Net architecture with skip connections
2. Multi-scale feature pyramid
3. Deeper network (8 encoder + 8 decoder blocks)
4. Residual dense blocks
5. Attention gates in skip connections
6. Multi-scale output supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np


class AttentionGate3D(nn.Module):
    """Attention gate for skip connections (focuses on relevant features)"""
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of feature maps from gating signal (decoder)
            F_l: Number of feature maps from skip connection (encoder)
            F_int: Number of intermediate feature maps
        """
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (B, F_g, D, H, W)
            x: Skip connection from encoder (B, F_l, D, H, W)
        Returns:
            Attention-weighted skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResidualDenseBlock3D(nn.Module):
    """Dense residual block with multiple convolution paths"""
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm3d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels + i * growth_rate, growth_rate, 
                         kernel_size=3, padding=1, bias=False)
            )
            self.layers.append(layer)
        
        # 1x1 conv to reduce channels
        self.bottleneck = nn.Conv3d(
            in_channels + num_layers * growth_rate,
            in_channels,
            kernel_size=1,
            bias=False
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        
        dense_out = torch.cat(features, dim=1)
        out = self.bottleneck(dense_out)
        return out + x  # Residual connection


class MultiScaleFeaturePyramid(nn.Module):
    """Extract features at multiple scales from X-rays"""
    def __init__(self, in_channels=1):
        super().__init__()
        
        # Scale 1: Full resolution
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Scale 2: 1/2 resolution
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Scale 3: 1/4 resolution
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Scale 4: 1/8 resolution
        self.scale4 = nn.Sequential(
            nn.Conv2d(in_channels, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Conv2d(32 + 64 + 128 + 256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Extract multi-scale features
        feat1 = self.scale1(x)  # (B, 32, H, W)
        feat2 = self.scale2(x)  # (B, 64, H/2, W/2)
        feat3 = self.scale3(x)  # (B, 128, H/4, W/4)
        feat4 = self.scale4(x)  # (B, 256, H/8, W/8)
        
        # Upsample all to 1/8 resolution for fusion
        target_size = feat4.shape[-2:]
        feat1_down = F.adaptive_avg_pool2d(feat1, target_size)
        feat2_down = F.adaptive_avg_pool2d(feat2, target_size)
        feat3_down = F.adaptive_avg_pool2d(feat3, target_size)
        
        # Concatenate and fuse
        fused = torch.cat([feat1_down, feat2_down, feat3_down, feat4], dim=1)
        fused = self.fusion(fused)
        
        return fused, [feat1, feat2, feat3, feat4]


class EncoderBlock3D(nn.Module):
    """U-Net encoder block with residual dense connections"""
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.downsample = downsample
        
        if downsample:
            self.down = nn.Conv3d(in_channels, in_channels, 3, stride=2, padding=1)
        
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            ResidualDenseBlock3D(out_channels, growth_rate=out_channels//4, num_layers=3),
        )
    
    def forward(self, x):
        if self.downsample:
            x = self.down(x)
        return self.conv_block(x)


class DecoderBlock3D(nn.Module):
    """U-Net decoder block with attention-gated skip connections"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels, 
            kernel_size=2, stride=2
        )
        
        self.attention_gate = AttentionGate3D(
            F_g=in_channels,
            F_l=skip_channels,
            F_int=out_channels
        )
        
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            ResidualDenseBlock3D(out_channels, growth_rate=out_channels//4, num_layers=3),
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Attention-weighted skip connection
        skip_att = self.attention_gate(x, skip)
        
        # Concatenate and convolve
        x = torch.cat([x, skip_att], dim=1)
        x = self.conv_block(x)
        
        return x


class MedicalGradeUNet3D(nn.Module):
    """
    Medical-grade 3D U-Net for CT reconstruction from dual-view X-rays.
    
    Architecture:
      - Multi-scale X-ray feature extraction
      - 8-level U-Net encoder-decoder
      - Attention gates in skip connections
      - Residual dense blocks
      - Deep supervision (multi-scale outputs)
    
    Target: 18-20 dB PSNR (milestone 1 toward 30+ dB)
    """
    def __init__(
        self,
        volume_size=(64, 64, 64),
        xray_size=512,
        base_channels=32
    ):
        super().__init__()
        self.volume_size = volume_size
        self.xray_size = xray_size
        
        # Multi-scale X-ray feature extractor
        self.xray_pyramid = MultiScaleFeaturePyramid(in_channels=1)
        
        # Cross-view fusion
        self.view_fusion = nn.Sequential(
            nn.Conv2d(1024, 512, 1),  # 512 * 2 views
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 2D to 3D projection
        D = volume_size[0]
        self.to_3d = nn.Sequential(
            nn.Conv2d(512, D * base_channels, 1),
        )
        
        # U-Net Encoder (8 levels for depth)
        self.enc1 = EncoderBlock3D(base_channels, base_channels * 2, downsample=False)
        self.enc2 = EncoderBlock3D(base_channels * 2, base_channels * 4)
        self.enc3 = EncoderBlock3D(base_channels * 4, base_channels * 8)
        self.enc4 = EncoderBlock3D(base_channels * 8, base_channels * 16)
        self.enc5 = EncoderBlock3D(base_channels * 16, base_channels * 24)
        self.enc6 = EncoderBlock3D(base_channels * 24, base_channels * 32)
        self.enc7 = EncoderBlock3D(base_channels * 32, base_channels * 40)
        self.enc8 = EncoderBlock3D(base_channels * 40, base_channels * 48)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels * 48, base_channels * 64, 3, padding=1),
            nn.BatchNorm3d(base_channels * 64),
            nn.ReLU(inplace=True),
            ResidualDenseBlock3D(base_channels * 64, growth_rate=base_channels * 16, num_layers=4),
            nn.Conv3d(base_channels * 64, base_channels * 48, 3, padding=1),
            nn.BatchNorm3d(base_channels * 48),
            nn.ReLU(inplace=True),
        )
        
        # U-Net Decoder (8 levels with attention gates)
        self.dec8 = DecoderBlock3D(base_channels * 48, base_channels * 40, base_channels * 40)
        self.dec7 = DecoderBlock3D(base_channels * 40, base_channels * 32, base_channels * 32)
        self.dec6 = DecoderBlock3D(base_channels * 32, base_channels * 24, base_channels * 24)
        self.dec5 = DecoderBlock3D(base_channels * 24, base_channels * 16, base_channels * 16)
        self.dec4 = DecoderBlock3D(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec3 = DecoderBlock3D(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderBlock3D(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderBlock3D(base_channels * 2, base_channels * 2, base_channels)
        
        # Deep supervision outputs (multi-scale)
        self.output_dec4 = nn.Conv3d(base_channels * 8, 1, 1)
        self.output_dec3 = nn.Conv3d(base_channels * 4, 1, 1)
        self.output_dec2 = nn.Conv3d(base_channels * 2, 1, 1)
        self.output_final = nn.Conv3d(base_channels, 1, 1)
    
    def forward(self, xrays: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            xrays: (B, num_views, 1, H, W) - dual-view X-rays
        
        Returns:
            final_output: (B, 1, D, H, W) - reconstructed CT
            aux_outputs: Dict with multi-scale outputs for deep supervision
        """
        B, num_views = xrays.shape[:2]
        
        # Extract multi-scale features from each view
        view_features = []
        for v in range(num_views):
            feat, scales = self.xray_pyramid(xrays[:, v])
            view_features.append(feat)
        
        # Fuse frontal + lateral views
        fused = torch.cat(view_features, dim=1)  # (B, 1024, H', W')
        fused = self.view_fusion(fused)  # (B, 512, H', W')
        
        # Project to 3D
        feat_3d = self.to_3d(fused)  # (B, D * base_channels, H', W')
        B, C, H_feat, W_feat = feat_3d.shape
        D = self.volume_size[0]
        feat_3d = feat_3d.view(B, -1, D, H_feat, W_feat)  # (B, base_channels, D, H', W')
        
        # Interpolate to target volume size
        if (H_feat, W_feat) != (self.volume_size[1], self.volume_size[2]):
            feat_3d = F.interpolate(
                feat_3d, size=self.volume_size,
                mode='trilinear', align_corners=True
            )
        
        # U-Net Encoder
        enc1 = self.enc1(feat_3d)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        enc8 = self.enc8(enc7)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc8)
        
        # U-Net Decoder with skip connections
        dec8 = self.dec8(bottleneck, enc7)
        dec7 = self.dec7(dec8, enc6)
        dec6 = self.dec6(dec7, enc5)
        dec5 = self.dec5(dec6, enc4)
        dec4 = self.dec4(dec5, enc3)
        dec3 = self.dec3(dec4, enc2)
        dec2 = self.dec2(dec3, enc1)
        dec1 = self.dec1(dec2, enc1)
        
        # Multi-scale outputs for deep supervision
        out_dec4 = self.output_dec4(dec4)
        out_dec3 = self.output_dec3(dec3)
        out_dec2 = self.output_dec2(dec2)
        out_final = self.output_final(dec1)
        
        # Clamp to valid range
        out_final = torch.clamp(out_final, 0, 1)
        
        aux_outputs = {
            'output_dec4': out_dec4,
            'output_dec3': out_dec3,
            'output_dec2': out_dec2,
            'output_final': out_final
        }
        
        return out_final, aux_outputs


class MedicalGradeLoss(nn.Module):
    """
    Multi-component loss for medical-grade reconstruction.
    
    Components:
      1. L1 Loss: Basic reconstruction
      2. SSIM Loss: Structural similarity
      3. Perceptual Loss: Feature-level similarity
      4. Edge Loss: Sharp boundaries
      5. Deep Supervision: Multi-scale guidance
      6. Frequency Loss: High-frequency preservation
    """
    def __init__(
        self,
        l1_weight=1.0,
        ssim_weight=0.3,
        perceptual_weight=0.2,
        edge_weight=0.2,
        deep_supervision_weight=0.2,
        frequency_weight=0.3
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.deep_supervision_weight = deep_supervision_weight
        self.frequency_weight = frequency_weight
    
    def ssim_loss(self, pred, target, window_size=11):
        """SSIM loss"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool3d(pred, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool3d(target, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool3d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool3d(target ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool3d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()
    
    def edge_loss(self, pred, target):
        """Gradient-based edge loss"""
        # Sobel filters for 3D gradients
        pred_dx = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        pred_dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        pred_dz = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        
        target_dx = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        target_dy = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        target_dz = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        
        loss_dx = F.l1_loss(pred_dx, target_dx)
        loss_dy = F.l1_loss(pred_dy, target_dy)
        loss_dz = F.l1_loss(pred_dz, target_dz)
        
        return (loss_dx + loss_dy + loss_dz) / 3
    
    def perceptual_loss_3d(self, pred, target):
        """Multi-scale perceptual loss"""
        loss = 0
        for scale in [1, 2, 4]:
            if scale > 1:
                pred_scaled = F.avg_pool3d(pred, scale, stride=scale)
                target_scaled = F.avg_pool3d(target, scale, stride=scale)
            else:
                pred_scaled = pred
                target_scaled = target
            
            # Local statistics
            pred_mean = F.avg_pool3d(pred_scaled, 3, stride=1, padding=1)
            target_mean = F.avg_pool3d(target_scaled, 3, stride=1, padding=1)
            
            pred_var = F.avg_pool3d((pred_scaled - pred_mean)**2, 3, stride=1, padding=1)
            target_var = F.avg_pool3d((target_scaled - target_mean)**2, 3, stride=1, padding=1)
            
            loss += F.l1_loss(pred_mean, target_mean)
            loss += F.l1_loss(torch.sqrt(pred_var + 1e-8), torch.sqrt(target_var + 1e-8))
        
        return loss / 3
    
    def frequency_loss(self, pred, target):
        """High-frequency emphasis in FFT domain"""
        pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
        target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))
        
        # High-pass filter
        D, H, W = pred.shape[-3:]
        d_freq = torch.fft.fftfreq(D, device=pred.device)
        h_freq = torch.fft.fftfreq(H, device=pred.device)
        w_freq = torch.fft.fftfreq(W, device=pred.device)
        
        D_grid, H_grid, W_grid = torch.meshgrid(d_freq, h_freq, w_freq, indexing='ij')
        freq_magnitude = torch.sqrt(D_grid**2 + H_grid**2 + W_grid**2)
        
        high_pass = torch.sigmoid(15 * (freq_magnitude - 0.15))
        high_pass = high_pass.unsqueeze(0).unsqueeze(0)
        
        freq_diff = torch.abs(pred_fft - target_fft) * high_pass
        return freq_diff.mean()
    
    def deep_supervision_loss(self, aux_outputs, target):
        """Multi-scale output supervision"""
        loss = 0
        
        # Downsample target to match each output scale
        for key in ['output_dec4', 'output_dec3', 'output_dec2']:
            output = aux_outputs[key]
            scale_factor = target.shape[-1] // output.shape[-1]
            
            if scale_factor > 1:
                target_scaled = F.avg_pool3d(target, scale_factor, stride=scale_factor)
            else:
                target_scaled = target
            
            loss += F.l1_loss(output, target_scaled)
        
        return loss / 3
    
    def forward(self, pred, target, aux_outputs):
        """
        Args:
            pred: (B, 1, D, H, W) - final prediction
            target: (B, 1, D, H, W) - ground truth
            aux_outputs: Dict with intermediate outputs
        
        Returns:
            total_loss: scalar
            loss_dict: Dict with individual losses
        """
        l1_loss = F.l1_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        edge_loss = self.edge_loss(pred, target)
        perc_loss = self.perceptual_loss_3d(pred, target)
        freq_loss = self.frequency_loss(pred, target)
        ds_loss = self.deep_supervision_loss(aux_outputs, target)
        
        total_loss = (
            self.l1_weight * l1_loss +
            self.ssim_weight * ssim_loss +
            self.perceptual_weight * perc_loss +
            self.edge_weight * edge_loss +
            self.deep_supervision_weight * ds_loss +
            self.frequency_weight * freq_loss
        )
        
        loss_dict = {
            'l1': l1_loss.item(),
            'ssim': ssim_loss.item(),
            'perceptual': perc_loss.item(),
            'edge': edge_loss.item(),
            'deep_supervision': ds_loss.item(),
            'frequency': freq_loss.item()
        }
        
        return total_loss, loss_dict
