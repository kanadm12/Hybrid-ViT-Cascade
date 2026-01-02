"""
Improved Refinement Network for 64³ → 256³ Upsampling
Key improvements:
- Residual learning (predict correction not full volume)
- Sub-pixel convolution for better upsampling
- Multi-scale features
- SSIM-aware architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class PixelShuffle3D(nn.Module):
    """3D Pixel Shuffle for sub-pixel convolution upsampling"""
    
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        batch_size, channels, d, h, w = x.shape
        channels = channels // (self.scale_factor ** 3)
        
        out_d = d * self.scale_factor
        out_h = h * self.scale_factor
        out_w = w * self.scale_factor
        
        x = x.view(batch_size, channels, self.scale_factor, self.scale_factor, self.scale_factor, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, channels, out_d, out_h, out_w)
        
        return x


class ResidualBlock3D(nn.Module):
    """3D Residual block with normalization"""
    
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
        out = out + residual
        out = self.relu(out)
        return out


class AttentionBlock3D(nn.Module):
    """3D Channel attention for feature refinement"""
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ImprovedRefinementNetwork(nn.Module):
    """
    Improved refinement network with:
    - Residual learning (predicts correction)
    - Sub-pixel convolution upsampling
    - Multi-scale features
    - Channel attention
    """
    
    def __init__(self, base_channels=32):
        super().__init__()
        
        # Initial feature extraction from 64³
        self.initial = nn.Sequential(
            nn.Conv3d(1, base_channels, 3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 64³ → 128³ (2x upsampling)
        self.stage1_res = nn.Sequential(
            ResidualBlock3D(base_channels),
            ResidualBlock3D(base_channels),
        )
        
        # Sub-pixel convolution for 2x upsampling
        self.upsample1 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 8, 3, padding=1),  # 8 = 2³
            PixelShuffle3D(scale_factor=2),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.stage1_attention = AttentionBlock3D(base_channels)
        
        # Stage 2: 128³ → 256³ (2x upsampling)
        self.stage2_res = nn.Sequential(
            ResidualBlock3D(base_channels),
            ResidualBlock3D(base_channels),
        )
        
        # Sub-pixel convolution for 2x upsampling
        self.upsample2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 8, 3, padding=1),  # 8 = 2³
            PixelShuffle3D(scale_factor=2),
            nn.Conv3d(base_channels, base_channels // 2, 3, padding=1),
            nn.InstanceNorm3d(base_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.stage2_attention = AttentionBlock3D(base_channels // 2)
        
        # Final refinement
        self.final_refine = nn.Sequential(
            ResidualBlock3D(base_channels // 2),
            nn.Conv3d(base_channels // 2, 1, 3, padding=1)
        )
        
        # Scale for residual correction (learnable, initialized to 0.3)
        self.correction_scale = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, coarse_volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coarse_volume: (B, 1, 64, 64, 64)
        Returns:
            refined_volume: (B, 1, 256, 256, 256)
        """
        # Extract features
        x = self.initial(coarse_volume)
        
        # Stage 1: 64³ → 128³
        x = self.stage1_res(x)
        x = self.upsample1(x)
        x = self.stage1_attention(x)
        
        # Stage 2: 128³ → 256³
        x = self.stage2_res(x)
        x = self.upsample2(x)
        x = self.stage2_attention(x)
        
        # Predict correction
        correction = self.final_refine(x)
        correction = correction * self.correction_scale
        
        # Upscale base prediction with trilinear
        base_upsampled = F.interpolate(
            coarse_volume, 
            scale_factor=4, 
            mode='trilinear', 
            align_corners=True
        )
        
        # Add residual correction
        refined = base_upsampled + correction
        
        return refined


class SSIMLoss(nn.Module):
    """3D SSIM Loss"""
    
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window_3d(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window_3d(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _2D_window.unsqueeze(0) * _1D_window.unsqueeze(0).unsqueeze(2)
        
        window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        window = self.window
        mu1 = F.conv3d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv3d(img2, window, padding=self.window_size // 2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv3d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 on 2D slices"""
    
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except:
            vgg = vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
        
        # Freeze
        for param in self.features.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize CT values [0, 1] to ImageNet distribution"""
        # Clamp to [0, 1] range
        x = torch.clamp(x, 0, 1)
        # Repeat channels for RGB
        x = x.repeat(1, 3, 1, 1)
        # Normalize
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, pred, target):
        # Sample middle slices along each axis
        pred_slices = []
        target_slices = []
        
        B, C, D, H, W = pred.shape
        
        # Axial slice (middle)
        pred_slices.append(pred[:, :, D//2, :, :])
        target_slices.append(target[:, :, D//2, :, :])
        
        # Normalize for VGG
        pred_slices = [self.normalize(s) for s in pred_slices]
        target_slices = [self.normalize(s) for s in target_slices]
        
        loss = 0
        for pred_s, target_s in zip(pred_slices, target_slices):
            pred_feat = self.features(pred_s)
            target_feat = self.features(target_s)
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(pred_slices)


class EdgeAwareLoss(nn.Module):
    """Edge-aware loss using Sobel filters"""
    
    def __init__(self):
        super().__init__()
        
        # 3D Sobel kernels for each axis
        self.register_buffer('sobel_x', torch.FloatTensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ]).unsqueeze(0).unsqueeze(0) / 16.0)
        
        self.register_buffer('sobel_y', torch.FloatTensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ]).unsqueeze(0).unsqueeze(0) / 16.0)
        
        self.register_buffer('sobel_z', torch.FloatTensor([
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        ]).unsqueeze(0).unsqueeze(0) / 16.0)
    
    def get_edges(self, x):
        edge_x = F.conv3d(x, self.sobel_x, padding=1)
        edge_y = F.conv3d(x, self.sobel_y, padding=1)
        edge_z = F.conv3d(x, self.sobel_z, padding=1)
        
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + edge_z ** 2 + 1e-6)
        return edges
    
    def forward(self, pred, target):
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        
        return F.l1_loss(pred_edges, target_edges)


class MultiScaleLoss(nn.Module):
    """Multi-scale L1 loss for better detail preservation"""
    
    def __init__(self, scales=[1.0, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
    
    def forward(self, pred, target):
        total_loss = 0
        
        for scale in self.scales:
            if scale == 1.0:
                scaled_pred = pred
                scaled_target = target
            else:
                size = [int(dim * scale) for dim in pred.shape[2:]]
                scaled_pred = F.interpolate(pred, size=size, mode='trilinear', align_corners=True)
                scaled_target = F.interpolate(target, size=size, mode='trilinear', align_corners=True)
            
            total_loss += F.l1_loss(scaled_pred, scaled_target)
        
        return total_loss / len(self.scales)
