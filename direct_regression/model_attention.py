"""
Enhanced Direct Regression Model with Attention Mechanisms
Includes: SE blocks, CBAM, and Cross-View Attention for improved accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import torchvision.models as models


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention3D(nn.Module):
    """Spatial attention module"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial = torch.cat([max_pool, avg_pool], dim=1)
        return x * self.sigmoid(self.conv(spatial))


class CBAM3D(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = SEBlock3D(channels, reduction)
        self.spatial_att = SpatialAttention3D()
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ResidualBlock3DWithAttention(nn.Module):
    """3D Residual block with SE attention"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(channels)
        self.se = SEBlock3D(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.relu(out)


class CrossViewAttention(nn.Module):
    """Attention mechanism for fusing frontal and lateral views"""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, view1, view2):
        """
        Args:
            view1, view2: (B, C, H, W) features from two views
        Returns:
            attended_view1: (B, C, H, W)
        """
        B, C, H, W = view1.size()
        
        # Query from view1, Key/Value from view2
        q = self.query(view1).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        k = self.key(view2).view(B, -1, H * W)  # (B, C', HW)
        
        # Attention scores
        attention = self.softmax(torch.bmm(q, k))  # (B, HW, HW)
        
        # Apply attention to values
        v = self.value(view2).view(B, C, -1)  # (B, C, HW)
        out = torch.bmm(v, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable weight
        return self.gamma * out + view1


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for 3D volumes"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        # Take middle slice
        pred_slice = pred[:, :, pred.shape[2]//2, :, :]
        target_slice = target[:, :, target.shape[2]//2, :, :]
        
        # Repeat to 3 channels
        pred_rgb = pred_slice.repeat(1, 3, 1, 1)
        target_rgb = target_slice.repeat(1, 3, 1, 1)
        
        # Extract features
        with torch.amp.autocast('cuda', enabled=False):
            pred_feat = self.vgg(pred_rgb.float())
            target_feat = self.vgg(target_rgb.float())
        
        return F.mse_loss(pred_feat, target_feat)


class EdgeAwareLoss(nn.Module):
    """Edge-aware loss using Sobel filters"""
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32) / 4.0
        self.sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32) / 4.0
    
    def forward(self, pred, target):
        device = pred.device
        sobel_x = self.sobel_x.unsqueeze(0).to(device)
        sobel_y = self.sobel_y.unsqueeze(0).to(device)
        
        # Middle slice
        pred_slice = pred[:, :, pred.shape[2]//2, :, :]
        target_slice = target[:, :, target.shape[2]//2, :, :]
        
        with torch.amp.autocast('cuda', enabled=False):
            pred_edges_x = F.conv2d(pred_slice.float(), sobel_x, padding=1)
            pred_edges_y = F.conv2d(pred_slice.float(), sobel_y, padding=1)
            target_edges_x = F.conv2d(target_slice.float(), sobel_x, padding=1)
            target_edges_y = F.conv2d(target_slice.float(), sobel_y, padding=1)
            
            pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + 1e-6)
            target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + 1e-6)
        
        return F.l1_loss(pred_edges, target_edges)


class AttentionEnhancedModel(nn.Module):
    """
    Enhanced Direct Regression Model with Attention Mechanisms
    Architecture: Dual-view encoder → Cross-view attention → 3D decoder with CBAM
    """
    
    def __init__(self, 
                 volume_size: Tuple[int, int, int] = (64, 64, 64),
                 base_channels: int = 96):
        super().__init__()
        
        self.volume_size = volume_size
        D, H, W = volume_size
        
        # X-ray encoder (ResNet-style)
        self.xray_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Cross-view attention for fusing frontal and lateral
        self.cross_view_attention = CrossViewAttention(512)
        
        # Feature fusion
        self.view_fusion = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 2D to 3D lifting
        self.depth_lifter = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, D * 8, 1),
        )
        
        # 3D decoder with attention
        self.decoder = nn.ModuleList([
            # Block 1: 8 → 32 channels
            nn.Sequential(
                nn.Conv3d(8, 32, 3, padding=1),
                nn.InstanceNorm3d(32),
                nn.ReLU(inplace=True),
                ResidualBlock3DWithAttention(32),
                CBAM3D(32),
            ),
            # Block 2: 32 → 64 channels
            nn.Sequential(
                nn.Conv3d(32, 64, 3, padding=1),
                nn.InstanceNorm3d(64),
                nn.ReLU(inplace=True),
                ResidualBlock3DWithAttention(64),
                ResidualBlock3DWithAttention(64),
                CBAM3D(64),
            ),
            # Block 3: 64 → base_channels
            nn.Sequential(
                nn.Conv3d(64, base_channels, 3, padding=1),
                nn.InstanceNorm3d(base_channels),
                nn.ReLU(inplace=True),
                ResidualBlock3DWithAttention(base_channels),
                ResidualBlock3DWithAttention(base_channels),
                CBAM3D(base_channels),
            ),
        ])
        
        # Multi-scale outputs
        self.output_32 = nn.Conv3d(32, 1, 1)
        self.output_64 = nn.Conv3d(64, 1, 1)
        self.output_final = nn.Conv3d(base_channels, 1, 1)
    
    def forward(self, xrays: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            xrays: (B, num_views, H, W) or (B, num_views, C, H, W)
        Returns:
            final_output: (B, 1, D, H, W)
            aux_outputs: Dict with multi-scale outputs
        """
        # Handle both (B, num_views, H, W) and (B, num_views, C, H, W)
        if xrays.dim() == 4:
            xrays = xrays.unsqueeze(2)
        
        B, num_views = xrays.shape[:2]
        
        # Encode each view
        view_features = []
        for v in range(num_views):
            feat = self.xray_encoder(xrays[:, v])
            view_features.append(feat)
        
        # Cross-view attention (frontal attends to lateral)
        if len(view_features) == 2:
            frontal_attended = self.cross_view_attention(view_features[0], view_features[1])
            lateral_attended = self.cross_view_attention(view_features[1], view_features[0])
            fused = (frontal_attended + lateral_attended) / 2
        else:
            fused = torch.stack(view_features, dim=0).mean(dim=0)
        
        fused = self.view_fusion(fused)
        
        # Lift to 3D
        depth_features = self.depth_lifter(fused)
        B, _, H_feat, W_feat = depth_features.shape
        D = self.volume_size[0]
        
        # Reshape to 3D
        x = depth_features.view(B, 8, D, H_feat, W_feat)
        
        # Interpolate to target size if needed
        if (H_feat, W_feat) != (self.volume_size[1], self.volume_size[2]):
            x = F.interpolate(x, size=self.volume_size, mode='trilinear', align_corners=True)
        
        # Multi-scale decoder with attention
        aux_outputs = {}
        
        for i, block in enumerate(self.decoder):
            x = block(x)
            
            if i == 0:
                aux_outputs['output_32'] = self.output_32(x)
            elif i == 1:
                aux_outputs['output_64'] = self.output_64(x)
            elif i == 2:
                aux_outputs['output_final'] = self.output_final(x)
        
        final_output = aux_outputs['output_final']
        
        return final_output, aux_outputs


class EnhancedLossWithAttention(nn.Module):
    """Combined loss with all components"""
    
    def __init__(self, 
                 l1_weight: float = 1.0,
                 ssim_weight: float = 0.5,
                 perceptual_weight: float = 0.2,
                 edge_weight: float = 0.2,
                 multiscale_weight: float = 0.3):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.multiscale_weight = multiscale_weight
        
        self.perceptual_loss = PerceptualLoss()
        self.edge_loss = EdgeAwareLoss()
    
    def ssim_loss(self, pred, target):
        """SSIM loss (structural similarity)"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = F.avg_pool3d(pred, 3, stride=1, padding=1)
        mu_target = F.avg_pool3d(target, 3, stride=1, padding=1)
        
        sigma_pred = F.avg_pool3d(pred ** 2, 3, stride=1, padding=1) - mu_pred ** 2
        sigma_target = F.avg_pool3d(target ** 2, 3, stride=1, padding=1) - mu_target ** 2
        sigma_pred_target = F.avg_pool3d(pred * target, 3, stride=1, padding=1) - mu_pred * mu_target
        
        ssim_map = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
        
        return 1 - ssim_map.mean()
    
    def forward(self, pred, target, aux_outputs):
        """Compute combined loss"""
        # Main losses
        l1_loss = F.l1_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        perceptual_loss = self.perceptual_loss(pred, target)
        edge_loss = self.edge_loss(pred, target)
        
        # Multi-scale supervision
        multiscale_loss = 0
        if 'output_32' in aux_outputs:
            target_32 = F.interpolate(target, scale_factor=0.5, mode='trilinear', align_corners=True)
            multiscale_loss += F.l1_loss(aux_outputs['output_32'], target_32)
        if 'output_64' in aux_outputs:
            multiscale_loss += F.l1_loss(aux_outputs['output_64'], target)
        
        total_loss = (
            self.l1_weight * l1_loss +
            self.ssim_weight * ssim_loss +
            self.perceptual_weight * perceptual_loss +
            self.edge_weight * edge_loss +
            self.multiscale_weight * multiscale_loss
        )
        
        return total_loss, {
            'l1': l1_loss.item(),
            'ssim': ssim_loss.item(),
            'perceptual': perceptual_loss.item(),
            'edge': edge_loss.item(),
            'multiscale': multiscale_loss.item() if isinstance(multiscale_loss, torch.Tensor) else multiscale_loss
        }
