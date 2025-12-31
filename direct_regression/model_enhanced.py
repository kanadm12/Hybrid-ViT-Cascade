"""
Enhanced Direct Regression Model with Refinement Network
- Better base model (64³) with perceptual + edge-aware losses
- Lightweight refinement network (64³ → 256³)
- Multi-scale outputs for better detail preservation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for anatomical structure preservation"""
    
    def __init__(self):
        super().__init__()
        # Use VGG16 features
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]) # relu3_3
        
        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, D, H, W) predicted volume
            target: (B, 1, D, H, W) target volume
        Returns:
            perceptual_loss: scalar
        """
        # Extract middle slices (treat as 2D images)
        D = pred.shape[2]
        pred_slice = pred[:, :, D//2, :, :]  # (B, 1, H, W)
        target_slice = target[:, :, D//2, :, :]
        
        # Convert to 3-channel for VGG and to float32 for VGG compatibility
        pred_3ch = pred_slice.repeat(1, 3, 1, 1).float()
        target_3ch = target_slice.repeat(1, 3, 1, 1).float()
        
        # Extract features at multiple layers
        loss = 0
        x_pred, x_target = pred_3ch, target_3ch
        
        for slice_net in [self.slice1, self.slice2, self.slice3]:
            x_pred = slice_net(x_pred)
            x_target = slice_net(x_target)
            loss += F.l1_loss(x_pred, x_target)
        
        return loss / 3.0


class EdgeAwareLoss(nn.Module):
    """Edge-aware loss to preserve sharp boundaries"""
    
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0))
    
    def compute_edges(self, x):
        """Compute edge magnitude using Sobel"""
        # Convert to float32 for Sobel filters
        x = x.float()
        
        # Process each depth slice
        B, C, D, H, W = x.shape
        edges = []
        
        for d in range(D):
            slice_2d = x[:, :, d, :, :]  # (B, 1, H, W)
            edge_x = F.conv2d(slice_2d, self.sobel_x, padding=1)
            edge_y = F.conv2d(slice_2d, self.sobel_y, padding=1)
            edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
            edges.append(edge_mag)
        
        return torch.stack(edges, dim=2)  # (B, 1, D, H, W)
    
    def forward(self, pred, target):
        pred_edges = self.compute_edges(pred)
        target_edges = self.compute_edges(target)
        return F.l1_loss(pred_edges, target_edges)


class ResidualBlock3D(nn.Module):
    """3D Residual block with instance norm"""
    
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


class EnhancedDirectModel(nn.Module):
    """Enhanced direct regression with multi-scale outputs"""
    
    def __init__(self, 
                 volume_size: Tuple[int, int, int] = (64, 64, 64),
                 base_channels: int = 64):
        super().__init__()
        
        self.volume_size = volume_size
        D, H, W = volume_size
        
        # X-ray encoder (similar to original but deeper)
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
        
        # Feature fusion for dual views
        self.view_fusion = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 2D to 3D lifting with depth attention
        self.depth_lifter = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, D * 8, 1),  # Output channels for depth
        )
        
        # 3D decoder with residual blocks
        self.decoder = nn.ModuleList([
            # Block 1: (8, D, H, W) → (32, D, H, W)
            nn.Sequential(
                nn.Conv3d(8, 32, 3, padding=1),
                nn.InstanceNorm3d(32),
                nn.ReLU(inplace=True),
                ResidualBlock3D(32),
            ),
            # Block 2: (32, D, H, W) → (64, D, H, W)
            nn.Sequential(
                nn.Conv3d(32, 64, 3, padding=1),
                nn.InstanceNorm3d(64),
                nn.ReLU(inplace=True),
                ResidualBlock3D(64),
                ResidualBlock3D(64),
            ),
            # Block 3: (64, D, H, W) → (128, D, H, W)
            nn.Sequential(
                nn.Conv3d(64, 128, 3, padding=1),
                nn.InstanceNorm3d(128),
                nn.ReLU(inplace=True),
                ResidualBlock3D(128),
                ResidualBlock3D(128),
            ),
        ])
        
        # Multi-scale outputs
        self.output_32 = nn.Conv3d(32, 1, 1)  # From block 1
        self.output_64 = nn.Conv3d(64, 1, 1)  # From block 2
        self.output_128 = nn.Conv3d(128, 1, 1)  # From block 3 (final)
    
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
            # Add channel dimension: (B, num_views, H, W) -> (B, num_views, 1, H, W)
            xrays = xrays.unsqueeze(2)
        
        B, num_views = xrays.shape[:2]
        
        # Encode each view
        view_features = []
        for v in range(num_views):
            feat = self.xray_encoder(xrays[:, v])  # (B, 1, H, W) -> features
            view_features.append(feat)
        
        # Fuse views (average)
        fused = torch.stack(view_features, dim=0).mean(dim=0)  # (B, 512, H', W')
        fused = self.view_fusion(fused)
        
        # Lift to 3D
        depth_features = self.depth_lifter(fused)  # (B, D*8, H', W')
        B, _, H_feat, W_feat = depth_features.shape
        D = self.volume_size[0]
        
        # Reshape to 3D: (B, D*8, H', W') → (B, 8, D, H', W')
        x = depth_features.view(B, 8, D, H_feat, W_feat)
        
        # Interpolate to target size if needed
        if (H_feat, W_feat) != (self.volume_size[1], self.volume_size[2]):
            x = F.interpolate(x, size=self.volume_size, mode='trilinear', align_corners=True)
        
        # Multi-scale decoder
        aux_outputs = {}
        
        for i, block in enumerate(self.decoder):
            x = block(x)
            
            # Output at this scale
            if i == 0:
                aux_outputs['output_32'] = self.output_32(x)
            elif i == 1:
                aux_outputs['output_64'] = self.output_64(x)
            elif i == 2:
                aux_outputs['output_128'] = self.output_128(x)
        
        final_output = aux_outputs['output_128']
        
        return final_output, aux_outputs


class RefinementNetwork(nn.Module):
    """Lightweight network to refine 64³ → 256³"""
    
    def __init__(self):
        super().__init__()
        
        # Upsampling with residual blocks (ESRGAN-inspired)
        self.upsample1 = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1),
            nn.PixelShuffle(2),  # 64³ → 128³
            nn.ReLU(inplace=True),
        )
        
        self.refine1 = nn.Sequential(
            ResidualBlock3D(8),  # After pixel shuffle: 64/8 = 8 channels
            ResidualBlock3D(8),
        )
        
        self.upsample2 = nn.Sequential(
            nn.Conv3d(8, 64, 3, padding=1),
            nn.PixelShuffle(2),  # 128³ → 256³
            nn.ReLU(inplace=True),
        )
        
        self.refine2 = nn.Sequential(
            ResidualBlock3D(8),
            ResidualBlock3D(8),
            nn.Conv3d(8, 1, 3, padding=1),
        )
    
    def forward(self, coarse_volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coarse_volume: (B, 1, 64, 64, 64)
        Returns:
            refined_volume: (B, 1, 256, 256, 256)
        """
        x = self.upsample1(coarse_volume)
        x = self.refine1(x)
        x = self.upsample2(x)
        x = self.refine2(x)
        
        # Add skip connection from upsampled input
        coarse_upsampled = F.interpolate(coarse_volume, scale_factor=4, mode='trilinear', align_corners=True)
        x = x + coarse_upsampled
        
        return x


class EnhancedLoss(nn.Module):
    """Combined loss: L1 + SSIM + Perceptual + Edge + Multi-scale"""
    
    def __init__(self, 
                 l1_weight: float = 1.0,
                 ssim_weight: float = 0.5,
                 perceptual_weight: float = 0.1,
                 edge_weight: float = 0.1,
                 multiscale_weight: float = 0.3):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.multiscale_weight = multiscale_weight
        
        self.perceptual_loss = PerceptualLoss()
        self.edge_loss = EdgeAwareLoss()
    
    def ssim_3d(self, pred, target, window_size=11):
        """Simplified 3D SSIM on middle slices"""
        D = pred.shape[2]
        pred_slice = pred[:, :, D//2, :, :]
        target_slice = target[:, :, D//2, :, :]
        
        # SSIM computation (simplified)
        mu1 = F.avg_pool2d(pred_slice, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(target_slice, window_size, stride=1, padding=window_size//2)
        
        sigma1_sq = F.avg_pool2d(pred_slice**2, window_size, stride=1, padding=window_size//2) - mu1**2
        sigma2_sq = F.avg_pool2d(target_slice**2, window_size, stride=1, padding=window_size//2) - mu2**2
        sigma12 = F.avg_pool2d(pred_slice * target_slice, window_size, stride=1, padding=window_size//2) - mu1 * mu2
        
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()
    
    def forward(self, pred, target, aux_outputs=None):
        """
        Args:
            pred: (B, 1, D, H, W) final prediction
            target: (B, 1, D, H, W) ground truth
            aux_outputs: Dict with multi-scale outputs
        Returns:
            loss_dict: Dictionary of losses
        """
        # Main losses
        l1_loss = F.l1_loss(pred, target)
        ssim_loss = self.ssim_3d(pred, target)
        perceptual_loss = self.perceptual_loss(pred, target)
        edge_loss = self.edge_loss(pred, target)
        
        # Multi-scale losses
        multiscale_loss = 0
        if aux_outputs is not None:
            for key in ['output_32', 'output_64']:
                if key in aux_outputs:
                    # Downsample target to match
                    target_down = F.interpolate(target, size=aux_outputs[key].shape[2:], 
                                                mode='trilinear', align_corners=True)
                    multiscale_loss += F.l1_loss(aux_outputs[key], target_down)
            multiscale_loss /= 2.0
        
        # Total loss
        total_loss = (self.l1_weight * l1_loss + 
                     self.ssim_weight * ssim_loss +
                     self.perceptual_weight * perceptual_loss +
                     self.edge_weight * edge_loss +
                     self.multiscale_weight * multiscale_loss)
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'ssim_loss': ssim_loss,
            'perceptual_loss': perceptual_loss,
            'edge_loss': edge_loss,
            'multiscale_loss': multiscale_loss
        }
