"""
Frequency-Aware Multi-Scale Loss System
Stage 1: L1 + SSIM (structure)
Stage 2: + VGG perceptual (texture)
Stage 3: + Gradient magnitude (edges) + DRR reprojection consistency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSIMLoss(nn.Module):
    """3D SSIM loss for structural similarity"""
    def __init__(self, window_size=11, channel=1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        
    def forward(self, pred, target):
        """
        Args:
            pred, target: (B, 1, D, H, W)
        Returns:
            ssim_loss: scalar
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        window_size = min(self.window_size, pred.shape[2], pred.shape[3], pred.shape[4])
        
        mu_pred = F.avg_pool3d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool3d(target, window_size, stride=1, padding=window_size//2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool3d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool3d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool3d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target
        
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return 1 - ssim_map.mean()


class TriPlanarVGGLoss(nn.Module):
    """
    2D VGG perceptual loss on tri-planar slices
    Captures texture and mid-frequency patterns
    """
    def __init__(self):
        super().__init__()
        
        # Use VGG16 features (pretrained on ImageNet)
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except:
            # Fallback for older torchvision
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)
        
        # Extract feature layers
        self.features = nn.ModuleList([
            vgg.features[:4],   # relu1_2
            vgg.features[:9],   # relu2_2
            vgg.features[:16],  # relu3_3
        ])
        
        # Freeze VGG weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.layer_weights = [1.0, 1.0, 1.0]
        
    def extract_features(self, x):
        """Extract VGG features from 2D image"""
        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x)
        return features
    
    def forward(self, pred_volume, target_volume):
        """
        Extract tri-planar slices and compute VGG perceptual loss
        Args:
            pred_volume, target_volume: (B, 1, D, H, W)
        Returns:
            perceptual_loss: scalar
        """
        B, C, D, H, W = pred_volume.shape
        
        # Debug: check input shape
        if C != 1:
            raise ValueError(f"VGG loss expected 1 channel, got {C} channels. pred_volume shape: {pred_volume.shape}")
        
        # Sample slices from middle of volume
        mid_d, mid_h, mid_w = D // 2, H // 2, W // 2
        
        # Axial, Sagittal, Coronal slices
        pred_slices = [
            pred_volume[:, :, mid_d, :, :],    # Axial
            pred_volume[:, :, :, mid_h, :],    # Sagittal
            pred_volume[:, :, :, :, mid_w],    # Coronal
        ]
        target_slices = [
            target_volume[:, :, mid_d, :, :],
            target_volume[:, :, :, mid_h, :],
            target_volume[:, :, :, :, mid_w],
        ]
        
        total_loss = 0.0
        
        for pred_slice, target_slice in zip(pred_slices, target_slices):
            # Debug: check slice shape before processing
            if pred_slice.shape[1] != 1:
                raise ValueError(f"pred_slice has wrong channels before expand: shape={pred_slice.shape}, expected (B, 1, H, W)")
            
            # Normalize to [0, 1] and replicate to 3 channels (VGG expects RGB)
            pred_slice = (pred_slice + 1) / 2  # Assuming input is [-1, 1]
            target_slice = (target_slice + 1) / 2
            
            # Repeat grayscale to 3-channel RGB (not expand - that requires dim=1)
            pred_slice = pred_slice.repeat(1, 3, 1, 1)
            target_slice = target_slice.repeat(1, 3, 1, 1)
            
            # Extract features
            pred_feats = self.extract_features(pred_slice)
            target_feats = self.extract_features(target_slice)
            
            # Compute perceptual loss
            for pred_feat, target_feat, weight in zip(pred_feats, target_feats, self.layer_weights):
                total_loss += weight * F.l1_loss(pred_feat, target_feat)
        
        return total_loss / 3  # Average over 3 planes


class GradientMagnitudeLoss(nn.Module):
    """
    Gradient magnitude loss for capturing edges and fine details
    Emphasizes high-frequency content
    """
    def __init__(self):
        super().__init__()
        
    def compute_gradients_3d(self, volume):
        """Compute 3D gradients using Sobel-like operators"""
        # Gradient in D dimension
        grad_d = volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :]
        # Gradient in H dimension
        grad_h = volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :]
        # Gradient in W dimension
        grad_w = volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1]
        
        return grad_d, grad_h, grad_w
    
    def forward(self, pred_volume, target_volume):
        """
        Args:
            pred_volume, target_volume: (B, 1, D, H, W)
        Returns:
            gradient_loss: scalar
        """
        pred_grads = self.compute_gradients_3d(pred_volume)
        target_grads = self.compute_gradients_3d(target_volume)
        
        loss = 0.0
        for pred_grad, target_grad in zip(pred_grads, target_grads):
            # Compute gradient magnitude
            pred_mag = torch.abs(pred_grad)
            target_mag = torch.abs(target_grad)
            loss += F.l1_loss(pred_mag, target_mag)
        
        return loss / 3  # Average over 3 dimensions


class DRRReprojectionLoss(nn.Module):
    """
    X-ray Reprojection Consistency Loss
    Differentiable DRR generation from predicted CT volume
    Ensures geometric consistency with input X-rays
    """
    def __init__(self, img_size=512):
        super().__init__()
        self.img_size = img_size
        
    def generate_drr(self, ct_volume, view_angle=0):
        """
        Generate DRR via differentiable raycasting
        Simplified version using maximum intensity projection (MIP)
        
        Args:
            ct_volume: (B, 1, D, H, W)
            view_angle: 0 for AP (front), 90 for lateral
        Returns:
            drr: (B, 1, H, W)
        """
        if view_angle == 0:
            # AP view: project along depth (D) dimension
            drr = torch.mean(ct_volume, dim=2)  # Average intensity projection
        else:
            # Lateral view: project along width (W) dimension
            drr = torch.mean(ct_volume, dim=4)
        
        # Resize to match X-ray resolution
        drr = F.interpolate(drr, size=(self.img_size, self.img_size), 
                           mode='bilinear', align_corners=False)
        
        return drr
    
    def forward(self, pred_volume, input_xrays):
        """
        Args:
            pred_volume: (B, 1, D, H, W) - predicted CT volume
            input_xrays: (B, 2, 1, 512, 512) - input X-ray images [AP, Lateral]
        Returns:
            reprojection_loss: scalar
        """
        # Generate DRRs from predicted volume
        drr_ap = self.generate_drr(pred_volume, view_angle=0)
        drr_lateral = self.generate_drr(pred_volume, view_angle=90)
        
        # Extract input X-rays
        xray_ap = input_xrays[:, 0, :, :, :]      # (B, 1, 512, 512)
        xray_lateral = input_xrays[:, 1, :, :, :] # (B, 1, 512, 512)
        
        # Compute consistency loss
        loss_ap = F.l1_loss(drr_ap, xray_ap)
        loss_lateral = F.l1_loss(drr_lateral, xray_lateral)
        
        return (loss_ap + loss_lateral) / 2


class Stage1Loss(nn.Module):
    """
    Stage 1 Loss: L1 + SSIM
    Focus: Coarse structure and overall anatomy
    """
    def __init__(self, l1_weight=1.0, ssim_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.ssim_loss = SSIMLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred, target: (B, 1, 64, 64, 64)
        Returns:
            loss_dict: dict with losses
        """
        l1_loss = F.l1_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        
        total_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'ssim_loss': ssim_loss
        }


class Stage2Loss(nn.Module):
    """
    Stage 2 Loss: L1 + SSIM + VGG Perceptual
    Focus: Add texture and medium-frequency details
    """
    def __init__(self, l1_weight=1.0, ssim_weight=0.5, vgg_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.vgg_weight = vgg_weight
        
        self.ssim_loss = SSIMLoss()
        self.vgg_loss = TriPlanarVGGLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred, target: (B, 1, 128, 128, 128)
        Returns:
            loss_dict: dict with losses
        """
        l1_loss = F.l1_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        vgg_loss = self.vgg_loss(pred, target)
        
        total_loss = (self.l1_weight * l1_loss + 
                     self.ssim_weight * ssim_loss + 
                     self.vgg_weight * vgg_loss)
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'ssim_loss': ssim_loss,
            'vgg_loss': vgg_loss
        }


class Stage3Loss(nn.Module):
    """
    Stage 3 Loss: L1 + SSIM + VGG + Gradient + DRR Reprojection
    Focus: Capture fine details, edges, and geometric consistency
    """
    def __init__(self, l1_weight=1.0, ssim_weight=0.5, vgg_weight=0.1, 
                 gradient_weight=0.2, drr_weight=0.3):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.vgg_weight = vgg_weight
        self.gradient_weight = gradient_weight
        self.drr_weight = drr_weight
        
        self.ssim_loss = SSIMLoss()
        self.vgg_loss = TriPlanarVGGLoss()
        self.gradient_loss = GradientMagnitudeLoss()
        self.drr_loss = DRRReprojectionLoss()
    
    def forward(self, pred, target, input_xrays=None):
        """
        Args:
            pred, target: (B, 1, 256, 256, 256)
            input_xrays: (B, 2, 1, 512, 512) - optional for DRR loss
        Returns:
            loss_dict: dict with losses
        """
        l1_loss = F.l1_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        vgg_loss = self.vgg_loss(pred, target)
        gradient_loss = self.gradient_loss(pred, target)
        
        total_loss = (self.l1_weight * l1_loss + 
                     self.ssim_weight * ssim_loss + 
                     self.vgg_weight * vgg_loss +
                     self.gradient_weight * gradient_loss)
        
        loss_dict = {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'ssim_loss': ssim_loss,
            'vgg_loss': vgg_loss,
            'gradient_loss': gradient_loss
        }
        
        # Add DRR reprojection loss if X-rays are provided
        if input_xrays is not None:
            drr_loss = self.drr_loss(pred, input_xrays)
            total_loss = total_loss + self.drr_weight * drr_loss
            loss_dict['total_loss'] = total_loss
            loss_dict['drr_loss'] = drr_loss
        
        return loss_dict


class MultiScaleLoss(nn.Module):
    """
    Unified multi-scale loss system
    Automatically selects appropriate losses based on stage
    """
    def __init__(self, config=None):
        super().__init__()
        
        # Default weights
        if config is None:
            config = {
                'stage1': {'l1': 1.0, 'ssim': 0.5},
                'stage2': {'l1': 1.0, 'ssim': 0.5, 'vgg': 0.1},
                'stage3': {'l1': 1.0, 'ssim': 0.5, 'vgg': 0.1, 'gradient': 0.2, 'drr': 0.3}
            }
        
        self.stage1_loss = Stage1Loss(
            l1_weight=config['stage1']['l1'],
            ssim_weight=config['stage1']['ssim']
        )
        
        self.stage2_loss = Stage2Loss(
            l1_weight=config['stage2']['l1'],
            ssim_weight=config['stage2']['ssim'],
            vgg_weight=config['stage2']['vgg']
        )
        
        self.stage3_loss = Stage3Loss(
            l1_weight=config['stage3']['l1'],
            ssim_weight=config['stage3']['ssim'],
            vgg_weight=config['stage3']['vgg'],
            gradient_weight=config['stage3']['gradient'],
            drr_weight=config['stage3']['drr']
        )
    
    def forward(self, pred, target, stage=1, input_xrays=None):
        """
        Args:
            pred: predicted volume (size depends on stage)
            target: target volume (same size as pred)
            stage: 1, 2, or 3
            input_xrays: (B, 2, 1, 512, 512) - for stage 3 DRR loss
        Returns:
            loss_dict: dict with all losses for the stage
        """
        if stage == 1:
            return self.stage1_loss(pred, target)
        elif stage == 2:
            return self.stage2_loss(pred, target)
        elif stage == 3:
            return self.stage3_loss(pred, target, input_xrays)
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3.")


def compute_psnr(pred, target):
    """Compute Peak Signal-to-Noise Ratio"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    # Assuming data range is [-1, 1], so max value is 2.0
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
    return psnr.item()


def compute_ssim_metric(pred, target):
    """Compute SSIM as a metric (not loss)"""
    # Compute SSIM directly without creating new loss instance
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    window_size = min(11, pred.shape[2], pred.shape[3], pred.shape[4])
    
    mu_pred = F.avg_pool3d(pred, window_size, stride=1, padding=window_size//2)
    mu_target = F.avg_pool3d(target, window_size, stride=1, padding=window_size//2)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = F.avg_pool3d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
    sigma_target_sq = F.avg_pool3d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
    sigma_pred_target = F.avg_pool3d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target
    
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    
    return ssim_map.mean().item()


if __name__ == "__main__":
    # Test losses
    print("Testing Multi-Scale Loss System...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = MultiScaleLoss().to(device)
    
    # Test Stage 1 (64³)
    print("\n=== Stage 1 Loss (64³) ===")
    pred1 = torch.randn(2, 1, 64, 64, 64).to(device)
    target1 = torch.randn(2, 1, 64, 64, 64).to(device)
    loss_dict1 = loss_fn(pred1, target1, stage=1)
    for k, v in loss_dict1.items():
        print(f"{k}: {v.item():.4f}")
    
    # Test Stage 2 (128³)
    print("\n=== Stage 2 Loss (128³) ===")
    pred2 = torch.randn(2, 1, 128, 128, 128).to(device)
    target2 = torch.randn(2, 1, 128, 128, 128).to(device)
    loss_dict2 = loss_fn(pred2, target2, stage=2)
    for k, v in loss_dict2.items():
        print(f"{k}: {v.item():.4f}")
    
    # Test Stage 3 (256³) with DRR
    print("\n=== Stage 3 Loss (256³) with DRR ===")
    pred3 = torch.randn(2, 1, 256, 256, 256).to(device)
    target3 = torch.randn(2, 1, 256, 256, 256).to(device)
    xrays = torch.randn(2, 2, 1, 512, 512).to(device)
    loss_dict3 = loss_fn(pred3, target3, stage=3, input_xrays=xrays)
    for k, v in loss_dict3.items():
        print(f"{k}: {v.item():.4f}")
    
    # Test metrics
    print("\n=== Metrics ===")
    psnr = compute_psnr(pred1, target1)
    ssim = compute_ssim_metric(pred1, target1)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
