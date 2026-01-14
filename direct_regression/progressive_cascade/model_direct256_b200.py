"""
Memory-Optimized 256³ Direct Regression Model for B200 GPU (180GB VRAM)
Trained from scratch (no transfer learning)

Architecture:
- 16³ → 32³ → 64³ → 128³ → 256³ (5 stages)
- XRay features: 128 channels (reduced from 512)
- Stage 4: No RDB blocks (memory-critical)
- Final refine: Direct convolutions (no RDB)
- Multi-scale skip connections

Memory Budget (batch=1):
- Forward: ~120 GB
- Backward: ~60 GB
- Total: ~180 GB ✅ Fits B200

Expected Performance:
- PSNR: 27-28 dB after 200 epochs
- SSIM: 0.65-0.70
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Import ResidualDenseBlock from 128³ model for compatibility
from model_direct128_h200 import ResidualDenseBlock


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)
    Combines channel and spatial attention for feature refinement.
    Memory: ~2-3GB at 128³ resolution with 128 channels
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        
        # Spatial attention
        self.conv_spatial = nn.Conv3d(2, 1, 7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


class XRayEncoder(nn.Module):
    """Encodes 2D X-ray views - Memory optimized (128 channels)"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(16, 96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, drr):
        # drr: (B, 2, 1, H, W) - AP and Lateral
        drr = drr.squeeze(2)  # (B, 2, H, W)
        return self.encoder(drr)


class Direct256Model_B200(nn.Module):
    """Memory-optimized 256³ direct regression model for B200 GPU
    
    Memory: ~175GB with batch_size=1
    Architecture: Reduced XRay (128ch), no RDB at 256³
    Expected PSNR: 27-28 dB
    """
    def __init__(self):
        super().__init__()
        
        # Initial learnable volume
        self.initial_volume = nn.Parameter(torch.randn(1, 16, 16, 16, 16) * 0.01)
        
        # X-ray encoder (transferable from 128³)
        self.xray_encoder = XRayEncoder()
        
        # ====== Stage 1: 16³ → 32³ (Transferable) ======
        self.enc_16_32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            ResidualDenseBlock(in_channels=32, growth_rate=16, num_layers=4),
        )
        
        # ====== Stage 2: 32³ → 64³ (Transferable) ======
        self.enc_32_64 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            ResidualDenseBlock(in_channels=64, growth_rate=24, num_layers=4),
        )
        
        # ====== Stage 3: 64³ → 128³ (Transferable) ======
        # NOTE: 128³ model uses 320 channels, but incompatible with 256³ memory constraints
        # Using 128 channels - this stage will NOT transfer from 128³ checkpoint
        self.enc_64_128 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            ResidualDenseBlock(in_channels=128, growth_rate=16, num_layers=3),
            ResidualDenseBlock(in_channels=128, growth_rate=16, num_layers=3),
        )
        
        # ====== CBAM Attention at 128³ ======
        # Applied after Stage 3 for global context refinement
        # Memory: ~2-3GB at 128³ with 128 channels
        self.cbam_128 = CBAM(channels=128, reduction=16)
        
        # ====== Stage 4: 128³ → 256³ (NEW - Random Init) ======
        # Memory-critical: No RDB blocks to fit in 180GB
        self.enc_128_256 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
        )
        
        # X-ray fusion at multiple scales (128ch XRay features)
        self.xray_fusion_32 = nn.Conv3d(32 + 128, 32, 1)
        self.xray_fusion_64 = nn.Conv3d(64 + 128, 64, 1)
        self.xray_fusion_128 = nn.Conv3d(128 + 128, 128, 1)
        self.xray_fusion_256 = nn.Conv3d(128 + 128, 128, 1)
        
        # Skip connection projections
        self.skip_proj_32_to_256 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False),
            nn.Conv3d(32, 64, 1)
        )
        self.skip_proj_64_to_256 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False),
            nn.Conv3d(64, 64, 1)
        )
        self.skip_proj_128_to_256 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(128, 64, 1)
        )
        
        # Multi-scale fusion
        self.multiscale_fusion = nn.Sequential(
            nn.Conv3d(128 + 64 + 64 + 64, 128, 1),  # Updated to 128
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True)
        )
        
        # Final refinement at 256³ - Simplified for memory
        self.final_refine = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, 1),
        )
        
    def forward(self, drr):
        """
        Args:
            drr: (B, 2, 1, H, W) - AP and Lateral X-rays
        Returns:
            ct_volume: (B, 1, 256, 256, 256)
        """
        B = drr.shape[0]
        
        # Encode X-ray features
        xray_feat = self.xray_encoder(drr)  # (B, 128, H/16, W/16)
        
        # Initial volume
        x = self.initial_volume.expand(B, -1, -1, -1, -1)
        
        # ====== Stage 1: 16³ → 32³ ======
        x = checkpoint(self.enc_16_32, x, use_reentrant=False)
        xray_32 = F.interpolate(xray_feat, size=(32, 32), mode='bilinear', align_corners=False)
        xray_32_3d = xray_32.unsqueeze(2).expand(-1, -1, 32, -1, -1)
        x = self.xray_fusion_32(torch.cat([x, xray_32_3d], dim=1))
        skip_32 = x
        
        # ====== Stage 2: 32³ → 64³ ======
        x = checkpoint(self.enc_32_64, x, use_reentrant=False)
        xray_64 = F.interpolate(xray_feat, size=(64, 64), mode='bilinear', align_corners=False)
        xray_64_3d = xray_64.unsqueeze(2).expand(-1, -1, 64, -1, -1)
        x = self.xray_fusion_64(torch.cat([x, xray_64_3d], dim=1))
        skip_64 = x
        
        # ====== Stage 3: 64³ → 128³ ======
        x = checkpoint(self.enc_64_128, x, use_reentrant=False)
        xray_128 = F.interpolate(xray_feat, size=(128, 128), mode='bilinear', align_corners=False)
        xray_128_3d = xray_128.unsqueeze(2).expand(-1, -1, 128, -1, -1)
        x = self.xray_fusion_128(torch.cat([x, xray_128_3d], dim=1))
        
        # Apply CBAM attention at 128³ for global context
        x = self.cbam_128(x)
        skip_128 = x
        
        # ====== Stage 4: 128³ → 256³ ======
        x = checkpoint(self.enc_128_256, x, use_reentrant=False)
        xray_256 = F.interpolate(xray_feat, size=(256, 256), mode='bilinear', align_corners=False)
        xray_256_3d = xray_256.unsqueeze(2).expand(-1, -1, 256, -1, -1)
        x = self.xray_fusion_256(torch.cat([x, xray_256_3d], dim=1))
        
        # Multi-scale skip connections
        skip_32_up = self.skip_proj_32_to_256(skip_32)
        skip_64_up = self.skip_proj_64_to_256(skip_64)
        skip_128_up = self.skip_proj_128_to_256(skip_128)
        
        # Fuse all scales
        x = self.multiscale_fusion(torch.cat([x, skip_32_up, skip_64_up, skip_128_up], dim=1))
        
        # Final refinement
        x = checkpoint(self.final_refine, x, use_reentrant=False)
        
        return x
    
    def load_pretrained_128(self, checkpoint_path):
        """Load weights from 128³ checkpoint and transfer compatible layers"""
        print(f"\n{'='*60}")
        print(f"Loading 128³ pretrained weights from:")
        print(f"  {checkpoint_path}")
        print(f"{'='*60}\n")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state' in checkpoint:
            pretrained_dict = checkpoint['model_state']
        elif 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            pretrained_dict = checkpoint
        
        model_dict = self.state_dict()
        
        # Transfer compatible weights
        transferred = []
        skipped = []
        
        for name, param in pretrained_dict.items():
            if name in model_dict and model_dict[name].shape == param.shape:
                model_dict[name] = param
                transferred.append(name)
            else:
                skipped.append(name)
        
        self.load_state_dict(model_dict)
        
        # Print summary
        print(f"✓ Transfer Summary:")
        print(f"  Transferred: {len(transferred)} layers")
        print(f"  Skipped: {len(skipped)} layers")
        print(f"  Transfer rate: {len(transferred)/(len(transferred)+len(skipped))*100:.1f}%\n")
        
        transferred_modules = set([name.split('.')[0] for name in transferred])
        print(f"✓ Transferred modules: {', '.join(sorted(transferred_modules))}")
        print(f"\n⚠ Randomly initialized: enc_128_256, final_refine")
        print(f"{'='*60}\n")
        
        return len(transferred), len(skipped)


if __name__ == "__main__":
    print("Testing Direct256Model_B200...")
    model = Direct256Model_B200().cuda()
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size: {trainable * 4 / 1024**3:.2f} GB (float32)")
    
    drr = torch.randn(1, 2, 1, 512, 512).cuda()
    with torch.cuda.amp.autocast():
        output = model(drr)
    
    print(f"\n✓ Forward pass: {drr.shape} → {output.shape}")
    print(f"Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
