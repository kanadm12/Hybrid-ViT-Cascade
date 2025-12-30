"""
CNN Branch for Local Features
EfficientNet-based branch for precise anatomical details
Paired with ViT for complementary strengths
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block
    Core building block of EfficientNet
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: int = 6,
                 kernel_size: int = 3,
                 stride: int = 1,
                 se_ratio: float = 0.25):
        super().__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size, stride=stride,
                     padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze-and-Excitation
        se_channels = max(1, int(hidden_dim * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(hidden_dim, se_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(se_channels, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Projection phase
        self.project_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Expansion
        out = self.expand_conv(x)
        
        # Depthwise
        out = self.depthwise_conv(out)
        
        # SE
        out = out * self.se(out)
        
        # Projection
        out = self.project_conv(out)
        
        # Residual connection
        if self.use_residual:
            out = out + identity
        
        return out


class EfficientNet3D(nn.Module):
    """
    EfficientNet-inspired 3D CNN for local feature extraction
    Lightweight and parameter-efficient
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 32,
                 feature_dim: int = 256):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.SiLU(inplace=True)
        )
        
        # MBConv blocks with progressive downsampling
        # Stage 1: 32 -> 64 channels, stride=1
        self.stage1 = nn.Sequential(
            MBConvBlock(base_channels, base_channels * 2, expand_ratio=1, stride=1),
            MBConvBlock(base_channels * 2, base_channels * 2, expand_ratio=6, stride=1)
        )
        
        # Stage 2: 64 -> 128 channels, stride=2
        self.stage2 = nn.Sequential(
            MBConvBlock(base_channels * 2, base_channels * 4, expand_ratio=6, stride=2),
            MBConvBlock(base_channels * 4, base_channels * 4, expand_ratio=6, stride=1)
        )
        
        # Stage 3: 128 -> 256 channels, stride=2
        self.stage3 = nn.Sequential(
            MBConvBlock(base_channels * 4, feature_dim, expand_ratio=6, stride=2),
            MBConvBlock(feature_dim, feature_dim, expand_ratio=6, stride=1)
        )
        
        # Feature pyramid for multi-scale features
        self.feature_pyramid = nn.ModuleList([
            nn.Conv3d(base_channels * 2, feature_dim, 1),  # Stage 1
            nn.Conv3d(base_channels * 4, feature_dim, 1),  # Stage 2
            nn.Conv3d(feature_dim, feature_dim, 1)         # Stage 3
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, 1, D, H, W) input volume
        Returns:
            final_features: (B, feature_dim, D', H', W') final feature map
            pyramid_features: List of multi-scale features
        """
        x = self.stem(x)  # (B, 32, D/2, H/2, W/2)
        
        # Extract multi-scale features
        s1 = self.stage1(x)   # (B, 64, D/2, H/2, W/2)
        s2 = self.stage2(s1)  # (B, 128, D/4, H/4, W/4)
        s3 = self.stage3(s2)  # (B, 256, D/8, H/8, W/8)
        
        # Build feature pyramid
        pyramid = [
            self.feature_pyramid[0](s1),
            self.feature_pyramid[1](s2),
            self.feature_pyramid[2](s3)
        ]
        
        return s3, pyramid


class HybridCNNViTFusion(nn.Module):
    """
    Fuses CNN local features with ViT global features
    Uses cross-attention and adaptive gating
    """
    
    def __init__(self,
                 feature_dim: int = 256,
                 num_heads: int = 4):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Cross-attention: CNN queries ViT features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Adaptive gating to balance CNN and ViT
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self,
                cnn_features: torch.Tensor,
                vit_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_features: (B, C, D, H, W) from CNN branch
            vit_features: (B, N, C) from ViT branch
        Returns:
            fused_features: (B, N, C) fused features
        """
        B, C, D, H, W = cnn_features.shape
        
        # Flatten CNN features to tokens
        cnn_tokens = cnn_features.flatten(2).transpose(1, 2)  # (B, D*H*W, C)
        
        # Cross-attention: CNN queries, ViT key/value
        attended_cnn, _ = self.cross_attn(
            query=cnn_tokens,
            key=vit_features,
            value=vit_features
        )
        
        # Upsample ViT features to match CNN spatial resolution if needed
        N_vit = vit_features.shape[1]
        N_cnn = cnn_tokens.shape[1]
        
        if N_vit != N_cnn:
            # Reshape ViT features and interpolate
            D_vit = int(round(N_vit ** (1/3)))
            vit_3d = vit_features.reshape(B, D_vit, D_vit, D_vit, C).permute(0, 4, 1, 2, 3)
            vit_upsampled = F.interpolate(vit_3d, size=(D, H, W), mode='trilinear', align_corners=True)
            vit_tokens = vit_upsampled.permute(0, 2, 3, 4, 1).reshape(B, N_cnn, C)
        else:
            vit_tokens = vit_features
        
        # Adaptive gating
        combined = torch.cat([attended_cnn, vit_tokens], dim=-1)  # (B, N, 2C)
        gate_weights = self.gate(combined)  # (B, N, C)
        
        # Weighted fusion
        gated_cnn = attended_cnn * gate_weights
        gated_vit = vit_tokens * (1 - gate_weights)
        
        # Final fusion
        fused_input = torch.cat([gated_cnn, gated_vit], dim=-1)
        fused_features = self.fusion(fused_input)
        
        return fused_features


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Testing EfficientNet3D ===")
    B = 2
    x = torch.randn(B, 1, 64, 64, 64).to(device)
    
    cnn = EfficientNet3D(in_channels=1, base_channels=32, feature_dim=256).to(device)
    final_features, pyramid = cnn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Final features: {final_features.shape}")
    print(f"Pyramid scales: {[p.shape for p in pyramid]}")
    
    # Parameter count
    total_params = sum(p.numel() for p in cnn.parameters())
    print(f"CNN parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print("\n=== Testing Hybrid CNN-ViT Fusion ===")
    cnn_features = final_features  # (B, 256, 8, 8, 8)
    vit_features = torch.randn(B, 512, 256).to(device)  # (B, N, C)
    
    fusion = HybridCNNViTFusion(feature_dim=256, num_heads=4).to(device)
    fused = fusion(cnn_features, vit_features)
    
    print(f"CNN features: {cnn_features.shape}")
    print(f"ViT features: {vit_features.shape}")
    print(f"Fused features: {fused.shape}")
    
    fusion_params = sum(p.numel() for p in fusion.parameters())
    print(f"Fusion parameters: {fusion_params:,} ({fusion_params/1e6:.2f}M)")
    
    print("\nCNN local branch test completed!")
