"""
Direct CT Regression Model for 256³ Resolution
Memory-optimized with gradient checkpointing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ViTBlock3D(nn.Module):
    """Transformer block with 3D attention"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Attention3D(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DirectCTRegression256(nn.Module):
    """
    Direct regression from dual-view X-rays to 256³ CT volumes
    Memory-optimized with gradient checkpointing
    """
    def __init__(
        self,
        xray_size=512,
        volume_size=256,
        vit_patch_size=16,
        vit_dim=512,
        vit_depth=6,
        vit_heads=8,
        mlp_ratio=4.,
        dropout=0.1,
        use_checkpointing=True
    ):
        super().__init__()
        
        self.xray_size = xray_size
        self.volume_size = volume_size
        self.vit_patch_size = vit_patch_size
        self.vit_dim = vit_dim
        self.use_checkpointing = use_checkpointing
        
        # Input: 2 views stacked = 2 channels
        num_patches = (xray_size // vit_patch_size) ** 2 * 2  # 2 views
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(1, vit_dim, kernel_size=vit_patch_size, stride=vit_patch_size)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, vit_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # ViT encoder with gradient checkpointing
        self.blocks = nn.ModuleList([
            ViTBlock3D(vit_dim, vit_heads, mlp_ratio, qkv_bias=True, drop=dropout, attn_drop=dropout)
            for _ in range(vit_depth)
        ])
        
        self.norm = nn.LayerNorm(vit_dim)
        
        # 3D volume decoder (progressive upsampling)
        # Start from 32³, upsample to 256³
        initial_vol_size = 32
        self.to_volume_features = nn.Linear(vit_dim, initial_vol_size ** 3)
        
        # Progressive upsampling: 32³ → 64³ → 128³ → 256³
        self.decoder = nn.ModuleList([
            # 32³ → 64³
            nn.Sequential(
                nn.Conv3d(1, 64, 3, padding=1),
                nn.InstanceNorm3d(64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 64, 3, padding=1),
                nn.InstanceNorm3d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ),
            # 64³ → 128³
            nn.Sequential(
                nn.Conv3d(64, 32, 3, padding=1),
                nn.InstanceNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 32, 3, padding=1),
                nn.InstanceNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ),
            # 128³ → 256³
            nn.Sequential(
                nn.Conv3d(32, 16, 3, padding=1),
                nn.InstanceNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 16, 3, padding=1),
                nn.InstanceNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ),
        ])
        
        # Final refinement
        self.final_conv = nn.Sequential(
            nn.Conv3d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, 3, padding=1)
        )
        
    def forward(self, x_rays):
        """
        Args:
            x_rays: (B, 2, 1, 512, 512) - dual view X-rays
        Returns:
            volume: (B, 1, 256, 256, 256) - predicted CT volume
        """
        B = x_rays.shape[0]
        
        # Process each view
        view1 = x_rays[:, 0]  # (B, 1, 512, 512)
        view2 = x_rays[:, 1]  # (B, 1, 512, 512)
        
        # Patch embedding
        patches1 = self.patch_embed(view1)  # (B, vit_dim, 32, 32)
        patches2 = self.patch_embed(view2)
        
        # Flatten patches
        patches1 = patches1.flatten(2).transpose(1, 2)  # (B, 1024, vit_dim)
        patches2 = patches2.flatten(2).transpose(1, 2)
        
        # Concatenate views
        x = torch.cat([patches1, patches2], dim=1)  # (B, 2048, vit_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # ViT encoder with optional checkpointing
        for blk in self.blocks:
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        
        x = self.norm(x)
        
        # Average pooling
        x = x.mean(dim=1)  # (B, vit_dim)
        
        # Project to 3D volume
        x = self.to_volume_features(x)  # (B, 32768)
        x = x.view(B, 1, 32, 32, 32)
        
        # Progressive upsampling with checkpointing
        for i, decoder_block in enumerate(self.decoder):
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(decoder_block, x, use_reentrant=False)
            else:
                x = decoder_block(x)
        
        # Final convolution
        volume = self.final_conv(x)
        
        return volume


if __name__ == "__main__":
    # Test model
    model = DirectCTRegression256(
        xray_size=512,
        volume_size=256,
        vit_patch_size=16,
        vit_dim=512,
        vit_depth=6,
        vit_heads=8
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Test forward pass
    x = torch.randn(1, 2, 1, 512, 512)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
