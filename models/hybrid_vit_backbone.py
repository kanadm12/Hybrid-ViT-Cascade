"""
Hybrid-ViT Backbone
Combines Vision Transformer with Voxel Processing and Physics Constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Import components
from .vit_components import (
    AdaLNModulation, 
    MultiHeadSelfAttention, 
    MultiHeadCrossAttention,
    SinusoidalTimeEmbedding
)


class HybridViTBlock3D(nn.Module):
    """
    Hybrid ViT block operating on voxel features (not patches)
    
    Combines:
    - Self-attention within 3D volume (from ViT)
    - Cross-attention to X-ray features (from Hybrid)
    - AdaLN modulation with time + X-ray + prev-stage (from ViT)
    - Voxel processing (from Hybrid)
    """
    
    def __init__(self, 
                 voxel_dim: int,
                 num_heads: int = 8,
                 context_dim: int = 512,
                 cond_dim: int = 1024,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 use_prev_stage: bool = False,
                 return_attention: bool = False):
        super().__init__()
        
        self.voxel_dim = voxel_dim
        self.use_prev_stage = use_prev_stage
        self.return_attention = return_attention
        
        # AdaLN modulation (time + xray + optional prev_stage)
        adaln_input_dim = cond_dim
        if use_prev_stage:
            adaln_input_dim += 256  # Previous stage embedding dim
        
        self.adaln = AdaLNModulation(
            embed_dim=voxel_dim,
            cond_dim=adaln_input_dim
        )
        
        # Self-attention on voxel features
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=voxel_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-attention to X-ray features
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim=voxel_dim,
            num_heads=num_heads,
            context_dim=context_dim,
            dropout=dropout,
            store_attention=return_attention
        )
        
        # MLP
        mlp_hidden_dim = int(voxel_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(voxel_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, voxel_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(voxel_dim)
        self.norm2 = nn.LayerNorm(voxel_dim)
        self.norm3 = nn.LayerNorm(voxel_dim)
    
    def forward(self, 
                voxel_features: torch.Tensor,
                xray_context: torch.Tensor,
                cond: torch.Tensor,
                prev_stage_embed: Optional[torch.Tensor] = None):
        """
        Args:
            voxel_features: (B, D*H*W, voxel_dim) - 3D voxel features
            xray_context: (B, N_xray, context_dim) - X-ray features for cross-attention
            cond: (B, cond_dim) - time + xray embedding
            prev_stage_embed: (B, 256) - optional previous stage embedding
        Returns:
            output: (B, D*H*W, voxel_dim)
            attention_map: (B, num_heads, D*H*W, N_xray) if return_attention=True, else None
        """
        batch_size = voxel_features.shape[0]
        
        # Prepare conditioning
        if self.use_prev_stage:
            if prev_stage_embed is None:
                # Create zero embedding if not provided
                prev_stage_embed = torch.zeros(batch_size, 256, 
                                              device=voxel_features.device, 
                                              dtype=voxel_features.dtype)
            combined_cond = torch.cat([cond, prev_stage_embed], dim=-1)
        else:
            combined_cond = cond
        
        # Get AdaLN modulation parameters (6 params: shift/scale/gate for self-attn and MLP)
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.adaln(voxel_features, combined_cond)
        
        # Self-attention with AdaLN
        x_norm = self.norm1(voxel_features)
        x_mod = (1 + scale_sa) * x_norm + shift_sa
        x_attn = self.self_attn(x_mod)
        voxel_features = voxel_features + gate_sa * x_attn
        
        # Cross-attention to X-ray (no AdaLN here)
        x_norm2 = self.norm2(voxel_features)
        x_cross = self.cross_attn(x_norm2, xray_context)
        voxel_features = voxel_features + x_cross
        
        # Store attention map if requested (accessed from cross_attn module)
        attention_map = None
        if self.return_attention and hasattr(self.cross_attn, 'attention_weights'):
            attention_map = self.cross_attn.attention_weights
        
        # MLP with AdaLN
        x_norm3 = self.norm3(voxel_features)
        x_mod2 = (1 + scale_mlp) * x_norm3 + shift_mlp
        x_mlp = self.mlp(x_mod2)
        voxel_features = voxel_features + gate_mlp * x_mlp
        
        if self.return_attention:
            return voxel_features, attention_map
        return voxel_features


class HybridViT3D(nn.Module):
    """
    Complete Hybrid-ViT model for one cascade stage
    
    Architecture:
    1. Voxel embedding (instead of patch embedding)
    2. Positional encoding (3D)
    3. Stack of HybridViTBlock3D
    4. Output projection
    """
    
    def __init__(self,
                 volume_size: Tuple[int, int, int] = (64, 64, 64),
                 in_channels: int = 1,
                 voxel_dim: int = 384,
                 depth: int = 6,
                 num_heads: int = 6,
                 context_dim: int = 512,
                 cond_dim: int = 1024,
                 use_prev_stage: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        
        self.volume_size = volume_size
        self.in_channels = in_channels
        self.voxel_dim = voxel_dim
        self.use_prev_stage = use_prev_stage
        
        # Adaptive downsampling based on volume size to keep token count manageable
        # FIXED: Less aggressive downsampling to preserve details
        # Stage 1 (64³): 16³ tokens (4096), Stage 2 (128³): ~24³ tokens (13824), Stage 3 (256³): ~32³ tokens (32768)
        D, H, W = volume_size
        if D <= 64:
            target_size = 16  # Stage 1: 64³ → 16³
        elif D <= 128:
            target_size = 16  # Stage 2: 128³ → 24³ (5.3x downsample)
        else:
            target_size = 32  # Stage 3: 256³ → 32³ (8x downsample)
        downsample_factor = max(D // target_size, H // target_size, W // target_size)
        downsample_factor = max(downsample_factor, 1)  # At least 1
        
        self.downsampled_size = tuple(d // downsample_factor for d in volume_size)
        D_down, H_down, W_down = self.downsampled_size
        
        # Voxel embedding with adaptive downsampling
        layers = []
        current_dim = in_channels
        remaining_downsample = downsample_factor
        
        while remaining_downsample > 1:
            stride = min(remaining_downsample, 2)
            out_dim = voxel_dim // 4 if current_dim == in_channels else voxel_dim // 2 if len(layers) < 4 else voxel_dim
            layers.extend([
                nn.Conv3d(current_dim, out_dim, kernel_size=3, stride=stride, padding=1),
                nn.GroupNorm(min(8, out_dim), out_dim),
                nn.SiLU()
            ])
            current_dim = out_dim
            remaining_downsample = remaining_downsample // stride
        
        # Final conv to reach voxel_dim
        if current_dim != voxel_dim:
            layers.append(nn.Conv3d(current_dim, voxel_dim, kernel_size=3, padding=1))
        
        self.voxel_embed = nn.Sequential(*layers)
        
        # Positional encoding (learnable) for downsampled size
        self.pos_embed = nn.Parameter(torch.randn(1, D_down * H_down * W_down, voxel_dim) * 0.02)
        
        # ViT blocks
        self.blocks = nn.ModuleList([
            HybridViTBlock3D(
                voxel_dim=voxel_dim,
                num_heads=num_heads,
                context_dim=context_dim,
                cond_dim=cond_dim,
                use_prev_stage=use_prev_stage,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(voxel_dim)
        # Always output 1 channel (denoised volume), regardless of input channels
        self.output_proj = nn.Linear(voxel_dim, 1)
    
    def forward(self,
                x: torch.Tensor,
                context: torch.Tensor,
                cond: torch.Tensor,
                prev_stage_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) - input volume
            context: (B, N, context_dim) - X-ray features
            cond: (B, cond_dim) - time + xray embedding
            prev_stage_embed: (B, 256) - optional previous stage
        Returns:
            output: (B, C, D, H, W) - predicted output
        """
        batch_size = x.shape[0]
        D, H, W = self.volume_size
        D_down, H_down, W_down = self.downsampled_size
        
        # Voxel embedding with downsampling
        x = self.voxel_embed(x)  # (B, voxel_dim, D_down, H_down, W_down)
        
        # Reshape to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, D_down*H_down*W_down, voxel_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Apply ViT blocks
        for block in self.blocks:
            x = block(x, context, cond, prev_stage_embed)
        
        # Output
        x = self.norm(x)
        x = self.output_proj(x)  # (B, D_down*H_down*W_down, 1)
        
        # Reshape back to downsampled volume - always 1 channel output
        x = x.transpose(1, 2).reshape(batch_size, 1, D_down, H_down, W_down)
        
        # Upsample back to original resolution
        x = F.interpolate(x, size=(D, H, W), mode='trilinear', align_corners=True)
        
        return x


# Test code
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Hybrid-ViT 3D Model...")
    
    # Test single block
    print("\n=== Testing HybridViTBlock3D ===")
    block = HybridViTBlock3D(
        voxel_dim=384,
        num_heads=6,
        context_dim=512,
        cond_dim=1024,
        use_prev_stage=True
    ).to(device)
    
    batch_size = 2
    voxel_features = torch.randn(batch_size, 64*64*64, 384).to(device)
    xray_context = torch.randn(batch_size, 256, 512).to(device)
    cond = torch.randn(batch_size, 1024).to(device)
    prev_stage_embed = torch.randn(batch_size, 256).to(device)
    
    output = block(voxel_features, xray_context, cond, prev_stage_embed)
    print(f"Input shape: {voxel_features.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test full model
    print("\n=== Testing Full HybridViT3D ===")
    model = HybridViT3D(
        volume_size=(64, 64, 64),
        in_channels=1,
        voxel_dim=384,
        depth=6,
        num_heads=6,
        context_dim=512,
        cond_dim=1024,
        use_prev_stage=True
    ).to(device)
    
    x_input = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    output = model(x_input, xray_context, cond, prev_stage_embed)
    
    print(f"Input volume shape: {x_input.shape}")
    print(f"Output volume shape: {output.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Model Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    
    # Memory estimate
    print(f"\n=== Memory Estimates ===")
    input_mem = x_input.element_size() * x_input.nelement() / 1024**2
    output_mem = output.element_size() * output.nelement() / 1024**2
    print(f"Input memory: {input_mem:.2f} MB")
    print(f"Output memory: {output_mem:.2f} MB")
    print(f"Approximate activation memory (training): ~{input_mem * 10:.2f} MB")
    
    print("\nHybrid-ViT 3D test completed!")
