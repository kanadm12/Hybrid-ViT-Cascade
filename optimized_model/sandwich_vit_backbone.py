"""
Optimized Sandwich ViT Backbone
Implements "sandwich layout" with more FFN layers and fewer attention layers
Memory-efficient with hierarchical adaptive conditioning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math

from cascaded_group_attention import CascadedGroupAttention, GroupedMultiScaleAttention


class FeedForward(nn.Module):
    """Enhanced FFN with GEGLU activation"""
    
    def __init__(self,
                 dim: int,
                 mult: int = 4,
                 dropout: float = 0.):
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)  # Slightly smaller for efficiency
        
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SandwichTransformerBlock(nn.Module):
    """
    Sandwich layout block: FFN -> Attention -> FFN
    More parameter-efficient and faster than standard Attention -> FFN
    """
    
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 use_multi_scale: bool = False,
                 ffn_mult: int = 4,
                 dropout: float = 0.1,
                 use_cascaded_attn: bool = True):
        super().__init__()
        
        # Pre-attention FFN
        self.pre_ffn = FeedForward(dim, mult=ffn_mult, dropout=dropout)
        
        # Attention mechanism
        if use_multi_scale:
            self.attn = GroupedMultiScaleAttention(
                dim=dim,
                num_heads=num_heads,
                scales=[1, 2],
                attn_drop=dropout,
                proj_drop=dropout
            )
        elif use_cascaded_attn:
            self.attn = CascadedGroupAttention(
                dim=dim,
                num_heads=num_heads,
                num_groups=2,
                attn_drop=dropout,
                proj_drop=dropout
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        self.attn_norm = nn.LayerNorm(dim)
        
        # Post-attention FFN
        self.post_ffn = FeedForward(dim, mult=ffn_mult, dropout=dropout)
        
        # Skip connections with learnable gates
        self.pre_gate = nn.Parameter(torch.ones(1) * 0.5)
        self.attn_gate = nn.Parameter(torch.ones(1) * 0.5)
        self.post_gate = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self,
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) input tokens
            context: (B, M, C) optional cross-attention context
        Returns:
            output: (B, N, C) transformed tokens
        """
        # Pre-attention FFN with gated residual
        x = x + self.pre_gate * self.pre_ffn(x)
        
        # Attention with gated residual
        x_norm = self.attn_norm(x)
        if isinstance(self.attn, (CascadedGroupAttention, GroupedMultiScaleAttention)):
            attn_out = self.attn(x_norm, context)
        else:
            attn_out = self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.attn_gate * attn_out
        
        # Post-attention FFN with gated residual
        x = x + self.post_gate * self.post_ffn(x)
        
        return x


class HierarchicalAdaptiveConditioning(nn.Module):
    """
    Adaptive conditioning that incorporates previous stage predictions
    Uses attention to selectively integrate multi-resolution information
    """
    
    def __init__(self,
                 dim: int,
                 num_heads: int = 4):
        super().__init__()
        
        self.dim = dim
        
        # Cross-attention to integrate previous stage
        self.prev_stage_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Adaptive weighting
        self.weight_predictor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self,
                current: torch.Tensor,
                previous: torch.Tensor,
                xray_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            current: (B, N, C) current stage features
            previous: (B, N', C) previous stage features (different resolution)
            xray_context: (B, M, C) X-ray conditioning
        Returns:
            conditioned: (B, N, C) hierarchically conditioned features
        """
        B, N, C = current.shape
        
        # Upsample previous stage if needed
        if previous.shape[1] != N:
            N_prev = previous.shape[1]
            D_prev = int(round(N_prev ** (1/3)))
            D_curr = int(round(N ** (1/3)))
            
            # Reshape and interpolate
            prev_3d = previous.reshape(B, D_prev, D_prev, D_prev, C).permute(0, 4, 1, 2, 3)
            prev_upsampled = F.interpolate(prev_3d, size=(D_curr, D_curr, D_curr), 
                                          mode='trilinear', align_corners=True)
            previous = prev_upsampled.permute(0, 2, 3, 4, 1).reshape(B, N, C)
        
        # Cross-attention: query from current, key/value from previous
        attended_prev, _ = self.prev_stage_attn(current, previous, previous)
        
        # Adaptive weighting based on feature similarity
        combined = torch.cat([current, attended_prev], dim=-1)  # (B, N, 2C)
        weights = self.weight_predictor(combined)  # (B, N, 1)
        
        # Weighted combination
        weighted = current * (1 - weights) + attended_prev * weights
        
        # Final fusion
        fused_input = torch.cat([current, weighted], dim=-1)
        conditioned = self.fusion(fused_input)
        
        return conditioned


class SandwichViT3D(nn.Module):
    """
    Optimized 3D ViT with sandwich layout
    - 2 attention blocks with multi-scale
    - 6 FFN-only blocks for efficiency
    - Hierarchical adaptive conditioning
    """
    
    def __init__(self,
                 volume_size: tuple = (64, 64, 64),
                 in_channels: int = 1,
                 voxel_dim: int = 256,
                 num_attn_blocks: int = 2,
                 num_ffn_blocks: int = 4,
                 num_heads: int = 8,
                 context_dim: int = 512,
                 dropout: float = 0.1,
                 use_prev_stage: bool = False):
        super().__init__()
        
        self.volume_size = volume_size
        self.voxel_dim = voxel_dim
        self.use_prev_stage = use_prev_stage
        
        D, H, W = volume_size
        num_voxels = D * H * W
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv3d(in_channels, voxel_dim // 2, 3, padding=1),
            nn.GroupNorm(8, voxel_dim // 2),
            nn.SiLU(),
            nn.Conv3d(voxel_dim // 2, voxel_dim, 3, padding=1)
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_voxels, voxel_dim) * 0.02)
        
        # Context projection
        self.context_proj = nn.Linear(context_dim, voxel_dim)
        
        # Hierarchical conditioning (if using previous stage)
        if use_prev_stage:
            self.hierarchical_cond = HierarchicalAdaptiveConditioning(
                dim=voxel_dim,
                num_heads=4
            )
        
        # Sandwich layout: FFN -> Attn -> FFN -> FFN -> Attn -> FFN -> FFN
        blocks = []
        
        # Start with FFN
        blocks.append(FeedForward(voxel_dim, mult=4, dropout=dropout))
        
        # Interleave attention and FFN blocks
        for i in range(num_attn_blocks):
            # Attention block (use multi-scale for first block)
            use_ms = (i == 0)
            blocks.append(SandwichTransformerBlock(
                dim=voxel_dim,
                num_heads=num_heads,
                use_multi_scale=use_ms,
                ffn_mult=4,
                dropout=dropout,
                use_cascaded_attn=True
            ))
            
            # Follow with 2 FFN blocks
            for _ in range(num_ffn_blocks // num_attn_blocks):
                blocks.append(FeedForward(voxel_dim, mult=4, dropout=dropout))
        
        self.blocks = nn.ModuleList(blocks)
        
        # Output normalization
        self.norm_out = nn.LayerNorm(voxel_dim)
        
        # Output projection to single channel
        self.output_proj = nn.Sequential(
            nn.Linear(voxel_dim, voxel_dim // 2),
            nn.LayerNorm(voxel_dim // 2),
            nn.GELU(),
            nn.Linear(voxel_dim // 2, 1)
        )
        
    def forward(self,
                x: torch.Tensor,
                context: torch.Tensor,
                prev_stage_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) input volume
            context: (B, M, context_dim) X-ray features
            prev_stage_embed: (B, N', voxel_dim) previous stage features
        Returns:
            output: (B, 1, D, H, W) predicted volume
        """
        B, C, D, H, W = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (B, voxel_dim, D, H, W)
        
        # Flatten to tokens
        x_tokens = x.flatten(2).transpose(1, 2)  # (B, D*H*W, voxel_dim)
        
        # Add positional embedding
        x_tokens = x_tokens + self.pos_embed
        
        # Project context
        context = self.context_proj(context)
        
        # Hierarchical conditioning if previous stage provided
        if prev_stage_embed is not None and self.use_prev_stage:
            x_tokens = self.hierarchical_cond(x_tokens, prev_stage_embed, context)
        
        # Apply sandwich transformer blocks
        for i, block in enumerate(self.blocks):
            if isinstance(block, SandwichTransformerBlock):
                x_tokens = block(x_tokens, context)
            else:
                # FFN block with residual
                x_tokens = x_tokens + block(x_tokens)
        
        # Output normalization
        x_tokens = self.norm_out(x_tokens)
        
        # Project to single channel
        output = self.output_proj(x_tokens)  # (B, D*H*W, 1)
        
        # Reshape to volume
        output = output.transpose(1, 2).reshape(B, 1, D, H, W)
        
        return output


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Testing Sandwich ViT 3D ===")
    B = 2
    volume_size = (64, 64, 64)
    
    model = SandwichViT3D(
        volume_size=volume_size,
        in_channels=1,
        voxel_dim=256,
        num_attn_blocks=2,
        num_ffn_blocks=4,
        num_heads=8,
        context_dim=512,
        use_prev_stage=False
    ).to(device)
    
    # Test forward pass
    x = torch.randn(B, 1, *volume_size).to(device)
    context = torch.randn(B, 4096, 512).to(device)
    
    output = model(x, context)
    
    print(f"Input shape: {x.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Output shape: {output.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Test with previous stage
    print("\n=== Testing with Previous Stage ===")
    model_cascade = SandwichViT3D(
        volume_size=volume_size,
        in_channels=1,
        voxel_dim=256,
        num_attn_blocks=2,
        num_ffn_blocks=4,
        num_heads=8,
        context_dim=512,
        use_prev_stage=True
    ).to(device)
    
    prev_stage = torch.randn(B, 32**3, 256).to(device)
    output = model_cascade(x, context, prev_stage)
    print(f"Output with cascading: {output.shape}")
    
    cascade_params = sum(p.numel() for p in model_cascade.parameters())
    print(f"Cascade parameters: {cascade_params:,} ({cascade_params/1e6:.2f}M)")
    
    print("\nSandwich ViT test completed!")
