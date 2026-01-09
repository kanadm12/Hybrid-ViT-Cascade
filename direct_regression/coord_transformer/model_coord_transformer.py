"""
Coordinate-Based Transformer for CT Reconstruction
Revolutionary architecture that processes volume as a SET of 3D coordinates

Key innovations:
1. Treats 3D volume as point cloud / coordinate set
2. Cross-attention between coordinates and X-ray features
3. Self-attention among coordinates for spatial relationships
4. Efficient sparse processing (query only needed points)
5. Set-based operations (permutation invariant)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Dict, Optional


class FourierFeatureEmbedding(nn.Module):
    """
    Fourier feature mapping for coordinates
    Maps R^3 → R^D with learnable frequencies
    """
    
    def __init__(self, input_dim: int = 3, mapping_size: int = 256, scale: float = 10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        
        # Learnable frequency matrix
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size // 2) * scale)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, 3) coordinates in [-1, 1]
        Returns:
            features: (B, N, mapping_size)
        """
        # x @ B → (B, N, mapping_size//2)
        x_proj = 2 * np.pi * x @ self.B
        
        # Apply sin and cos
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention: Query from coordinates, Key/Value from X-ray features
    """
    
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(context_dim, query_dim)
        self.v_proj = nn.Linear(context_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, context, mask=None):
        """
        Args:
            query: (B, N_q, D_q) coordinate features
            context: (B, N_c, D_c) X-ray features
            mask: optional attention mask
        Returns:
            output: (B, N_q, D_q)
        """
        B, N_q, D = query.shape
        N_c = context.shape[1]
        
        # Project to Q, K, V
        Q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).reshape(B, N_c, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).reshape(B, N_c, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, N_q, N_c)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ V  # (B, H, N_q, head_dim)
        out = out.transpose(1, 2).reshape(B, N_q, D)
        
        return self.out_proj(out)


class MultiHeadSelfAttention(nn.Module):
    """Self-attention among coordinates"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, D)
        Returns:
            output: (B, N, D)
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, head_dim)
        
        # Attention scores
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = attn @ V  # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """
    Transformer block with:
    1. Self-attention among coordinates
    2. Cross-attention with X-ray features
    3. Feed-forward network
    """
    
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        
        # Cross-attention
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadCrossAttention(dim, context_dim, num_heads, dropout)
        
        # Feed-forward
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, context):
        """
        Args:
            x: (B, N, D) coordinate features
            context: (B, M, D_c) X-ray features
        Returns:
            output: (B, N, D)
        """
        # Self-attention
        x = x + self.self_attn(self.norm1(x))
        
        # Cross-attention
        x = x + self.cross_attn(self.norm2(x), context)
        
        # Feed-forward
        x = x + self.mlp(self.norm3(x))
        
        return x


class XrayTransformerEncoder(nn.Module):
    """Encode X-ray images with Vision Transformer"""
    
    def __init__(self, img_size: int = 512, patch_size: int = 32, 
                 in_channels: int = 1, embed_dim: int = 512, depth: int = 6, num_heads: int = 8):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) X-ray image
        Returns:
            features: (B, num_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        return self.norm(x)


class CoordinateTransformer(nn.Module):
    """
    Complete Coordinate-Based Transformer for CT Reconstruction
    
    Pipeline:
    1. Encode X-rays with Vision Transformer → patch features
    2. Embed 3D coordinates with Fourier features
    3. Cross-attend coordinates to X-ray patches
    4. Self-attend among coordinates for spatial coherence
    5. Predict density at each coordinate
    """
    
    def __init__(self,
                 volume_size: Tuple[int, int, int] = (64, 64, 64),
                 xray_img_size: int = 512,
                 xray_patch_size: int = 32,
                 xray_embed_dim: int = 512,
                 xray_depth: int = 6,
                 coord_embed_dim: int = 512,
                 num_transformer_blocks: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.volume_size = volume_size
        self.coord_embed_dim = coord_embed_dim
        
        # X-ray encoder (dual-view)
        self.xray_encoder = XrayTransformerEncoder(
            img_size=xray_img_size,
            patch_size=xray_patch_size,
            in_channels=1,
            embed_dim=xray_embed_dim,
            depth=xray_depth,
            num_heads=num_heads
        )
        
        # Coordinate embedding
        self.coord_embedding = FourierFeatureEmbedding(
            input_dim=3,
            mapping_size=coord_embed_dim,
            scale=10.0
        )
        
        # Project to transformer dimension
        self.coord_proj = nn.Linear(coord_embed_dim, coord_embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=coord_embed_dim,
                context_dim=xray_embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout
            ) for _ in range(num_transformer_blocks)
        ])
        
        # Density prediction head
        self.density_head = nn.Sequential(
            nn.LayerNorm(coord_embed_dim),
            nn.Linear(coord_embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Precompute coordinate grid
        self.register_buffer('coord_grid', self._create_coordinate_grid(volume_size))
        
        # View fusion
        self.view_fusion = nn.Linear(xray_embed_dim * 2, xray_embed_dim)
    
    def _create_coordinate_grid(self, volume_size: Tuple[int, int, int]) -> torch.Tensor:
        """Create normalized 3D coordinate grid"""
        D, H, W = volume_size
        
        z = torch.linspace(-1, 1, D)
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1)  # (D, H, W, 3)
        
        return coords
    
    def encode_xrays(self, xrays):
        """
        Encode dual-view X-rays
        Args:
            xrays: (B, num_views, 1, H, W)
        Returns:
            context: (B, num_patches, embed_dim) fused features
        """
        B, num_views = xrays.shape[0], xrays.shape[1]
        
        # Encode each view
        view_features = []
        for i in range(num_views):
            view = xrays[:, i, :, :, :]  # (B, 1, H, W)
            features = self.xray_encoder(view)  # (B, num_patches, embed_dim)
            view_features.append(features)
        
        # Concatenate and fuse views
        if len(view_features) == 2:
            combined = torch.cat(view_features, dim=-1)  # (B, num_patches, 2*embed_dim)
            fused = self.view_fusion(combined)  # (B, num_patches, embed_dim)
        else:
            fused = view_features[0]
        
        return fused
    
    def forward(self, xrays: torch.Tensor, 
                coords: Optional[torch.Tensor] = None,
                query_resolution: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """
        Args:
            xrays: (B, num_views, 1, H, W) X-ray images
            coords: Optional (B, N, 3) custom coordinates to query
            query_resolution: Optional resolution for regular grid
        Returns:
            volume: (B, 1, D, H, W) reconstructed CT volume
        """
        B = xrays.shape[0]
        
        # Encode X-rays
        xray_context = self.encode_xrays(xrays)  # (B, num_patches, embed_dim)
        
        # Get coordinates
        if coords is not None:
            # Custom coordinates provided
            query_coords = coords
            N = coords.shape[1]
            output_shape = None
        else:
            # Use regular grid
            if query_resolution is not None and query_resolution != self.volume_size:
                grid = self._create_coordinate_grid(query_resolution).to(xrays.device)
            else:
                grid = self.coord_grid
                query_resolution = self.volume_size
            
            D, H, W = query_resolution
            query_coords = grid.reshape(-1, 3).unsqueeze(0).expand(B, -1, -1)  # (B, D*H*W, 3)
            N = D * H * W
            output_shape = (D, H, W)
        
        # Process in chunks to handle large coordinate sets
        chunk_size = 4096
        all_densities = []
        
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            coord_chunk = query_coords[:, i:end_idx, :]  # (B, chunk_size, 3)
            
            # Embed coordinates
            coord_features = self.coord_embedding(coord_chunk)  # (B, chunk_size, coord_embed_dim)
            coord_features = self.coord_proj(coord_features)
            
            # Apply transformer blocks
            for block in self.transformer_blocks:
                coord_features = block(coord_features, xray_context)
            
            # Predict density
            density = self.density_head(coord_features)  # (B, chunk_size, 1)
            all_densities.append(density)
        
        # Concatenate all chunks
        densities = torch.cat(all_densities, dim=1)  # (B, N, 1)
        
        # Reshape to volume if regular grid
        if output_shape is not None:
            D, H, W = output_shape
            volume = densities.reshape(B, 1, D, H, W)
            return volume
        else:
            return densities
    
    def query_points(self, xrays: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Query density at arbitrary 3D coordinates
        
        Args:
            xrays: (B, num_views, 1, H, W)
            coords: (B, N, 3) coordinates in [-1, 1]
        Returns:
            density: (B, N, 1)
        """
        return self.forward(xrays, coords=coords)


class CoordinateTransformerLoss(nn.Module):
    """
    Loss function for coordinate-based transformer
    """
    
    def __init__(self,
                 l1_weight: float = 1.0,
                 gradient_weight: float = 0.3,
                 smoothness_weight: float = 0.05):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.smoothness_weight = smoothness_weight
    
    def compute_gradients_3d(self, x):
        """Compute 3D gradients"""
        grad_z = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        grad_y = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        grad_x = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        return grad_z, grad_y, grad_x
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: (B, 1, D, H, W)
            target: (B, 1, D, H, W)
        Returns:
            loss_dict
        """
        # L1 reconstruction loss
        l1_loss = F.l1_loss(pred, target)
        
        # Gradient matching
        pred_gz, pred_gy, pred_gx = self.compute_gradients_3d(pred)
        tgt_gz, tgt_gy, tgt_gx = self.compute_gradients_3d(target)
        
        gradient_loss = (
            F.l1_loss(pred_gz, tgt_gz) +
            F.l1_loss(pred_gy, tgt_gy) +
            F.l1_loss(pred_gx, tgt_gx)
        ) / 3.0
        
        # Smoothness (TV loss)
        smoothness_loss = (
            torch.mean(pred_gz) + 
            torch.mean(pred_gy) + 
            torch.mean(pred_gx)
        ) / 3.0
        
        # Total loss
        total_loss = (
            self.l1_weight * l1_loss +
            self.gradient_weight * gradient_loss +
            self.smoothness_weight * smoothness_loss
        )
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'gradient_loss': gradient_loss,
            'smoothness_loss': smoothness_loss
        }


if __name__ == "__main__":
    print("Testing Coordinate-Based Transformer...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = CoordinateTransformer(
        volume_size=(64, 64, 64),
        xray_img_size=512,
        xray_patch_size=32,
        xray_embed_dim=512,
        xray_depth=6,
        coord_embed_dim=512,
        num_transformer_blocks=6,
        num_heads=8
    ).to(device)
    
    # Test input
    xrays = torch.randn(2, 2, 1, 512, 512).to(device)
    
    print(f"Input X-rays: {xrays.shape}")
    
    # Test regular grid inference
    volume_64 = model(xrays, query_resolution=(64, 64, 64))
    print(f"Output volume (64³): {volume_64.shape}")
    
    # Test higher resolution
    volume_96 = model(xrays, query_resolution=(96, 96, 96))
    print(f"Output volume (96³): {volume_96.shape}")
    
    # Test arbitrary point queries
    random_coords = torch.randn(2, 100, 3).to(device)  # 100 random points
    densities = model.query_points(xrays, random_coords)
    print(f"Query 100 arbitrary points: {densities.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    
    # Test loss
    target = torch.randn(2, 1, 64, 64, 64).to(device)
    loss_fn = CoordinateTransformerLoss()
    losses = loss_fn(volume_64, target)
    print(f"\nLoss test:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    print("\n✓ Coordinate-based transformer test successful!")
