"""
Enhanced Attention Mechanisms for Spatial Clustering CT Generator
Implements: Cross-Modal, 3D Spatial, Channel, Hierarchical, and Cluster-Interaction Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between frontal and lateral X-rays
    Allows each view to attend to the other to resolve depth ambiguity
    """
    
    def __init__(self, feature_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Projections for frontal
        self.frontal_q = nn.Linear(feature_dim, feature_dim)
        self.frontal_k = nn.Linear(feature_dim, feature_dim)
        self.frontal_v = nn.Linear(feature_dim, feature_dim)
        
        # Projections for lateral
        self.lateral_q = nn.Linear(feature_dim, feature_dim)
        self.lateral_k = nn.Linear(feature_dim, feature_dim)
        self.lateral_v = nn.Linear(feature_dim, feature_dim)
        
        # Output projections
        self.frontal_proj = nn.Linear(feature_dim, feature_dim)
        self.lateral_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, frontal_features: torch.Tensor, lateral_features: torch.Tensor):
        """
        Args:
            frontal_features: (B, N_f, C) frontal X-ray features
            lateral_features: (B, N_l, C) lateral X-ray features
        Returns:
            attended_frontal, attended_lateral: Enhanced features
        """
        B = frontal_features.shape[0]
        
        # Frontal queries attend to lateral keys/values
        q_f = self.frontal_q(frontal_features).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_l = self.lateral_k(lateral_features).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_l = self.lateral_v(lateral_features).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_f2l = (q_f @ k_l.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_f2l = F.softmax(attn_f2l, dim=-1)
        out_f = (attn_f2l @ v_l).transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        
        # Lateral queries attend to frontal keys/values
        q_l = self.lateral_q(lateral_features).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_f = self.frontal_k(frontal_features).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_f = self.frontal_v(frontal_features).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_l2f = (q_l @ k_f.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_l2f = F.softmax(attn_l2f, dim=-1)
        out_l = (attn_l2f @ v_f).transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        
        # Residual connections
        attended_frontal = frontal_features + self.frontal_proj(out_f)
        attended_lateral = lateral_features + self.lateral_proj(out_l)
        
        return attended_frontal, attended_lateral


class Spatial3DAttention(nn.Module):
    """
    3D spatial attention to highlight important volume regions
    """
    
    def __init__(self, channels: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) volume features
        Returns:
            attended: (B, C, D, H, W) spatially weighted features
        """
        spatial_weights = self.attention(x)  # (B, 1, D, H, W)
        return x * spatial_weights


class ChannelAttention(nn.Module):
    """
    Channel attention to weight feature channels adaptively
    """
    
    def __init__(self, channels: int = 256, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) or (B, C, N) features
        Returns:
            attended: Channel-weighted features
        """
        # Handle both (B, N, C) and (B, C, N) formats
        if x.dim() == 3:
            B, N, C = x.shape
            x_permuted = x.transpose(1, 2)  # (B, C, N)
            
            avg_out = self.avg_pool(x_permuted).squeeze(-1)  # (B, C)
            max_out = self.max_pool(x_permuted).squeeze(-1)  # (B, C)
            
            channel_weights = self.mlp(avg_out + max_out)  # (B, C)
            return x * channel_weights.unsqueeze(1)  # (B, N, C)
        else:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D")


class HierarchicalAttention(nn.Module):
    """
    Multi-scale hierarchical attention for coarse-to-fine processing
    """
    
    def __init__(self, channels: int = 256, scales: list = [32, 64]):
        super().__init__()
        self.scales = scales
        
        # Attention at each scale
        self.attentions = nn.ModuleDict({
            f'scale_{s}': nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, kernel_size=1)
            ) for s in scales
        })
        
        # Fusion layer
        self.fusion = nn.Conv3d(channels * len(scales), channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor, target_size: Tuple[int, int, int] = (64, 64, 64)) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) volume features
            target_size: Target spatial dimensions
        Returns:
            attended: Multi-scale attended features
        """
        B, C, D, H, W = x.shape
        
        multi_scale_features = []
        
        for scale in self.scales:
            # Resize to scale
            if (D, H, W) != (scale, scale, scale):
                x_scaled = F.interpolate(x, size=(scale, scale, scale), mode='trilinear', align_corners=False)
            else:
                x_scaled = x
            
            # Apply attention at this scale
            attended = self.attentions[f'scale_{scale}'](x_scaled)
            
            # Resize back to target
            if (scale, scale, scale) != target_size:
                attended = F.interpolate(attended, size=target_size, mode='trilinear', align_corners=False)
            
            multi_scale_features.append(attended)
        
        # Concatenate and fuse
        fused = torch.cat(multi_scale_features, dim=1)
        output = self.fusion(fused)
        
        return x + output  # Residual connection


class ClusterInteractionAttention(nn.Module):
    """
    Novel cluster-to-cluster attention mechanism
    Allows anatomical clusters to interact and share information
    """
    
    def __init__(self, cluster_dim: int = 256, num_clusters: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = cluster_dim // num_heads
        
        self.q_proj = nn.Linear(cluster_dim, cluster_dim)
        self.k_proj = nn.Linear(cluster_dim, cluster_dim)
        self.v_proj = nn.Linear(cluster_dim, cluster_dim)
        self.out_proj = nn.Linear(cluster_dim, cluster_dim)
        
        # Learnable cluster relationships (e.g., heart-lung, bone-tissue)
        self.cluster_bias = nn.Parameter(torch.zeros(num_clusters, num_clusters))
        
    def forward(self, cluster_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cluster_features: (B, K, C) cluster-aggregated features
        Returns:
            interacted_features: (B, K, C) clusters after interaction
        """
        B, K, C = cluster_features.shape
        
        # Multi-head projections
        q = self.q_proj(cluster_features).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(cluster_features).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(cluster_features).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Cluster-to-cluster attention with learned bias
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn + self.cluster_bias.unsqueeze(0).unsqueeze(0)  # Add anatomical bias
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(B, K, C)
        out = self.out_proj(out)
        
        return cluster_features + out  # Residual


class EnhancedAttentionBlock(nn.Module):
    """
    Complete attention block combining all mechanisms
    """
    
    def __init__(self, 
                 voxel_dim: int = 256,
                 num_clusters: int = 64,
                 num_heads: int = 8):
        super().__init__()
        
        self.spatial_attn = Spatial3DAttention(voxel_dim)
        self.channel_attn = ChannelAttention(voxel_dim)
        self.cluster_interaction = ClusterInteractionAttention(voxel_dim, num_clusters, num_heads=4)
        
    def forward(self, 
                voxel_features: torch.Tensor,
                cluster_assignments: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel_features: (B, N, C)
            cluster_assignments: (B, N, K)
        Returns:
            enhanced_features: (B, N, C)
        """
        B, N, C = voxel_features.shape
        K = cluster_assignments.shape[-1]
        
        # Reshape to 3D for spatial attention
        D = int(N ** (1/3))
        voxel_3d = voxel_features.transpose(1, 2).view(B, C, D, D, D)
        
        # Apply 3D spatial attention
        voxel_3d = self.spatial_attn(voxel_3d)
        voxel_features = voxel_3d.view(B, C, N).transpose(1, 2)
        
        # Apply channel attention
        voxel_features = self.channel_attn(voxel_features)
        
        # Aggregate to clusters
        cluster_features = torch.einsum('bnc,bnk->bkc', voxel_features, cluster_assignments)
        cluster_sizes = cluster_assignments.sum(dim=1, keepdim=True).transpose(-2, -1)  # (B, K, 1)
        cluster_features = cluster_features / (cluster_sizes + 1e-8)
        
        # Cluster interaction
        cluster_features = self.cluster_interaction(cluster_features)
        
        # Distribute back to voxels
        enhanced_features = torch.einsum('bkc,bnk->bnc', cluster_features, cluster_assignments)
        
        return voxel_features + enhanced_features  # Residual
