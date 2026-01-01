"""
Enhanced Spatial Clustering CT Generator with Full Attention Mechanisms
Integrates: Cross-Modal, 3D Spatial, Channel, Hierarchical, and Cluster-Interaction Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

# Import base architecture
from spatial_cluster_architecture import (
    PositionEncodingModule,
    VoxelClusteringModule,
    ClusterAwareAttention,
    PositionIntensityTracker
)

# Import enhanced attention modules
from enhanced_attention_modules import (
    CrossModalAttention,
    Spatial3DAttention,
    ChannelAttention,
    HierarchicalAttention,
    ClusterInteractionAttention,
    EnhancedAttentionBlock
)


class EnhancedSpatialClusteringCTGenerator(nn.Module):
    """
    Enhanced version with all attention mechanisms for maximum performance
    
    Enhancements:
    1. Cross-modal attention between frontal/lateral X-rays
    2. 3D spatial attention on volume features
    3. Channel attention in transformer blocks
    4. Hierarchical multi-scale processing
    5. Cluster-to-cluster interaction attention
    """
    
    def __init__(self,
                 volume_size: Tuple[int, int, int] = (64, 64, 64),
                 voxel_dim: int = 256,
                 num_clusters: int = 64,
                 num_heads: int = 8,
                 num_blocks: int = 6,
                 xray_channels: int = 1):
        super().__init__()
        
        self.volume_size = volume_size
        self.voxel_dim = voxel_dim
        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.num_voxels = volume_size[0] * volume_size[1] * volume_size[2]
        
        # X-ray encoder (shared for both views)
        self.xray_encoder = nn.Sequential(
            nn.Conv2d(xray_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # NEW: Cross-modal attention between X-ray views
        self.cross_modal_attention = CrossModalAttention(feature_dim=256*8*8, num_heads=8)
        
        # Efficient fusion with smaller intermediate projection
        fusion_dim = 1024  # Bottleneck dimension
        self.fusion = nn.Sequential(
            nn.Linear(256*8*8*2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        # Project to 3D volume via reshape + conv
        self.to_volume = nn.Sequential(
            nn.Linear(fusion_dim, voxel_dim * 16 * 16 * 16),  # Smaller initial volume
            nn.Unflatten(1, (voxel_dim, 16, 16, 16)),
            nn.ConvTranspose3d(voxel_dim, voxel_dim, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(voxel_dim, voxel_dim, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(inplace=True)
        )
        
        # Position encoding
        self.position_encoder = PositionEncodingModule(num_freq_bands=10, d_model=128)
        
        # Initial projection to 3D
        self.to_3d = nn.Sequential(
            nn.Linear(voxel_dim + 128, voxel_dim),
            nn.LayerNorm(voxel_dim),
            nn.ReLU(inplace=True)
        )
        
        # NEW: 3D Spatial attention before clustering
        self.spatial_attention_pre = Spatial3DAttention(voxel_dim)
        
        # Clustering module
        self.clustering = VoxelClusteringModule(
            voxel_dim=voxel_dim,
            num_clusters=num_clusters,
            use_position=True,
            use_intensity=True
        )
        
        # NEW: Cluster interaction attention
        self.cluster_interaction = ClusterInteractionAttention(
            cluster_dim=voxel_dim,
            num_clusters=num_clusters,
            num_heads=4
        )
        
        # Transformer blocks with enhanced attention
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(voxel_dim),
                'cluster_attn': ClusterAwareAttention(voxel_dim, num_clusters, num_heads),
                'channel_attn': ChannelAttention(voxel_dim),  # NEW
                'norm2': nn.LayerNorm(voxel_dim),
                'ffn': nn.Sequential(
                    nn.Linear(voxel_dim, voxel_dim * 4),
                    nn.GELU(),
                    nn.Linear(voxel_dim * 4, voxel_dim)
                ),
                'enhanced_attn': EnhancedAttentionBlock(voxel_dim, num_clusters, num_heads)  # NEW
            })
            for _ in range(num_blocks)
        ])
        
        # NEW: Hierarchical multi-scale processing
        self.hierarchical_attention = HierarchicalAttention(
            channels=voxel_dim,
            scales=[32, 64]
        )
        
        # Volume decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(voxel_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Position-intensity tracker
        self.tracker = PositionIntensityTracker()
    
    def forward(self, 
                frontal_xray: torch.Tensor,
                lateral_xray: torch.Tensor,
                gt_volume: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with all attention mechanisms
        
        Args:
            frontal_xray: (B, 1, H, W)
            lateral_xray: (B, 1, H, W)
            gt_volume: (B, 1, D, H, W) optional ground truth
        Returns:
            output: Dictionary with predictions and tracking metrics
        """
        B = frontal_xray.shape[0]
        D, H, W = self.volume_size
        
        # Extract features from X-rays
        frontal_features = self.xray_encoder(frontal_xray)  # (B, 256, 8, 8)
        lateral_features = self.xray_encoder(lateral_xray)  # (B, 256, 8, 8)
        
        # Flatten for cross-modal attention
        frontal_flat = frontal_features.view(B, -1).unsqueeze(1)  # (B, 1, 256*64)
        lateral_flat = lateral_features.view(B, -1).unsqueeze(1)  # (B, 1, 256*64)
        
        # NEW: Cross-modal attention (frontal ↔ lateral)
        frontal_attended, lateral_attended = self.cross_modal_attention(frontal_flat, lateral_flat)
        
        frontal_attended = frontal_attended.squeeze(1)  # (B, 256*64)
        lateral_attended = lateral_attended.squeeze(1)  # (B, 256*64)
        
        # Fuse both views
        fused = torch.cat([frontal_attended, lateral_attended], dim=1)  # (B, 256*64*2)
        
        # Efficient fusion through bottleneck
        fused_features = self.fusion(fused)  # (B, fusion_dim)
        
        # Project to 3D volume
        x_3d = self.to_volume(fused_features)  # (B, voxel_dim, D, H, W)
        
        # Flatten to voxel sequence
        x = x_3d.view(B, self.voxel_dim, self.num_voxels).transpose(1, 2)  # (B, N, voxel_dim)
        
        # Position encoding
        position_features = self.position_encoder(self.volume_size, x.device)  # (N, 128)
        position_features = position_features.unsqueeze(0).expand(B, -1, -1)  # (B, N, 128)
        
        # Combine with position
        x = torch.cat([x, position_features], dim=-1)  # (B, N, voxel_dim + 128)
        x = self.to_3d(x)  # (B, N, voxel_dim)
        
        # NEW: 3D spatial attention before clustering
        x_3d = x.transpose(1, 2).view(B, self.voxel_dim, D, H, W)
        x_3d = self.spatial_attention_pre(x_3d)
        x = x_3d.view(B, self.voxel_dim, self.num_voxels).transpose(1, 2)
        
        # Compute intensity features for clustering
        intensity_features = x.mean(dim=-1, keepdim=True)  # (B, N, 1)
        
        # Cluster voxels
        cluster_assignments, x = self.clustering(x, position_features, intensity_features)
        
        # NEW: Cluster interaction (let clusters talk to each other)
        cluster_features = torch.einsum('bnc,bnk->bkc', x, cluster_assignments)
        cluster_sizes = cluster_assignments.sum(dim=1, keepdim=True).transpose(-2, -1)
        cluster_features = cluster_features / (cluster_sizes + 1e-8)
        
        cluster_features = self.cluster_interaction(cluster_features)
        
        # Distribute enhanced cluster features back to voxels
        x_enhanced = torch.einsum('bkc,bnk->bnc', cluster_features, cluster_assignments)
        x = x + x_enhanced
        
        # Transformer blocks with ALL attention mechanisms
        for block in self.blocks:
            # Cluster-aware attention
            x_norm = block['norm1'](x)
            x = x + block['cluster_attn'](x_norm, cluster_assignments)
            
            # NEW: Channel attention
            x = x + block['channel_attn'](x)
            
            # NEW: Enhanced attention block (spatial + cluster)
            x = x + block['enhanced_attn'](x, cluster_assignments)
            
            # FFN
            x = x + block['ffn'](block['norm2'](x))
        
        # Reshape to 3D
        x = x.transpose(1, 2).view(B, self.voxel_dim, D, H, W)
        
        # NEW: Hierarchical multi-scale refinement
        x = self.hierarchical_attention(x, target_size=(D, H, W))
        
        # Decode to volume
        pred_volume = self.decoder(x)  # (B, 1, D, H, W)
        
        # Tracking metrics
        if gt_volume is not None:
            position_accuracy = self.tracker.compute_position_accuracy(pred_volume, gt_volume)
            intensity_metrics = self.tracker.compute_intensity_accuracy(pred_volume, gt_volume)
        else:
            position_accuracy = torch.zeros(B, D, H, W, device=pred_volume.device)
            intensity_metrics = {
                'intensity_mae': torch.tensor(0.0, device=pred_volume.device),
                'contrast_error': torch.tensor(0.0, device=pred_volume.device),
                'voxel_wise_mae': torch.zeros_like(pred_volume)
            }
        
        return {
            'pred_volume': pred_volume,
            'cluster_assignments': cluster_assignments,
            'voxel_features': x.view(B, self.voxel_dim, self.num_voxels).transpose(1, 2),
            'position_accuracy': position_accuracy,
            'intensity_metrics': intensity_metrics
        }


if __name__ == "__main__":
    # Test enhanced model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = EnhancedSpatialClusteringCTGenerator(
        volume_size=(64, 64, 64),
        voxel_dim=256,
        num_clusters=64,
        num_heads=8,
        num_blocks=6
    ).to(device)
    
    print(f"Enhanced Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test forward pass
    frontal = torch.randn(1, 1, 512, 512).to(device)
    lateral = torch.randn(1, 1, 512, 512).to(device)
    gt = torch.randn(1, 1, 64, 64, 64).to(device)
    
    output = model(frontal, lateral, gt)
    print(f"Output shape: {output['pred_volume'].shape}")
    print("✓ Enhanced model test passed!")
