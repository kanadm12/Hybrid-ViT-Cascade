"""
Spatial Clustering Architecture for CT Volume Generation from Bilateral X-rays

Key Features:
1. Position-aware voxel clustering
2. Pixel-wise generation tracking (position + intensity)
3. Cluster-based feature learning
4. Accuracy tracking per cluster and voxel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class PositionEncodingModule(nn.Module):
    """
    3D Positional Encoding for voxel spatial information
    Encodes (x, y, z) coordinates into feature space
    """
    
    def __init__(self, num_freq_bands: int = 10, d_model: int = 128):
        super().__init__()
        self.num_freq_bands = num_freq_bands
        self.d_model = d_model
        
        # Fourier features for position encoding
        freq_bands = 2.0 ** torch.linspace(0, num_freq_bands - 1, num_freq_bands)
        self.register_buffer('freq_bands', freq_bands)
        
        # Project encoded position to feature space
        # Each coordinate (x, y, z) gets: [sin(2^k * pi * x), cos(2^k * pi * x)] for k freq bands
        # Total: 3 coords * 2 (sin/cos) * num_freq_bands = 6 * num_freq_bands
        self.pos_proj = nn.Linear(6 * num_freq_bands, d_model)
    
    def forward(self, volume_size: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
        """
        Args:
            volume_size: (D, H, W) volume dimensions
            device: torch device
        Returns:
            pos_features: (D*H*W, d_model) position features for each voxel
        """
        D, H, W = volume_size
        
        # Create normalized 3D grid coordinates [0, 1]
        z = torch.linspace(0, 1, D, device=device)
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')  # (D, H, W) each
        
        # Flatten coordinates: (D*H*W, 3)
        coords = torch.stack([zz.flatten(), yy.flatten(), xx.flatten()], dim=-1)
        
        # Apply sinusoidal encoding
        # coords: (D*H*W, 3)
        # freq_bands: (num_freq_bands,)
        coords_freq = coords.unsqueeze(-1) * self.freq_bands.unsqueeze(0).unsqueeze(0)  # (D*H*W, 3, num_freq_bands)
        coords_freq = coords_freq.view(-1, 3 * self.num_freq_bands)  # (D*H*W, 3 * num_freq_bands)
        
        # Compute sin and cos
        sin_features = torch.sin(math.pi * coords_freq)
        cos_features = torch.cos(math.pi * coords_freq)
        
        # Concatenate
        pos_encoding = torch.cat([sin_features, cos_features], dim=-1)  # (D*H*W, 6 * num_freq_bands)
        
        # Project to d_model
        pos_features = self.pos_proj(pos_encoding)  # (D*H*W, d_model)
        
        return pos_features


class VoxelClusteringModule(nn.Module):
    """
    Learnable K-Means style clustering for voxels
    Groups similar voxels based on position + intensity + features
    """
    
    def __init__(self, 
                 voxel_dim: int,
                 num_clusters: int = 64,
                 use_position: bool = True,
                 use_intensity: bool = True,
                 temperature: float = 1.0):
        super().__init__()
        self.voxel_dim = voxel_dim
        self.num_clusters = num_clusters
        self.use_position = use_position
        self.use_intensity = use_intensity
        self.temperature = temperature
        
        # Learnable cluster centroids
        self.cluster_centroids = nn.Parameter(torch.randn(num_clusters, voxel_dim))
        
        # Feature fusion for clustering
        fusion_dim = voxel_dim
        if use_position:
            fusion_dim += 128  # Position encoding dim
        if use_intensity:
            fusion_dim += 1  # Intensity value
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, voxel_dim * 2),
            nn.LayerNorm(voxel_dim * 2),
            nn.GELU(),
            nn.Linear(voxel_dim * 2, voxel_dim)
        )
    
    def forward(self, 
                voxel_features: torch.Tensor,
                position_features: Optional[torch.Tensor] = None,
                intensities: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            voxel_features: (B, N, voxel_dim) voxel features
            position_features: (N, 128) position encodings
            intensities: (B, N, 1) voxel intensity values
        Returns:
            cluster_assignments: (B, N, num_clusters) soft assignment probabilities
            fused_features: (B, N, voxel_dim) enhanced features for clustering
        """
        B, N, _ = voxel_features.shape
        
        # Fuse features
        features_list = [voxel_features]
        
        if self.use_position and position_features is not None:
            # Expand position features to batch
            pos_expanded = position_features.unsqueeze(0).expand(B, -1, -1)  # (B, N, 128)
            features_list.append(pos_expanded)
        
        if self.use_intensity and intensities is not None:
            features_list.append(intensities)
        
        fused_features = torch.cat(features_list, dim=-1)  # (B, N, fusion_dim)
        fused_features = self.feature_fusion(fused_features)  # (B, N, voxel_dim)
        
        # Compute similarity to cluster centroids
        # fused_features: (B, N, voxel_dim)
        # cluster_centroids: (num_clusters, voxel_dim)
        
        # Normalize for cosine similarity
        fused_norm = F.normalize(fused_features, p=2, dim=-1)  # (B, N, voxel_dim)
        centroids_norm = F.normalize(self.cluster_centroids, p=2, dim=-1)  # (num_clusters, voxel_dim)
        
        # Compute similarity scores
        similarity = torch.matmul(fused_norm, centroids_norm.T)  # (B, N, num_clusters)
        
        # Soft assignment with temperature
        cluster_assignments = F.softmax(similarity / self.temperature, dim=-1)  # (B, N, num_clusters)
        
        return cluster_assignments, fused_features


class ClusterAwareAttention(nn.Module):
    """
    Attention mechanism that respects cluster structure
    Voxels within same cluster attend more to each other
    """
    
    def __init__(self, voxel_dim: int, num_heads: int = 8, num_clusters: int = 64):
        super().__init__()
        self.voxel_dim = voxel_dim
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.head_dim = voxel_dim // num_heads
        
        self.qkv = nn.Linear(voxel_dim, voxel_dim * 3, bias=False)
        self.proj = nn.Linear(voxel_dim, voxel_dim)
        
        # Cluster-based attention bias
        self.cluster_bias = nn.Parameter(torch.zeros(num_clusters, num_clusters))
    
    def forward(self, 
                voxel_features: torch.Tensor,
                cluster_assignments: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel_features: (B, N, voxel_dim)
            cluster_assignments: (B, N, num_clusters) soft cluster memberships
        Returns:
            attended_features: (B, N, voxel_dim)
        """
        B, N, C = voxel_features.shape
        K = cluster_assignments.shape[-1]  # num_clusters
        
        # Generate Q, K, V
        qkv = self.qkv(voxel_features).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Cluster-compressed attention: aggregate to cluster centroids to reduce N -> K
        # This avoids O(N^2) attention by using O(NK + K^2) instead
        
        # Step 1: Aggregate voxels to cluster centroids using soft assignments
        cluster_weights = cluster_assignments.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, K, 1)
        k_reshaped = k.transpose(1, 2).unsqueeze(3)  # (B, N, num_heads, 1, head_dim)
        v_reshaped = v.transpose(1, 2).unsqueeze(3)  # (B, N, num_heads, 1, head_dim)
        
        # Weighted sum to get cluster centroids
        k_cluster = (k_reshaped * cluster_weights).sum(dim=1) / (cluster_weights.sum(dim=1) + 1e-8)  # (B, K, num_heads, head_dim)
        v_cluster = (v_reshaped * cluster_weights).sum(dim=1) / (cluster_weights.sum(dim=1) + 1e-8)  # (B, K, num_heads, head_dim)
        
        k_cluster = k_cluster.transpose(1, 2)  # (B, num_heads, K, head_dim)
        v_cluster = v_cluster.transpose(1, 2)  # (B, num_heads, K, head_dim)
        
        # Step 2: Voxel-to-cluster attention (N x K instead of N x N)
        attn = (q @ k_cluster.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B, num_heads, N, K)
        
        # Add cluster bias (K x K) - broadcast through voxels
        cluster_bias_expanded = self.cluster_bias.unsqueeze(0).unsqueeze(1)  # (1, 1, K, K)
        attn = attn + torch.matmul(cluster_assignments.unsqueeze(1), cluster_bias_expanded).squeeze(2)  # (B, num_heads, N, K)
        
        attn = F.softmax(attn, dim=-1)  # (B, num_heads, N, K)
        
        # Step 3: Apply attention to cluster centroids
        x = (attn @ v_cluster).transpose(1, 2).reshape(B, N, C)  # (B, N, voxel_dim)
        x = self.proj(x)
        
        return x


class PositionIntensityTracker(nn.Module):
    """
    Tracks per-voxel accuracy based on:
    1. Spatial position accuracy
    2. Intensity/contrast accuracy
    """
    
    def __init__(self):
        super().__init__()
    
    def compute_position_accuracy(self,
                                  pred_volume: torch.Tensor,
                                  gt_volume: torch.Tensor,
                                  volume_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Compute spatial position-aware accuracy
        Voxels at different positions should have different importance
        
        Args:
            pred_volume: (B, 1, D, H, W) predicted volume
            gt_volume: (B, 1, D, H, W) ground truth volume
            volume_size: (D, H, W)
        Returns:
            position_accuracy: (B, D, H, W) per-voxel position-weighted accuracy
        """
        B, _, D, H, W = pred_volume.shape
        
        # Compute voxel-wise error
        voxel_error = torch.abs(pred_volume - gt_volume).squeeze(1)  # (B, D, H, W)
        
        # Create position-based importance weights
        # Center regions (heart, major vessels) are more important
        z_center, y_center, x_center = D // 2, H // 2, W // 2
        
        z = torch.arange(D, device=pred_volume.device).view(D, 1, 1)
        y = torch.arange(H, device=pred_volume.device).view(1, H, 1)
        x = torch.arange(W, device=pred_volume.device).view(1, 1, W)
        
        # Distance from center (normalized)
        dist_z = torch.abs(z - z_center) / (D / 2)
        dist_y = torch.abs(y - y_center) / (H / 2)
        dist_x = torch.abs(x - x_center) / (W / 2)
        
        # Gaussian importance (center = higher importance)
        position_weights = torch.exp(-(dist_z**2 + dist_y**2 + dist_x**2))  # (D, H, W)
        position_weights = position_weights.unsqueeze(0).expand(B, -1, -1, -1)  # (B, D, H, W)
        
        # Position-weighted accuracy (lower error = higher accuracy)
        position_accuracy = position_weights * (1.0 - voxel_error)
        
        return position_accuracy
    
    def compute_intensity_accuracy(self,
                                   pred_volume: torch.Tensor,
                                   gt_volume: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute intensity/contrast accuracy
        
        Args:
            pred_volume: (B, 1, D, H, W)
            gt_volume: (B, 1, D, H, W)
        Returns:
            accuracy_dict: Dictionary of intensity-based metrics
        """
        # Voxel-wise intensity accuracy
        intensity_mae = torch.abs(pred_volume - gt_volume)  # (B, 1, D, H, W)
        
        # Contrast accuracy (gradient matching)
        pred_grad_z = torch.abs(pred_volume[:, :, 1:, :, :] - pred_volume[:, :, :-1, :, :])
        gt_grad_z = torch.abs(gt_volume[:, :, 1:, :, :] - gt_volume[:, :, :-1, :, :])
        
        pred_grad_y = torch.abs(pred_volume[:, :, :, 1:, :] - pred_volume[:, :, :, :-1, :])
        gt_grad_y = torch.abs(gt_volume[:, :, :, 1:, :] - gt_volume[:, :, :, :-1, :])
        
        pred_grad_x = torch.abs(pred_volume[:, :, :, :, 1:] - pred_volume[:, :, :, :, :-1])
        gt_grad_x = torch.abs(gt_volume[:, :, :, :, 1:] - gt_volume[:, :, :, :, :-1])
        
        contrast_error_z = torch.abs(pred_grad_z - gt_grad_z)
        contrast_error_y = torch.abs(pred_grad_y - gt_grad_y)
        contrast_error_x = torch.abs(pred_grad_x - gt_grad_x)
        
        return {
            'intensity_mae': intensity_mae.mean(),
            'contrast_error': (contrast_error_z.mean() + contrast_error_y.mean() + contrast_error_x.mean()) / 3,
            'voxel_wise_mae': intensity_mae  # (B, 1, D, H, W) for per-voxel tracking
        }


class SpatialClusteringCTGenerator(nn.Module):
    """
    Complete Spatial Clustering Architecture for CT Volume Generation
    
    Pipeline:
    1. Extract features from bilateral X-rays
    2. Create position encodings for 3D volume
    3. Cluster voxels based on position + predicted intensity
    4. Apply cluster-aware attention
    5. Generate volume with per-voxel tracking
    """
    
    def __init__(self,
                 volume_size: Tuple[int, int, int] = (64, 64, 64),
                 voxel_dim: int = 256,
                 num_clusters: int = 64,
                 num_heads: int = 8,
                 num_blocks: int = 6):
        super().__init__()
        
        self.volume_size = volume_size
        self.voxel_dim = voxel_dim
        self.num_clusters = num_clusters
        D, H, W = volume_size
        
        # X-ray feature extractor (bilateral views)
        self.xray_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((H, W))
        )
        
        # Cross-view fusion
        self.view_fusion = nn.Sequential(
            nn.Conv2d(512, 256, 1),  # 256 * 2 views
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Position encoding
        self.position_encoder = PositionEncodingModule(num_freq_bands=10, d_model=128)
        
        # Initial 3D volume from 2D features
        self.depth_lifter = nn.Sequential(
            nn.Conv2d(256, D * 16, 1),
            nn.BatchNorm2d(D * 16),
            nn.ReLU(inplace=True)
        )
        
        # Project to voxel features
        self.voxel_proj = nn.Conv3d(16, voxel_dim, 3, padding=1)
        
        # Voxel clustering
        self.clustering = VoxelClusteringModule(
            voxel_dim=voxel_dim,
            num_clusters=num_clusters,
            use_position=True,
            use_intensity=True
        )
        
        # Cluster-aware transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                'cluster_attn': ClusterAwareAttention(voxel_dim, num_heads, num_clusters),
                'ffn': nn.Sequential(
                    nn.LayerNorm(voxel_dim),
                    nn.Linear(voxel_dim, voxel_dim * 4),
                    nn.GELU(),
                    nn.Linear(voxel_dim * 4, voxel_dim)
                ),
                'norm1': nn.LayerNorm(voxel_dim),
                'norm2': nn.LayerNorm(voxel_dim)
            })
            for _ in range(num_blocks)
        ])
        
        # Output head
        self.output_proj = nn.Sequential(
            nn.Linear(voxel_dim, voxel_dim // 2),
            nn.GELU(),
            nn.Linear(voxel_dim // 2, 1)
        )
        
        # Accuracy tracker
        self.accuracy_tracker = PositionIntensityTracker()
    
    def forward(self, 
                frontal_xray: torch.Tensor,
                lateral_xray: torch.Tensor,
                gt_volume: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            frontal_xray: (B, 1, H_img, W_img) frontal X-ray
            lateral_xray: (B, 1, H_img, W_img) lateral X-ray
            gt_volume: (B, 1, D, H, W) ground truth (for training only)
        Returns:
            output_dict: Dictionary with predicted volume and tracking metrics
        """
        B = frontal_xray.shape[0]
        D, H, W = self.volume_size
        
        # Extract X-ray features
        frontal_feat = self.xray_encoder(frontal_xray)  # (B, 256, H, W)
        lateral_feat = self.xray_encoder(lateral_xray)  # (B, 256, H, W)
        
        # Fuse views
        fused_xray = self.view_fusion(torch.cat([frontal_feat, lateral_feat], dim=1))  # (B, 256, H, W)
        
        # Lift to 3D
        lifted = self.depth_lifter(fused_xray)  # (B, D*16, H, W)
        lifted = lifted.view(B, 16, D, H, W)  # (B, 16, D, H, W)
        
        # Project to voxel features
        voxel_features = self.voxel_proj(lifted)  # (B, voxel_dim, D, H, W)
        
        # Reshape to sequence
        voxel_features = voxel_features.view(B, self.voxel_dim, -1).transpose(1, 2)  # (B, D*H*W, voxel_dim)
        
        # Get position encodings
        position_features = self.position_encoder(self.volume_size, voxel_features.device)  # (D*H*W, 128)
        
        # Extract initial intensities for clustering
        initial_intensities = self.output_proj(voxel_features).squeeze(-1).unsqueeze(-1)  # (B, D*H*W, 1)
        
        # Perform clustering
        cluster_assignments, fused_features = self.clustering(
            voxel_features, 
            position_features,
            initial_intensities
        )  # (B, D*H*W, num_clusters), (B, D*H*W, voxel_dim)
        
        # Apply transformer blocks with cluster-aware attention
        x = fused_features
        for block in self.transformer_blocks:
            # Cluster-aware attention
            x_norm = block['norm1'](x)
            x = x + block['cluster_attn'](x_norm, cluster_assignments)
            
            # FFN
            x = x + block['ffn'](block['norm2'](x))
        
        # Generate final volume
        pred_intensities = self.output_proj(x)  # (B, D*H*W, 1)
        pred_volume = pred_intensities.transpose(1, 2).view(B, 1, D, H, W)  # (B, 1, D, H, W)
        
        # Prepare output
        output = {
            'pred_volume': pred_volume,
            'cluster_assignments': cluster_assignments,
            'voxel_features': x
        }
        
        # If ground truth provided, compute tracking metrics
        if gt_volume is not None:
            position_accuracy = self.accuracy_tracker.compute_position_accuracy(
                pred_volume, gt_volume, self.volume_size
            )
            intensity_metrics = self.accuracy_tracker.compute_intensity_accuracy(
                pred_volume, gt_volume
            )
            
            output.update({
                'position_accuracy': position_accuracy,
                'intensity_metrics': intensity_metrics
            })
        
        return output


class ClusterTrackingLoss(nn.Module):
    """
    Loss function that tracks accuracy per cluster and per position
    """
    
    def __init__(self, 
                 lambda_position: float = 1.0,
                 lambda_intensity: float = 1.0,
                 lambda_contrast: float = 0.5,
                 lambda_cluster: float = 0.3):
        super().__init__()
        self.lambda_position = lambda_position
        self.lambda_intensity = lambda_intensity
        self.lambda_contrast = lambda_contrast
        self.lambda_cluster = lambda_cluster
    
    def forward(self, 
                pred_volume: torch.Tensor,
                gt_volume: torch.Tensor,
                position_accuracy: torch.Tensor,
                intensity_metrics: Dict[str, torch.Tensor],
                cluster_assignments: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss with per-voxel tracking
        
        Args:
            pred_volume: (B, 1, D, H, W)
            gt_volume: (B, 1, D, H, W)
            position_accuracy: (B, D, H, W) position-weighted accuracy
            intensity_metrics: Dictionary from tracker
            cluster_assignments: (B, N, num_clusters)
        Returns:
            loss_dict: Dictionary of losses and metrics
        """
        # Position-weighted loss
        position_loss = -position_accuracy.mean()  # Negative because higher accuracy = better
        
        # Intensity loss
        intensity_loss = intensity_metrics['intensity_mae']
        
        # Contrast loss
        contrast_loss = intensity_metrics['contrast_error']
        
        # Cluster consistency loss
        # Encourage voxels in same cluster to have similar intensities
        B, N, K = cluster_assignments.shape
        pred_flat = pred_volume.view(B, N)  # (B, N)
        
        cluster_variance = 0
        for k in range(K):
            # Get voxels assigned to this cluster
            cluster_weights = cluster_assignments[:, :, k]  # (B, N)
            weighted_pred = pred_flat * cluster_weights  # (B, N)
            
            # Compute weighted mean and variance
            cluster_mean = weighted_pred.sum(dim=1, keepdim=True) / (cluster_weights.sum(dim=1, keepdim=True) + 1e-8)
            cluster_var = ((weighted_pred - cluster_mean * cluster_weights) ** 2).sum(dim=1).mean()
            cluster_variance += cluster_var
        
        cluster_consistency_loss = cluster_variance / K
        
        # Total loss
        total_loss = (
            self.lambda_position * position_loss +
            self.lambda_intensity * intensity_loss +
            self.lambda_contrast * contrast_loss +
            self.lambda_cluster * cluster_consistency_loss
        )
        
        return {
            'total_loss': total_loss,
            'position_loss': position_loss,
            'intensity_loss': intensity_loss,
            'contrast_loss': contrast_loss,
            'cluster_consistency': cluster_consistency_loss
        }


# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = SpatialClusteringCTGenerator(
        volume_size=(64, 64, 64),
        voxel_dim=256,
        num_clusters=64,
        num_heads=8,
        num_blocks=6
    ).to(device)
    
    # Test forward pass
    frontal = torch.randn(2, 1, 512, 512).to(device)
    lateral = torch.randn(2, 1, 512, 512).to(device)
    gt = torch.randn(2, 1, 64, 64, 64).to(device)
    
    output = model(frontal, lateral, gt)
    
    print("Output keys:", output.keys())
    print("Predicted volume:", output['pred_volume'].shape)
    print("Cluster assignments:", output['cluster_assignments'].shape)
    print("Position accuracy:", output['position_accuracy'].shape)
    print("Intensity MAE:", output['intensity_metrics']['intensity_mae'].item())
    
    # Test loss
    loss_fn = ClusterTrackingLoss()
    loss_dict = loss_fn(
        output['pred_volume'],
        gt,
        output['position_accuracy'],
        output['intensity_metrics'],
        output['cluster_assignments']
    )
    
    print("\nLoss breakdown:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
