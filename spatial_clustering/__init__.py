"""
Spatial Clustering Architecture for CT Volume Generation from Bilateral X-rays

This package implements a novel spatial clustering approach for generating 3D CT volumes
from bilateral X-ray images, with pixel-wise tracking of position and intensity accuracy.

Main components:
- PositionEncodingModule: 3D Fourier position encoding
- VoxelClusteringModule: Learnable soft clustering based on position + intensity
- ClusterAwareAttention: Attention mechanism respecting cluster structure
- PositionIntensityTracker: Per-voxel accuracy tracking
- SpatialClusteringCTGenerator: Complete end-to-end model
- ClusterTrackingLoss: Multi-component loss function

Quick start:
    >>> from spatial_clustering import SpatialClusteringCTGenerator
    >>> model = SpatialClusteringCTGenerator(
    ...     volume_size=(64, 64, 64),
    ...     num_clusters=64
    ... ).cuda()
    >>> output = model(frontal_xray, lateral_xray, gt_volume)
    >>> print(output['pred_volume'].shape)  # (B, 1, 64, 64, 64)

For detailed documentation, see README.md and QUICKSTART.md
"""

from .spatial_cluster_architecture import (
    PositionEncodingModule,
    VoxelClusteringModule,
    ClusterAwareAttention,
    PositionIntensityTracker,
    SpatialClusteringCTGenerator,
    ClusterTrackingLoss
)

__version__ = '1.0.0'
__author__ = 'Your Name'
__all__ = [
    'PositionEncodingModule',
    'VoxelClusteringModule',
    'ClusterAwareAttention',
    'PositionIntensityTracker',
    'SpatialClusteringCTGenerator',
    'ClusterTrackingLoss'
]
