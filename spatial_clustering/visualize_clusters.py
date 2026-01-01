"""
Visualization tools for Spatial Clustering CT Generator
Visualizes cluster assignments, position accuracy, and tracking metrics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import argparse

from spatial_cluster_architecture import SpatialClusteringCTGenerator


class ClusterVisualizer:
    """
    Visualize cluster assignments and metrics
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def visualize_clusters_3d(self, frontal_xray, lateral_xray, save_path='cluster_3d.png'):
        """
        Visualize 3D cluster assignment map
        """
        output = self.model(frontal_xray, lateral_xray)
        cluster_assignments = output['cluster_assignments'][0]  # (N, K)
        
        # Get dominant cluster for each voxel
        cluster_ids = torch.argmax(cluster_assignments, dim=-1).cpu().numpy()  # (N,)
        
        D, H, W = self.model.volume_size
        cluster_volume = cluster_ids.reshape(D, H, W)
        
        # Create 3D visualization
        fig = plt.figure(figsize=(15, 5))
        
        # Axial view
        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(cluster_volume[D//2], cmap='tab20', vmin=0, vmax=self.model.num_clusters-1)
        ax1.set_title('Cluster Map (Axial)')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='Cluster ID')
        
        # Coronal view
        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(cluster_volume[:, H//2, :], cmap='tab20', vmin=0, vmax=self.model.num_clusters-1)
        ax2.set_title('Cluster Map (Coronal)')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, label='Cluster ID')
        
        # Sagittal view
        ax3 = fig.add_subplot(133)
        im3 = ax3.imshow(cluster_volume[:, :, W//2], cmap='tab20', vmin=0, vmax=self.model.num_clusters-1)
        ax3.set_title('Cluster Map (Sagittal)')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, label='Cluster ID')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved 3D cluster visualization to {save_path}")
        
        return cluster_volume
    
    @torch.no_grad()
    def visualize_cluster_statistics(self, frontal_xray, lateral_xray, gt_volume, save_path='cluster_stats.png'):
        """
        Visualize per-cluster statistics
        """
        output = self.model(frontal_xray, lateral_xray, gt_volume)
        
        cluster_assignments = output['cluster_assignments'][0]  # (N, K)
        pred_volume = output['pred_volume'][0, 0].cpu().numpy().flatten()  # (N,)
        gt_volume_flat = gt_volume[0, 0].cpu().numpy().flatten()  # (N,)
        
        K = self.model.num_clusters
        
        # Compute per-cluster metrics
        cluster_sizes = []
        cluster_mean_intensity = []
        cluster_variance = []
        cluster_mae = []
        
        for k in range(K):
            weights = cluster_assignments[:, k].cpu().numpy()  # (N,)
            cluster_size = weights.sum()
            
            if cluster_size > 0.1:  # Only consider non-empty clusters
                # Weighted statistics
                weighted_pred = pred_volume * weights
                weighted_gt = gt_volume_flat * weights
                
                mean_intensity = weighted_pred.sum() / cluster_size
                variance = ((pred_volume - mean_intensity) ** 2 * weights).sum() / cluster_size
                mae = np.abs(weighted_pred - weighted_gt).sum() / cluster_size
                
                cluster_sizes.append(cluster_size)
                cluster_mean_intensity.append(mean_intensity)
                cluster_variance.append(variance)
                cluster_mae.append(mae)
            else:
                cluster_sizes.append(0)
                cluster_mean_intensity.append(0)
                cluster_variance.append(0)
                cluster_mae.append(0)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Cluster sizes
        axes[0, 0].bar(range(K), cluster_sizes, color='steelblue')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Cluster Size (# voxels)')
        axes[0, 0].set_title('Cluster Sizes')
        axes[0, 0].grid(alpha=0.3)
        
        # Mean intensities
        axes[0, 1].bar(range(K), cluster_mean_intensity, color='forestgreen')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Mean Intensity (HU)')
        axes[0, 1].set_title('Cluster Mean Intensities')
        axes[0, 1].grid(alpha=0.3)
        
        # Variance
        axes[1, 0].bar(range(K), cluster_variance, color='coral')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Intensity Variance')
        axes[1, 0].set_title('Cluster Variance (Consistency)')
        axes[1, 0].grid(alpha=0.3)
        
        # MAE per cluster
        axes[1, 1].bar(range(K), cluster_mae, color='crimson')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('Per-Cluster Accuracy')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved cluster statistics to {save_path}")
        
        return {
            'sizes': cluster_sizes,
            'mean_intensity': cluster_mean_intensity,
            'variance': cluster_variance,
            'mae': cluster_mae
        }
    
    @torch.no_grad()
    def visualize_position_accuracy(self, frontal_xray, lateral_xray, gt_volume, save_path='position_accuracy.png'):
        """
        Visualize position-weighted accuracy heatmaps
        """
        output = self.model(frontal_xray, lateral_xray, gt_volume)
        
        position_acc = output['position_accuracy'][0].cpu().numpy()  # (D, H, W)
        pred_volume = output['pred_volume'][0, 0].cpu().numpy()
        gt_vol = gt_volume[0, 0].cpu().numpy()
        
        D, H, W = position_acc.shape
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Predicted volume (axial)
        axes[0, 0].imshow(pred_volume[D//2], cmap='gray')
        axes[0, 0].set_title('Predicted Volume (Axial)')
        axes[0, 0].axis('off')
        
        # Ground truth (axial)
        axes[0, 1].imshow(gt_vol[D//2], cmap='gray')
        axes[0, 1].set_title('Ground Truth (Axial)')
        axes[0, 1].axis('off')
        
        # Position accuracy (axial)
        im1 = axes[0, 2].imshow(position_acc[D//2], cmap='RdYlGn', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Position Accuracy (Axial)\nMean: {position_acc[D//2].mean():.3f}')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2])
        
        # Coronal views
        axes[1, 0].imshow(pred_volume[:, H//2, :], cmap='gray')
        axes[1, 0].set_title('Predicted Volume (Coronal)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(gt_vol[:, H//2, :], cmap='gray')
        axes[1, 1].set_title('Ground Truth (Coronal)')
        axes[1, 1].axis('off')
        
        im2 = axes[1, 2].imshow(position_acc[:, H//2, :], cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 2].set_title(f'Position Accuracy (Coronal)\nMean: {position_acc[:, H//2, :].mean():.3f}')
        axes[1, 2].axis('off')
        plt.colorbar(im2, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved position accuracy visualization to {save_path}")
    
    @torch.no_grad()
    def visualize_cluster_intensity_distribution(self, frontal_xray, lateral_xray, save_path='cluster_intensity_dist.png'):
        """
        Visualize intensity distribution per cluster
        """
        output = self.model(frontal_xray, lateral_xray)
        
        cluster_assignments = output['cluster_assignments'][0]  # (N, K)
        pred_volume = output['pred_volume'][0, 0].cpu().numpy().flatten()  # (N,)
        
        # Get dominant cluster for each voxel
        cluster_ids = torch.argmax(cluster_assignments, dim=-1).cpu().numpy()  # (N,)
        
        K = self.model.num_clusters
        
        # Group intensities by cluster
        cluster_intensities = [pred_volume[cluster_ids == k] for k in range(K)]
        
        # Plot histograms
        fig, axes = plt.subplots(8, 8, figsize=(20, 20))
        axes = axes.flatten()
        
        for k in range(min(K, 64)):
            if len(cluster_intensities[k]) > 0:
                axes[k].hist(cluster_intensities[k], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                axes[k].set_title(f'Cluster {k} (n={len(cluster_intensities[k])})', fontsize=8)
                axes[k].set_xlabel('Intensity', fontsize=7)
                axes[k].set_ylabel('Count', fontsize=7)
                axes[k].tick_params(labelsize=6)
            else:
                axes[k].text(0.5, 0.5, 'Empty', ha='center', va='center', transform=axes[k].transAxes)
                axes[k].set_title(f'Cluster {k}', fontsize=8)
            axes[k].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved cluster intensity distributions to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Spatial Clustering CT Generator')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    model = SpatialClusteringCTGenerator(
        volume_size=tuple(config['model']['volume_size']),
        voxel_dim=config['model']['voxel_dim'],
        num_clusters=config['model']['num_clusters'],
        num_heads=config['model']['num_heads'],
        num_blocks=config['model']['num_blocks']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create visualizer
    visualizer = ClusterVisualizer(model, device)
    
    # Create dummy data (replace with real data)
    frontal = torch.randn(1, 1, 512, 512).to(device)
    lateral = torch.randn(1, 1, 512, 512).to(device)
    gt = torch.randn(1, 1, 64, 64, 64).to(device)
    
    print("\nGenerating visualizations...")
    
    # 3D cluster map
    visualizer.visualize_clusters_3d(
        frontal, lateral,
        save_path=output_dir / 'cluster_3d.png'
    )
    
    # Cluster statistics
    visualizer.visualize_cluster_statistics(
        frontal, lateral, gt,
        save_path=output_dir / 'cluster_stats.png'
    )
    
    # Position accuracy
    visualizer.visualize_position_accuracy(
        frontal, lateral, gt,
        save_path=output_dir / 'position_accuracy.png'
    )
    
    # Intensity distributions
    visualizer.visualize_cluster_intensity_distribution(
        frontal, lateral,
        save_path=output_dir / 'cluster_intensity_dist.png'
    )
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
