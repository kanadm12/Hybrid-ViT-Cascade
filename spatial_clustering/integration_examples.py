"""
Integration Guide: Spatial Clustering Architecture with Existing Codebase
Shows how to use spatial clustering with your current training infrastructure
"""

import torch
import torch.nn as nn
from spatial_cluster_architecture import (
    SpatialClusteringCTGenerator,
    ClusterTrackingLoss
)

# Example 1: Integration with existing data pipeline
def integrate_with_dataloader():
    """
    Shows how to use spatial clustering with your existing dataset
    """
    print("\n" + "="*80)
    print("Example 1: Integration with DataLoader")
    print("="*80)
    
    # Assuming your dataset returns:
    # {
    #   'frontal': (B, 1, H, W),
    #   'lateral': (B, 1, H, W),
    #   'volume': (B, 1, D, H, W),
    #   'metadata': {...}
    # }
    
    from torch.utils.data import DataLoader
    
    # Create model
    model = SpatialClusteringCTGenerator(
        volume_size=(64, 64, 64),
        voxel_dim=256,
        num_clusters=64,
        num_heads=8,
        num_blocks=6
    ).cuda()
    
    # Loss function
    criterion = ClusterTrackingLoss(
        lambda_position=1.0,
        lambda_intensity=1.0,
        lambda_contrast=0.5,
        lambda_cluster=0.3
    )
    
    # Your existing dataloader
    # train_loader = DataLoader(your_dataset, batch_size=4, ...)
    
    # Training loop
    model.train()
    # for batch in train_loader:
    #     frontal = batch['frontal'].cuda()
    #     lateral = batch['lateral'].cuda()
    #     volume = batch['volume'].cuda()
    #     
    #     # Forward
    #     output = model(frontal, lateral, volume)
    #     
    #     # Compute loss
    #     loss_dict = criterion(
    #         output['pred_volume'],
    #         volume,
    #         output['position_accuracy'],
    #         output['intensity_metrics'],
    #         output['cluster_assignments']
    #     )
    #     
    #     # Backward
    #     loss_dict['total_loss'].backward()
    
    print("✓ Integration example ready")


# Example 2: Hybrid with existing models
def hybrid_with_existing_models():
    """
    Shows how to combine spatial clustering with your existing models
    """
    print("\n" + "="*80)
    print("Example 2: Hybrid Architecture")
    print("="*80)
    
    # Import your existing models
    import sys
    sys.path.append('..')
    # from models.unified_model import UnifiedCTReconstruction
    # from models.cascaded_depth_lifting import CascadedDepthWeightNetwork
    
    class HybridSpatialClusteringModel(nn.Module):
        """
        Combines spatial clustering with your existing cascaded architecture
        """
        
        def __init__(self):
            super().__init__()
            
            # Stage 1: Your existing coarse generation
            # self.stage1 = UnifiedCTReconstruction(...)
            
            # Stage 2: Spatial clustering refinement
            self.stage2_clustering = SpatialClusteringCTGenerator(
                volume_size=(64, 64, 64),
                voxel_dim=256,
                num_clusters=64
            )
        
        def forward(self, frontal, lateral):
            # Stage 1: Coarse prediction
            # coarse_volume = self.stage1(frontal, lateral)
            coarse_volume = torch.randn(frontal.shape[0], 1, 64, 64, 64).to(frontal.device)
            
            # Stage 2: Cluster-based refinement
            # Use coarse volume as initialization
            refined_output = self.stage2_clustering(frontal, lateral)
            
            # Combine coarse + refined
            final_volume = 0.3 * coarse_volume + 0.7 * refined_output['pred_volume']
            
            return {
                'coarse': coarse_volume,
                'refined': refined_output['pred_volume'],
                'final': final_volume,
                'clusters': refined_output['cluster_assignments']
            }
    
    model = HybridSpatialClusteringModel().cuda()
    print(f"Hybrid model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("✓ Hybrid architecture created")


# Example 3: Adding to your existing loss functions
def combine_with_existing_losses():
    """
    Shows how to add clustering losses to your existing loss functions
    """
    print("\n" + "="*80)
    print("Example 3: Combined Loss Functions")
    print("="*80)
    
    import sys
    sys.path.append('..')
    # from models.diagnostic_losses import DiagnosticLosses
    
    class CombinedLoss(nn.Module):
        """
        Combines your existing losses with spatial clustering losses
        """
        
        def __init__(self):
            super().__init__()
            
            # Your existing losses
            # self.diagnostic_losses = DiagnosticLosses(
            #     volume_size=(64, 64, 64),
            #     use_perceptual=True,
            #     use_feature_metrics=True
            # )
            
            # Spatial clustering losses
            self.cluster_loss = ClusterTrackingLoss(
                lambda_position=1.0,
                lambda_intensity=1.0,
                lambda_contrast=0.5,
                lambda_cluster=0.3
            )
        
        def forward(self, output, gt_volume, xrays=None):
            """
            Args:
                output: dict from SpatialClusteringCTGenerator
                gt_volume: ground truth
                xrays: optional X-ray inputs for projection loss
            """
            pred_volume = output['pred_volume']
            
            # Existing losses
            # diagnostic_dict = self.diagnostic_losses.compute_all_losses(
            #     predicted=pred_volume,
            #     target=gt_volume,
            #     pred_x0=pred_volume,
            #     gt_x0=gt_volume,
            #     xrays=xrays
            # )
            
            # Spatial clustering losses
            cluster_dict = self.cluster_loss(
                pred_volume,
                gt_volume,
                output['position_accuracy'],
                output['intensity_metrics'],
                output['cluster_assignments']
            )
            
            # Combine
            total_loss = cluster_dict['total_loss']  # + 0.5 * diagnostic_dict['total_loss']
            
            return {
                'total_loss': total_loss,
                'cluster_loss': cluster_dict['total_loss'],
                # 'diagnostic_loss': diagnostic_dict['total_loss'],
                **cluster_dict
            }
    
    combined_loss = CombinedLoss()
    print("✓ Combined loss function created")


# Example 4: Progressive training strategy
def progressive_training_strategy():
    """
    Shows a progressive training strategy with clustering
    """
    print("\n" + "="*80)
    print("Example 4: Progressive Training Strategy")
    print("="*80)
    
    # Phase 1: Train without clustering (warmup)
    print("\nPhase 1: Warmup (epochs 0-10)")
    print("  - Train base model without cluster loss")
    print("  - lambda_cluster = 0.0")
    
    # Phase 2: Introduce clustering gradually
    print("\nPhase 2: Gradual clustering (epochs 10-30)")
    print("  - Gradually increase lambda_cluster: 0.0 → 0.3")
    print("  - Start with fewer clusters (32), increase to 64")
    
    # Phase 3: Full clustering with refinement
    print("\nPhase 3: Full clustering (epochs 30+)")
    print("  - Full cluster loss: lambda_cluster = 0.3")
    print("  - Fine-tune position weights")
    
    class ProgressiveTrainer:
        def __init__(self):
            self.model = SpatialClusteringCTGenerator(
                volume_size=(64, 64, 64),
                num_clusters=32  # Start with fewer
            ).cuda()
            
            self.epoch = 0
        
        def get_loss_weights(self):
            """Returns loss weights based on training phase"""
            if self.epoch < 10:
                # Phase 1: Warmup
                return {
                    'lambda_position': 1.0,
                    'lambda_intensity': 1.0,
                    'lambda_contrast': 0.5,
                    'lambda_cluster': 0.0  # No clustering yet
                }
            elif self.epoch < 30:
                # Phase 2: Gradual introduction
                progress = (self.epoch - 10) / 20.0
                return {
                    'lambda_position': 1.0,
                    'lambda_intensity': 1.0,
                    'lambda_contrast': 0.5,
                    'lambda_cluster': 0.3 * progress  # Gradually increase
                }
            else:
                # Phase 3: Full clustering
                return {
                    'lambda_position': 1.0,
                    'lambda_intensity': 1.0,
                    'lambda_contrast': 0.5,
                    'lambda_cluster': 0.3
                }
    
    trainer = ProgressiveTrainer()
    print("\n✓ Progressive training strategy defined")


# Example 5: Inference and visualization
def inference_and_visualization():
    """
    Shows how to use the model for inference and visualize clusters
    """
    print("\n" + "="*80)
    print("Example 5: Inference and Visualization")
    print("="*80)
    
    from visualize_clusters import ClusterVisualizer
    
    # Load trained model
    model = SpatialClusteringCTGenerator(
        volume_size=(64, 64, 64),
        voxel_dim=256,
        num_clusters=64
    ).cuda()
    
    # checkpoint = torch.load('checkpoints/spatial_clustering/best.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Inference
    frontal = torch.randn(1, 1, 512, 512).cuda()
    lateral = torch.randn(1, 1, 512, 512).cuda()
    
    with torch.no_grad():
        output = model(frontal, lateral)
    
    print(f"Predicted volume: {output['pred_volume'].shape}")
    print(f"Cluster assignments: {output['cluster_assignments'].shape}")
    
    # Visualize
    visualizer = ClusterVisualizer(model, 'cuda')
    # visualizer.visualize_clusters_3d(frontal, lateral, 'output_clusters.png')
    
    print("✓ Inference example ready")


# Example 6: Cluster analysis
def analyze_learned_clusters():
    """
    Shows how to analyze what the model learned
    """
    print("\n" + "="*80)
    print("Example 6: Cluster Analysis")
    print("="*80)
    
    model = SpatialClusteringCTGenerator(
        volume_size=(64, 64, 64),
        num_clusters=64
    ).cuda()
    model.eval()
    
    # Run on test set
    frontal = torch.randn(1, 1, 512, 512).cuda()
    lateral = torch.randn(1, 1, 512, 512).cuda()
    
    with torch.no_grad():
        output = model(frontal, lateral)
    
    cluster_assignments = output['cluster_assignments'][0]  # (N, K)
    pred_volume = output['pred_volume'][0, 0].cpu().numpy()
    
    # Analyze clusters
    print("\nCluster Analysis:")
    for k in range(min(10, model.num_clusters)):
        weights = cluster_assignments[:, k].cpu().numpy()
        cluster_size = weights.sum()
        
        if cluster_size > 100:
            weighted_intensity = (pred_volume.flatten() * weights).sum() / cluster_size
            print(f"  Cluster {k}: size={cluster_size:.0f}, mean_intensity={weighted_intensity:.3f}")
    
    # Find dominant clusters
    dominant_clusters = cluster_assignments.sum(dim=0).topk(5)
    print(f"\nTop 5 dominant clusters: {dominant_clusters.indices.tolist()}")
    print(f"Their sizes: {dominant_clusters.values.cpu().numpy()}")
    
    print("\n✓ Cluster analysis complete")


# Run all examples
if __name__ == "__main__":
    print("\n" + "="*80)
    print("SPATIAL CLUSTERING INTEGRATION EXAMPLES")
    print("="*80)
    
    integrate_with_dataloader()
    hybrid_with_existing_models()
    combine_with_existing_losses()
    progressive_training_strategy()
    inference_and_visualization()
    analyze_learned_clusters()
    
    print("\n" + "="*80)
    print("✓ ALL INTEGRATION EXAMPLES COMPLETE")
    print("="*80 + "\n")
    
    print("\nNext steps:")
    print("1. Replace dummy datasets with your actual CT data")
    print("2. Adjust hyperparameters in config_spatial_clustering.json")
    print("3. Run test_spatial_clustering.py to validate setup")
    print("4. Start training with train_spatial_clustering.py")
    print("5. Visualize results with visualize_clusters.py")
