"""
Visualization utilities for feature maps and model outputs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple
from pathlib import Path

# Optional wandb - graceful fallback if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Install with: pip install wandb")
    WANDB_AVAILABLE = False
    wandb = None


def plot_feature_maps(features: Dict[str, torch.Tensor],
                     save_path: Optional[Path] = None,
                     title: str = "Feature Maps",
                     slice_idx: Optional[int] = None,
                     max_channels: int = 8,
                     wandb_log: bool = False,
                     wandb_prefix: str = "") -> plt.Figure:
    """
    Plot feature maps from multiple levels
    
    Args:
        features: Dictionary with keys like 'level_0', 'level_1', etc.
                 Each value has shape (B, C, D, H, W) or (B, C, H, W)
        save_path: Path to save the figure
        title: Figure title
        slice_idx: Which slice to visualize (for 3D features). If None, use middle slice
        max_channels: Maximum number of channels to display per level
        wandb_log: Whether to log to Weights & Biases
        wandb_prefix: Prefix for wandb logging key
    
    Returns:
        matplotlib Figure object
    """
    # Validate input
    if not features or len(features) == 0:
        raise ValueError("Features dictionary is empty")
    
    num_levels = len(features)
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_levels, max_channels, 
                            figsize=(max_channels * 2, num_levels * 2))
    
    # Ensure axes is 2D array
    if num_levels == 1:
        axes = axes.reshape(1, -1)
    if max_channels == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for level_idx, (level_name, feat_tensor) in enumerate(sorted(features.items())):
        # Validate tensor
        if feat_tensor is None or feat_tensor.numel() == 0:
            print(f"Warning: Empty tensor at {level_name}, skipping")
            continue
            
        # Move to CPU and get first batch item
        feat = feat_tensor[0].detach().cpu()  # (C, D, H, W) or (C, H, W)
        
        # Handle 3D features - extract middle slice (reset slice_idx per level)
        current_slice_idx = slice_idx
        if len(feat.shape) == 4:  # (C, D, H, W)
            if current_slice_idx is None:
                current_slice_idx = feat.shape[1] // 2  # Middle slice along depth
            # Clamp slice index to valid range
            current_slice_idx = max(0, min(current_slice_idx, feat.shape[1] - 1))
            feat = feat[:, current_slice_idx, :, :]  # (C, H, W)
        
        # Validate shape after slicing
        if len(feat.shape) != 3:
            print(f"Warning: Unexpected shape {feat.shape} at {level_name}, skipping")
            continue
        
        # Get number of channels
        num_channels = feat.shape[0]
        channels_to_show = min(num_channels, max_channels)
        
        # Plot each channel
        for ch_idx in range(max_channels):
            ax = axes[level_idx, ch_idx]
            
            if ch_idx < channels_to_show:
                # Plot feature map
                channel_data = feat[ch_idx].numpy()
                
                # Normalize for visualization
                vmin, vmax = channel_data.min(), channel_data.max()
                if vmax - vmin > 1e-6:
                    channel_data = (channel_data - vmin) / (vmax - vmin)
                else:
                    # If constant, just use the data as-is
                    pass
                
                im = ax.imshow(channel_data, cmap='viridis', aspect='auto')
                ax.set_title(f'{level_name}\nCh {ch_idx}', fontsize=8)
                ax.axis('off')
                
                # Add colorbar for first channel of each level
                if ch_idx == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                # Empty subplot
                ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature maps saved to {save_path}")
    
    # Log to wandb
    if wandb_log and WANDB_AVAILABLE:
        wandb_key = f"{wandb_prefix}feature_maps" if wandb_prefix else "feature_maps"
        wandb.log({wandb_key: wandb.Image(fig)})
    elif wandb_log and not WANDB_AVAILABLE:
        print("Warning: wandb logging requested but wandb not available")
    
    return fig


def plot_feature_comparison(features_base: Dict[str, torch.Tensor],
                           features_gen: Dict[str, torch.Tensor],
                           save_path: Optional[Path] = None,
                           title: str = "Feature Comparison: Base vs Generated",
                           slice_idx: Optional[int] = None,
                           max_levels: int = 4,
                           wandb_log: bool = False,
                           wandb_prefix: str = "") -> plt.Figure:
    """
    Plot side-by-side comparison of base CT and generated CT feature maps
    
    Args:
        features_base: Features from ground truth CT
        features_gen: Features from generated CT
        save_path: Path to save the figure
        title: Figure title
        slice_idx: Which slice to visualize (for 3D features)
        max_levels: Maximum number of levels to display
        wandb_log: Whether to log to Weights & Biases
        wandb_prefix: Prefix for wandb logging key
    
    Returns:
        matplotlib Figure object
    """
    num_levels = min(len(features_base), max_levels)
    level_names = sorted(list(features_base.keys()))[:num_levels]
    
    # Create figure: rows = levels, cols = [base, generated, difference]
    fig, axes = plt.subplots(num_levels, 3, figsize=(12, num_levels * 3))
    
    if num_levels == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for level_idx, level_name in enumerate(level_names):
        # Validate tensors exist
        if level_name not in features_base or level_name not in features_gen:
            print(f"Warning: {level_name} missing in one of the feature dicts, skipping")
            continue
            
        feat_base = features_base[level_name][0].detach().cpu()  # First batch item
        feat_gen = features_gen[level_name][0].detach().cpu()
        
        # Handle 3D features (reset slice_idx per level)
        current_slice_idx = slice_idx
        if len(feat_base.shape) == 4:  # (C, D, H, W)
            if current_slice_idx is None:
                current_slice_idx = feat_base.shape[1] // 2  # Middle slice
            # Clamp to valid range
            current_slice_idx = max(0, min(current_slice_idx, feat_base.shape[1] - 1))
            feat_base = feat_base[:, current_slice_idx, :, :]
            feat_gen = feat_gen[:, current_slice_idx, :, :]
        
        # Validate shapes match
        if feat_base.shape != feat_gen.shape:
            print(f"Warning: Shape mismatch at {level_name}: {feat_base.shape} vs {feat_gen.shape}")
            continue
        
        # Average across channels for visualization
        feat_base_avg = feat_base.mean(dim=0).numpy()
        feat_gen_avg = feat_gen.mean(dim=0).numpy()
        feat_diff = np.abs(feat_base_avg - feat_gen_avg)
        
        # Normalize
        vmin = min(feat_base_avg.min(), feat_gen_avg.min())
        vmax = max(feat_base_avg.max(), feat_gen_avg.max())
        
        # Plot base
        ax = axes[level_idx, 0]
        im1 = ax.imshow(feat_base_avg, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'{level_name}\nBase CT', fontsize=10)
        ax.axis('off')
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot generated
        ax = axes[level_idx, 1]
        im2 = ax.imshow(feat_gen_avg, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'{level_name}\nGenerated CT', fontsize=10)
        ax.axis('off')
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot difference
        ax = axes[level_idx, 2]
        im3 = ax.imshow(feat_diff, cmap='hot', aspect='auto')
        ax.set_title(f'{level_name}\nAbsolute Difference', fontsize=10)
        ax.axis('off')
        plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature comparison saved to {save_path}")
    
    # Log to wandb
    if wandb_log and WANDB_AVAILABLE:
        wandb_key = f"{wandb_prefix}feature_comparison" if wandb_prefix else "feature_comparison"
        wandb.log({wandb_key: wandb.Image(fig)})
    elif wandb_log and not WANDB_AVAILABLE:
        print("Warning: wandb logging requested but wandb not available")
    
    return fig


def plot_feature_accuracy_heatmap(accuracy_metrics: Dict[str, float],
                                  save_path: Optional[Path] = None,
                                  title: str = "Feature Map Accuracy Metrics",
                                  wandb_log: bool = False,
                                  wandb_prefix: str = "") -> plt.Figure:
    """
    Plot heatmap of feature accuracy metrics across levels
    
    Args:
        accuracy_metrics: Dictionary with keys like 'level_0_mse', 'level_1_cosine', etc.
        save_path: Path to save the figure
        title: Figure title
        wandb_log: Whether to log to Weights & Biases
        wandb_prefix: Prefix for wandb logging key
    
    Returns:
        matplotlib Figure object
    """
    # Parse metrics into matrix format
    levels = set()
    metric_types = set()
    
    for key in accuracy_metrics.keys():
        if key.startswith('level_'):
            parts = key.split('_')
            level = f"{parts[0]}_{parts[1]}"  # e.g., 'level_0'
            metric = '_'.join(parts[2:])  # e.g., 'mse', 'cosine'
            levels.add(level)
            metric_types.add(metric)
    
    levels = sorted(list(levels))
    metric_types = sorted(list(metric_types))
    
    # Create matrix
    matrix = np.zeros((len(levels), len(metric_types)))
    
    for i, level in enumerate(levels):
        for j, metric in enumerate(metric_types):
            key = f"{level}_{metric}"
            if key in accuracy_metrics:
                matrix[i, j] = accuracy_metrics[key]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(4, len(levels) * 0.5)))
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metric_types)))
    ax.set_yticks(np.arange(len(levels)))
    ax.set_xticklabels(metric_types)
    ax.set_yticklabels(levels)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(levels)):
        for j in range(len(metric_types)):
            text = ax.text(j, i, f'{matrix[i, j]:.4f}',
                         ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    # Save figure
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature accuracy heatmap saved to {save_path}")
    
    # Log to wandb
    if wandb_log and WANDB_AVAILABLE:
        wandb_key = f"{wandb_prefix}feature_accuracy_heatmap" if wandb_prefix else "feature_accuracy_heatmap"
        wandb.log({wandb_key: wandb.Image(fig)})
    elif wandb_log and not WANDB_AVAILABLE:
        print("Warning: wandb logging requested but wandb not available")
    
    return fig


def visualize_epoch_features(model,
                            val_batch: Dict[str, torch.Tensor],
                            epoch: int,
                            stage_name: str,
                            save_dir: Path,
                            device: str = 'cuda',
                            wandb_log: bool = False) -> Dict[str, plt.Figure]:
    """
    Comprehensive feature visualization at the end of an epoch
    
    Args:
        model: Trained model with feature extraction capability
        val_batch: Validation batch containing 'ct_volume' and 'drr_stacked'
        epoch: Current epoch number
        stage_name: Current training stage name
        save_dir: Directory to save visualizations
        device: Device to run on
        wandb_log: Whether to log to Weights & Biases
    
    Returns:
        Dictionary of matplotlib figures
    """
    model.eval()
    figures = {}
    
    with torch.no_grad():
        # Extract batch data
        gt_volume = val_batch['ct_volume'].to(device)
        xrays = val_batch['drr_stacked'].to(device)
        
        # Get prediction (simplified - use direct forward without full diffusion)
        # For visualization, we'll use the model's feature extractor on GT
        # and compare with a noisy version to demonstrate the capability
        
        # Generate a prediction (using model forward)
        loss_dict = model(gt_volume, xrays, stage_name, prev_stage_volume=None)
        
        # For feature visualization, we need actual predictions
        # Let's use a sample timestep
        batch_size = gt_volume.shape[0]
        t = torch.randint(0, model.num_timesteps // 2, (batch_size,), device=device).long()
        noise = torch.randn_like(gt_volume)
        noisy_volume = model.q_sample(gt_volume, t, noise)
        
        # Get time embeddings
        t_normalized = t.float() / model.num_timesteps
        t_embed = model.time_embed(t_normalized.unsqueeze(-1))
        
        # Encode X-rays
        xray_context, time_xray_cond, xray_features_2d = model.xray_encoder(xrays, t_embed)
        
        # Get stage and forward through it
        stage = model.stages[stage_name]
        predicted_noise = stage(
            noisy_volume=noisy_volume,
            xray_features=xray_features_2d,
            xray_context=xray_features_2d,
            time_xray_cond=time_xray_cond,
            prev_stage_volume=None,
            prev_stage_embed=None
        )
        
        # Predict clean volume
        if model.v_parameterization:
            sqrt_alphas_t = model.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
            sqrt_one_minus_alphas_t = model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
            pred_volume = sqrt_alphas_t * noisy_volume - sqrt_one_minus_alphas_t * predicted_noise
        else:
            sqrt_alphas_t = model.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
            sqrt_one_minus_alphas_t = model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
            pred_volume = (noisy_volume - sqrt_one_minus_alphas_t * predicted_noise) / (sqrt_alphas_t + 1e-8)
        
        # Clamp predicted volume to reasonable range
        pred_volume = torch.clamp(pred_volume, -10.0, 10.0)
        
        # Ensure predicted volume matches GT volume size
        if pred_volume.shape != gt_volume.shape:
            print(f"Warning: Shape mismatch - GT: {gt_volume.shape}, Pred: {pred_volume.shape}")
            pred_volume = torch.nn.functional.interpolate(
                pred_volume, size=gt_volume.shape[2:], mode='trilinear', align_corners=True
            )
        
        # Extract features from both GT and predicted
        features_gt = model.extract_feature_maps(gt_volume)
        features_pred = model.extract_feature_maps(pred_volume)
        
        # Compute accuracy metrics
        accuracy_metrics = model.compute_feature_accuracy(gt_volume, pred_volume)
        
        # Create visualizations
        epoch_dir = save_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature maps from GT
        fig_gt = plot_feature_maps(
            features_gt,
            save_path=epoch_dir / f"{stage_name}_features_gt.png",
            title=f"Ground Truth Features - Epoch {epoch}",
            wandb_log=wandb_log,
            wandb_prefix=f"{stage_name}/epoch_{epoch}/"
        )
        figures['features_gt'] = fig_gt
        plt.close(fig_gt)
        
        # 2. Feature maps from predicted
        fig_pred = plot_feature_maps(
            features_pred,
            save_path=epoch_dir / f"{stage_name}_features_pred.png",
            title=f"Predicted Features - Epoch {epoch}",
            wandb_log=wandb_log,
            wandb_prefix=f"{stage_name}/epoch_{epoch}/"
        )
        figures['features_pred'] = fig_pred
        plt.close(fig_pred)
        
        # 3. Feature comparison
        fig_comp = plot_feature_comparison(
            features_gt,
            features_pred,
            save_path=epoch_dir / f"{stage_name}_feature_comparison.png",
            title=f"Feature Comparison - Epoch {epoch}",
            wandb_log=wandb_log,
            wandb_prefix=f"{stage_name}/epoch_{epoch}/"
        )
        figures['feature_comparison'] = fig_comp
        plt.close(fig_comp)
        
        # 4. Accuracy heatmap
        fig_heatmap = plot_feature_accuracy_heatmap(
            accuracy_metrics,
            save_path=epoch_dir / f"{stage_name}_accuracy_heatmap.png",
            title=f"Feature Accuracy - Epoch {epoch}",
            wandb_log=wandb_log,
            wandb_prefix=f"{stage_name}/epoch_{epoch}/"
        )
        figures['accuracy_heatmap'] = fig_heatmap
        plt.close(fig_heatmap)
        
        # Log accuracy metrics to wandb
        if wandb_log and WANDB_AVAILABLE:
            wandb_metrics = {f"{stage_name}/feature_accuracy/{k}": v 
                           for k, v in accuracy_metrics.items()}
            wandb_metrics['epoch'] = epoch
            wandb.log(wandb_metrics)
        elif wandb_log and not WANDB_AVAILABLE:
            print("Warning: wandb logging requested but wandb not available")
    
    model.train()
    return figures
