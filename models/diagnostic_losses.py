"""
Diagnostic Losses for Hybrid-ViT Cascade
Each loss isolates a specific architectural component for debugging/ablation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import sys
from pathlib import Path

# Import feature metrics
from .feature_metrics import (
    MultiLevelFeatureExtractor,
    FeatureMapAccuracy,
    LPIPS3D,
    ComprehensiveFeatureMetrics
)


class DRRRenderer(nn.Module):
    """
    Digitally Reconstructed Radiograph (DRR) Renderer
    Simulates X-ray projection from 3D volume
    """
    def __init__(self, volume_shape: Tuple[int, int, int]):
        super().__init__()
        self.volume_shape = volume_shape
        
    def forward(self, volume: torch.Tensor, angle: float = 0) -> torch.Tensor:
        """
        Generate DRR by summing along depth dimension
        
        Args:
            volume: (B, D, H, W) - 3D volume
            angle: Rotation angle in degrees (0=AP, 90=lateral)
        
        Returns:
            drr: (B, H, W) - 2D projection
        """
        if angle == 90:
            # Lateral view: sum along width dimension
            drr = volume.sum(dim=-1)  # (B, D, H)
            # Transpose to match expected output shape
            drr = drr.transpose(1, 2)  # (B, H, D)
        else:
            # AP view: sum along depth dimension
            drr = volume.sum(dim=1)  # (B, H, W)
        
        # Normalize to [0, 1] range
        drr = (drr - drr.min()) / (drr.max() - drr.min() + 1e-8)
        
        return drr


class XrayConditioningModule(nn.Module):
    """
    Processes X-ray images for conditioning the 3D volume generation
    """
    def __init__(self, img_size: int = 512, in_channels: int = 1, embed_dim: int = 256,
                 num_views: int = 1, time_embed_dim: int = 256, cond_dim: int = 1024,
                 share_view_weights: bool = True):
        super().__init__()
        self.num_views = num_views
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, cond_dim)
        )
        
        # Projection to conditioning dimension
        self.to_cond = nn.Linear(embed_dim, cond_dim)
        
    def forward(self, xrays: torch.Tensor, t: torch.Tensor):
        """
        Args:
            xrays: (B, num_views, C, H, W) - input X-ray images
            t: (B,) - timestep embeddings
        Returns:
            xray_context: (B, cond_dim) - global conditioning
            time_xray_cond: (B, cond_dim) - time + xray conditioning
            xray_features_2d: (B, embed_dim, H', W') - 2D feature maps
        """
        B, num_views = xrays.shape[0], xrays.shape[1]
        
        # Process each view
        if num_views > 1:
            # Flatten views into batch
            xrays_flat = xrays.view(B * num_views, *xrays.shape[2:])
            features = self.encoder(xrays_flat)
            # Average across views
            features = features.view(B, num_views, *features.shape[1:]).mean(dim=1)
        else:
            features = self.encoder(xrays[:, 0])
        
        # Global average pooling for context
        xray_context = features.mean(dim=[-2, -1])  # (B, embed_dim)
        xray_context = self.to_cond(xray_context)  # (B, cond_dim)
        
        # Time conditioning
        time_embed = self.time_mlp(t)  # (B, cond_dim)
        time_xray_cond = time_embed + xray_context
        
        return xray_context, time_xray_cond, features


class ProjectionLoss(nn.Module):
    """
    Computes projection consistency loss between 3D volume and 2D X-rays
    """
    def __init__(self, volume_shape: Tuple[int, int, int]):
        super().__init__()
        self.drr_renderer = DRRRenderer(volume_shape)
        
    def forward(self, volume: torch.Tensor, xray_target: torch.Tensor, 
                angle: float = 0) -> torch.Tensor:
        """
        Args:
            volume: (B, 1, D, H, W) - predicted 3D volume
            xray_target: (B, 1, H, W) - target X-ray image
            angle: Projection angle
        Returns:
            loss: scalar tensor
        """
        # Remove channel dimension for DRR rendering
        drr = self.drr_renderer(volume.squeeze(1), angle=angle)
        
        # Match dimensions
        if drr.shape != xray_target.squeeze(1).shape:
            drr = F.interpolate(drr.unsqueeze(1), 
                               size=xray_target.shape[2:],
                               mode='bilinear', align_corners=True).squeeze(1)
        
        target = xray_target.squeeze(1)
        return F.mse_loss(drr, target)


class DiagnosticLosses(nn.Module):
    """
    Component-specific losses to identify what's working/failing
    
    Loss Breakdown:
    1. Diffusion Loss → Core denoising capability
    2. Projection Loss → Physics constraint working?
    3. Depth Consistency → Is depth lifting helping?
    4. Cross-Attention Alignment → Is X-ray conditioning effective?
    5. Stage Transition → Are cascade stages coherent?
    6. Frequency Analysis → High-freq details vs low-freq structure
    7. Anatomical Prior → Are priors being used?
    """
    
    def __init__(self,
                 volume_size: Tuple[int, int, int],
                 use_perceptual: bool = True,
                 use_frequency: bool = True,
                 use_feature_metrics: bool = True,
                 use_lpips: bool = True,
                 feature_dims: List[int] = [32, 64, 128, 256],
                 lpips_net: str = 'alex'):
        super().__init__()
        
        self.volume_size = volume_size
        D, H, W = volume_size
        
        # DRR renderer for projection loss
        self.drr_renderer = DRRRenderer(volume_shape=(D, H, W))
        
        # Perceptual loss (3D features)
        if use_perceptual:
            self.perceptual_net = Simple3DPerceptualNet()
        
        # Frequency decomposition
        self.use_frequency = use_frequency
        
        # Feature metrics and LPIPS
        self.use_feature_metrics = use_feature_metrics
        self.use_lpips = use_lpips
        
        if use_feature_metrics or use_lpips:
            self.comprehensive_metrics = ComprehensiveFeatureMetrics(
                feature_dims=feature_dims,
                lpips_net=lpips_net,
                num_lpips_slices=16,
                compute_lpips=use_lpips
            )
        
        # Loss weights (for ablation studies)
        self.loss_weights = {
            'diffusion': 1.0,
            'projection_single': 0.3,
            'projection_multi_view': 0.2,
            'projection_multi_scale': 0.1,
            'depth_consistency': 0.15,
            'cross_attention_align': 0.1,
            'stage_transition': 0.2,
            'perceptual': 0.1,
            'frequency_low': 0.05,
            'frequency_high': 0.05,
            'anatomical_prior': 0.1,
            # New feature-based losses
            'feature_mse': 0.15,
            'feature_cosine': 0.1,
            'feature_correlation': 0.05,
            'lpips': 0.2
        }
    
    def compute_all_losses(self,
                          predicted: torch.Tensor,
                          target: torch.Tensor,
                          pred_x0: torch.Tensor,
                          gt_x0: torch.Tensor,
                          xrays: torch.Tensor,
                          depth_prior: Optional[torch.Tensor] = None,
                          prev_stage_volume: Optional[torch.Tensor] = None,
                          attention_maps: Optional[Dict[str, torch.Tensor]] = None,
                          return_dict: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute all diagnostic losses
        
        Args:
            predicted: (B, 1, D, H, W) - predicted noise/velocity
            target: (B, 1, D, H, W) - target noise/velocity
            pred_x0: (B, 1, D, H, W) - predicted clean volume
            gt_x0: (B, 1, D, H, W) - ground truth clean volume
            xrays: (B, num_views, 1, H_xray, W_xray) - input X-rays
            depth_prior: (B, 1, D, H, W) - depth lifting prior
            prev_stage_volume: (B, 1, D_prev, H_prev, W_prev) - previous cascade stage
            attention_maps: Dict of attention weights from model
        """
        
        losses = {}
        
        # ============================================================
        # 1. DIFFUSION LOSS - Core denoising capability
        # ============================================================
        losses['diffusion'] = F.mse_loss(predicted, target)
        
        # ============================================================
        # 2. PROJECTION LOSSES - Physics constraint
        # ============================================================
        
        # 2a. Single-view projection
        drr_pred = self.drr_renderer(pred_x0.squeeze(1))  # (B, H, W)
        drr_gt = self.drr_renderer(gt_x0.squeeze(1))
        xray_target = xrays[:, 0, 0]  # First view
        
        if drr_pred.shape != xray_target.shape:
            drr_pred = F.interpolate(drr_pred.unsqueeze(1), 
                                    size=xray_target.shape[1:],
                                    mode='bilinear', align_corners=True).squeeze(1)
            drr_gt = F.interpolate(drr_gt.unsqueeze(1),
                                  size=xray_target.shape[1:],
                                  mode='bilinear', align_corners=True).squeeze(1)
        
        losses['projection_single'] = F.mse_loss(drr_pred, xray_target)
        
        # Diagnostic: Check if GT also matches (sanity check)
        losses['projection_gt_sanity'] = F.mse_loss(drr_gt, xray_target)
        
        # 2b. Multi-view projection (if available)
        if xrays.shape[1] > 1:
            # Lateral view (90° rotation)
            drr_lateral_pred = self.drr_renderer(pred_x0.squeeze(1), angle=90)
            xray_lateral = xrays[:, 1, 0]
            
            if drr_lateral_pred.shape != xray_lateral.shape:
                drr_lateral_pred = F.interpolate(drr_lateral_pred.unsqueeze(1),
                                                 size=xray_lateral.shape[1:],
                                                 mode='bilinear', align_corners=True).squeeze(1)
            
            losses['projection_multi_view'] = F.mse_loss(drr_lateral_pred, xray_lateral)
        else:
            losses['projection_multi_view'] = torch.tensor(0.0, device=predicted.device)
        
        # 2c. Multi-scale projection (coarse-to-fine)
        drr_pred_64 = F.interpolate(drr_pred.unsqueeze(1), size=(64, 64), 
                                     mode='bilinear', align_corners=True).squeeze(1)
        xray_64 = F.interpolate(xray_target.unsqueeze(1), size=(64, 64),
                               mode='bilinear', align_corners=True).squeeze(1)
        
        drr_pred_128 = F.interpolate(drr_pred.unsqueeze(1), size=(128, 128),
                                      mode='bilinear', align_corners=True).squeeze(1)
        xray_128 = F.interpolate(xray_target.unsqueeze(1), size=(128, 128),
                                mode='bilinear', align_corners=True).squeeze(1)
        
        losses['projection_multi_scale'] = (
            F.mse_loss(drr_pred_64, xray_64) +
            F.mse_loss(drr_pred_128, xray_128)
        ) / 2
        
        # ============================================================
        # 3. DEPTH CONSISTENCY - Is depth lifting helping?
        # ============================================================
        if depth_prior is not None:
            # Check if predicted volume correlates with depth prior
            # High correlation → depth prior is useful
            # Low correlation → depth prior being ignored
            
            # Cosine similarity in voxel space
            pred_flat = pred_x0.view(pred_x0.shape[0], -1)
            prior_flat = depth_prior.view(depth_prior.shape[0], -1)
            
            cosine_sim = F.cosine_similarity(pred_flat, prior_flat, dim=1).mean()
            
            # We want some correlation but not perfect (prior is just initialization)
            # Target: 0.3-0.6 correlation
            target_correlation = 0.45
            losses['depth_consistency'] = F.mse_loss(cosine_sim, 
                                                     torch.tensor(target_correlation, 
                                                                 device=cosine_sim.device))
            
            # Also check if depth prior reduces initial error
            prior_error = F.mse_loss(depth_prior, gt_x0)
            losses['depth_prior_quality'] = prior_error
            
        else:
            losses['depth_consistency'] = torch.tensor(0.0, device=predicted.device)
            losses['depth_prior_quality'] = torch.tensor(0.0, device=predicted.device)
        
        # ============================================================
        # 4. CROSS-ATTENTION ALIGNMENT - Is X-ray conditioning working?
        # ============================================================
        if attention_maps is not None and 'cross_attention' in attention_maps:
            # Check if cross-attention focuses on relevant X-ray regions
            
            cross_attn = attention_maps['cross_attention']  # (B, num_heads, D*H*W, N_xray)
            
            # Attention should be:
            # 1. Diverse (not all voxels attend to same X-ray token)
            # 2. Sparse (each voxel attends to few relevant tokens)
            
            # Diversity: Entropy of attention distribution
            attn_mean = cross_attn.mean(dim=1)  # (B, D*H*W, N_xray)
            attn_probs = F.softmax(attn_mean, dim=-1)
            
            entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1).mean()
            
            # Higher entropy → more diverse (good, but not too high)
            # Target: log(N_xray) * 0.6 (moderately diverse)
            target_entropy = torch.log(torch.tensor(attn_probs.shape[-1], 
                                                    dtype=torch.float32, 
                                                    device=entropy.device)) * 0.6
            
            losses['cross_attention_align'] = F.mse_loss(entropy, target_entropy)
            
            # Sparsity: L1 norm of attention (encourages peaky distribution)
            losses['cross_attention_sparsity'] = -attn_probs.max(dim=-1)[0].mean()
            
        else:
            losses['cross_attention_align'] = torch.tensor(0.0, device=predicted.device)
            losses['cross_attention_sparsity'] = torch.tensor(0.0, device=predicted.device)
        
        # ============================================================
        # 5. STAGE TRANSITION - Cascade coherence
        # ============================================================
        if prev_stage_volume is not None:
            # Upsampled previous stage should be close to current prediction
            # But current stage should add details (not just copy)
            
            prev_upsampled = F.interpolate(prev_stage_volume, 
                                          size=pred_x0.shape[2:],
                                          mode='trilinear', align_corners=True)
            
            # Structural similarity (low-frequency should match)
            pred_lowfreq = F.avg_pool3d(pred_x0, kernel_size=4, stride=1, padding=2)
            prev_lowfreq = F.avg_pool3d(prev_upsampled, kernel_size=4, stride=1, padding=2)
            
            losses['stage_transition'] = F.mse_loss(pred_lowfreq, prev_lowfreq)
            
            # But high-frequency should differ (adding details)
            pred_highfreq = pred_x0 - pred_lowfreq
            prev_highfreq = prev_upsampled - prev_lowfreq
            
            # We want high-freq to be different (negative loss encourages difference)
            losses['stage_detail_addition'] = -F.mse_loss(pred_highfreq, prev_highfreq)
            
        else:
            losses['stage_transition'] = torch.tensor(0.0, device=predicted.device)
            losses['stage_detail_addition'] = torch.tensor(0.0, device=predicted.device)
        
        # ============================================================
        # 6. FREQUENCY ANALYSIS - Structure vs Details
        # ============================================================
        if self.use_frequency:
            # Low-frequency (structure, anatomy) - proper downsampling then upsample
            pred_low = F.avg_pool3d(pred_x0, kernel_size=8, stride=8, padding=0)
            pred_low = F.interpolate(pred_low, size=pred_x0.shape[2:], mode='trilinear', align_corners=True)
            
            gt_low = F.avg_pool3d(gt_x0, kernel_size=8, stride=8, padding=0)
            gt_low = F.interpolate(gt_low, size=gt_x0.shape[2:], mode='trilinear', align_corners=True)
            
            losses['frequency_low'] = F.mse_loss(pred_low, gt_low)
            
            # High-frequency (details, edges)
            pred_high = pred_x0 - pred_low
            gt_high = gt_x0 - gt_low
            losses['frequency_high'] = F.mse_loss(pred_high, gt_high)
            
            # Diagnostic: Which is failing?
            # If frequency_low high → struggling with anatomy
            # If frequency_high high → missing fine details
            
        else:
            losses['frequency_low'] = torch.tensor(0.0, device=predicted.device)
            losses['frequency_high'] = torch.tensor(0.0, device=predicted.device)
        
        # ============================================================
        # 7. PERCEPTUAL LOSS - Semantic similarity
        # ============================================================
        if hasattr(self, 'perceptual_net'):
            feat_pred = self.perceptual_net(pred_x0)
            feat_gt = self.perceptual_net(gt_x0)
            
            losses['perceptual'] = F.mse_loss(feat_pred, feat_gt)
        else:
            losses['perceptual'] = torch.tensor(0.0, device=predicted.device)
        
        # ============================================================
        # 8. ANATOMICAL PRIOR - Are priors being utilized?
        # ============================================================
        if depth_prior is not None:
            # Check if prediction improves over prior
            prior_error = F.mse_loss(depth_prior, gt_x0)
            pred_error = F.mse_loss(pred_x0, gt_x0)
            
            # Improvement ratio
            improvement = (prior_error - pred_error) / (prior_error + 1e-8)
            
            # We want positive improvement (prediction better than prior)
            # But if improvement < 0, prior is being ignored
            losses['anatomical_prior'] = F.relu(-improvement)  # Penalize negative improvement
            
            losses['prior_improvement_ratio'] = improvement.detach()
        else:
            losses['anatomical_prior'] = torch.tensor(0.0, device=predicted.device)
            losses['prior_improvement_ratio'] = torch.tensor(0.0, device=predicted.device)
        
        # ============================================================
        # 9. FEATURE MAP ACCURACY & LPIPS - New additions
        # ============================================================
        if self.use_feature_metrics or self.use_lpips:
            # Compute comprehensive feature metrics
            feature_metrics = self.comprehensive_metrics(
                base_ct=gt_x0,
                generated_ct=pred_x0,
                compute_lpips=self.use_lpips
            )
            
            # Extract key metrics for loss computation
            if self.use_feature_metrics:
                losses['feature_mse'] = feature_metrics['overall_feature_mse']
                losses['feature_cosine'] = 1.0 - feature_metrics['overall_feature_cosine']  # Convert to loss
                losses['feature_correlation'] = 1.0 - feature_metrics['overall_feature_correlation']  # Convert to loss
                losses['feature_ssim'] = 1.0 - feature_metrics['overall_feature_ssim']  # Convert to loss
                losses['feature_style'] = feature_metrics['overall_feature_style']
                
                # Store per-level metrics for diagnostics (not used in total loss)
                for k, v in feature_metrics.items():
                    if k.startswith('level_'):
                        losses[f'diagnostic_{k}'] = v
            else:
                losses['feature_mse'] = torch.tensor(0.0, device=predicted.device)
                losses['feature_cosine'] = torch.tensor(0.0, device=predicted.device)
                losses['feature_correlation'] = torch.tensor(0.0, device=predicted.device)
                losses['feature_ssim'] = torch.tensor(0.0, device=predicted.device)
                losses['feature_style'] = torch.tensor(0.0, device=predicted.device)
            
            # LPIPS loss
            if self.use_lpips:
                losses['lpips'] = feature_metrics['lpips_average']
                losses['lpips_axial'] = feature_metrics['lpips_axial']
                losses['lpips_coronal'] = feature_metrics['lpips_coronal']
                losses['lpips_sagittal'] = feature_metrics['lpips_sagittal']
            else:
                losses['lpips'] = torch.tensor(0.0, device=predicted.device)
        else:
            losses['feature_mse'] = torch.tensor(0.0, device=predicted.device)
            losses['feature_cosine'] = torch.tensor(0.0, device=predicted.device)
            losses['feature_correlation'] = torch.tensor(0.0, device=predicted.device)
            losses['feature_ssim'] = torch.tensor(0.0, device=predicted.device)
            losses['feature_style'] = torch.tensor(0.0, device=predicted.device)
            losses['lpips'] = torch.tensor(0.0, device=predicted.device)
        
        # ============================================================
        # WEIGHTED TOTAL LOSS
        # ============================================================
        total_loss = torch.tensor(0.0, device=predicted.device)
        
        for loss_name, loss_value in losses.items():
            if loss_name in self.loss_weights and not loss_name.endswith('_sanity'):
                total_loss += self.loss_weights[loss_name] * loss_value
        
        losses['total'] = total_loss
        
        if return_dict:
            return losses
        
        return total_loss
    
    def analyze_component_health(self, losses: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """
        Diagnose which components are working/failing
        
        Returns status for each component:
        - 'EXCELLENT': Working very well
        - 'GOOD': Working as expected
        - 'WARNING': Underperforming
        - 'CRITICAL': Major failure
        """
        
        health = {}
        
        # Diffusion (core denoising)
        if losses['diffusion'] < 0.01:
            health['denoising'] = 'EXCELLENT'
        elif losses['diffusion'] < 0.05:
            health['denoising'] = 'GOOD'
        elif losses['diffusion'] < 0.1:
            health['denoising'] = 'WARNING'
        else:
            health['denoising'] = 'CRITICAL'
        
        # Physics constraint
        if losses['projection_single'] < 0.005:
            health['physics'] = 'EXCELLENT'
        elif losses['projection_single'] < 0.02:
            health['physics'] = 'GOOD'
        elif losses['projection_single'] < 0.05:
            health['physics'] = 'WARNING'
        else:
            health['physics'] = 'CRITICAL'
        
        # Depth lifting
        if 'depth_consistency' in losses and losses['depth_consistency'] > 0:
            depth_corr = 0.45 - losses['depth_consistency'].sqrt()  # Approximate correlation
            if depth_corr > 0.5:
                health['depth_lifting'] = 'EXCELLENT'
            elif depth_corr > 0.3:
                health['depth_lifting'] = 'GOOD'
            elif depth_corr > 0.1:
                health['depth_lifting'] = 'WARNING'
            else:
                health['depth_lifting'] = 'CRITICAL - Prior being ignored'
        
        # Cross-attention
        if 'cross_attention_align' in losses and losses['cross_attention_align'] > 0:
            if losses['cross_attention_align'] < 0.1:
                health['cross_attention'] = 'EXCELLENT'
            elif losses['cross_attention_align'] < 0.3:
                health['cross_attention'] = 'GOOD'
            elif losses['cross_attention_align'] < 0.5:
                health['cross_attention'] = 'WARNING'
            else:
                health['cross_attention'] = 'CRITICAL - Attention collapsed'
        
        # Frequency analysis
        if 'frequency_low' in losses and 'frequency_high' in losses:
            low_err = losses['frequency_low']
            high_err = losses['frequency_high']
            
            if low_err > 2 * high_err:
                health['structure_vs_details'] = 'WARNING - Struggling with anatomy'
            elif high_err > 2 * low_err:
                health['structure_vs_details'] = 'WARNING - Missing fine details'
            else:
                health['structure_vs_details'] = 'GOOD - Balanced'
        
        # Cascade transition
        if 'stage_transition' in losses and losses['stage_transition'] > 0:
            if losses['stage_transition'] < 0.01:
                health['cascade'] = 'EXCELLENT - Smooth transition'
            elif losses['stage_transition'] < 0.05:
                health['cascade'] = 'GOOD'
            elif losses['stage_transition'] < 0.1:
                health['cascade'] = 'WARNING - Stages disconnected'
            else:
                health['cascade'] = 'CRITICAL - Cascade not coherent'
        
        # Feature map accuracy
        if 'feature_mse' in losses and losses['feature_mse'] > 0:
            if losses['feature_mse'] < 0.01:
                health['feature_accuracy'] = 'EXCELLENT - Features match well'
            elif losses['feature_mse'] < 0.05:
                health['feature_accuracy'] = 'GOOD'
            elif losses['feature_mse'] < 0.1:
                health['feature_accuracy'] = 'WARNING - Feature mismatch'
            else:
                health['feature_accuracy'] = 'CRITICAL - Features very different'
        
        # LPIPS perceptual similarity
        if 'lpips' in losses and losses['lpips'] > 0:
            lpips_val = losses['lpips'].item() if torch.is_tensor(losses['lpips']) else losses['lpips']
            if lpips_val < 0.1:
                health['perceptual_similarity'] = 'EXCELLENT - Perceptually identical'
            elif lpips_val < 0.3:
                health['perceptual_similarity'] = 'GOOD'
            elif lpips_val < 0.5:
                health['perceptual_similarity'] = 'WARNING - Perceptual differences'
            else:
                health['perceptual_similarity'] = 'CRITICAL - Very different perceptually'
        
        return health


class Simple3DPerceptualNet(nn.Module):
    """Simple 3D feature extractor for perceptual loss"""
    
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )
    
    def forward(self, x):
        return self.features(x).view(x.shape[0], -1)


# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Diagnostic Losses...")
    
    # Create diagnostic loss module
    diagnostic = DiagnosticLosses(
        volume_size=(64, 64, 64),
        use_perceptual=True,
        use_frequency=True,
        use_feature_metrics=True,
        use_lpips=True
    ).to(device)
    
    # Dummy data
    batch_size = 2
    predicted = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    target = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    pred_x0 = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    gt_x0 = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    xrays = torch.randn(batch_size, 2, 1, 512, 512).to(device)
    depth_prior = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    
    # Compute all losses
    losses = diagnostic.compute_all_losses(
        predicted=predicted,
        target=target,
        pred_x0=pred_x0,
        gt_x0=gt_x0,
        xrays=xrays,
        depth_prior=depth_prior
    )
    
    print("\n=== Loss Breakdown ===")
    for loss_name, loss_value in losses.items():
        if not loss_name.endswith('_ratio'):
            print(f"{loss_name:30s}: {loss_value.item():.6f}")
    
    # Health analysis
    health = diagnostic.analyze_component_health(losses)
    
    print("\n=== Component Health Analysis ===")
    for component, status in health.items():
        print(f"{component:30s}: {status}")
    
    print("\n=== Interpretation Guide ===")
    print("• diffusion > 0.1 → Core denoising failing")
    print("• projection_single > 0.05 → Physics constraint not working")
    print("• depth_consistency → correlation with prior (target: ~0.45)")
    print("• frequency_low >> frequency_high → Struggling with anatomy")
    print("• frequency_high >> frequency_low → Missing fine details")
    print("• stage_transition > 0.1 → Cascade stages disconnected")
    print("• prior_improvement_ratio < 0 → Depth prior being ignored")
    print("• feature_mse > 0.1 → Feature representations very different")
    print("• lpips > 0.5 → Perceptually very different (0=identical, 1=max diff)")
    print("• feature_cosine < 0.5 → Feature directions misaligned")
    print("• feature_style > 0.01 → Texture/pattern mismatch")
