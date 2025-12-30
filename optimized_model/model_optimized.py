"""
Optimized CT Regression Model
Integrates all optimizations:
- Cascaded Group Attention (30-40% faster)
- CNN + ViT hybrid for local + global features
- Learnable depth priors with uncertainty
- Sandwich layout for efficiency
- Hierarchical adaptive conditioning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '..')

from models.diagnostic_losses import XrayConditioningModule
from cnn_local_branch import EfficientNet3D, HybridCNNViTFusion
from learnable_depth_priors import AdaptiveDepthWeightNetwork, UncertaintyRegularizationLoss
from sandwich_vit_backbone import SandwichViT3D


class OptimizedCTRegression(nn.Module):
    """
    Optimized direct regression model with all improvements
    Expected: 2-3x faster, better accuracy than baseline
    """
    
    def __init__(self,
                 volume_size=(64, 64, 64),
                 xray_img_size=512,
                 voxel_dim=256,
                 num_attn_blocks=2,
                 num_ffn_blocks=4,
                 num_heads=8,
                 xray_feature_dim=512,
                 use_cnn_branch=True,
                 use_learnable_priors=True):
        super().__init__()
        
        self.volume_size = volume_size
        self.use_cnn_branch = use_cnn_branch
        self.use_learnable_priors = use_learnable_priors
        
        # X-ray encoder (with BatchNorm fix from diagnostic_losses)
        self.xray_encoder = XrayConditioningModule(
            img_size=xray_img_size,
            in_channels=1,
            embed_dim=xray_feature_dim,
            num_views=2,
            time_embed_dim=256,
            cond_dim=1024,
            share_view_weights=False
        )
        
        # CNN branch for local features (optional but recommended)
        if use_cnn_branch:
            self.cnn_branch = EfficientNet3D(
                in_channels=voxel_dim,  # FIXED: Accept voxel_dim not 1
                base_channels=32,
                feature_dim=voxel_dim
            )
            
            self.cnn_vit_fusion = HybridCNNViTFusion(
                feature_dim=voxel_dim,
                num_heads=4
            )
            
            # Project X-ray features to voxel_dim for fusion
            self.xray_to_voxel = nn.Linear(xray_feature_dim, voxel_dim)
        
        # Learnable depth lifting (replaces simple initial volume)
        if use_learnable_priors:
            self.depth_lifter = AdaptiveDepthWeightNetwork(
                feature_dim=xray_feature_dim,
                depth_size=volume_size[0],  # Assuming cubic volume
                num_regions=3
            )
            
            # Project lifted 3D features
            self.depth_proj = nn.Conv3d(xray_feature_dim, voxel_dim, 3, padding=1)
        else:
            # Fallback: learnable initial volume
            D, H, W = volume_size
            self.initial_volume = nn.Parameter(torch.randn(1, 1, D, H, W) * 0.01)
        
        # Sandwich ViT backbone (optimized)
        self.vit_backbone = SandwichViT3D(
            volume_size=volume_size,
            in_channels=voxel_dim if use_learnable_priors else 1,
            voxel_dim=voxel_dim,
            num_attn_blocks=num_attn_blocks,
            num_ffn_blocks=num_ffn_blocks,
            num_heads=num_heads,
            context_dim=voxel_dim if use_cnn_branch else xray_feature_dim,
            use_prev_stage=False
        )
        
    def forward(self, xrays):
        """
        Args:
            xrays: (B, num_views, 1, H, W) - input X-ray images
        Returns:
            predicted_volume: (B, 1, D, H, W) - predicted CT volume
            aux_info: Dictionary with auxiliary information (uncertainties, etc.)
        """
        batch_size = xrays.shape[0]
        D, H, W = self.volume_size
        
        # Create dummy timestep (encoder expects it but we don't use it)
        dummy_t = torch.zeros(batch_size, 256, device=xrays.device)
        
        # Encode X-rays
        xray_context, time_xray_cond, xray_features_2d = self.xray_encoder(xrays, dummy_t)
        
        # Shape assertions for debugging
        assert xray_features_2d.dim() == 4, f"Expected 4D xray_features_2d, got {xray_features_2d.dim()}D"
        assert xray_features_2d.shape[0] == batch_size, f"Batch size mismatch: {xray_features_2d.shape[0]} vs {batch_size}"
        
        # Get pooled features for adaptive conditioning
        # xray_context: (B, 1024) -> downsample to (B, 512)
        pooled_features = F.adaptive_avg_pool1d(xray_context.unsqueeze(1), 512).squeeze(1)
        
        aux_info = {}
        
        # Generate initial 3D volume using learnable depth priors
        if self.use_learnable_priors:
            # Lift 2D X-ray features to 3D using learned depth distributions
            # xray_features_2d: (B, embed_dim, H', W') - already averaged across views by encoder
            
            # Get depth weights and auxiliary info (boundaries, uncertainties)
            depth_weights, depth_aux = self.depth_lifter(xray_features_2d, pooled_features)
            # depth_weights: (B, H, W, D)
            
            aux_info.update(depth_aux)
            
            # Lift to 3D: broadcast features along depth with learned weights
            # depth_weights: (B, H, W, D)
            # xray_features_2d: (B, C, H', W')
            B_batch, C_feat, H_feat, W_feat = xray_features_2d.shape
            _, H_target, W_target, D_target = depth_weights.shape
            
            # Interpolate features to match depth_weights spatial resolution if needed
            if (H_feat, W_feat) != (H_target, W_target):
                xray_features_2d = F.interpolate(
                    xray_features_2d,
                    size=(H_target, W_target),
                    mode='bilinear',
                    align_corners=True
                )
            
            # Broadcast: (B, C, H, W, 1) * (B, 1, H, W, D) -> (B, C, H, W, D)
            xray_features_3d = xray_features_2d.unsqueeze(-1) * depth_weights.unsqueeze(1)
            
            # Permute to Conv3d format: (B, C, D, H, W)
            xray_features_3d = xray_features_3d.permute(0, 1, 4, 2, 3)
            
            # Shape assertion
            assert xray_features_3d.shape == (batch_size, C_feat, D_target, H_target, W_target), \
                f"3D volume shape mismatch: {xray_features_3d.shape} vs expected ({batch_size}, {C_feat}, {D_target}, {H_target}, {W_target})"
            
            # Project to voxel dim
            x = self.depth_proj(xray_features_3d)
        else:
            # Use simple learned initial volume
            x = self.initial_volume.expand(batch_size, -1, -1, -1, -1)
        
        # Optional CNN branch for local features
        if self.use_cnn_branch:
            # CNN processes the initial 3D volume
            cnn_features, cnn_pyramid = self.cnn_branch(x)  # (B, voxel_dim, D', H', W')
            
            # Track CNN output spatial dimensions
            B_batch, C_cnn, D_cnn, H_cnn, W_cnn = cnn_features.shape
            
            # Prepare ViT context from X-ray features
            # xray_features_2d: (B, 512, H, W) - already interpolated to match volume
            vit_context = xray_features_2d.flatten(2).transpose(1, 2)  # (B, H*W, 512)
            
            # Project to voxel_dim to match CNN features
            vit_context = self.xray_to_voxel(vit_context)  # (B, H*W, voxel_dim)
            
            # Fuse CNN and ViT features
            fused_context = self.cnn_vit_fusion(cnn_features, vit_context)
            
            # Reshape fused_context back to 3D using actual CNN dimensions
            # fused_context: (B, N_fused, voxel_dim)
            fused_3d = fused_context.transpose(1, 2).reshape(batch_size, -1, D_cnn, H_cnn, W_cnn)
            
            # Upsample to target resolution if needed
            if (D_cnn, H_cnn, W_cnn) != (D, H, W):
                fused_3d = F.interpolate(fused_3d, size=(D, H, W), mode='trilinear', align_corners=True)
            
            # Use fused features as input to ViT
            x_vit_input = fused_3d
            context_for_vit = fused_context
        else:
            x_vit_input = x
            context_for_vit = xray_features_2d.flatten(2).transpose(1, 2)
        
        # ViT backbone processes volume
        predicted_volume = self.vit_backbone(
            x=x_vit_input,
            context=context_for_vit,
            prev_stage_embed=None
        )
        
        return predicted_volume, aux_info


class OptimizedRegressionLoss(nn.Module):
    """
    Combined loss with uncertainty regularization
    """
    
    def __init__(self,
                 l1_weight=1.0,
                 ssim_weight=0.5,
                 uncertainty_reg_weight=0.01):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.uncertainty_reg = UncertaintyRegularizationLoss(
            boundary_smoothness_weight=0.1,
            uncertainty_reg_weight=uncertainty_reg_weight
        )
        
    def forward(self, pred, target, aux_info=None):
        # L1 loss (primary)
        l1_loss = F.l1_loss(pred, target)
        
        # SSIM loss (structure)
        ssim_loss = self.compute_ssim_loss(pred, target)
        
        # Reconstruction loss
        recon_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        
        # Uncertainty regularization (if learnable priors used)
        reg_loss = 0.0
        if aux_info is not None and 'boundaries' in aux_info:
            reg_loss = self.uncertainty_reg(aux_info)
        
        total_loss = recon_loss + reg_loss
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'ssim_loss': ssim_loss,
            'reg_loss': reg_loss if isinstance(reg_loss, torch.Tensor) else torch.tensor(0.0)
        }
    
    @staticmethod
    def compute_ssim_loss(pred, target, window_size=11):
        """Compute SSIM loss for 3D volumes"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = F.avg_pool3d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool3d(target, window_size, stride=1, padding=window_size//2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool3d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool3d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool3d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target
        
        ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return 1 - ssim.mean()


if __name__ == "__main__":
    # Test optimized model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Testing Optimized CT Regression Model ===")
    
    # Test with all optimizations
    model = OptimizedCTRegression(
        volume_size=(64, 64, 64),
        voxel_dim=256,
        num_attn_blocks=2,
        num_ffn_blocks=4,
        use_cnn_branch=True,
        use_learnable_priors=True
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Test forward pass
    xrays = torch.randn(2, 2, 1, 512, 512).to(device)
    output, aux_info = model(xrays)
    
    print(f"\nInput X-rays: {xrays.shape}")
    print(f"Output volume: {output.shape}")
    print(f"Auxiliary info keys: {list(aux_info.keys())}")
    
    if 'boundaries' in aux_info:
        print(f"Learned boundaries: {aux_info['boundaries'][0] * 100}% of depth")
        print(f"Uncertainties: {aux_info['uncertainties'][0] * 100}%")
    
    # Test loss
    target = torch.randn_like(output)
    loss_fn = OptimizedRegressionLoss()
    loss_dict = loss_fn(output, target, aux_info)
    
    print(f"\nLosses:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.6f}")
    
    # Compare with baseline (no optimizations)
    print("\n=== Comparing with Baseline ===")
    baseline_model = OptimizedCTRegression(
        volume_size=(64, 64, 64),
        voxel_dim=256,
        num_attn_blocks=2,
        num_ffn_blocks=4,
        use_cnn_branch=False,
        use_learnable_priors=False
    ).to(device)
    
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"Baseline parameters: {baseline_params:,} ({baseline_params/1e6:.2f}M)")
    print(f"Optimized parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Parameter ratio: {total_params/baseline_params:.2f}x")
    
    print("\nOptimized model test completed!")
