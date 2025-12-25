"""
Unified Hybrid-ViT Cascading Model
Combines the best of both worlds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Import components with relative imports
from .cascaded_depth_lifting import CascadedDepthLifting, ResolutionDepthPriors
from .hybrid_vit_backbone import HybridViT3D
from .diagnostic_losses import XrayConditioningModule, DRRRenderer, ProjectionLoss


class UnifiedCascadeStage(nn.Module):
    """
    Single stage in unified hybrid-ViT cascade
    
    Flow:
    X-ray → Encoder → Depth Lifting → ViT 3D → Denoised Volume
                                        ↓
                              Physics Loss (DRR projection)
    """
    
    def __init__(self,
                 stage_name: str,
                 volume_size: Tuple[int, int, int],
                 in_channels: int = 1,
                 voxel_dim: int = 384,
                 vit_depth: int = 6,
                 num_heads: int = 6,
                 xray_feature_dim: int = 512,
                 use_prev_stage: bool = False,
                 use_depth_lifting: bool = True,
                 use_physics_loss: bool = True):
        super().__init__()
        
        self.stage_name = stage_name
        self.volume_size = volume_size
        self.use_prev_stage = use_prev_stage
        self.use_depth_lifting = use_depth_lifting
        self.use_physics_loss = use_physics_loss
        
        D, H, W = volume_size
        
        # Depth lifting (if enabled)
        if use_depth_lifting:
            self.depth_lifter = CascadedDepthLifting(
                feature_dim=xray_feature_dim,
                depth_sizes=[D],
                use_prev_stage=use_prev_stage
            )
            # Project depth features to 1 channel to match noisy_volume
            self.depth_to_volume = nn.Conv3d(xray_feature_dim, in_channels, kernel_size=1)
        
        # Hybrid-ViT backbone
        self.vit_backbone = HybridViT3D(
            volume_size=volume_size,
            in_channels=in_channels,
            voxel_dim=voxel_dim,
            depth=vit_depth,
            num_heads=num_heads,
            context_dim=xray_feature_dim,
            cond_dim=1024,  # time + xray embedding
            use_prev_stage=use_prev_stage
        )
        
        # Physics loss (DRR renderer)
        if use_physics_loss:
            self.drr_renderer = DRRRenderer(volume_shape=(D, H, W))
    
    def forward(self,
                noisy_volume: torch.Tensor,
                xray_features: torch.Tensor,
                xray_context: torch.Tensor,
                time_xray_cond: torch.Tensor,
                prev_stage_volume: Optional[torch.Tensor] = None,
                prev_stage_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            noisy_volume: (B, 1, D, H, W) - noisy input
            xray_features: (B, C, H_xray, W_xray) - X-ray features for depth lifting
            xray_context: (B, C, H_xray, W_xray) - X-ray features for cross-attention
            time_xray_cond: (B, cond_dim) - time + xray embedding for AdaLN
            prev_stage_volume: (B, 1, D_prev, H_prev, W_prev) - previous stage output
            prev_stage_embed: (B, 256) - previous stage embedding
        Returns:
            predicted: (B, 1, D, H, W) - predicted noise/velocity
        """
        batch_size = noisy_volume.shape[0]
        D, H, W = self.volume_size
        
        # Reshape xray_context for cross-attention: (B, C, H, W) -> (B, H*W, C)
        B, C, H_x, W_x = xray_context.shape
        xray_context_seq = xray_context.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Optional: Add depth prior as initial signal
        if self.use_depth_lifting:
            depth_prior = self.depth_lifter(
                xray_features,
                target_depth=D,
                prev_stage_volume=prev_stage_volume
            )
            # Project to 1 channel and combine with noisy volume
            depth_prior = self.depth_to_volume(depth_prior)
            
            # Ensure depth_prior matches noisy_volume size
            if depth_prior.shape != noisy_volume.shape:
                depth_prior = F.interpolate(depth_prior, size=(D, H, W), 
                                           mode='trilinear', align_corners=True)
            
            noisy_volume = noisy_volume + 0.1 * depth_prior  # Weak prior
        
        # ViT denoising
        predicted = self.vit_backbone(
            noisy_volume,
            xray_context_seq,  # Pass reshaped context
            time_xray_cond,
            prev_stage_embed
        )
        
        return predicted


class UnifiedHybridViTCascade(nn.Module):
    """
    Complete unified model with multi-stage cascading
    """
    
    def __init__(self,
                 stage_configs: List[Dict],
                 xray_img_size: int = 512,
                 xray_channels: int = 1,
                 num_views: int = 2,
                 xray_embed_dim: int = 512,
                 time_embed_dim: int = 256,
                 num_timesteps: int = 1000,
                 v_parameterization: bool = True,
                 share_view_weights: bool = False):
        super().__init__()
        
        self.stage_configs = stage_configs
        self.num_stages = len(stage_configs)
        self.v_parameterization = v_parameterization
        self.num_timesteps = num_timesteps
        self.num_views = num_views  # Store for reference
        self.time_embed_dim = time_embed_dim
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # X-ray encoder (shared across all stages)
        # Now supports both single-view (num_views=1) and multi-view (num_views=2+)
        self.xray_encoder = XrayConditioningModule(
            img_size=xray_img_size,
            in_channels=xray_channels,
            embed_dim=xray_embed_dim,
            num_views=num_views,
            time_embed_dim=time_embed_dim,
            cond_dim=1024,
            share_view_weights=share_view_weights
        )
        
        # Create cascade stages
        self.stages = nn.ModuleDict()
        for i, config in enumerate(stage_configs):
            stage_name = config['name']
            self.stages[stage_name] = UnifiedCascadeStage(
                stage_name=stage_name,
                volume_size=tuple(config['volume_size']),
                in_channels=1,
                voxel_dim=config['voxel_dim'],
                vit_depth=config['vit_depth'],
                num_heads=config['num_heads'],
                xray_feature_dim=xray_embed_dim,
                use_prev_stage=(i > 0),
                use_depth_lifting=config.get('use_depth_lifting', True),
                use_physics_loss=config.get('use_physics_loss', True)
            )
        
        # Store stage names and sizes
        self.stage_names = [config['name'] for config in stage_configs]
        self.stage_sizes = {config['name']: tuple(config['volume_size']) 
                           for config in stage_configs}
        
        # Pre-register prev_stage_projectors for stages 2+
        self.prev_stage_projectors = nn.ModuleDict()
        for i, config in enumerate(stage_configs):
            if i > 0:  # Skip first stage
                stage_name = config['name']
                self.prev_stage_projectors[stage_name] = nn.Linear(1, 256)
        
        # Register noise schedule
        self.register_noise_schedule()
    
    def register_noise_schedule(self, schedule_type='cosine'):
        """Register diffusion noise schedule"""
        import math
        
        if schedule_type == 'cosine':
            s = 0.008
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
    
    def q_sample(self, x_start, t, noise=None):
        """Add noise (forward diffusion)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        
        return sqrt_alphas_t * x_start + sqrt_one_minus_alphas_t * noise
    
    def get_v_target(self, x_start, noise, t):
        """Compute v-parameterization target: v = alpha*noise - sigma*x_start"""
        sqrt_alphas_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        
        return sqrt_alphas_t * noise - sqrt_one_minus_alphas_t * x_start
    
    def forward(self,
                x_start: torch.Tensor,
                xrays: torch.Tensor,
                stage_name: str,
                prev_stage_volume: Optional[torch.Tensor] = None,
                return_loss_dict: bool = True) -> Dict[str, torch.Tensor]:
        """
        Training forward pass
        
        Args:
            x_start: (B, 1, D, H, W) - clean volume at stage resolution
            xrays: (B, num_views, 1, H, W) for multi-view OR
                   (B, 1, H, W) for single-view - X-ray images
            stage_name: Which stage to train
            prev_stage_volume: Previous stage output (if not first stage)
            return_loss_dict: Return detailed losses
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Add noise
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Create timestep embeddings
        t_normalized = t.float() / self.num_timesteps  # Normalize to [0, 1]
        t_embed = self.time_embed(t_normalized.unsqueeze(-1))  # (B, time_embed_dim)
        
        # Encode X-rays - now returns 3 outputs
        xray_context, time_xray_cond, xray_features_2d = self.xray_encoder(xrays, t_embed)
        
        # Get stage
        stage = self.stages[stage_name]
        
        # Extract previous stage embedding if needed
        prev_stage_embed = None
        if prev_stage_volume is not None and stage.use_prev_stage:
            # Global average pooling over 3D volume
            prev_embed_raw = F.adaptive_avg_pool3d(prev_stage_volume, (1, 1, 1))
            prev_embed_raw = prev_embed_raw.view(batch_size, -1)  # (B, 1)
            
            # Project to 256 dimensions using pre-registered projector
            if stage_name in self.prev_stage_projectors:
                prev_stage_embed = self.prev_stage_projectors[stage_name](prev_embed_raw)
        
        # Forward through stage
        predicted = stage(
            noisy_volume=x_noisy,
            xray_features=xray_features_2d,
            xray_context=xray_features_2d,  # Use 2D features for cross-attention
            time_xray_cond=time_xray_cond,
            prev_stage_volume=prev_stage_volume,
            prev_stage_embed=prev_stage_embed
        )
        
        # Compute target
        if self.v_parameterization:
            target = self.get_v_target(x_start, noise, t)
        else:
            target = noise
        
        # Simple MSE loss
        diffusion_loss = F.mse_loss(predicted, target)
        
        # Physics loss (if enabled)
        physics_loss = torch.tensor(0.0, device=device)
        if stage.use_physics_loss and self.training:
            # Predict x_0 from predicted v/noise
            if self.v_parameterization:
                sqrt_alphas_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                sqrt_one_minus_alphas_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                pred_x_start = sqrt_alphas_t * x_noisy - sqrt_one_minus_alphas_t * predicted
            else:
                sqrt_alphas_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                sqrt_one_minus_alphas_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                # Clamp to prevent division by very small numbers
                sqrt_alphas_t = torch.clamp(sqrt_alphas_t, min=1e-8)
                pred_x_start = (x_noisy - sqrt_one_minus_alphas_t * predicted) / sqrt_alphas_t
            
            # Clamp predicted volume to reasonable range to prevent extreme DRR values
            pred_x_start = torch.clamp(pred_x_start, -10.0, 10.0)
            
            # Multi-view DRR loss (support both single and multi-view)
            num_views = xrays.shape[1]
            view_losses = []
            
            for view_idx in range(num_views):
                # Render DRR (squeeze channel dimension for renderer)
                drr_pred = stage.drr_renderer(pred_x_start.squeeze(1))  # (B, H, W)
                xray_target = xrays[:, view_idx, 0]  # Current view
                
                # Downsample if needed
                if drr_pred.shape != xray_target.shape:
                    drr_pred = F.interpolate(drr_pred.unsqueeze(1), size=xray_target.shape[1:], 
                                            mode='bilinear', align_corners=True).squeeze(1)
                
                view_loss = F.mse_loss(drr_pred, xray_target)
                view_losses.append(view_loss)
            
            # Average loss across all views
            physics_loss = sum(view_losses) / len(view_losses)
        
        # Total loss
        physics_weight = 0.3  # Can be stage-dependent
        total_loss = diffusion_loss + physics_weight * physics_loss
        
        if return_loss_dict:
            return {
                'loss': total_loss,
                'diffusion_loss': diffusion_loss,
                'physics_loss': physics_loss
            }
        
        return total_loss


# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Unified Hybrid-ViT Cascade Model...")
    
    # Define stages
    stage_configs = [
        {
            'name': 'stage1_low',
            'volume_size': [64, 64, 64],
            'voxel_dim': 256,
            'vit_depth': 4,
            'num_heads': 4,
            'use_depth_lifting': True,
            'use_physics_loss': True
        },
        {
            'name': 'stage2_mid',
            'volume_size': [128, 128, 128],
            'voxel_dim': 384,
            'vit_depth': 6,
            'num_heads': 6,
            'use_depth_lifting': True,
            'use_physics_loss': True
        }
    ]
    
    # Create model
    model = UnifiedHybridViTCascade(
        stage_configs=stage_configs,
        xray_img_size=512,
        v_parameterization=True
    ).to(device)
    
    # Test Stage 1
    print("\n=== Testing Stage 1 ===")
    batch_size = 2
    x_clean = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    xrays = torch.randn(batch_size, 2, 1, 512, 512).to(device)
    
    loss_dict = model(x_clean, xrays, 'stage1_low')
    
    print(f"Total loss: {loss_dict['loss'].item():.6f}")
    print(f"Diffusion loss: {loss_dict['diffusion_loss'].item():.6f}")
    print(f"Physics loss: {loss_dict['physics_loss'].item():.6f}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Model Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB")
    
    print("\nUnified model test completed!")
