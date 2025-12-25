"""
Feature Map Extraction and Accuracy Metrics
Provides multi-level feature extraction and comparison for CT volumes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import lpips


class MultiLevelFeatureExtractor(nn.Module):
    """
    Extract feature maps at multiple scales/levels from 3D CT volumes
    Useful for comparing base CT to generated CT at different abstraction levels
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 feature_dims: List[int] = [32, 64, 128, 256],
                 return_all_layers: bool = True):
        """
        Args:
            in_channels: Number of input channels (typically 1 for CT)
            feature_dims: Feature dimensions at each level
            return_all_layers: Return features from all layers (True) or just final (False)
        """
        super().__init__()
        
        self.return_all_layers = return_all_layers
        self.feature_dims = feature_dims
        
        # Build encoder layers
        self.layers = nn.ModuleList()
        prev_dim = in_channels
        
        for i, dim in enumerate(feature_dims):
            layer = nn.Sequential(
                nn.Conv3d(prev_dim, dim, kernel_size=3, padding=1, stride=2 if i > 0 else 1),
                nn.GroupNorm(8, dim),
                nn.ReLU(inplace=True),
                nn.Conv3d(dim, dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, dim),
                nn.ReLU(inplace=True)
            )
            self.layers.append(layer)
            prev_dim = dim
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features at multiple levels
        
        Args:
            x: (B, C, D, H, W) - input volume
            
        Returns:
            features: Dict with keys 'level_0', 'level_1', etc.
        """
        features = {}
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            features[f'level_{i}'] = x
        
        return features if self.return_all_layers else {f'level_{len(self.layers)-1}': x}


class FeatureMapAccuracy(nn.Module):
    """
    Compute accuracy metrics for feature maps between base CT and generated CT
    """
    
    def __init__(self, 
                 feature_dims: List[int] = [32, 64, 128, 256],
                 normalize_features: bool = True):
        """
        Args:
            feature_dims: Feature dimensions at each level (must match extractor)
            normalize_features: L2 normalize features before comparison
        """
        super().__init__()
        
        self.feature_extractor = MultiLevelFeatureExtractor(
            in_channels=1,
            feature_dims=feature_dims,
            return_all_layers=True
        )
        self.normalize_features = normalize_features
        
    def compute_feature_mse(self, 
                           feat_base: torch.Tensor, 
                           feat_gen: torch.Tensor) -> torch.Tensor:
        """Compute MSE between feature maps"""
        return F.mse_loss(feat_base, feat_gen)
    
    def compute_feature_cosine(self, 
                              feat_base: torch.Tensor, 
                              feat_gen: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between feature maps"""
        if self.normalize_features:
            feat_base = F.normalize(feat_base, p=2, dim=1)
            feat_gen = F.normalize(feat_gen, p=2, dim=1)
        
        # Compute per-channel cosine similarity
        cos_sim = (feat_base * feat_gen).sum(dim=1).mean()
        return cos_sim
    
    def compute_feature_correlation(self,
                                   feat_base: torch.Tensor,
                                   feat_gen: torch.Tensor) -> torch.Tensor:
        """Compute feature correlation (Pearson correlation coefficient)"""
        # Flatten spatial dimensions
        B, C = feat_base.shape[:2]
        feat_base_flat = feat_base.view(B, C, -1)
        feat_gen_flat = feat_gen.view(B, C, -1)
        
        # Compute correlation per channel
        feat_base_centered = feat_base_flat - feat_base_flat.mean(dim=2, keepdim=True)
        feat_gen_centered = feat_gen_flat - feat_gen_flat.mean(dim=2, keepdim=True)
        
        numerator = (feat_base_centered * feat_gen_centered).sum(dim=2)
        denominator = torch.sqrt((feat_base_centered ** 2).sum(dim=2) * 
                                (feat_gen_centered ** 2).sum(dim=2) + 1e-8)
        
        correlation = (numerator / denominator).mean()
        return correlation
    
    def compute_structural_similarity(self,
                                     feat_base: torch.Tensor,
                                     feat_gen: torch.Tensor,
                                     window_size: int = 11) -> torch.Tensor:
        """Compute feature-level SSIM"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Simple mean pooling for local statistics
        mu_base = F.avg_pool3d(feat_base, kernel_size=3, stride=1, padding=1)
        mu_gen = F.avg_pool3d(feat_gen, kernel_size=3, stride=1, padding=1)
        
        mu_base_sq = mu_base ** 2
        mu_gen_sq = mu_gen ** 2
        mu_base_gen = mu_base * mu_gen
        
        sigma_base_sq = F.avg_pool3d(feat_base ** 2, kernel_size=3, stride=1, padding=1) - mu_base_sq
        sigma_gen_sq = F.avg_pool3d(feat_gen ** 2, kernel_size=3, stride=1, padding=1) - mu_gen_sq
        sigma_base_gen = F.avg_pool3d(feat_base * feat_gen, kernel_size=3, stride=1, padding=1) - mu_base_gen
        
        ssim_map = ((2 * mu_base_gen + C1) * (2 * sigma_base_gen + C2)) / \
                   ((mu_base_sq + mu_gen_sq + C1) * (sigma_base_sq + sigma_gen_sq + C2))
        
        return ssim_map.mean()
    
    def compute_gram_matrix(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style comparison"""
        B, C, D, H, W = feat.shape
        feat_reshaped = feat.view(B, C, D * H * W)
        gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
        return gram / (C * D * H * W)
    
    def compute_style_loss(self,
                          feat_base: torch.Tensor,
                          feat_gen: torch.Tensor) -> torch.Tensor:
        """Compute style loss via Gram matrix comparison"""
        gram_base = self.compute_gram_matrix(feat_base)
        gram_gen = self.compute_gram_matrix(feat_gen)
        return F.mse_loss(gram_base, gram_gen)
    
    def forward(self, 
                base_ct: torch.Tensor, 
                generated_ct: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive feature map accuracy metrics
        
        Args:
            base_ct: (B, 1, D, H, W) - ground truth CT
            generated_ct: (B, 1, D, H, W) - generated CT
            
        Returns:
            metrics: Dictionary of accuracy metrics at each level and overall
        """
        # Extract features at multiple levels
        features_base = self.feature_extractor(base_ct)
        features_gen = self.feature_extractor(generated_ct)
        
        metrics = {}
        
        # Compute metrics at each level
        for level_name in features_base.keys():
            feat_base = features_base[level_name]
            feat_gen = features_gen[level_name]
            
            # MSE
            mse = self.compute_feature_mse(feat_base, feat_gen)
            metrics[f'{level_name}_mse'] = mse
            
            # Cosine similarity
            cos_sim = self.compute_feature_cosine(feat_base, feat_gen)
            metrics[f'{level_name}_cosine'] = cos_sim
            
            # Correlation
            corr = self.compute_feature_correlation(feat_base, feat_gen)
            metrics[f'{level_name}_correlation'] = corr
            
            # Structural similarity
            ssim = self.compute_structural_similarity(feat_base, feat_gen)
            metrics[f'{level_name}_ssim'] = ssim
            
            # Style loss (texture comparison)
            style_loss = self.compute_style_loss(feat_base, feat_gen)
            metrics[f'{level_name}_style'] = style_loss
        
        # Compute overall metrics (average across levels)
        mse_values = [v for k, v in metrics.items() if 'mse' in k]
        metrics['overall_feature_mse'] = sum(mse_values) / len(mse_values)
        
        cos_values = [v for k, v in metrics.items() if 'cosine' in k]
        metrics['overall_feature_cosine'] = sum(cos_values) / len(cos_values)
        
        corr_values = [v for k, v in metrics.items() if 'correlation' in k]
        metrics['overall_feature_correlation'] = sum(corr_values) / len(corr_values)
        
        ssim_values = [v for k, v in metrics.items() if 'ssim' in k]
        metrics['overall_feature_ssim'] = sum(ssim_values) / len(ssim_values)
        
        style_values = [v for k, v in metrics.items() if 'style' in k]
        metrics['overall_feature_style'] = sum(style_values) / len(style_values)
        
        return metrics


class LPIPS3D(nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) adapted for 3D CT volumes
    Uses 2D LPIPS on multiple slices and averages
    """
    
    def __init__(self, 
                 net: str = 'alex',  # 'alex', 'vgg', or 'squeeze'
                 spatial: bool = False,
                 num_slices: int = 16):
        """
        Args:
            net: Network backbone ('alex', 'vgg', 'squeeze')
            spatial: Return spatial map (True) or scalar (False)
            num_slices: Number of slices to sample from 3D volume
        """
        super().__init__()
        
        self.lpips_2d = lpips.LPIPS(net=net, spatial=spatial)
        self.num_slices = num_slices
        self.spatial = spatial
        
        # Freeze LPIPS weights
        for param in self.lpips_2d.parameters():
            param.requires_grad = False
    
    def forward(self, 
                base_ct: torch.Tensor, 
                generated_ct: torch.Tensor,
                dimension: str = 'axial') -> torch.Tensor:
        """
        Compute LPIPS for 3D volumes by averaging over slices
        
        Args:
            base_ct: (B, 1, D, H, W) - ground truth CT
            generated_ct: (B, 1, D, H, W) - generated CT
            dimension: Which dimension to slice ('axial', 'coronal', 'sagittal')
            
        Returns:
            lpips_score: Scalar or spatial map
        """
        B, C, D, H, W = base_ct.shape
        
        # Determine slice dimension
        if dimension == 'axial':
            slice_dim = 2  # Slice along D
            num_total_slices = D
        elif dimension == 'coronal':
            slice_dim = 3  # Slice along H
            num_total_slices = H
        elif dimension == 'sagittal':
            slice_dim = 4  # Slice along W
            num_total_slices = W
        else:
            raise ValueError(f"Unknown dimension: {dimension}")
        
        # Sample slice indices uniformly
        num_slices = min(self.num_slices, num_total_slices)
        slice_indices = torch.linspace(0, num_total_slices - 1, num_slices).long()
        
        lpips_scores = []
        
        for slice_idx in slice_indices:
            # Extract 2D slices
            if dimension == 'axial':
                slice_base = base_ct[:, :, slice_idx, :, :]  # (B, 1, H, W)
                slice_gen = generated_ct[:, :, slice_idx, :, :]
            elif dimension == 'coronal':
                slice_base = base_ct[:, :, :, slice_idx, :]  # (B, 1, D, W)
                slice_gen = generated_ct[:, :, :, slice_idx, :]
            else:  # sagittal
                slice_base = base_ct[:, :, :, :, slice_idx]  # (B, 1, D, H)
                slice_gen = generated_ct[:, :, :, :, slice_idx]
            
            # LPIPS expects 3-channel images, so replicate channel
            slice_base_3ch = slice_base.repeat(1, 3, 1, 1)
            slice_gen_3ch = slice_gen.repeat(1, 3, 1, 1)
            
            # Normalize to [-1, 1] (LPIPS expects this range)
            slice_base_3ch = 2 * slice_base_3ch - 1
            slice_gen_3ch = 2 * slice_gen_3ch - 1
            
            # Compute LPIPS for this slice
            lpips_score = self.lpips_2d(slice_base_3ch, slice_gen_3ch)
            lpips_scores.append(lpips_score)
        
        # Average over slices
        if self.spatial:
            lpips_avg = torch.stack(lpips_scores, dim=0).mean(dim=0)
        else:
            lpips_avg = torch.stack(lpips_scores, dim=0).mean()
        
        return lpips_avg
    
    def forward_multi_view(self,
                          base_ct: torch.Tensor,
                          generated_ct: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute LPIPS from all three anatomical views
        
        Returns:
            Dict with 'axial', 'coronal', 'sagittal', and 'average' LPIPS scores
        """
        lpips_axial = self.forward(base_ct, generated_ct, dimension='axial')
        lpips_coronal = self.forward(base_ct, generated_ct, dimension='coronal')
        lpips_sagittal = self.forward(base_ct, generated_ct, dimension='sagittal')
        
        return {
            'lpips_axial': lpips_axial,
            'lpips_coronal': lpips_coronal,
            'lpips_sagittal': lpips_sagittal,
            'lpips_average': (lpips_axial + lpips_coronal + lpips_sagittal) / 3
        }


class ComprehensiveFeatureMetrics(nn.Module):
    """
    Combined module for all feature-based metrics:
    - Multi-level feature extraction
    - Feature map accuracy
    - LPIPS perceptual similarity
    """
    
    def __init__(self,
                 feature_dims: List[int] = [32, 64, 128, 256],
                 lpips_net: str = 'alex',
                 num_lpips_slices: int = 16,
                 compute_lpips: bool = True):
        """
        Args:
            feature_dims: Feature dimensions at each level
            lpips_net: LPIPS network backbone
            num_lpips_slices: Number of slices for LPIPS computation
            compute_lpips: Whether to compute LPIPS (can be expensive)
        """
        super().__init__()
        
        self.feature_accuracy = FeatureMapAccuracy(feature_dims=feature_dims)
        
        self.compute_lpips = compute_lpips
        if compute_lpips:
            self.lpips_3d = LPIPS3D(net=lpips_net, num_slices=num_lpips_slices)
    
    def forward(self,
                base_ct: torch.Tensor,
                generated_ct: torch.Tensor,
                compute_lpips: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all feature-based metrics
        
        Args:
            base_ct: (B, 1, D, H, W) - ground truth CT
            generated_ct: (B, 1, D, H, W) - generated CT
            compute_lpips: Override whether to compute LPIPS
            
        Returns:
            Comprehensive dictionary of all metrics
        """
        metrics = {}
        
        # Feature map accuracy metrics
        feature_metrics = self.feature_accuracy(base_ct, generated_ct)
        metrics.update(feature_metrics)
        
        # LPIPS metrics (if enabled)
        if compute_lpips is None:
            compute_lpips = self.compute_lpips
        
        if compute_lpips:
            lpips_metrics = self.lpips_3d.forward_multi_view(base_ct, generated_ct)
            metrics.update(lpips_metrics)
        
        return metrics


# Example usage and testing
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Feature Metrics Module...")
    print("=" * 80)
    
    # Create test data
    batch_size = 2
    volume_size = (64, 64, 64)
    
    base_ct = torch.randn(batch_size, 1, *volume_size).to(device)
    generated_ct = base_ct + 0.1 * torch.randn_like(base_ct)  # Similar but not identical
    
    # Test 1: Multi-level feature extraction
    print("\n1. Testing Multi-Level Feature Extractor...")
    feature_extractor = MultiLevelFeatureExtractor(
        in_channels=1,
        feature_dims=[32, 64, 128, 256]
    ).to(device)
    
    features = feature_extractor(base_ct)
    print(f"   Extracted {len(features)} feature levels:")
    for level_name, feat in features.items():
        print(f"   - {level_name}: shape {feat.shape}")
    
    # Test 2: Feature map accuracy
    print("\n2. Testing Feature Map Accuracy...")
    feature_accuracy = FeatureMapAccuracy(
        feature_dims=[32, 64, 128, 256]
    ).to(device)
    
    accuracy_metrics = feature_accuracy(base_ct, generated_ct)
    print(f"   Computed {len(accuracy_metrics)} accuracy metrics:")
    for metric_name, value in accuracy_metrics.items():
        if not metric_name.startswith('level_'):
            print(f"   - {metric_name:30s}: {value.item():.6f}")
    
    # Test 3: LPIPS 3D
    print("\n3. Testing LPIPS for 3D...")
    lpips_3d = LPIPS3D(net='alex', num_slices=8).to(device)
    
    lpips_metrics = lpips_3d.forward_multi_view(base_ct, generated_ct)
    print("   LPIPS scores from different views:")
    for view_name, score in lpips_metrics.items():
        print(f"   - {view_name:20s}: {score.item():.6f}")
    
    # Test 4: Comprehensive metrics
    print("\n4. Testing Comprehensive Feature Metrics...")
    comprehensive = ComprehensiveFeatureMetrics(
        feature_dims=[32, 64, 128, 256],
        lpips_net='alex',
        num_lpips_slices=8,
        compute_lpips=True
    ).to(device)
    
    all_metrics = comprehensive(base_ct, generated_ct)
    
    print(f"\n   Total metrics computed: {len(all_metrics)}")
    print("\n   Summary of key metrics:")
    print(f"   - Overall Feature MSE:         {all_metrics['overall_feature_mse'].item():.6f}")
    print(f"   - Overall Feature Cosine:      {all_metrics['overall_feature_cosine'].item():.6f}")
    print(f"   - Overall Feature Correlation: {all_metrics['overall_feature_correlation'].item():.6f}")
    print(f"   - Overall Feature SSIM:        {all_metrics['overall_feature_ssim'].item():.6f}")
    print(f"   - LPIPS (average):             {all_metrics['lpips_average'].item():.6f}")
    
    # Memory usage
    total_params = sum(p.numel() for p in comprehensive.parameters())
    print(f"\n   Model parameters: {total_params:,}")
    print(f"   Model size: {total_params * 4 / 1024**2:.2f} MB")
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    
    print("\n=== Interpretation Guide ===")
    print("• Lower feature MSE → More similar feature representations")
    print("• Higher feature cosine → More aligned feature directions")
    print("• Higher correlation → More linear relationship in features")
    print("• Higher feature SSIM → More structurally similar features")
    print("• Lower LPIPS → More perceptually similar (0=identical, 1=very different)")
    print("• Style loss → Texture/pattern similarity (lower is better)")
