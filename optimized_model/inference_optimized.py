"""
Inference Script for Optimized CT Regression Model
Generates full CT volume from X-ray images with optional upscaling
"""
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
import json
import nibabel as nib
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import sys

sys.path.insert(0, '..')

from model_optimized import OptimizedCTRegression


class OptimizedCTInference:
    """Inference wrapper for optimized CT reconstruction"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 device: str = 'cuda',
                 upscale_to: Optional[Tuple[int, int, int]] = None):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
            upscale_to: Optional target size (D, H, W) for upscaling
        """
        self.device = torch.device(device)
        self.upscale_to = upscale_to
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract config
        if 'config' in checkpoint:
            config = checkpoint['config']
            if isinstance(config, dict) and 'model' in config:
                model_config = config['model']
            else:
                model_config = config
        else:
            # Default config
            model_config = {
                'volume_size': [64, 64, 64],
                'xray_feature_dim': 512,
                'voxel_dim': 256,
                'use_learnable_priors': True
            }
            print("Warning: No config in checkpoint, using defaults")
        
        # Create model
        self.model = OptimizedCTRegression(
            volume_size=tuple(model_config['volume_size']),
            xray_feature_dim=model_config['xray_feature_dim'],
            voxel_dim=model_config['voxel_dim'],
            use_learnable_priors=model_config.get('use_learnable_priors', True)
        ).to(self.device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        self.base_volume_size = tuple(model_config['volume_size'])
        print(f"Model loaded successfully!")
        print(f"Base volume size: {self.base_volume_size}")
        if upscale_to:
            print(f"Will upscale to: {upscale_to}")
    
    @torch.no_grad()
    def predict(self, 
                xrays: torch.Tensor,
                return_aux: bool = False) -> torch.Tensor:
        """
        Generate CT volume from X-rays
        
        Args:
            xrays: (B, num_views, C, H, W) or (num_views, C, H, W) X-ray images
            return_aux: Whether to return auxiliary info (boundaries, etc.)
            
        Returns:
            predicted_volume: (B, 1, D, H, W) or (1, D, H, W) CT volume
            aux_info: Optional dict with boundaries and uncertainties
        """
        # Handle single sample input
        if xrays.dim() == 4:
            xrays = xrays.unsqueeze(0)
        
        xrays = xrays.to(self.device)
        
        # Forward pass
        predicted, aux_info = self.model(xrays)
        
        # Upscale if requested
        if self.upscale_to is not None:
            predicted = self._upscale_volume(predicted, self.upscale_to)
        
        if return_aux:
            return predicted, aux_info
        return predicted
    
    def _upscale_volume(self, 
                       volume: torch.Tensor,
                       target_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Upscale volume using trilinear interpolation
        
        Args:
            volume: (B, C, D, H, W) volume
            target_size: (D_target, H_target, W_target)
            
        Returns:
            upscaled: (B, C, D_target, H_target, W_target)
        """
        print(f"Upscaling from {volume.shape[2:]} to {target_size}...")
        
        upscaled = F.interpolate(
            volume,
            size=target_size,
            mode='trilinear',
            align_corners=True
        )
        
        return upscaled
    
    def predict_from_files(self,
                          xray_paths: list,
                          output_path: str,
                          visualize: bool = True):
        """
        Generate CT from X-ray file paths and save results
        
        Args:
            xray_paths: List of paths to X-ray images (PNG, NIFTI, etc.)
            output_path: Path to save output CT volume (will be .nii.gz)
            visualize: Whether to save visualization
        """
        # Load X-rays
        xrays = self._load_xrays(xray_paths)
        
        # Predict
        print("Running inference...")
        predicted, aux_info = self.predict(xrays, return_aux=True)
        
        # Convert to numpy
        volume_np = predicted[0, 0].cpu().numpy()
        
        # Save as NIFTI
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        nifti_img = nib.Nifti1Image(volume_np, affine=np.eye(4))
        nib.save(nifti_img, str(output_path))
        print(f"Saved CT volume to {output_path}")
        
        # Visualization
        if visualize:
            vis_path = output_path.parent / f"{output_path.stem}_visualization.png"
            self._visualize_result(volume_np, xrays[0], aux_info, vis_path)
            print(f"Saved visualization to {vis_path}")
        
        return volume_np, aux_info
    
    def _load_xrays(self, xray_paths: list) -> torch.Tensor:
        """Load X-ray images from file paths"""
        xrays = []
        
        for path in xray_paths:
            path = Path(path)
            
            if path.suffix in ['.nii', '.gz']:
                # Load NIFTI
                img = nib.load(str(path))
                data = img.get_fdata()
            else:
                # Load image (PNG, JPG, etc.)
                from PIL import Image
                img = Image.open(path).convert('L')
                data = np.array(img)
            
            # Normalize to [0, 1]
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            
            # Resize to 512x512 if needed
            if data.shape != (512, 512):
                data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
                data = F.interpolate(data, size=(512, 512), mode='bilinear', align_corners=True)
                data = data.squeeze().numpy()
            
            xrays.append(data)
        
        # Stack: (num_views, 1, 512, 512)
        xrays = np.stack(xrays, axis=0)[:, None, :, :]
        xrays = torch.from_numpy(xrays).float()
        
        print(f"Loaded {len(xray_paths)} X-ray views with shape {xrays.shape}")
        return xrays
    
    def _visualize_result(self,
                         volume: np.ndarray,
                         xrays: torch.Tensor,
                         aux_info: dict,
                         save_path: Path):
        """Create visualization of results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # X-ray views
        for i, ax in enumerate(axes[0, :2]):
            if i < xrays.shape[0]:
                ax.imshow(xrays[i, 0].cpu().numpy(), cmap='gray')
                ax.set_title(f'X-ray View {i+1}')
            ax.axis('off')
        
        # Volume slices
        D, H, W = volume.shape
        axes[0, 2].imshow(volume[D//2], cmap='gray')
        axes[0, 2].set_title('Axial Slice (Mid)')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(volume[:, H//2, :], cmap='gray')
        axes[1, 0].set_title('Coronal Slice (Mid)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(volume[:, :, W//2], cmap='gray')
        axes[1, 1].set_title('Sagittal Slice (Mid)')
        axes[1, 1].axis('off')
        
        # Learned boundaries visualization
        if 'boundaries' in aux_info:
            boundaries = aux_info['boundaries'][0].cpu().numpy() * 100
            ax = axes[1, 2]
            ax.barh(range(len(boundaries)), boundaries, color='steelblue')
            ax.set_yticks(range(len(boundaries)))
            ax.set_yticklabels([f'B{i}' for i in range(len(boundaries))])
            ax.set_xlabel('Depth Position (%)')
            ax.set_title('Learned Anatomical Boundaries')
            ax.grid(axis='x', alpha=0.3)
        else:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Optimized CT Reconstruction Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--xrays', type=str, nargs='+', required=True,
                       help='Paths to X-ray images (1 or 2 views)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for CT volume (.nii.gz)')
    parser.add_argument('--upscale', type=int, nargs=3, default=None,
                       help='Upscale to size (D H W), e.g., --upscale 256 256 256')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.xrays) > 2:
        print("Warning: Model expects 1-2 X-ray views, using first 2")
        args.xrays = args.xrays[:2]
    
    # Create inference engine
    inferencer = OptimizedCTInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        upscale_to=tuple(args.upscale) if args.upscale else None
    )
    
    # Run inference
    volume, aux_info = inferencer.predict_from_files(
        xray_paths=args.xrays,
        output_path=args.output,
        visualize=not args.no_visualize
    )
    
    # Print stats
    print("\n" + "="*50)
    print("Inference Complete!")
    print("="*50)
    print(f"Output volume shape: {volume.shape}")
    print(f"Value range: [{volume.min():.4f}, {volume.max():.4f}]")
    print(f"Mean: {volume.mean():.4f}, Std: {volume.std():.4f}")
    
    if 'boundaries' in aux_info:
        boundaries = aux_info['boundaries'][0].cpu().numpy() * 100
        print(f"\nLearned Anatomical Boundaries (%):")
        for i, b in enumerate(boundaries):
            print(f"  Boundary {i}: {b:.1f}%")


if __name__ == '__main__':
    main()
