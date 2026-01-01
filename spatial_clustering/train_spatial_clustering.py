"""
Training script for Spatial Clustering CT Generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from spatial_cluster_architecture import (
    SpatialClusteringCTGenerator,
    ClusterTrackingLoss
)


def compute_psnr(pred, target, data_range=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        pred: (B, C, D, H, W) predicted volume
        target: (B, C, D, H, W) target volume
        data_range: maximum possible pixel value (default 1.0 for normalized data)
    Returns:
        psnr: scalar PSNR value in dB
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(data_range).to(pred.device) / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(pred, target, window_size=11, data_range=1.0):
    """
    Compute Structural Similarity Index (SSIM) for 3D volumes
    Applies SSIM on axial slices and averages
    
    Args:
        pred: (B, C, D, H, W) predicted volume
        target: (B, C, D, H, W) target volume
        window_size: size of Gaussian window
        data_range: dynamic range of pixel values
    Returns:
        ssim: scalar SSIM value
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Compute SSIM on axial slices
    B, C, D, H, W = pred.shape
    ssim_values = []
    
    for d in range(D):
        pred_slice = pred[:, :, d, :, :]  # (B, C, H, W)
        target_slice = target[:, :, d, :, :]  # (B, C, H, W)
        
        # Compute local statistics with average pooling
        mu_pred = F.avg_pool2d(pred_slice, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target_slice, window_size, stride=1, padding=window_size//2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool2d(pred_slice ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target_slice ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred_slice * target_slice, window_size, stride=1, padding=window_size//2) - mu_pred_target
        
        # SSIM formula
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        ssim_values.append(ssim_map.mean().item())
    
    return np.mean(ssim_values)


# Import your existing dataset (assuming you have one)
import sys
sys.path.append('..')
# from utils.dataset import CTDataset  # Adjust based on your actual dataset


class SpatialClusteringTrainer:
    """
    Trainer for Spatial Clustering CT Generation
    Tracks per-voxel and per-cluster accuracy throughout training
    """
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = SpatialClusteringCTGenerator(
            volume_size=tuple(self.config['model']['volume_size']),
            voxel_dim=self.config['model']['voxel_dim'],
            num_clusters=self.config['model']['num_clusters'],
            num_heads=self.config['model']['num_heads'],
            num_blocks=self.config['model']['num_blocks']
        ).to(self.device)
        
        # Create loss function
        self.criterion = ClusterTrackingLoss(
            lambda_position=self.config['loss_weights']['lambda_position'],
            lambda_intensity=self.config['loss_weights']['lambda_intensity'],
            lambda_contrast=self.config['loss_weights']['lambda_contrast'],
            lambda_cluster=self.config['loss_weights']['lambda_cluster']
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=self.config['optimizer']['betas'],
            eps=self.config['optimizer']['eps']
        )
        
        # Scheduler
        if self.config['scheduler']['type'] == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['scheduler']['T_0'],
                T_mult=self.config['scheduler']['T_mult'],
                eta_min=self.config['scheduler']['eta_min']
            )
        
        # Mixed precision
        self.scaler = GradScaler() if self.config['training']['mixed_precision'] else None
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Directories
        self.checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        self.log_dir = Path(self.config['logging']['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.train_metrics = {
            'total_loss': [],
            'position_loss': [],
            'intensity_loss': [],
            'contrast_loss': [],
            'cluster_consistency': [],
            'psnr': [],
            'ssim': []
        }
        self.val_metrics = {
            'total_loss': [],
            'position_loss': [],
            'intensity_loss': [],
            'contrast_loss': [],
            'cluster_consistency': [],
            'psnr': [],
            'ssim': []
        }
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {k: 0.0 for k in ['total_loss', 'position_loss', 'intensity_loss', 'contrast_loss', 'cluster_consistency']}
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch (adjust based on your dataset)
            frontal_xray = batch['frontal'].to(self.device)  # (B, 1, H, W)
            lateral_xray = batch['lateral'].to(self.device)  # (B, 1, H, W)
            gt_volume = batch['volume'].to(self.device)      # (B, 1, D, H, W)
            
            # Forward pass
            if self.scaler:
                with autocast():
                    output = self.model(frontal_xray, lateral_xray, gt_volume)
                    loss_dict = self.criterion(
                        output['pred_volume'],
                        gt_volume,
                        output['position_accuracy'],
                        output['intensity_metrics'],
                        output['cluster_assignments']
                    )
                    loss = loss_dict['total_loss'] / self.config['training']['accumulation_steps']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config['training']['accumulation_steps'] == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip']
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                output = self.model(frontal_xray, lateral_xray, gt_volume)
                loss_dict = self.criterion(
                    output['pred_volume'],
                    gt_volume,
                    output['position_accuracy'],
                    output['intensity_metrics'],
                    output['cluster_assignments']
                )
                loss = loss_dict['total_loss'] / self.config['training']['accumulation_steps']
                loss.backward()
                
                if (batch_idx + 1) % self.config['training']['accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            # Update metrics
            for k in epoch_losses.keys():
                epoch_losses[k] += loss_dict[k].item()
            
            # Compute PSNR and SSIM (on a subset for speed)
            if batch_idx % 5 == 0:  # Compute every 5 batches to save time
                with torch.no_grad():
                    batch_psnr = compute_psnr(output['pred_volume'], gt_volume)
                    batch_ssim = compute_ssim(output['pred_volume'], gt_volume)
                    epoch_psnr += batch_psnr
                    epoch_ssim += batch_ssim
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['total_loss'].item(),
                'pos': loss_dict['position_loss'].item(),
                'int': loss_dict['intensity_loss'].item()
            })
            
            self.global_step += 1
            
            # Logging
        num_psnr_ssim_samples = len(train_loader) // 5 + 1
        for k in epoch_losses.keys():
            epoch_losses[k] /= len(train_loader)
            self.train_metrics[k].append(epoch_losses[k])
        
        epoch_losses['psnr'] = epoch_psnr / num_psnr_ssim_samples
        epoch_losses['ssim'] = epoch_ssim / num_psnr_ssim_samples
        self.train_metrics['psnr'].append(epoch_losses['psnr'])
        self.train_metrics['ssim'].append(epoch_losses['ssim'
            # Visualization
            if self.global_step % self.config['logging']['visualize_interval'] == 0:
                self._visualize_predictions(output, gt_volume, batch_idx)
        
        # Average epoch losses
        for k in epoch_losses.keys():
            epoch_losses[k] /= len(train_loader)
            self.train_metrics[k].append(epoch_losses[k])
        
        return epoch_losses['total_loss', 'position_loss', 'intensity_loss', 'contrast_loss', 'cluster_consistency']}
        val_psnr = 0.0
        val_ssim = 0.0
        
        for batch in tqdm(val_loader, desc="Validation"):
            frontal_xray = batch['frontal'].to(self.device)
            lateral_xray = batch['lateral'].to(self.device)
            gt_volume = batch['volume'].to(self.device)
            
            output = self.model(frontal_xray, lateral_xray, gt_volume)
            loss_dict = self.criterion(
                output['pred_volume'],
                gt_volume,
                output['position_accuracy'],
                output['intensity_metrics'],
                output['cluster_assignments']
            )
            
            for k in val_losses.keys():
                val_losses[k] += loss_dict[k].item()
            
            # Compute PSNR and SSIM
            batch_psnr = compute_psnr(output['pred_volume'], gt_volume)
            batch_ssim = compute_ssim(output['pred_volume'], gt_volume)
            val_psnr += batch_psnr
            val_ssim += batch_ssim
        
        # Average validation losses
        for k in val_losses.keys():
            val_losses[k] /= len(val_loader)
            self.val_metrics[k].append(val_losses[k])
        
        val_losses['psnr'] = val_psnr / len(val_loader)
        val_losses['ssim'] = val_ssim / len(val_loader)
        self.val_metrics['psnr'].append(val_losses['psnr'])
        self.val_metrics['ssim'].append(val_losses['ssim'
                val_losses[k] += loss_dict[k].item()
        
        # Average validation losses
        for k in val_losses.keys():
            val_losses[k] /= len(val_loader)
            self.val_metrics[k].append(val_losses[k])
        
        return val_losses
    
    def _log_metrics(self, metrics: dict, split: str):
        """Log metrics"""
        log_str = f"[{split.upper()}] Step {self.global_step} - "
        log_str += " | ".join([f"{k}: {v.item():.4f}" for k, v in metrics.items()])
        print(log_str)
    
    def _visualize_predictions(self, output: dict, gt_volume: torch.Tensor, batch_idx: int):
        """Visualize predictions and cluster assignments"""
        pred = output['pred_volume'][0, 0].cpu().numpy()  # (D, H, W)
        gt = gt_volume[0, 0].cpu().numpy()
        position_acc = output['position_accuracy'][0].cpu().numpy()
        
        D, H, W = pred.shape
        mid_slice = D // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Predicted volume
        axes[0, 0].imshow(pred[mid_slice], cmap='gray')
        axes[0, 0].set_title('Predicted (mid-axial)')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(gt[mid_slice], cmap='gray')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # Error map
        error = np.abs(pred[mid_slice] - gt[mid_slice])
        im = axes[0, 2].imshow(error, cmap='hot')
        axes[0, 2].set_title(f'Error (MAE: {error.mean():.4f})')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2])
        
        # Position accuracy
        im = axes[1, 0].imshow(position_acc[mid_slice], cmap='viridis')
        axes[1, 0].set_title('Position-weighted Accuracy')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Coronal view
        axes[1, 1].imshow(pred[:, H//2, :], cmap='gray')
        axes[1, 1].set_title('Predicted (mid-coronal)')
        axes[1, 1].axis('off')
        
        # Sagittal view
        axes[1, 2].imshow(pred[:, :, W//2], cmap='gray')
        axes[1, 2].set_title('Predicted (mid-sagittal)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        save_path = self.log_dir / f'pred_step_{self.global_step}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {save_path}")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save epoch checkpoint
        if self.epoch % self.config['logging']['save_interval'] == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{self.epoch}.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            print(f"✓ Saved best model at epoch {self.epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['globametrics:")
            for k, v in train_losses.items():
                if k in ['psnr', 'ssim']:
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v:.6f}")
            
            # Validate
            val_losses = self.validate(val_loader)
            print(f"Epoch {epoch} - Val metrics:")
            for k, v in val_losses.items():
                if k in ['psnr', 'ssim']:
                    print(f"  {k}: {v:.4f}")
                else:
                self.load_checkpoint(resume_from)
        
        print("\n" + "="*80)
        print("Starting training...")
        print("="*80 + "\n")
        
        for epoch in range(self.epoch, self.config['training']['num_epochs']):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch(train_loader)
            print(f"\nEpoch {epoch} - Train losses:")
            for k, v in train_losses.items():
                print(f"  {k}: {v:.6f}")
            
            # Validate
            val_losses = self.validate(val_loader)
            print(f"Epoch {epoch} - Val losses:")
            for k, v in val_losses.items():
                print(f"  {k}: {v:.6f}")
            
            # Check for best model
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            print(f"\nCompleted epoch {epoch}/{self.config['training']['num_epochs']}\n")
        
        print("\n" + "="*80)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print("="*80 + "\n")


# Example usage
if __name__ == "__main__":
    # Create trainer
    trainer = SpatialClusteringTrainer('config_spatial_clustering.json')
    
    # Create dummy dataset (replace with your actual dataset)
    # Your dataset should have 500 patients total
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=500):
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return {
                'frontal': torch.randn(1, 512, 512),
                'lateral': torch.randn(1, 512, 512),
                'volume': torch.randn(1, 64, 64, 64)
            }
    
    # Split into train/val (80/20 split for 500 patients)
    train_dataset = DummyDataset(400)  # 400 train patients
    val_dataset = DummyDataset(100)    # 100 val patients
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4
    )
    
    print(f"\nDataset Info:")
    print(f"  Train patients: {len(train_dataset)}")
    print(f"  Val patients: {len(val_dataset)}")
    print(f"  Total epochs: 20")
    print(f"  Batches per epoch: {len(train_loader)}\n")
    
    # Train
    trainer.train(train_loader, val_loader)
