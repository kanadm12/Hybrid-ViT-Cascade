"""
Training script for Spatial Clustering CT Generator - 4 GPU Version
Uses DistributedDataParallel for multi-GPU training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import nibabel as nib
from PIL import Image

from spatial_cluster_architecture import (
    ClusterTrackingLoss
)
from enhanced_spatial_clustering import EnhancedSpatialClusteringCTGenerator


def compute_psnr(pred, target, data_range=1.0):
    """Compute Peak Signal-to-Noise Ratio (PSNR)"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(data_range).to(pred.device) / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(pred, target, window_size=11, data_range=1.0):
    """Compute Structural Similarity Index (SSIM) for 3D volumes"""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    B, C, D, H, W = pred.shape
    ssim_values = []
    
    for d in range(D):
        pred_slice = pred[:, :, d, :, :]
        target_slice = target[:, :, d, :, :]
        
        mu_pred = F.avg_pool2d(pred_slice, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target_slice, window_size, stride=1, padding=window_size//2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool2d(pred_slice ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target_slice ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred_slice * target_slice, window_size, stride=1, padding=window_size//2) - mu_pred_target
        
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        ssim_values.append(ssim_map.mean().item())
    
    return np.mean(ssim_values)


class CTDRRDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='/workspace/drr_patient_data', num_samples=500):
        self.data_dir = data_dir
        self.patient_dirs = sorted(glob.glob(f"{data_dir}/*"))[:num_samples]
        
    def __len__(self):
        return len(self.patient_dirs)
    
    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_id = os.path.basename(patient_dir)
        
        # Load PA (frontal) DRR
        pa_path = os.path.join(patient_dir, f"{patient_id}_pa_drr.png")
        frontal = np.array(Image.open(pa_path).convert('L'))
        frontal = np.flipud(frontal).copy()  # FLIP VERTICALLY + copy to fix negative strides
        frontal = torch.from_numpy(frontal).float().unsqueeze(0) / 255.0
        
        # Load LAT (lateral) DRR
        lat_path = os.path.join(patient_dir, f"{patient_id}_lat_drr.png")
        lateral = np.array(Image.open(lat_path).convert('L'))
        lateral = np.flipud(lateral).copy()  # FLIP VERTICALLY + copy to fix negative strides
        lateral = torch.from_numpy(lateral).float().unsqueeze(0) / 255.0
        
        # Load CT volume
        ct_path = os.path.join(patient_dir, f"{patient_id}.nii.gz")
        ct_img = nib.load(ct_path)
        volume = ct_img.get_fdata()
        
        # Resize volume to (64, 64, 64)
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        volume_resized = F.interpolate(volume_tensor, size=(64, 64, 64), mode='trilinear', align_corners=False)
        volume_resized = volume_resized.squeeze(0)
        
        # Normalize volume to [0, 1]
        volume_resized = (volume_resized - volume_resized.min()) / (volume_resized.max() - volume_resized.min() + 1e-8)
        
        # Resize X-rays to 512x512 if needed
        if frontal.shape[-2:] != (512, 512):
            frontal = F.interpolate(frontal.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
        if lateral.shape[-2:] != (512, 512):
            lateral = F.interpolate(lateral.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
        
        return {
            'frontal': frontal,
            'lateral': lateral,
            'volume': volume_resized
        }


class SpatialClusteringTrainer:
    """Multi-GPU Trainer for Spatial Clustering CT Generation"""
    
    def __init__(self, config_path: str, local_rank: int):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)
        
        # Initialize distributed training
        if local_rank == 0:
            print(f"Using {dist.get_world_size()} GPUs")
        
        # Create model
        self.model = EnhancedSpatialClusteringCTGenerator(
            volume_size=tuple(self.config['model']['volume_size']),
            voxel_dim=self.config['model']['voxel_dim'],
            num_clusters=self.config['model']['num_clusters'],
            num_heads=self.config['model']['num_heads'],
            num_blocks=self.config['model']['num_blocks']
        ).to(self.device)
        
        # Wrap with DDP
        self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
        
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
        
        # Directories (only rank 0)
        if local_rank == 0:
            self.checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
            self.log_dir = Path(self.config['logging']['log_dir'])
            self.viz_dir = Path(self.config['logging'].get('viz_dir', 'visualizations/spatial_clustering'))
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.train_metrics = {
            'total_loss': [], 'position_loss': [], 'intensity_loss': [],
            'contrast_loss': [], 'cluster_consistency': [], 'psnr': [], 'ssim': []
        }
        self.val_metrics = {
            'total_loss': [], 'position_loss': [], 'intensity_loss': [],
            'contrast_loss': [], 'cluster_consistency': [], 'psnr': [], 'ssim': []
        }
        
        if local_rank == 0:
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
    
    def save_visualization(self, pred_volume, gt_volume, epoch, phase='train'):
        """Save visualization of GT vs Pred CT slices (only rank 0)"""
        if self.local_rank != 0:
            return
        
        # Take first sample from batch
        pred = pred_volume[0, 0].detach().cpu().numpy()  # (D, H, W)
        gt = gt_volume[0, 0].detach().cpu().numpy()  # (D, H, W)
        
        D, H, W = pred.shape
        
        # Extract middle slices
        axial_idx = D // 2
        sagittal_idx = W // 2
        coronal_idx = H // 2
        
        # Create figure with 2 rows (GT, Pred) x 3 columns (Axial, Sagittal, Coronal)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Epoch {epoch} - {phase.upper()} | GT (top) vs Pred (bottom)', fontsize=16)
        
        # Row 0: Ground Truth
        axes[0, 0].imshow(gt[axial_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title(f'GT Axial (z={axial_idx})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gt[:, :, sagittal_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title(f'GT Sagittal (x={sagittal_idx})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(gt[:, coronal_idx, :], cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title(f'GT Coronal (y={coronal_idx})')
        axes[0, 2].axis('off')
        
        # Row 1: Prediction
        axes[1, 0].imshow(pred[axial_idx], cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Pred Axial (z={axial_idx})')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred[:, :, sagittal_idx], cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Pred Sagittal (x={sagittal_idx})')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(pred[:, coronal_idx, :], cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title(f'Pred Coronal (y={coronal_idx})')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save to disk
        save_path = self.viz_dir / f'epoch_{epoch:03d}_{phase}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Saved visualization: {save_path}")

    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {k: 0.0 for k in ['total_loss', 'position_loss', 'intensity_loss', 'contrast_loss', 'cluster_consistency']}
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        
        # Initialize gradients to zero at epoch start
        self.optimizer.zero_grad()
        
        if self.local_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            frontal_xray = batch['frontal'].to(self.device)
            lateral_xray = batch['lateral'].to(self.device)
            gt_volume = batch['volume'].to(self.device)
            
            # Forward pass
            if self.scaler:
                with autocast():
                    output = self.model(frontal_xray, lateral_xray, gt_volume)
                    loss_dict = self.criterion(
                        output['pred_volume'], gt_volume,
                        output['position_accuracy'],
                        output['intensity_metrics'],
                        output['cluster_assignments']
                    )
                    loss = loss_dict['total_loss'] / self.config['training']['accumulation_steps']
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config['training']['accumulation_steps'] == 0:
                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                    # Step optimizer (scaler may skip if inf/nan gradients)
                    self.scaler.step(self.optimizer)
                    # Update scaler scale factor
                    self.scaler.update()
                    # Zero gradients for next accumulation
                    self.optimizer.zero_grad()
                    # Step scheduler
                    self.scheduler.step()
            else:
                output = self.model(frontal_xray, lateral_xray, gt_volume)
                loss_dict = self.criterion(
                    output['pred_volume'], gt_volume,
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
            
            # Compute PSNR and SSIM (every 5 batches)
            if batch_idx % 5 == 0:
                with torch.no_grad():
                    batch_psnr = compute_psnr(output['pred_volume'], gt_volume)
                    batch_ssim = compute_ssim(output['pred_volume'], gt_volume)
                    epoch_psnr += batch_psnr
                    epoch_ssim += batch_ssim
            
            # Update progress bar (rank 0 only)
            if self.local_rank == 0:
                pbar.set_postfix({
                    'loss': loss_dict['total_loss'].item(),
                    'psnr': batch_psnr if batch_idx % 5 == 0 else 0
                })
            
            self.global_step += 1
        
        # Average epoch losses
        num_psnr_ssim_samples = len(train_loader) // 5 + 1
        for k in epoch_losses.keys():
            epoch_losses[k] /= len(train_loader)
            self.train_metrics[k].append(epoch_losses[k])
        
        epoch_losses['psnr'] = epoch_psnr / num_psnr_ssim_samples
        epoch_losses['ssim'] = epoch_ssim / num_psnr_ssim_samples
        self.train_metrics['psnr'].append(epoch_losses['psnr'])
        self.train_metrics['ssim'].append(epoch_losses['ssim'])
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_losses = {k: 0.0 for k in ['total_loss', 'position_loss', 'intensity_loss', 'contrast_loss', 'cluster_consistency']}
        val_psnr = 0.0
        val_ssim = 0.0
        
        if self.local_rank == 0:
            pbar = tqdm(val_loader, desc="Validation")
        else:
            pbar = val_loader
        
        for batch in pbar:
            frontal_xray = batch['frontal'].to(self.device)
            lateral_xray = batch['lateral'].to(self.device)
            gt_volume = batch['volume'].to(self.device)
            
            output = self.model(frontal_xray, lateral_xray, gt_volume)
            loss_dict = self.criterion(
                output['pred_volume'], gt_volume,
                output['position_accuracy'],
                output['intensity_metrics'],
                output['cluster_assignments']
            )
            
            for k in val_losses.keys():
                val_losses[k] += loss_dict[k].item()
            
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
        self.val_metrics['ssim'].append(val_losses['ssim'])
        
        # Save visualization of last batch
        if self.local_rank == 0:
            self.save_visualization(output['pred_volume'], gt_volume, self.epoch, phase='val')
        
        return val_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint (rank 0 only)"""
        if self.local_rank != 0:
            return
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        if self.epoch % self.config['logging']['save_interval'] == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{self.epoch}.pth')
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            print(f"✓ Saved best model at epoch {self.epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint and resume training"""
        if self.local_rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Try to load model state dict with strict=False to handle architecture mismatches
        missing_keys, unexpected_keys = self.model.module.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        
        if self.local_rank == 0:
            if missing_keys:
                print(f"\n⚠️  WARNING: Checkpoint architecture mismatch!")
                print(f"   Missing {len(missing_keys)} keys (new model has these, checkpoint doesn't)")
                print(f"   Unexpected {len(unexpected_keys)} keys (checkpoint has these, new model doesn't)")
                print(f"   → Starting FRESH training with enhanced model (cannot resume from base model)")
                print(f"   → Enhanced model will be trained from random initialization\n")
                # Don't load optimizer/scheduler/metrics since we're starting fresh
                return
        
        # Only load optimizer/scheduler/metrics if architectures match
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch'] + 1  # Start from next epoch
        self.global_step = checkpoint['global_step']
        self.train_metrics = checkpoint['train_metrics']
        self.val_metrics = checkpoint['val_metrics']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.local_rank == 0:
            print(f"✓ Resumed from epoch {checkpoint['epoch']}, step {self.global_step}")
            print(f"✓ Best val loss so far: {self.best_val_loss:.6f}")

    
    def train(self, train_loader, val_loader, resume_from: str = None):
        """Main training loop"""
        
        # Load checkpoint if resuming
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
        
        if self.local_rank == 0:
            print("\n" + "="*80)
            if resume_from:
                print(f"Resuming 4-GPU training from epoch {self.epoch}...")
            else:
                print("Starting 4-GPU training with ENHANCED ATTENTION...")
            print("="*80 + "\n")
        
        for epoch in range(self.epoch, self.config['training']['num_epochs']):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            train_loader.sampler.set_epoch(epoch)
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            if self.local_rank == 0:
                print(f"\nEpoch {epoch} - Train metrics:")
                for k, v in train_losses.items():
                    if k in ['psnr', 'ssim']:
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v:.6f}")
            
            # Validate
            val_losses = self.validate(val_loader)
            
            if self.local_rank == 0:
                print(f"Epoch {epoch} - Val metrics:")
                for k, v in val_losses.items():
                    if k in ['psnr', 'ssim']:
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v:.6f}")
            
            # Check for best model
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            if self.local_rank == 0:
                print(f"\nCompleted epoch {epoch}/{self.config['training']['num_epochs']}\n")
        
        if self.local_rank == 0:
            print("\n" + "="*80)
            print("Training complete!")
            print(f"Best validation loss: {self.best_val_loss:.6f}")
            print("="*80 + "\n")


def main():
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Create trainer
    trainer = SpatialClusteringTrainer('config_spatial_clustering.json', local_rank)
    
    # Create dataset
    full_dataset = CTDRRDataset(num_samples=500)
    
    if local_rank == 0:
        print(f"Found {len(full_dataset)} patient folders")
    
    # Split into train/val
    train_size = 400
    val_size = 100
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # Per-GPU batch size
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    if local_rank == 0:
        print(f"\nDataset Info:")
        print(f"  Train patients: {len(train_dataset)}")
        print(f"  Val patients: {len(val_dataset)}")
        print(f"  Per-GPU batch size: 2")
        print(f"  Effective batch size: 8 (2 x 4 GPUs)")
        print(f"  Total epochs: 20")
        print(f"  Batches per epoch: {len(train_loader)}")
        print(f"\n🚀 ENHANCED MODEL with ALL attention mechanisms:")
        print(f"  ✓ Cross-Modal Attention (Frontal ↔ Lateral)")
        print(f"  ✓ 3D Spatial Attention")
        print(f"  ✓ Channel Attention")
        print(f"  ✓ Hierarchical Multi-Scale")
        print(f"  ✓ Cluster Interaction Attention\n")
    
    # Check for best checkpoint to resume from
    checkpoint_dir = Path('checkpoints/spatial_clustering')
    resume_from = None
    if checkpoint_dir.exists() and (checkpoint_dir / 'best.pth').exists():
        if local_rank == 0:
            response = input("Found existing checkpoint. Resume from best.pth? (y/n): ")
            if response.lower() == 'y':
                resume_from = str(checkpoint_dir / 'best.pth')
    
    # Broadcast resume decision to all ranks
    if dist.get_world_size() > 1:
        resume_tensor = torch.tensor(1 if resume_from else 0, device=torch.device(f'cuda:{local_rank}'))
        dist.broadcast(resume_tensor, src=0)
        if local_rank != 0 and resume_tensor.item() == 1:
            resume_from = str(checkpoint_dir / 'best.pth')
    
    # Train
    trainer.train(train_loader, val_loader, resume_from=resume_from)
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
