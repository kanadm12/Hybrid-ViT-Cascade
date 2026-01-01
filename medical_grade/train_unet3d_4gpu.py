"""
Medical-Grade Hybrid CNN-ViT Training Script (4 GPU DDP)

Trains the medical-grade 3D U-Net with Vision Transformers for CT reconstruction.
Expected performance: 18-20 dB PSNR (Milestone 1)

Usage:
    python train_unet3d_4gpu.py --config config_unet3d.json --world_size 4
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import json
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import nibabel as nib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset import PatientDRRDataset
from medical_grade.model_unet3d import HybridCNNViTUNet3D, MedicalGradeLoss


def setup_ddp(rank, world_size, port=12359):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://localhost:{port}',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def compute_metrics(pred, target):
    """Compute PSNR and SSIM"""
    # PSNR
    mse = torch.mean((pred - target) ** 2)
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    
    # SSIM (simplified)
    mu_x = torch.mean(pred)
    mu_y = torch.mean(target)
    sigma_x = torch.std(pred)
    sigma_y = torch.std(target)
    sigma_xy = torch.mean((pred - mu_x) * (target - mu_y))
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))
    
    return psnr.item(), ssim.item()


def train_epoch(model, loader, criterion, optimizer, scaler, rank, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    
    if rank == 0:
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
    else:
        pbar = loader
    
    for batch_idx, batch in enumerate(pbar):
        # Move to GPU
        xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
        target_ct = batch['ct_volume'].cuda(rank, non_blocking=True)
        
        # Reshape X-rays: (B, 2, H, W) -> (B, 2, 1, H, W)
        if xrays.dim() == 4:
            xrays = xrays.unsqueeze(2)
        
        optimizer.zero_grad()
        
        # Forward with mixed precision
        with autocast():
            pred_ct, aux_outputs = model(xrays)
            loss, loss_dict = criterion(pred_ct, target_ct, aux_outputs)
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        with torch.no_grad():
            psnr, ssim = compute_metrics(pred_ct, target_ct)
        
        total_loss += loss.item()
        total_psnr += psnr
        total_ssim += ssim
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{psnr:.2f}',
                'ssim': f'{ssim:.4f}'
            })
    
    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / len(loader)
    
    return avg_loss, avg_psnr, avg_ssim


def validate(model, loader, criterion, rank):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    
    with torch.no_grad():
        for batch in loader:
            xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
            target_ct = batch['ct_volume'].cuda(rank, non_blocking=True)
            
            if xrays.dim() == 4:
                xrays = xrays.unsqueeze(2)
            
            with autocast():
                pred_ct, aux_outputs = model(xrays)
                loss, loss_dict = criterion(pred_ct, target_ct, aux_outputs)
            
            psnr, ssim = compute_metrics(pred_ct, target_ct)
            
            total_loss += loss.item()
            total_psnr += psnr
            total_ssim += ssim
    
    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / len(loader)
    
    return avg_loss, avg_psnr, avg_ssim


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path, rank):
    """Save checkpoint (rank 0 only)"""
    if rank != 0:
        return
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    print(f'Saved checkpoint to {save_path}')


def main_worker(rank, world_size, config):
    """Main training worker"""
    # Setup DDP
    setup_ddp(rank, world_size, port=config['ddp_port'])
    
    if rank == 0:
        print(f'\n=== Medical-Grade Hybrid CNN-ViT Training ===')
        print(f'Target: 18-20 dB PSNR (Milestone 1)')
        print(f'GPUs: {world_size}')
        print(f'Batch size per GPU: {config["batch_size"]}')
        print(f'Total batch size: {config["batch_size"] * world_size}')
        print(f'Epochs: {config["num_epochs"]}\n')
    
    # Create model
    model = HybridCNNViTUNet3D(
        volume_size=tuple(config['volume_size']),
        xray_size=config['xray_size'],
        base_channels=config['base_channels']
    ).cuda(rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params / 1e6:.2f}M')
        print(f'Trainable parameters: {trainable_params / 1e6:.2f}M\n')
    
    # Loss function
    criterion = MedicalGradeLoss(
        l1_weight=config['loss_weights']['l1'],
        ssim_weight=config['loss_weights']['ssim'],
        perceptual_weight=config['loss_weights']['perceptual'],
        edge_weight=config['loss_weights']['edge'],
        deep_supervision_weight=config['loss_weights']['deep_supervision'],
        frequency_weight=config['loss_weights']['frequency']
    ).cuda(rank)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['learning_rate'] * 0.01
    )
    
    # AMP
    scaler = GradScaler()
    
    # Dataset
    train_dataset = PatientDRRDataset(
        data_dir=config['data_dir'],
        mode='train',
        volume_size=tuple(config['volume_size'])
    )
    
    val_dataset = PatientDRRDataset(
        data_dir=config['data_dir'],
        mode='val',
        volume_size=tuple(config['volume_size'])
    )
    
    # Samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    if rank == 0:
        print(f'Train samples: {len(train_dataset)}')
        print(f'Val samples: {len(val_dataset)}\n')
    
    # Training loop
    best_psnr = 0
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, config['num_epochs'] + 1):
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_psnr, train_ssim = train_epoch(
            model, train_loader, criterion, optimizer, scaler, rank, epoch
        )
        
        # Validate
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, rank)
        
        # Step scheduler
        scheduler.step()
        
        if rank == 0:
            print(f'\nEpoch {epoch}/{config["num_epochs"]}:')
            print(f'  Train - Loss: {train_loss:.4f} | PSNR: {train_psnr:.2f} dB | SSIM: {train_ssim:.4f}')
            print(f'  Val   - Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}')
            print(f'  LR: {scheduler.get_last_lr()[0]:.6f}\n')
        
        # Save checkpoint
        metrics = {
            'train_loss': train_loss,
            'train_psnr': train_psnr,
            'train_ssim': train_ssim,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim
        }
        
        # Save every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                save_dir / f'checkpoint_epoch_{epoch:03d}.pth',
                rank
            )
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                save_dir / 'best_model.pth',
                rank
            )
            if rank == 0:
                print(f'✓ New best model! PSNR: {best_psnr:.2f} dB\n')
    
    # Final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, config['num_epochs'], metrics,
        save_dir / 'final_model.pth',
        rank
    )
    
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--world_size', type=int, default=4, help='Number of GPUs')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Spawn processes
    torch.multiprocessing.spawn(
        main_worker,
        args=(args.world_size, config),
        nprocs=args.world_size,
        join=True
    )


if __name__ == '__main__':
    main()
