"""
Training script for 256³ Direct CT Regression with 4 GPUs
Memory-optimized with gradient accumulation and checkpointing
"""
import os
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import OneCycleLR

import sys
sys.path.append(str(Path(__file__).parent.parent))

from model_direct_256_v2 import DirectCTRegression256
from utils.dataset import PatientDRRDataset


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'  # Different port from refinement training
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def ssim_loss(pred, target, window_size=11):
    """3D SSIM loss (simplified for speed)"""
    # Use smaller patches for memory efficiency
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool3d(pred, window_size, stride=window_size//2)
    mu2 = F.avg_pool3d(target, window_size, stride=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool3d(pred * pred, window_size, stride=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool3d(target * target, window_size, stride=window_size//2) - mu2_sq
    sigma12 = F.avg_pool3d(pred * target, window_size, stride=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return 1 - ssim_map.mean()


def train_epoch(model, dataloader, optimizer, scaler, rank, epoch, config):
    """Training loop with gradient accumulation"""
    model.train()
    
    epoch_losses = {'total': 0, 'l1': 0, 'ssim': 0}
    num_batches = 0
    
    start_time = time.time()
    accumulation_steps = config['training']['gradient_accumulation_steps']
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, dict):
            xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
            target = batch['ct_volume'].cuda(rank, non_blocking=True)
        else:
            xrays, target = batch
            xrays = xrays.cuda(rank, non_blocking=True)
            target = target.cuda(rank, non_blocking=True)
        
        # Forward with mixed precision
        with torch.amp.autocast('cuda'):
            predicted = model(xrays)
            
            # Compute losses
            l1_loss = F.l1_loss(predicted, target)
            ssim_loss_val = ssim_loss(predicted, target)
            
            total_loss = (
                config['loss']['l1_weight'] * l1_loss +
                config['loss']['ssim_weight'] * ssim_loss_val
            )
            
            # Scale loss for gradient accumulation
            total_loss = total_loss / accumulation_steps
        
        # Backward
        scaler.scale(total_loss).backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip_max_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Accumulate losses (unscaled)
        epoch_losses['total'] += total_loss.item() * accumulation_steps
        epoch_losses['l1'] += l1_loss.item()
        epoch_losses['ssim'] += ssim_loss_val.item()
        num_batches += 1
        
        # Logging
        if rank == 0 and batch_idx % config['logging']['log_interval'] == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * config['training']['batch_size'] * config['world_size'] / elapsed
            
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {total_loss.item() * accumulation_steps:.4f} | "
                  f"L1: {l1_loss.item():.4f} | "
                  f"SSIM: {ssim_loss_val.item():.4f} | "
                  f"{samples_per_sec:.2f} samples/s")
    
    # Average
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def validate(model, dataloader, rank):
    """Validation loop"""
    model.eval()
    
    total_loss = 0
    total_l1 = 0
    total_psnr = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
                target = batch['ct_volume'].cuda(rank, non_blocking=True)
            else:
                xrays, target = batch
                xrays = xrays.cuda(rank, non_blocking=True)
                target = target.cuda(rank, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                predicted = model(xrays)
                
                # Losses
                l1_loss = F.l1_loss(predicted, target)
                ssim_loss_val = ssim_loss(predicted, target)
                
                total_loss += l1_loss.item() + 0.1 * ssim_loss_val.item()
                total_l1 += l1_loss.item()
                
                # PSNR
                mse = F.mse_loss(predicted, target)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
                total_psnr += psnr.item()
                
                num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_l1 = total_l1 / num_batches
    avg_psnr = total_psnr / num_batches
    
    return avg_loss, avg_l1, avg_psnr


def train_ddp(rank, world_size, config, resume_checkpoint=None):
    """Main training function"""
    
    setup_ddp(rank, world_size)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Direct 256³ CT Regression Training")
        print(f"{'='*60}")
    
    # Model (same architecture as successful 64³ model)
    model = DirectCTRegression256(
        volume_size=(256, 256, 256),
        xray_img_size=config['model']['xray_size'],
        voxel_dim=256,
        vit_depth=4,
        num_heads=4,
        xray_feature_dim=512,
        use_checkpointing=config['model']['use_checkpointing']
    ).cuda(rank)
    
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler (CosineAnnealing like 64³ model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=1e-7
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Resume from checkpoint if provided
    start_epoch = 1
    best_psnr = 0
    
    if resume_checkpoint is not None:
        if rank == 0:
            print(f"\n✓ Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=f'cuda:{rank}', weights_only=False)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('psnr', 0)
        if rank == 0:
            print(f"✓ Resuming from epoch {checkpoint['epoch']}, Best PSNR: {best_psnr:.2f} dB")
            print(f"✓ New learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Datasets with 80/20 split
    full_dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_xray_size=config['model']['xray_size'],
        target_volume_size=(config['model']['volume_size'],) * 3,
        max_patients=config['data']['max_patients']
    )
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    indices = list(range(total_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    if rank == 0:
        print(f"\nDataset split: {train_size} train, {val_size} val (from {total_size} total patients)")
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['data']['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['data']['num_workers'] > 0 else False
    )
    
    # Training loop
    best_psnr = best_psnr  # Use loaded value if resuming
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Starting Training from Epoch {start_epoch}")
        print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
            print(f"{'='*60}")
        
        train_losses = train_epoch(
            model, train_loader, optimizer, scaler, rank, epoch, config
        )
        
        if rank == 0:
            print(f"\nEpoch {epoch} Training:")
            print(f"  Total Loss: {train_losses['total']:.4f}")
            print(f"  L1: {train_losses['l1']:.4f}")
            print(f"  SSIM: {train_losses['ssim']:.4f}")
        
        # Validate
        if epoch % config['logging']['val_interval'] == 0:
            val_loss, val_l1, val_psnr = validate(model, val_loader, rank)
            
            if rank == 0:
                print(f"\nValidation:")
                print(f"  PSNR (256³): {val_psnr:.2f} dB")
                print(f"  L1 Loss: {val_l1:.4f}")
                
                # Save best
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    Path(config['checkpoint_dir']).mkdir(exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'psnr': val_psnr,
                        'config': config
                    }, f"{config['checkpoint_dir']}/best_model_256.pt")
                    print(f"  ✓ New best model! PSNR: {best_psnr:.2f} dB")
        
        # Save checkpoint
        
        # Step scheduler after validation
        scheduler.step()
        if rank == 0 and epoch % config['logging']['save_checkpoint_interval'] == 0:
            Path(config['checkpoint_dir']).mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': val_psnr if 'val_psnr' in locals() else 0,
                'config': config
            }, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch}.pt")
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training Complete! Best PSNR: {best_psnr:.2f} dB")
        print(f"{'='*60}\n")
    
    cleanup_ddp()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / "config_direct_256.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    world_size = config['world_size']
    
    mp.spawn(
        train_ddp,
        args=(world_size, config, args.resume),
        nprocs=world_size,
        join=True
    )
