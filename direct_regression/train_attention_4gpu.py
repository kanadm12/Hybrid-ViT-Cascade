"""
Training script for Attention-Enhanced Model
Supports 4-GPU distributed training with DDP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import sys
import json
from pathlib import Path
import time

sys.path.insert(0, '..')

from model_attention import AttentionEnhancedModel, EnhancedLossWithAttention
from utils.dataset import PatientDRRDataset


def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    
    dist.init_process_group(backend='nccl', init_method='env://', 
                           rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def train_epoch(model, dataloader, criterion, optimizer, scaler, rank, epoch, config):
    """Train for one epoch"""
    model.train()
    
    epoch_losses = {'total': 0, 'l1': 0, 'ssim': 0, 'perceptual': 0, 'edge': 0, 'multiscale': 0}
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
        ct_volume = batch['ct_volume'].cuda(rank, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with torch.amp.autocast('cuda'):
            predicted, aux_outputs = model(xrays)
            total_loss, loss_dict = criterion(predicted, ct_volume, aux_outputs)
        
        # Backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip_max_norm'])
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        epoch_losses['total'] += total_loss.item()
        for key, val in loss_dict.items():
            if key in epoch_losses:
                epoch_losses[key] += val
        num_batches += 1
        
        # Logging
        if rank == 0 and batch_idx % config['logging']['log_interval'] == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * config['training']['batch_size'] * dist.get_world_size() / elapsed
            
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {total_loss.item():.4f} | "
                  f"L1: {loss_dict['l1']:.4f} | "
                  f"SSIM: {loss_dict['ssim']:.4f} | "
                  f"Perc: {loss_dict['perceptual']:.4f} | "
                  f"Edge: {loss_dict['edge']:.4f} | "
                  f"{samples_per_sec:.2f} samples/s")
    
    # Average
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def validate(model, dataloader, criterion, rank):
    """Validation loop"""
    model.eval()
    
    val_losses = {'total': 0, 'l1': 0, 'ssim': 0, 'perceptual': 0, 'edge': 0, 'multiscale': 0}
    num_batches = 0
    total_psnr = 0
    
    with torch.no_grad():
        for batch in dataloader:
            xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
            ct_volume = batch['ct_volume'].cuda(rank, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                predicted, aux_outputs = model(xrays)
                total_loss, loss_dict = criterion(predicted, ct_volume, aux_outputs)
            
            # PSNR
            mse = F.mse_loss(predicted, ct_volume)
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
            total_psnr += psnr.item()
            
            val_losses['total'] += total_loss.item()
            for key, val in loss_dict.items():
                if key in val_losses:
                    val_losses[key] += val
            num_batches += 1
    
    for key in val_losses:
        val_losses[key] /= num_batches
    avg_psnr = total_psnr / num_batches
    
    return val_losses, avg_psnr


def train_ddp(rank, world_size, config):
    """Main training function for each GPU"""
    setup_ddp(rank, world_size)
    
    if rank == 0:
        print("\n" + "="*60)
        print("Starting Training - Attention-Enhanced Model")
        print("="*60)
    
    # Model
    model = AttentionEnhancedModel(
        volume_size=tuple(config['model']['volume_size']),
        base_channels=config['model']['base_channels']
    ).cuda(rank)
    
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Loss
    criterion = EnhancedLossWithAttention(
        l1_weight=config['loss']['l1_weight'],
        ssim_weight=config['loss']['ssim_weight'],
        perceptual_weight=config['loss']['perceptual_weight'],
        edge_weight=config['loss']['edge_weight'],
        multiscale_weight=config['loss']['multiscale_weight']
    ).cuda(rank)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=1e-6
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Datasets
    train_dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_xray_size=512,
        target_volume_size=tuple(config['model']['volume_size']),
        max_patients=config['data']['max_patients']
    )
    
    val_dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_xray_size=512,
        target_volume_size=tuple(config['model']['volume_size']),
        max_patients=100
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Training loop
    best_psnr = 0
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        if rank == 0:
            print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
            print("="*60)
        
        train_sampler.set_epoch(epoch)
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler, rank, epoch, config
        )
        
        if rank == 0:
            print(f"\nEpoch {epoch} Training Summary:")
            print(f"  Total Loss: {train_losses['total']:.4f}")
            print(f"  L1: {train_losses['l1']:.4f}")
            print(f"  SSIM: {train_losses['ssim']:.4f}")
            print(f"  Perceptual: {train_losses['perceptual']:.4f}")
            print(f"  Edge: {train_losses['edge']:.4f}")
            print(f"  Multiscale: {train_losses['multiscale']:.4f}")
        
        # Validate
        if epoch % config['logging']['val_interval'] == 0:
            val_losses, val_psnr = validate(model, val_loader, criterion, rank)
            
            if rank == 0:
                print(f"\nValidation Results:")
                print(f"  Total Loss: {val_losses['total']:.4f}")
                print(f"  PSNR: {val_psnr:.2f} dB")
                
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    Path('checkpoints_attention').mkdir(exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_psnr': val_psnr,
                        'best_psnr': best_psnr,
                        'config': config
                    }, 'checkpoints_attention/best_model.pt')
                    print(f"  ✓ New best model saved! PSNR: {best_psnr:.2f} dB")
        
        scheduler.step()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"{'='*60}")
    
    cleanup_ddp()


def main():
    config_path = 'config_attention.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training")
    
    mp.spawn(
        train_ddp,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
