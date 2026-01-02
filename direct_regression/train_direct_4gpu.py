"""
Training script for Direct CT Regression (no diffusion)
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

from model_direct import DirectCTRegression, DirectRegressionLoss
from utils.dataset import PatientDRRDataset


def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend='nccl', init_method='env://', 
                           rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def compute_psnr(pred, target):
    """Compute PSNR between predicted and target volumes"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
    return psnr.item()


def train_epoch(model, dataloader, criterion, optimizer, scaler, rank, epoch, config):
    """Train for one epoch"""
    model.train()
    
    epoch_losses = {'total': 0, 'l1': 0, 'ssim': 0}
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
        ct_volume = batch['ct_volume'].cuda(rank, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with torch.amp.autocast('cuda'):
            predicted = model(xrays)
            loss_dict = criterion(predicted, ct_volume)
            total_loss = loss_dict['total_loss']
        
        # Backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        epoch_losses['total'] += total_loss.item()
        epoch_losses['l1'] += loss_dict['l1_loss'].item()
        epoch_losses['ssim'] += loss_dict['ssim_loss'].item()
        num_batches += 1
        
        # Logging
        if rank == 0 and batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * config['training']['batch_size'] * dist.get_world_size() / elapsed
            
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {total_loss.item():.4f} | "
                  f"L1: {loss_dict['l1_loss'].item():.4f} | "
                  f"SSIM: {loss_dict['ssim_loss'].item():.4f} | "
                  f"{samples_per_sec:.2f} samples/s")
    
    # Average
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def validate(model, dataloader, criterion, rank):
    """Validation loop"""
    model.eval()
    
    val_losses = {'total': 0, 'l1': 0, 'ssim': 0}
    num_batches = 0
    total_psnr = 0
    
    with torch.no_grad():
        for batch in dataloader:
            xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
            ct_volume = batch['ct_volume'].cuda(rank, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                predicted = model(xrays)
                loss_dict = criterion(predicted, ct_volume)
                total_loss = loss_dict['total_loss']
            
            # PSNR
            psnr = compute_psnr(predicted, ct_volume)
            total_psnr += psnr
            
            val_losses['total'] += total_loss.item()
            val_losses['l1'] += loss_dict['l1_loss'].item()
            val_losses['ssim'] += loss_dict['ssim_loss'].item()
            num_batches += 1
    
    for key in val_losses:
        val_losses[key] /= num_batches
    avg_psnr = total_psnr / num_batches
    
    return val_losses, avg_psnr


def train_ddp(rank, world_size, config, resume_from=None):
    """Main training function for each GPU"""
    setup_ddp(rank, world_size)
    
    if rank == 0:
        print("\n" + "="*80)
        print("DIRECT CT REGRESSION (NO DIFFUSION) - 4-GPU Training")
        print("="*80)
    
    # Model
    model = DirectCTRegression(**config['model']).cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Loss
    criterion = DirectRegressionLoss(
        l1_weight=config['training']['l1_weight'],
        ssim_weight=config['training']['ssim_weight']
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
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_psnr = 0
    
    if resume_from is not None:
        if rank == 0:
            print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=f'cuda:{rank}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', checkpoint.get('val_psnr', 0))
        if rank == 0:
            print(f"Resuming from epoch {checkpoint['epoch']}")
            print(f"Best PSNR so far: {best_psnr:.2f} dB")
    
    # Datasets
    train_dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_xray_size=512,
        target_volume_size=tuple(config['model']['volume_size']),
        max_patients=config['data']['max_patients'],
        validate_alignment=False
    )
    
    val_dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_xray_size=512,
        target_volume_size=tuple(config['model']['volume_size']),
        max_patients=min(50, config['data']['max_patients']),  # Use subset for validation
        validate_alignment=False
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
    
    if rank == 0:
        print(f"\nDataset Info:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Batch size per GPU: {config['training']['batch_size']}")
        print(f"  Effective batch size: {config['training']['batch_size'] * world_size}")
    
    # Training loop
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{config['training']['num_epochs']}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
            print("="*80)
        
        train_sampler.set_epoch(epoch)
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler, rank, epoch, config
        )
        
        if rank == 0:
            print(f"\nEpoch {epoch} Training Summary:")
            print(f"  Total Loss: {train_losses['total']:.4f}")
            print(f"  L1 Loss: {train_losses['l1']:.4f}")
            print(f"  SSIM Loss: {train_losses['ssim']:.4f}")
        
        # Validate every epoch
        val_losses, val_psnr = validate(model, val_loader, criterion, rank)
        
        if rank == 0:
            print(f"\nValidation Results:")
            print(f"  Total Loss: {val_losses['total']:.4f}")
            print(f"  L1 Loss: {val_losses['l1']:.4f}")
            print(f"  SSIM Loss: {val_losses['ssim']:.4f}")
            print(f"  PSNR: {val_psnr:.2f} dB")
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                Path(config['checkpoints']['save_dir']).mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_psnr': val_psnr,
                    'best_psnr': best_psnr,
                    'config': config
                }, f"{config['checkpoints']['save_dir']}/best_model.pt")
                print(f"  ✓ New best model saved! PSNR: {best_psnr:.2f} dB")
            
            # Save periodic checkpoint (only at specified intervals)
            if epoch % config['checkpoints']['save_every'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_psnr': val_psnr,
                    'config': config
                }, f"{config['checkpoints']['save_dir']}/checkpoint_epoch_{epoch}.pt")
                print(f"  ✓ Periodic checkpoint saved at epoch {epoch}")
        
        scheduler.step()
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"{'='*80}")
    
    cleanup_ddp()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_direct.json')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    world_size = torch.cuda.device_count()
    
    print(f"\n{'='*80}")
    print(f"Direct CT Regression Training")
    print(f"Using {world_size} GPUs")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print(f"{'='*80}")
    
    mp.spawn(
        train_ddp,
        args=(world_size, config, args.resume),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
