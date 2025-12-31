"""
4 GPU Distributed Training for Enhanced Direct Regression Model
Uses PyTorch DDP (DistributedDataParallel) for efficient multi-GPU training
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

from model_enhanced import EnhancedDirectModel, EnhancedLoss
from utils.dataset import PatientDRRDataset


def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(backend='nccl', init_method='env://', 
                           rank=rank, world_size=world_size)
    
    # Set device
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
        # Handle dict format
        if isinstance(batch, dict):
            xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
            target = batch['ct_volume'].cuda(rank, non_blocking=True)
        else:
            xrays = batch['xrays'].cuda(rank, non_blocking=True)
            target = batch['ct_volume'].cuda(rank, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with torch.amp.autocast('cuda'):
            predicted, aux_outputs = model(xrays)
            loss_dict = criterion(predicted, target, aux_outputs)
            loss = loss_dict['total_loss']
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip_max_norm'])
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        epoch_losses['total'] += loss_dict['total_loss'].item()
        epoch_losses['l1'] += loss_dict['l1_loss'].item()
        epoch_losses['ssim'] += loss_dict['ssim_loss'].item()
        epoch_losses['perceptual'] += loss_dict['perceptual_loss'].item()
        epoch_losses['edge'] += loss_dict['edge_loss'].item()
        epoch_losses['multiscale'] += loss_dict['multiscale_loss'].item()
        num_batches += 1
        
        # Logging (only rank 0)
        if rank == 0 and batch_idx % config['logging']['log_interval'] == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * config['training']['batch_size'] * config['world_size'] / elapsed
            
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} | "
                  f"L1: {loss_dict['l1_loss'].item():.4f} | "
                  f"SSIM: {loss_dict['ssim_loss'].item():.4f} | "
                  f"Perc: {loss_dict['perceptual_loss'].item():.4f} | "
                  f"Edge: {loss_dict['edge_loss'].item():.4f} | "
                  f"{samples_per_sec:.2f} samples/s")
    
    # Average losses
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
            if isinstance(batch, dict):
                xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
                target = batch['ct_volume'].cuda(rank, non_blocking=True)
            else:
                xrays = batch['xrays'].cuda(rank, non_blocking=True)
                target = batch['ct_volume'].cuda(rank, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                predicted, aux_outputs = model(xrays)
                loss_dict = criterion(predicted, target, aux_outputs)
            
            # Accumulate losses
            val_losses['total'] += loss_dict['total_loss'].item()
            val_losses['l1'] += loss_dict['l1_loss'].item()
            val_losses['ssim'] += loss_dict['ssim_loss'].item()
            val_losses['perceptual'] += loss_dict['perceptual_loss'].item()
            val_losses['edge'] += loss_dict['edge_loss'].item()
            val_losses['multiscale'] += loss_dict['multiscale_loss'].item()
            
            # Compute PSNR
            mse = F.mse_loss(predicted, target)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            total_psnr += psnr.item()
            
            num_batches += 1
    
    # Average
    for key in val_losses:
        val_losses[key] /= num_batches
    avg_psnr = total_psnr / num_batches
    
    return val_losses, avg_psnr


def train_ddp(rank, world_size, config):
    """Main training function for each GPU"""
    
    # Setup DDP
    setup_ddp(rank, world_size)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Enhanced Direct Regression - 4 GPU Training")
        print(f"{'='*60}")
        print(f"World size: {world_size} GPUs")
        print(f"Batch size per GPU: {config['training']['batch_size']}")
        print(f"Effective batch size: {config['training']['batch_size'] * world_size}")
        print(f"Epochs: {config['training']['num_epochs']}")
    
    # Create model
    model = EnhancedDirectModel(
        volume_size=tuple(config['model']['volume_size']),
        base_channels=config['model']['base_channels']
    ).cuda(rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Create loss and optimizer
    criterion = EnhancedLoss(**config['loss'])
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=1e-6
    )
    
    # Gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    # Create datasets
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
    
    # Distributed samplers
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
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}\n")
    
    # Training loop
    best_psnr = 0.0
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # Set epoch for sampler (important for proper shuffling)
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*60}")
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            rank, epoch, config
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
                
                # Save best model (only rank 0)
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    Path('checkpoints_enhanced').mkdir(exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'psnr': val_psnr,
                        'config': config
                    }, 'checkpoints_enhanced/best_model.pt')
                    print(f"  ✓ New best model saved! PSNR: {best_psnr:.2f} dB")
        
        # Save checkpoint periodically (only rank 0)
        if rank == 0 and epoch % config['logging']['save_checkpoint_interval'] == 0:
            Path('checkpoints_enhanced').mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
            }, f'checkpoints_enhanced/epoch_{epoch}.pt')
        
        # Step scheduler
        scheduler.step()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"{'='*60}")
    
    cleanup_ddp()


def main():
    # Load config
    config_path = 'config_enhanced.json'
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        print("Creating default config...")
        
        default_config = {
            "model": {
                "volume_size": [64, 64, 64],
                "base_channels": 64
            },
            "training": {
                "num_epochs": 50,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "grad_clip_max_norm": 1.0
            },
            "loss": {
                "l1_weight": 1.0,
                "ssim_weight": 0.5,
                "perceptual_weight": 0.1,
                "edge_weight": 0.1,
                "multiscale_weight": 0.3
            },
            "data": {
                "dataset_path": "/workspace/drr_patient_data",
                "max_patients": 1000,
                "num_workers": 4
            },
            "logging": {
                "log_interval": 10,
                "val_interval": 1,
                "save_checkpoint_interval": 5
            },
            "world_size": 4
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        config = default_config
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Get world size (number of GPUs)
    world_size = config['world_size']
    
    # Spawn processes for DDP
    mp.spawn(
        train_ddp,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
