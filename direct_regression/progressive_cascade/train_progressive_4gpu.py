"""
Progressive Training Pipeline for Multi-Scale CT Reconstruction
Stage 1: Train 64³ base for 50 epochs
Stage 2: Freeze stage 1, train 128³ refiner for 30 epochs
Stage 3: Freeze stages 1+2, train 256³ refiner for 20 epochs

Supports 4-GPU DDP training with mixed precision and gradient checkpointing
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
from datetime import datetime
import numpy as np

sys.path.insert(0, '../..')

from progressive_cascade.model_progressive import ProgressiveCascadeModel
from progressive_cascade.loss_multiscale import MultiScaleLoss, compute_psnr, compute_ssim_metric
from utils.dataset import PatientDRRDataset


def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12366'  # Different port to avoid conflicts
    
    dist.init_process_group(backend='nccl', init_method='env://', 
                           rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def resize_ct_volume(ct_volume, target_size):
    """
    Resize CT volume to target resolution
    Args:
        ct_volume: (B, 1, D, H, W)
        target_size: tuple (D, H, W)
    Returns:
        resized_volume: (B, 1, D', H', W')
    """
    return F.interpolate(ct_volume, size=target_size, 
                        mode='trilinear', align_corners=False)


def train_epoch(model, dataloader, criterion, optimizer, scaler, rank, epoch, 
                stage, config, max_stage=1):
    """
    Train for one epoch
    Args:
        max_stage: Maximum stage to train (1, 2, or 3)
    """
    model.train()
    
    epoch_losses = {'total': 0}
    num_batches = 0
    
    # Determine target size based on stage
    if max_stage == 1:
        target_sizes = [(64, 64, 64)]
    elif max_stage == 2:
        target_sizes = [(64, 64, 64), (128, 128, 128)]
    else:  # max_stage == 3
        target_sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
        ct_volume_orig = batch['ct_volume'].cuda(rank, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with torch.amp.autocast('cuda'):
            # Forward through cascade
            if max_stage == 1:
                pred = model(xrays, max_stage=1)
                target = resize_ct_volume(ct_volume_orig, target_sizes[0])
                loss_dict = criterion(pred, target, stage=1)
            
            elif max_stage == 2:
                outputs = model(xrays, return_intermediate=True, max_stage=2)
                
                # Compute stage 2 loss (on 128³)
                pred = outputs['stage2']
                target = resize_ct_volume(ct_volume_orig, target_sizes[1])
                loss_dict = criterion(pred, target, stage=2)
            
            else:  # max_stage == 3
                outputs = model(xrays, return_intermediate=True, max_stage=3)
                
                # Compute stage 3 loss (on 256³)
                pred = outputs['stage3']
                target = resize_ct_volume(ct_volume_orig, target_sizes[2])
                loss_dict = criterion(pred, target, stage=3, input_xrays=xrays)
            
            total_loss = loss_dict['total_loss']
        
        # Backward with gradient scaling
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        for k, v in loss_dict.items():
            if k not in epoch_losses:
                epoch_losses[k] = 0
            epoch_losses[k] += v.item()
        
        num_batches += 1
        
        # Log progress
        if rank == 0 and batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed
            print(f"Epoch {epoch} | Stage {max_stage} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {total_loss.item():.4f} | {batches_per_sec:.2f} batch/s")
    
    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= num_batches
    
    return epoch_losses


def validate(model, dataloader, criterion, rank, stage, max_stage=1):
    """Validate the model"""
    model.eval()
    
    val_losses = {'total': 0}
    val_psnr = 0
    val_ssim = 0
    num_batches = 0
    
    # Determine target size
    if max_stage == 1:
        target_size = (64, 64, 64)
    elif max_stage == 2:
        target_size = (128, 128, 128)
    else:
        target_size = (256, 256, 256)
    
    with torch.no_grad():
        for batch in dataloader:
            xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
            ct_volume_orig = batch['ct_volume'].cuda(rank, non_blocking=True)
            target = resize_ct_volume(ct_volume_orig, target_size)
            
            with torch.amp.autocast('cuda'):
                # Forward
                pred = model(xrays, max_stage=max_stage)
                
                # Compute loss
                if max_stage == 3:
                    loss_dict = criterion(pred, target, stage=max_stage, input_xrays=xrays)
                else:
                    loss_dict = criterion(pred, target, stage=max_stage)
            
            # Accumulate losses
            for k, v in loss_dict.items():
                if k not in val_losses:
                    val_losses[k] = 0
                val_losses[k] += v.item()
            
            # Compute metrics
            val_psnr += compute_psnr(pred, target)
            val_ssim += compute_ssim_metric(pred, target)
            
            num_batches += 1
    
    # Average
    for k in val_losses:
        val_losses[k] /= num_batches
    val_psnr /= num_batches
    val_ssim /= num_batches
    
    val_losses['psnr'] = val_psnr
    val_losses['ssim'] = val_ssim
    
    return val_losses


def train_stage(rank, world_size, config, stage, checkpoint_dir):
    """
    Train a specific stage
    Args:
        stage: 1, 2, or 3
    """
    setup_ddp(rank, world_size)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training Stage {stage}")
        print(f"{'='*60}\n")
    
    # Create model
    model = ProgressiveCascadeModel(
        xray_img_size=config['model']['xray_img_size'],
        xray_feature_dim=config['model']['xray_feature_dim'],
        voxel_dim=config['model']['voxel_dim'],
        use_gradient_checkpointing=(stage == 3)  # Only for stage 3
    ).cuda(rank)
    
    # Load previous stage weights if not stage 1
    if stage > 1:
        prev_checkpoint = checkpoint_dir / f"stage{stage-1}_best.pth"
        if prev_checkpoint.exists():
            if rank == 0:
                print(f"Loading Stage {stage-1} checkpoint: {prev_checkpoint}")
            checkpoint = torch.load(prev_checkpoint, map_location=f'cuda:{rank}')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Freeze previous stages
            for prev_stage in range(1, stage):
                model.freeze_stage(prev_stage)
        else:
            if rank == 0:
                print(f"Warning: Stage {stage-1} checkpoint not found!")
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Create loss function
    loss_config = {
        'stage1': config['loss']['stage1'],
        'stage2': config['loss']['stage2'],
        'stage3': config['loss']['stage3']
    }
    criterion = MultiScaleLoss(config=loss_config).cuda(rank)
    
    # Optimizer and scheduler
    stage_key = f'stage{stage}'
    lr = config['training'][stage_key]['learning_rate']
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training'][stage_key]['num_epochs'],
        eta_min=lr * 0.1
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Dataset
    dataset_path = config['data']['dataset_path']
    train_dataset = PatientDRRDataset(
        root_dir=dataset_path,
        max_patients=config['data']['max_patients'],
        split='train',
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split']
    )
    
    val_dataset = PatientDRRDataset(
        root_dir=dataset_path,
        max_patients=config['data']['max_patients'],
        split='val',
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split']
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    # Adjust batch size for stage 3 (memory constraints)
    batch_size = config['training'][stage_key]['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Training loop
    num_epochs = config['training'][stage_key]['num_epochs']
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        train_sampler.set_epoch(epoch)
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            rank, epoch, stage, config, max_stage=stage
        )
        
        # Validate
        val_losses = validate(
            model, val_loader, criterion, rank, stage, max_stage=stage
        )
        
        # Step scheduler
        scheduler.step()
        
        # Log
        if rank == 0:
            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"Train Loss: {train_losses['total']:.4f}")
            print(f"Val Loss: {val_losses['total']:.4f} | "
                  f"PSNR: {val_losses['psnr']:.2f} dB | "
                  f"SSIM: {val_losses['ssim']:.4f}")
            
            # Save checkpoint
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_losses['total'],
                    'val_psnr': val_losses['psnr'],
                    'val_ssim': val_losses['ssim'],
                    'config': config
                }
                save_path = checkpoint_dir / f"stage{stage}_best.pth"
                torch.save(checkpoint, save_path)
                print(f"✓ Saved best checkpoint: {save_path}")
            
            # Save regular checkpoint
            if epoch % config['checkpoints']['save_every'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_losses['total'],
                    'config': config
                }
                save_path = checkpoint_dir / f"stage{stage}_epoch{epoch}.pth"
                torch.save(checkpoint, save_path)
                print(f"✓ Saved checkpoint: {save_path}")
    
    cleanup_ddp()


def main_worker(rank, world_size, config):
    """Main training worker"""
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoints']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train each stage sequentially
    for stage in range(1, 4):
        if rank == 0:
            print(f"\n{'#'*60}")
            print(f"# STAGE {stage} TRAINING")
            print(f"{'#'*60}\n")
        
        train_stage(rank, world_size, config, stage, checkpoint_dir)
        
        # Synchronize before moving to next stage
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def main():
    """Main entry point"""
    # Load config
    config_path = Path(__file__).parent / "config_progressive.json"
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Please create config_progressive.json first!")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("="*60)
    print("Progressive Multi-Scale CT Reconstruction Training")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config_path}")
    print("="*60)
    
    # Number of GPUs
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    
    if world_size < 1:
        print("Error: No GPUs available!")
        return
    
    # Launch distributed training
    mp.spawn(
        main_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
