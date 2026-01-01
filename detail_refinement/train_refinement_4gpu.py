"""
Training script for detail refinement network (4 GPUs)
Two-stage approach: Base model → Detail refinement
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import PatientDRRDataset
from direct_regression.model_enhanced import EnhancedDirectModel
from model_refinement import DetailRefinementNetwork, DetailRefinementLoss


def setup_ddp(rank, world_size, port=12358):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def load_base_model(checkpoint_path, device):
    """Load frozen base model for generating coarse predictions"""
    base_model = EnhancedDirectModel(
        volume_size=(64, 64, 64),
        base_channels=128
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    
    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False
    
    return base_model


def calculate_psnr(pred, target):
    """Calculate PSNR"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def train_epoch(base_model, refinement_model, dataloader, criterion, optimizer, scaler, rank, epoch, config):
    """Train for one epoch"""
    refinement_model.train()
    
    epoch_losses = {'total': 0, 'l1': 0, 'frequency': 0, 'gradient': 0, 'perceptual': 0, 'consistency': 0}
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
        ct_volume = batch['ct_volume'].cuda(rank, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Stage 1: Generate coarse prediction with frozen base model
        with torch.no_grad():
            coarse_pred, _ = base_model(xrays)
        
        # Stage 2: Refine with detail network
        with torch.amp.autocast('cuda'):
            refined_pred, aux_outputs = refinement_model(coarse_pred, xrays)
            total_loss, loss_dict = criterion(refined_pred, ct_volume, coarse_pred, aux_outputs)
        
        # Backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(refinement_model.parameters(), config['training']['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        epoch_losses['total'] += total_loss.item()
        for key in loss_dict:
            epoch_losses[key] += loss_dict[key]
        num_batches += 1
        
        # Print progress
        if rank == 0 and (batch_idx % config['logging']['print_interval'] == 0):
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * config['training']['batch_size'] * dist.get_world_size() / elapsed
            
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {total_loss.item():.4f} | "
                  f"L1: {loss_dict['l1']:.4f} | "
                  f"Freq: {loss_dict['frequency']:.4f} | "
                  f"Grad: {loss_dict['gradient']:.4f} | "
                  f"{samples_per_sec:.2f} samples/s")
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


@torch.no_grad()
def validate(base_model, refinement_model, dataloader, criterion, rank, config):
    """Validation loop"""
    refinement_model.eval()
    
    val_losses = {'total': 0, 'l1': 0, 'frequency': 0, 'gradient': 0, 'perceptual': 0, 'consistency': 0}
    total_psnr_coarse = 0
    total_psnr_refined = 0
    num_batches = 0
    
    for batch in dataloader:
        xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
        ct_volume = batch['ct_volume'].cuda(rank, non_blocking=True)
        
        # Stage 1: Coarse prediction
        coarse_pred, _ = base_model(xrays)
        
        # Stage 2: Refined prediction
        refined_pred, aux_outputs = refinement_model(coarse_pred, xrays)
        
        # Calculate loss
        total_loss, loss_dict = criterion(refined_pred, ct_volume, coarse_pred, aux_outputs)
        
        # Calculate PSNR
        psnr_coarse = calculate_psnr(coarse_pred, ct_volume)
        psnr_refined = calculate_psnr(refined_pred, ct_volume)
        
        val_losses['total'] += total_loss.item()
        for key in loss_dict:
            val_losses[key] += loss_dict[key]
        total_psnr_coarse += psnr_coarse
        total_psnr_refined += psnr_refined
        num_batches += 1
    
    for key in val_losses:
        val_losses[key] /= num_batches
    avg_psnr_coarse = total_psnr_coarse / num_batches
    avg_psnr_refined = total_psnr_refined / num_batches
    
    return val_losses, avg_psnr_coarse, avg_psnr_refined


def train_ddp(rank, world_size, config):
    """Main training function for each GPU"""
    setup_ddp(rank, world_size)
    
    if rank == 0:
        print("\n" + "="*60)
        print("Detail Refinement Training")
        print("Two-Stage: Base Model → Detail Network")
        print("="*60)
    
    # Load frozen base model
    if rank == 0:
        print(f"\nLoading base model from: {config['model']['base_model_checkpoint']}")
    
    base_model = load_base_model(config['model']['base_model_checkpoint'], rank)
    
    # Create refinement model
    refinement_model = DetailRefinementNetwork(
        volume_size=tuple(config['model']['volume_size']),
        xray_size=config['model']['xray_size'],
        hidden_channels=config['model']['hidden_channels']
    ).cuda(rank)
    
    refinement_model = DDP(refinement_model, device_ids=[rank])
    
    if rank == 0:
        total_params = sum(p.numel() for p in refinement_model.parameters())
        trainable_params = sum(p.numel() for p in refinement_model.parameters() if p.requires_grad)
        print(f"\nRefinement model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Loss
    criterion = DetailRefinementLoss(
        l1_weight=config['loss']['l1_weight'],
        frequency_weight=config['loss']['frequency_weight'],
        gradient_weight=config['loss']['gradient_weight'],
        perceptual_weight=config['loss']['perceptual_weight'],
        consistency_weight=config['loss']['consistency_weight']
    ).cuda(rank)
    
    # Optimizer (only for refinement model)
    optimizer = torch.optim.AdamW(
        refinement_model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=1e-7
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Datasets
    train_dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_xray_size=config['model']['xray_size'],
        target_volume_size=tuple(config['model']['volume_size']),
        max_patients=config['data']['max_patients']
    )
    
    val_dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_xray_size=config['model']['xray_size'],
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
    
    if rank == 0:
        print(f"\nDataset: {len(train_dataset)} training patients")
        print(f"Batches per epoch: {len(train_loader)}")
        print(f"Effective batch size: {config['training']['batch_size'] * world_size}")
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    
    # Training loop
    best_psnr = 0.0
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        if rank == 0:
            print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            print("="*60)
        
        train_sampler.set_epoch(epoch)
        
        # Train
        train_losses = train_epoch(
            base_model, refinement_model, train_loader,
            criterion, optimizer, scaler, rank, epoch, config
        )
        
        # Validate
        val_losses, val_psnr_coarse, val_psnr_refined = validate(
            base_model, refinement_model, val_loader,
            criterion, rank, config
        )
        
        scheduler.step()
        
        if rank == 0:
            print(f"\nEpoch {epoch} Training Summary:")
            print(f"  Total Loss: {train_losses['total']:.4f}")
            print(f"  L1: {train_losses['l1']:.4f}")
            print(f"  Frequency: {train_losses['frequency']:.4f}")
            print(f"  Gradient: {train_losses['gradient']:.4f}")
            print(f"  Perceptual: {train_losses['perceptual']:.4f}")
            print(f"  Consistency: {train_losses['consistency']:.4f}")
            
            print(f"\nValidation Results:")
            print(f"  Total Loss: {val_losses['total']:.4f}")
            print(f"  PSNR Coarse: {val_psnr_coarse:.2f} dB")
            print(f"  PSNR Refined: {val_psnr_refined:.2f} dB")
            print(f"  PSNR Gain: +{val_psnr_refined - val_psnr_coarse:.2f} dB")
            
            # Save checkpoint
            if val_psnr_refined > best_psnr:
                best_psnr = val_psnr_refined
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': refinement_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_psnr_coarse': val_psnr_coarse,
                    'val_psnr_refined': val_psnr_refined,
                    'config': config
                }
                torch.save(checkpoint, os.path.join(config['checkpoint']['save_dir'], 'best_model.pt'))
                print(f"  ✓ New best model saved! PSNR: {val_psnr_refined:.2f} dB")
            
            # Save periodic checkpoint
            if epoch % config['checkpoint']['save_interval'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': refinement_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_psnr_coarse': val_psnr_coarse,
                    'val_psnr_refined': val_psnr_refined,
                    'config': config
                }
                torch.save(checkpoint, os.path.join(config['checkpoint']['save_dir'], f'epoch_{epoch}.pt'))
    
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='Train Detail Refinement Network')
    parser.add_argument('--config', type=str, default='config_refinement.json',
                        help='Path to config file')
    parser.add_argument('--world_size', type=int, default=4,
                        help='Number of GPUs')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Launch DDP training
    mp.spawn(
        train_ddp,
        args=(args.world_size, config),
        nprocs=args.world_size,
        join=True
    )


if __name__ == '__main__':
    main()
