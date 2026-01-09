"""
Training script for Coordinate-Based Transformer - 4 GPU DDP
Processes 3D volume as a set of coordinates with transformer attention
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import json
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coord_transformer.model_coord_transformer import CoordinateTransformer, CoordinateTransformerLoss
from utils.dataset import PatientDRRDataset


def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def compute_psnr(pred, target):
    """Compute PSNR"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
    return psnr.item()


def compute_ssim_3d(pred, target, window_size=11):
    """Compute SSIM for 3D volumes"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_pred = torch.nn.functional.avg_pool3d(pred, window_size, stride=1, padding=window_size//2)
    mu_target = torch.nn.functional.avg_pool3d(target, window_size, stride=1, padding=window_size//2)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = torch.nn.functional.avg_pool3d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
    sigma_target_sq = torch.nn.functional.avg_pool3d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
    sigma_pred_target = torch.nn.functional.avg_pool3d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target
    
    ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
           ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    
    return ssim.mean().item()


def train_epoch(model, train_loader, optimizer, loss_fn, scaler, device, rank, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_l1 = 0
    total_gradient = 0
    total_smoothness = 0
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch_idx, batch_data in enumerate(pbar):
        if isinstance(batch_data, dict):
            volumes = batch_data['ct_volume'].to(device)
            xrays = batch_data['drr_stacked'].to(device)
        else:
            volumes, xrays = batch_data
            volumes = volumes.to(device)
            xrays = xrays.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            pred_volume = model(xrays)
            loss_dict = loss_fn(pred_volume, volumes)
            loss = loss_dict['total_loss']
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_l1 += loss_dict['l1_loss'].item()
        total_gradient += loss_dict['gradient_loss'].item()
        total_smoothness += loss_dict['smoothness_loss'].item()
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'l1': f"{loss_dict['l1_loss'].item():.4f}",
                'grad': f"{loss_dict['gradient_loss'].item():.4f}"
            })
    
    n = len(train_loader)
    return {
        'loss': total_loss / n,
        'l1': total_l1 / n,
        'gradient': total_gradient / n,
        'smoothness': total_smoothness / n
    }


def validate(model, val_loader, loss_fn, device, rank):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    n_samples = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            if isinstance(batch_data, dict):
                volumes = batch_data['ct_volume'].to(device)
                xrays = batch_data['drr_stacked'].to(device)
            else:
                volumes, xrays = batch_data
                volumes = volumes.to(device)
                xrays = xrays.to(device)
            
            pred_volume = model(xrays)
            loss_dict = loss_fn(pred_volume, volumes)
            
            total_loss += loss_dict['total_loss'].item()
            
            # Compute metrics
            for i in range(pred_volume.shape[0]):
                psnr = compute_psnr(pred_volume[i:i+1], volumes[i:i+1])
                ssim = compute_ssim_3d(pred_volume[i:i+1], volumes[i:i+1])
                total_psnr += psnr
                total_ssim += ssim
                n_samples += 1
    
    return {
        'loss': total_loss / len(val_loader),
        'psnr': total_psnr / n_samples,
        'ssim': total_ssim / n_samples
    }


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main_worker(rank, world_size, config):
    """Main training function for each GPU"""
    
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print("=" * 80)
        print("COORDINATE-BASED TRANSFORMER CT RECONSTRUCTION - 4 GPU DDP")
        print("Revolutionary Set-Based Architecture with Cross-Attention")
        print("=" * 80)
        print(f"Training on {world_size} GPUs")
        print(f"Volume size: {config['model']['volume_size']}")
        print(f"Batch size per GPU: {config['training']['batch_size']}")
        print(f"Effective batch size: {config['training']['batch_size'] * world_size}")
        print(f"Transformer blocks: {config['model']['num_transformer_blocks']}")
        print(f"Attention heads: {config['model']['num_heads']}")
        print("=" * 80)
    
    # Create model
    model = CoordinateTransformer(
        volume_size=tuple(config['model']['volume_size']),
        xray_img_size=config['model']['xray_img_size'],
        xray_patch_size=config['model']['xray_patch_size'],
        xray_embed_dim=config['model']['xray_embed_dim'],
        xray_depth=config['model']['xray_depth'],
        coord_embed_dim=config['model']['coord_embed_dim'],
        num_transformer_blocks=config['model']['num_transformer_blocks'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout']
    ).to(device)
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Create loss
    loss_fn = CoordinateTransformerLoss(
        l1_weight=config['training']['loss_weights']['l1_weight'],
        gradient_weight=config['training']['loss_weights']['gradient_weight'],
        smoothness_weight=config['training']['loss_weights']['smoothness_weight']
    ).to(device)
    
    # Create optimizer with layer-wise learning rate decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Create dataset
    full_dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_volume_size=tuple(config['model']['volume_size']),
        max_patients=config['data']['max_patients'],
        validate_alignment=False
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(config['data']['train_split'] * total_size)
    val_size = int(config['data']['val_split'] * total_size)
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    if rank == 0:
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_dataset)} patients")
        print(f"  Val: {len(val_dataset)} patients")
    
    # Create distributed samplers
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
    
    # Create dataloaders
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
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    num_warmup_steps = len(train_loader) * config['training']['warmup_epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Create checkpoint directory
    if rank == 0:
        checkpoint_dir = Path(config['checkpoints']['save_dir'])
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, rank, epoch
        )
        scheduler.step()
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device, rank)
        
        if rank == 0:
            print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | L1: {train_metrics['l1']:.4f} | Grad: {train_metrics['gradient']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | PSNR: {val_metrics['psnr']:.2f} | SSIM: {val_metrics['ssim']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if epoch % config['checkpoints']['save_every'] == 0:
                checkpoint_path = checkpoint_dir / f"coord_transformer_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'config': config
                }, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if config['checkpoints']['keep_best'] and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_path = checkpoint_dir / "coord_transformer_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'config': config
                }, best_path)
                print(f"  ★ New best model saved! Val Loss: {best_val_loss:.4f}")
    
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='Train Coordinate-Based Transformer')
    parser.add_argument('--config', type=str, default='coord_transformer/config_coord_transformer.json',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    
    if world_size < 4:
        print(f"Warning: Found {world_size} GPUs, expected 4")
        print("Training will proceed with available GPUs")
    
    # Launch training
    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
