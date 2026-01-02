"""
Stage 2: Refinement Network Training (64³ → 256³)
Train lightweight upsampling network after base model is trained
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

from model_direct import DirectCTRegression
from model_enhanced import RefinementNetwork, PerceptualLoss, EdgeAwareLoss
from utils.dataset import PatientDRRDataset


def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    dist.init_process_group(backend='nccl', init_method='env://', 
                           rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def train_epoch(base_model, refinement, dataloader, criterion_dict, optimizer, scaler, rank, epoch, config):
    """Train refinement network for one epoch"""
    base_model.eval()  # Frozen
    refinement.train()
    
    epoch_losses = {'total': 0, 'l1': 0, 'perceptual': 0, 'edge': 0}
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, dict):
            xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
            target_256 = batch['ct_volume'].cuda(rank, non_blocking=True)  # 256³ target
        else:
            xrays = batch['xrays'].cuda(rank, non_blocking=True)
            target_256 = batch['ct_volume'].cuda(rank, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with torch.amp.autocast('cuda'):
            # Get 64³ prediction from frozen base model
            with torch.no_grad():
                predicted_64 = base_model(xrays)  # DirectCTRegression returns only volume
            
            # Refine to 256³
            predicted_256 = refinement(predicted_64)
            
            # Compute losses
            l1_loss = F.l1_loss(predicted_256, target_256)
            perceptual_loss = criterion_dict['perceptual'](predicted_256, target_256)
            edge_loss = criterion_dict['edge'](predicted_256, target_256)
            
            total_loss = (
                config['loss']['l1_weight'] * l1_loss +
                config['loss']['perceptual_weight'] * perceptual_loss +
                config['loss']['edge_weight'] * edge_loss
            )
        
        # Backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(refinement.parameters(), config['training']['grad_clip_max_norm'])
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        epoch_losses['total'] += total_loss.item()
        epoch_losses['l1'] += l1_loss.item()
        epoch_losses['perceptual'] += perceptual_loss.item()
        epoch_losses['edge'] += edge_loss.item()
        num_batches += 1
        
        # Logging
        if rank == 0 and batch_idx % config['logging']['log_interval'] == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * config['training']['batch_size'] * config['world_size'] / elapsed
            
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {total_loss.item():.4f} | "
                  f"L1: {l1_loss.item():.4f} | "
                  f"Perc: {perceptual_loss.item():.4f} | "
                  f"Edge: {edge_loss.item():.4f} | "
                  f"{samples_per_sec:.2f} samples/s")
    
    # Average
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def validate(base_model, refinement, dataloader, criterion_dict, rank):
    """Validation loop"""
    base_model.eval()
    refinement.eval()
    
    val_losses = {'total': 0, 'l1': 0, 'perceptual': 0, 'edge': 0}
    num_batches = 0
    total_psnr = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                xrays = batch['drr_stacked'].cuda(rank, non_blocking=True)
                target_256 = batch['ct_volume'].cuda(rank, non_blocking=True)
            else:
                xrays = batch['xrays'].cuda(rank, non_blocking=True)
                target_256 = batch['ct_volume'].cuda(rank, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                # Base prediction
                predicted_64 = base_model(xrays)  # DirectCTRegression returns only volume
                
                # Refinement
                predicted_256 = refinement(predicted_64)
                
                # Losses
                l1_loss = F.l1_loss(predicted_256, target_256)
                perceptual_loss = criterion_dict['perceptual'](predicted_256, target_256)
                edge_loss = criterion_dict['edge'](predicted_256, target_256)
                
                total_loss = l1_loss + 0.1 * perceptual_loss + 0.1 * edge_loss
            
            val_losses['total'] += total_loss.item()
            val_losses['l1'] += l1_loss.item()
            val_losses['perceptual'] += perceptual_loss.item()
            val_losses['edge'] += edge_loss.item()
            
            # PSNR
            mse = F.mse_loss(predicted_256, target_256)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            total_psnr += psnr.item()
            
            num_batches += 1
    
    for key in val_losses:
        val_losses[key] /= num_batches
    avg_psnr = total_psnr / num_batches
    
    return val_losses, avg_psnr


def train_refinement_ddp(rank, world_size, config):
    """Main training function for refinement network"""
    
    setup_ddp(rank, world_size)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Stage 2: Refinement Network Training (64³ → 256³)")
        print(f"{'='*60}")
    
    # Load frozen base model (DirectCTRegression)
    checkpoint = torch.load(config['base_model_checkpoint'], map_location=f'cuda:{rank}')
    
    # Get model config from checkpoint
    model_config = checkpoint['config']['model']
    
    base_model = DirectCTRegression(
        volume_size=tuple(model_config['volume_size']),
        xray_img_size=model_config['xray_img_size'],
        voxel_dim=model_config['voxel_dim'],
        vit_depth=model_config['vit_depth'],
        num_heads=model_config['num_heads'],
        xray_feature_dim=model_config['xray_feature_dim']
    ).cuda(rank)
    
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    
    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False
    
    if rank == 0:
        print(f"✓ Loaded frozen base model from: {config['base_model_checkpoint']}")
    
    # Create refinement network
    refinement = RefinementNetwork().cuda(rank)
    refinement = DDP(refinement, device_ids=[rank])
    
    if rank == 0:
        refinement_params = sum(p.numel() for p in refinement.parameters())
        print(f"✓ Refinement parameters: {refinement_params:,} ({refinement_params/1e6:.2f}M)")
    
    # Loss functions
    criterion_dict = {
        'perceptual': PerceptualLoss().cuda(rank),
        'edge': EdgeAwareLoss().cuda(rank)
    }
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        refinement.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=1e-6
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Datasets (need 256³ targets)
    # Load full dataset and split into train/val (80/20)
    full_dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_xray_size=512,
        target_volume_size=(256, 256, 256),  # 256³
        max_patients=config['data']['max_patients']
    )
    
    # 80/20 train/val split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Create indices for split
    indices = list(range(total_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
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
        print(f"\n{'='*60}")
        print(f"Starting Refinement Training")
        print(f"{'='*60}\n")
    
    best_psnr = 0.0
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
            print(f"{'='*60}")
        
        train_losses = train_epoch(
            base_model, refinement, train_loader, criterion_dict,
            optimizer, scaler, rank, epoch, config
        )
        
        if rank == 0:
            print(f"\nEpoch {epoch} Training:")
            print(f"  Total Loss: {train_losses['total']:.4f}")
            print(f"  L1: {train_losses['l1']:.4f}")
            print(f"  Perceptual: {train_losses['perceptual']:.4f}")
            print(f"  Edge: {train_losses['edge']:.4f}")
        
        # Validate
        if epoch % config['logging']['val_interval'] == 0:
            val_losses, val_psnr = validate(
                base_model, refinement, val_loader, criterion_dict, rank
            )
            
            if rank == 0:
                print(f"\nValidation:")
                print(f"  PSNR (256³): {val_psnr:.2f} dB")
                
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    Path('checkpoints_refinement').mkdir(exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'refinement_state_dict': refinement.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'psnr': val_psnr,
                        'config': config
                    }, 'checkpoints_refinement/best_refinement.pt')
                    print(f"  ✓ New best refinement! PSNR: {best_psnr:.2f} dB")
        
        scheduler.step()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Refinement Training Complete!")
        print(f"Best 256³ PSNR: {best_psnr:.2f} dB")
        print(f"{'='*60}")
    
    cleanup_ddp()


def main():
    config_path = 'config_refinement.json'
    
    if not Path(config_path).exists():
        print(f"Creating default config: {config_path}")
        
        default_config = {
            "model": {
                "base_channels": 64
            },
            "base_model_checkpoint": "checkpoints_enhanced/best_model.pt",
            "training": {
                "num_epochs": 20,
                "batch_size": 2,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "grad_clip_max_norm": 1.0
            },
            "loss": {
                "l1_weight": 1.0,
                "perceptual_weight": 0.1,
                "edge_weight": 0.1
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
    
    world_size = config['world_size']
    
    mp.spawn(
        train_refinement_ddp,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
