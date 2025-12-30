"""
Training script for direct CT regression (no diffusion)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import json
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_direct import DirectCTRegression, DirectRegressionLoss
from utils.dataset import PatientDRRDataset


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        
        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1


def compute_psnr(pred, target):
    """Compute PSNR between predicted and target volumes"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
    return psnr.item()


def train_epoch(model, train_loader, optimizer, loss_fn, scaler, device, is_distributed, is_main_process):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_l1 = 0
    total_ssim = 0
    
    if is_main_process:
        pbar = tqdm(train_loader, desc="Training")
    else:
        pbar = train_loader
    
    for batch_data in pbar:
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
        total_ssim += loss_dict['ssim_loss'].item()
        
        if is_main_process:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'l1': f"{loss_dict['l1_loss'].item():.4f}"
            })
    
    n = len(train_loader)
    return total_loss / n, total_l1 / n, total_ssim / n


def validate(model, val_loader, loss_fn, device, is_main_process):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_psnr = 0
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
            
            # Compute PSNR for each sample
            for i in range(pred_volume.shape[0]):
                psnr = compute_psnr(pred_volume[i:i+1], volumes[i:i+1])
                total_psnr += psnr
                n_samples += 1
    
    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / n_samples if n_samples > 0 else 0
    
    return avg_loss, avg_psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_direct.json')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    # Setup distributed
    is_distributed, rank, local_rank, world_size = setup_distributed()
    is_main_process = rank == 0
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process:
        print("=" * 80)
        print("DIRECT CT REGRESSION (NO DIFFUSION)")
        print("=" * 80)
        print(f"Distributed: {is_distributed}")
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
    
    # Create datasets
    train_dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_volume_size=tuple(config['model']['volume_size']),
        max_patients=config['data']['max_patients'],
        validate_alignment=False
    )
    
    val_dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_volume_size=tuple(config['model']['volume_size']),
        max_patients=config['data']['max_patients'],
        validate_alignment=False
    )
    
    # Create dataloaders
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = DirectCTRegression(**config['model']).to(device)
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters: {total_params:,}")
    
    # Wrap with DDP
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Loss and optimizer
    loss_fn = DirectRegressionLoss(
        l1_weight=config['training']['l1_weight'],
        ssim_weight=config['training']['ssim_weight']
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scaler = GradScaler()
    
    # Create checkpoint dir
    if is_main_process:
        os.makedirs(config['checkpoints']['save_dir'], exist_ok=True)
    
    # Training loop
    best_psnr = 0
    
    for epoch in range(config['training']['num_epochs']):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_l1, train_ssim = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, is_distributed, is_main_process
        )
        
        # Validate
        val_loss, val_psnr = validate(model, val_loader, loss_fn, device, is_main_process)
        
        if is_main_process:
            print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}:")
            print(f"  Train - Loss: {train_loss:.4f}, L1: {train_l1:.4f}, SSIM: {train_ssim:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f} dB")
            
            # Save checkpoint
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'psnr': val_psnr,
                    'config': config
                }
                torch.save(checkpoint, os.path.join(config['checkpoints']['save_dir'], 'best_model.pt'))
                print(f"  ✓ Saved best model (PSNR: {val_psnr:.2f} dB)")
    
    if is_main_process:
        print(f"\n{'='*80}")
        print(f"Training complete! Best PSNR: {best_psnr:.2f} dB")
        print(f"{'='*80}")
    
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
