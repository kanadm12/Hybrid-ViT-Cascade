"""
Training Script for Optimized CT Regression Model
Single GPU, incorporates all optimizations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '..')

from model_optimized import OptimizedCTRegression, OptimizedRegressionLoss
from utils.dataset import CTReconDataset


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, config):
    """Train for one epoch"""
    model.train()
    epoch_losses = {'total': 0, 'l1': 0, 'ssim': 0, 'reg': 0}
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        xrays = batch['xrays'].to(device)
        target = batch['ct_volume'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=config['optimization']['use_mixed_precision']):
            # Forward pass
            predicted, aux_info = model(xrays)
            
            # Compute loss
            loss_dict = criterion(predicted, target, aux_info)
            loss = loss_dict['total_loss']
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['training']['grad_clip_max_norm']
        )
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        epoch_losses['total'] += loss_dict['total_loss'].item()
        epoch_losses['l1'] += loss_dict['l1_loss'].item()
        epoch_losses['ssim'] += loss_dict['ssim_loss'].item()
        epoch_losses['reg'] += loss_dict['reg_loss'].item()
        num_batches += 1
        
        # Logging
        if batch_idx % config['logging']['log_interval'] == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * config['training']['batch_size'] / elapsed
            
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} | "
                  f"L1: {loss_dict['l1_loss'].item():.4f} | "
                  f"SSIM: {loss_dict['ssim_loss'].item():.4f} | "
                  f"Reg: {loss_dict['reg_loss'].item():.4f} | "
                  f"{samples_per_sec:.2f} samples/s")
            
            # Print learned boundaries if available
            if 'boundaries' in aux_info:
                boundaries = aux_info['boundaries'][0] * 100
                print(f"  Boundaries: {boundaries.tolist()}")
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def validate(model, dataloader, criterion, device):
    """Validation loop"""
    model.eval()
    val_losses = {'total': 0, 'l1': 0, 'ssim': 0, 'reg': 0}
    psnr_sum = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            xrays = batch['xrays'].to(device)
            target = batch['ct_volume'].to(device)
            
            # Forward pass
            predicted, aux_info = model(xrays)
            
            # Compute loss
            loss_dict = criterion(predicted, target, aux_info)
            
            # Accumulate losses
            val_losses['total'] += loss_dict['total_loss'].item()
            val_losses['l1'] += loss_dict['l1_loss'].item()
            val_losses['ssim'] += loss_dict['ssim_loss'].item()
            val_losses['reg'] += loss_dict['reg_loss'].item()
            
            # Compute PSNR
            psnr = compute_psnr(predicted, target)
            psnr_sum += psnr
            
            num_batches += 1
    
    # Average metrics
    for key in val_losses:
        val_losses[key] /= num_batches
    avg_psnr = psnr_sum / num_batches
    
    return val_losses, avg_psnr


def compute_psnr(pred, target):
    """Compute PSNR"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    
    data_range = target.max() - target.min()
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()


def main():
    # Load config
    with open('config_optimized.json', 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda:0')
    print(f"Using device: {device}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Create model
    print("\n=== Creating Optimized Model ===")
    model = OptimizedCTRegression(**config['model']).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Create loss function
    criterion = OptimizedRegressionLoss(**config['loss'])
    
    # Create optimizer with warmup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config['training']['warmup_epochs']:
            return (epoch + 1) / config['training']['warmup_epochs']
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(
        enabled=config['optimization']['use_mixed_precision']
    )
    
    # Create datasets
    print("\n=== Loading Dataset ===")
    train_dataset = CTReconDataset(
        data_dir=config['data']['dataset_path'],
        split='train',
        max_patients=config['data']['max_patients']
    )
    
    val_dataset = CTReconDataset(
        data_dir=config['data']['dataset_path'],
        split='val',
        max_patients=100  # Smaller validation set
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        prefetch_factor=config['data']['prefetch_factor']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Training loop
    print("\n=== Starting Training ===")
    best_psnr = 0.0
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, config
        )
        
        print(f"\nEpoch {epoch} Training Summary:")
        print(f"  Total Loss: {train_losses['total']:.4f}")
        print(f"  L1 Loss: {train_losses['l1']:.4f}")
        print(f"  SSIM Loss: {train_losses['ssim']:.4f}")
        print(f"  Reg Loss: {train_losses['reg']:.6f}")
        
        # Validate
        if epoch % config['logging']['val_interval'] == 0:
            print(f"\nRunning validation...")
            val_losses, val_psnr = validate(model, val_loader, criterion, device)
            
            print(f"Validation Results:")
            print(f"  Total Loss: {val_losses['total']:.4f}")
            print(f"  L1 Loss: {val_losses['l1']:.4f}")
            print(f"  SSIM Loss: {val_losses['ssim']:.4f}")
            print(f"  PSNR: {val_psnr:.2f} dB")
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'psnr': val_psnr,
                    'config': config
                }, 'best_model_optimized.pth')
                print(f"  ✓ New best model saved! PSNR: {best_psnr:.2f} dB")
        
        # Save checkpoint
        if epoch % config['logging']['save_checkpoint_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
            }, f'checkpoint_epoch_{epoch}.pth')
        
        # Step scheduler
        scheduler.step()
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation PSNR: {best_psnr:.2f} dB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
