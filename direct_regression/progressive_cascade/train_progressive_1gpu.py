"""
Progressive Training Pipeline for Multi-Scale CT Reconstruction - Single GPU
Stage 1: Train 64³ base for 100 epochs
Stage 2: Freeze stage 1, train 128³ refiner for 100 epochs
Stage 3: Freeze stages 1+2, train 256³ refiner for 100 epochs

Single GPU training with PSNR and SSIM metrics for 100 patients
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import json
from pathlib import Path
import time
from datetime import datetime
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_progressive import ProgressiveCascadeModel
from loss_multiscale import MultiScaleLoss, compute_psnr, compute_ssim_metric
from dataset_simple import PatientDRRDataset


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


def train_epoch(model, dataloader, criterion, optimizer, scaler, epoch, 
                stage, config, max_stage=1):
    """
    Train for one epoch
    Args:
        max_stage: Maximum stage to train (1, 2, or 3)
    """
    model.train()
    
    epoch_losses = {'total': 0}
    epoch_psnr = 0
    epoch_ssim = 0
    num_batches = 0
    
    # Determine target size based on stage
    if max_stage == 1:
        target_size = (64, 64, 64)
    elif max_stage == 2:
        target_size = (128, 128, 128)
    else:  # max_stage == 3
        target_size = (256, 256, 256)
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        xrays = batch['drr_stacked'].cuda(non_blocking=True)
        ct_volume_orig = batch['ct_volume'].cuda(non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with torch.amp.autocast('cuda'):
            # Forward through cascade
            pred = model(xrays, max_stage=max_stage)
            target = resize_ct_volume(ct_volume_orig, target_size)
            
            # Compute loss
            if max_stage == 3:
                loss_dict = criterion(pred, target, stage=max_stage, input_xrays=xrays)
            else:
                loss_dict = criterion(pred, target, stage=max_stage)
            
            total_loss = loss_dict['total_loss']
        
        # Backward with gradient scaling
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()
        
        # Compute metrics
        with torch.no_grad():
            batch_psnr = compute_psnr(pred, target)
            batch_ssim = compute_ssim_metric(pred, target)
        
        # Accumulate losses and metrics
        for k, v in loss_dict.items():
            if k not in epoch_losses:
                epoch_losses[k] = 0
            epoch_losses[k] += v.item()
        
        epoch_psnr += batch_psnr
        epoch_ssim += batch_ssim
        num_batches += 1
        
        # Log progress
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed
            print(f"Epoch {epoch} | Stage {max_stage} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {total_loss.item():.4f} | PSNR: {batch_psnr:.2f} dB | "
                  f"SSIM: {batch_ssim:.4f} | {batches_per_sec:.2f} batch/s")
    
    # Average losses and metrics
    for k in epoch_losses:
        epoch_losses[k] /= num_batches
    epoch_psnr /= num_batches
    epoch_ssim /= num_batches
    
    epoch_losses['psnr'] = epoch_psnr
    epoch_losses['ssim'] = epoch_ssim
    
    return epoch_losses


def validate(model, dataloader, criterion, stage, max_stage=1):
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
            xrays = batch['drr_stacked'].cuda(non_blocking=True)
            ct_volume_orig = batch['ct_volume'].cuda(non_blocking=True)
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


def train_stage(config, stage, checkpoint_dir):
    """
    Train a specific stage
    Args:
        stage: 1, 2, or 3
    """
    print(f"\n{'='*60}")
    print(f"Training Stage {stage}")
    print(f"{'='*60}\n")
    
    # Create model
    model = ProgressiveCascadeModel(
        xray_img_size=config['model']['xray_img_size'],
        xray_feature_dim=config['model']['xray_feature_dim'],
        voxel_dim=config['model']['voxel_dim'],
        use_gradient_checkpointing=(stage == 3)  # Only for stage 3
    ).cuda()
    
    # Load previous stage weights if not stage 1
    if stage > 1:
        prev_checkpoint = checkpoint_dir / f"stage{stage-1}_best.pth"
        if prev_checkpoint.exists():
            print(f"Loading Stage {stage-1} checkpoint: {prev_checkpoint}")
            checkpoint = torch.load(prev_checkpoint, map_location='cuda')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Freeze previous stages
            for prev_stage in range(1, stage):
                model.freeze_stage(prev_stage)
                print(f"✓ Frozen Stage {prev_stage}")
        else:
            print(f"Warning: Stage {stage-1} checkpoint not found!")
    
    # Create loss function
    loss_config = {
        'stage1': config['loss']['stage1'],
        'stage2': config['loss']['stage2'],
        'stage3': config['loss']['stage3']
    }
    criterion = MultiScaleLoss(config=loss_config).cuda()
    
    # Optimizer and scheduler
    stage_key = f'stage{stage}'
    lr = config['training'][stage_key]['learning_rate']
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config['training']['weight_decay']
    )
    
    num_epochs = config['training'][stage_key]['num_epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=lr * 0.1
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Dataset
    dataset_path = config['data']['dataset_path']
    train_dataset = PatientDRRDataset(
        dataset_path=dataset_path,
        max_patients=config['data']['max_patients'],
        split='train',
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split']
    )
    
    val_dataset = PatientDRRDataset(
        dataset_path=dataset_path,
        max_patients=config['data']['max_patients'],
        split='val',
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split']
    )
    
    # Adjust batch size for stage 3 (memory constraints)
    batch_size = config['training'][stage_key]['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
    print(f"Learning rate: {lr}\n")
    
    # Training loop
    best_val_loss = float('inf')
    best_psnr = 0
    best_ssim = 0
    
    # Open log file
    log_file = checkpoint_dir / f"stage{stage}_training_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"Stage {stage} Training Log\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Patients: {config['data']['max_patients']}, Epochs: {num_epochs}\n")
        f.write("="*80 + "\n\n")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, stage, config, max_stage=stage
        )
        
        # Validate
        val_losses = validate(
            model, val_loader, criterion, stage, max_stage=stage
        )
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log
        log_msg = (
            f"\nEpoch {epoch}/{num_epochs} - Time: {epoch_time:.1f}s\n"
            f"Train - Loss: {train_losses['total']:.4f} | "
            f"PSNR: {train_losses['psnr']:.2f} dB | SSIM: {train_losses['ssim']:.4f}\n"
            f"Val   - Loss: {val_losses['total']:.4f} | "
            f"PSNR: {val_losses['psnr']:.2f} dB | SSIM: {val_losses['ssim']:.4f}\n"
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        print(log_msg)
        
        # Write to log file
        with open(log_file, 'a') as f:
            f.write(log_msg + "\n")
        
        # Save best checkpoint
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_psnr = val_losses['psnr']
            best_ssim = val_losses['ssim']
            
            checkpoint = {
                'epoch': epoch,
                'stage': stage,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_losses['total'],
                'val_psnr': val_losses['psnr'],
                'val_ssim': val_losses['ssim'],
                'train_psnr': train_losses['psnr'],
                'train_ssim': train_losses['ssim'],
                'config': config
            }
            save_path = checkpoint_dir / f"stage{stage}_best.pth"
            torch.save(checkpoint, save_path)
            print(f"✓ Saved best checkpoint: {save_path} (PSNR: {best_psnr:.2f} dB, SSIM: {best_ssim:.4f})")
        
        # Save periodic checkpoint
        if epoch % config['checkpoints']['save_every'] == 0:
            checkpoint = {
                'epoch': epoch,
                'stage': stage,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_losses['total'],
                'val_psnr': val_losses['psnr'],
                'val_ssim': val_losses['ssim'],
                'config': config
            }
            save_path = checkpoint_dir / f"stage{stage}_epoch{epoch}.pth"
            torch.save(checkpoint, save_path)
            print(f"✓ Saved checkpoint: {save_path}")
    
    print(f"\n{'='*60}")
    print(f"Stage {stage} Training Complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best SSIM: {best_ssim:.4f}")
    print(f"{'='*60}\n")
    
    # Write summary to log
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Stage {stage} Training Complete!\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Best PSNR: {best_psnr:.2f} dB\n")
        f.write(f"Best SSIM: {best_ssim:.4f}\n")
        f.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")


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
    
    # Override with 100 patients and 20 epochs per stage
    config['data']['max_patients'] = 100
    config['training']['stage1']['num_epochs'] = 20
    config['training']['stage2']['num_epochs'] = 20
    config['training']['stage3']['num_epochs'] = 20
    
    print("="*60)
    print("Progressive Multi-Scale CT Reconstruction Training")
    print("Single GPU - 100 Patients - 20 Epochs per Stage")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config_path}")
    print(f"Dataset: {config['data']['dataset_path']}")
    print(f"Max Patients: {config['data']['max_patients']}")
    print("="*60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("Error: No GPU available!")
        return
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoints']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train each stage sequentially
    total_start = time.time()
    
    for stage in range(1, 4):
        print(f"\n{'#'*60}")
        print(f"# STAGE {stage} TRAINING")
        print(f"{'#'*60}\n")
        
        stage_start = time.time()
        train_stage(config, stage, checkpoint_dir)
        stage_time = time.time() - stage_start
        
        print(f"Stage {stage} completed in {stage_time/3600:.2f} hours\n")
    
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print("ALL STAGES TRAINING COMPLETE!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\nCheckpoints saved in:", checkpoint_dir)
    print("- stage1_best.pth")
    print("- stage2_best.pth")
    print("- stage3_best.pth")
    print("="*60)


if __name__ == "__main__":
    main()
