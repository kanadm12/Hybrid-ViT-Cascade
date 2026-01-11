"""
Extended Stage 2 Training - 100 Epochs for Better Detail Capture
Resumes from stage1_best.pth and trains Stage 2 with:
- 100 epochs instead of 20
- Gradient loss for sharper edges
- Cosine annealing for better convergence
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
    """Resize CT volume to target resolution"""
    return F.interpolate(ct_volume, size=target_size, 
                        mode='trilinear', align_corners=False)


def train_epoch(model, dataloader, criterion, optimizer, scaler, epoch, device):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    l1_losses = 0
    ssim_losses = 0
    vgg_losses = 0
    grad_losses = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        ct_volume = batch['ct'].to(device)  # (B, 1, 256, 256, 256)
        xray_images = batch['xrays'].to(device)  # (B, 2, 1, 512, 512)
        
        # Resize ground truth to 128³ for Stage 2
        ct_128 = resize_ct_volume(ct_volume, (128, 128, 128))
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # Forward pass through Stage 1 + Stage 2
            pred_ct = model(xray_images, max_stage=2)  # Returns 128³
            
            # Compute loss
            loss_dict = criterion(pred_ct, ct_128, stage=2)
            loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate metrics
        total_loss += loss.item()
        l1_losses += loss_dict.get('l1_loss', 0).item()
        ssim_losses += loss_dict.get('ssim_loss', 0).item()
        vgg_losses += loss_dict.get('vgg_loss', 0).item()
        grad_losses += loss_dict.get('gradient_loss', 0).item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}: "
                  f"Loss={loss.item():.4f}, "
                  f"L1={loss_dict.get('l1_loss', 0).item():.4f}, "
                  f"Grad={loss_dict.get('gradient_loss', 0).item():.4f}")
    
    return {
        'total_loss': total_loss / num_batches,
        'l1_loss': l1_losses / num_batches,
        'ssim_loss': ssim_losses / num_batches,
        'vgg_loss': vgg_losses / num_batches,
        'gradient_loss': grad_losses / num_batches
    }


def validate_epoch(model, dataloader, criterion, epoch, device):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0
    psnr_values = []
    ssim_values = []
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            ct_volume = batch['ct'].to(device)
            xray_images = batch['xrays'].to(device)
            
            # Resize ground truth to 128³
            ct_128 = resize_ct_volume(ct_volume, (128, 128, 128))
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                pred_ct = model(xray_images, max_stage=2)
                loss_dict = criterion(pred_ct, ct_128, stage=2)
                loss = loss_dict['total_loss']
            
            # Compute metrics
            psnr = compute_psnr(pred_ct, ct_128)
            ssim = compute_ssim_metric(pred_ct, ct_128)
            
            total_loss += loss.item()
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            num_batches += 1
    
    return {
        'val_loss': total_loss / num_batches,
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values)
    }


def main():
    """Main training loop"""
    print("="*60)
    print("Extended Stage 2 Training - 100 Epochs")
    print("Improved loss with gradient term for anatomical detail")
    print("="*60)
    
    # Load config
    config_path = Path(__file__).parent / "config_progressive.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Dataset
    print("Loading dataset...")
    train_dataset = PatientDRRDataset(
        dataset_path=config['data']['dataset_path'],
        split='train',
        max_patients=config['data']['max_patients']
    )
    val_dataset = PatientDRRDataset(
        dataset_path=config['data']['dataset_path'],
        split='val',
        max_patients=10
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # Stage 2 batch size
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} patients")
    print(f"Val: {len(val_dataset)} patients\n")
    
    # Model
    print("Initializing model...")
    model = ProgressiveCascadeModel().to(device)
    
    # Load Stage 1 checkpoint
    checkpoint_dir = Path(config['checkpoints']['save_dir'])
    stage1_path = checkpoint_dir / "stage1_best.pth"
    
    if stage1_path.exists():
        print(f"Loading Stage 1 checkpoint: {stage1_path}")
        checkpoint = torch.load(stage1_path, map_location=device)
        
        # Filter to only Stage 1 weights
        stage1_weights = {k: v for k, v in checkpoint['model_state_dict'].items() 
                         if 'stage1' in k or 'xray_encoder' in k}
        model.load_state_dict(stage1_weights, strict=False)
        print(f"✓ Loaded Stage 1 weights\n")
    else:
        print(f"ERROR: Stage 1 checkpoint not found: {stage1_path}")
        return
    
    # Freeze Stage 1
    for name, param in model.named_parameters():
        if 'stage1' in name or 'xray_encoder' in name:
            param.requires_grad = False
    
    # Enable gradient checkpointing for Stage 2
    if hasattr(model.stage2, 'hybrid_vit'):
        model.stage2.hybrid_vit.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled for Stage 2")
    
    # Loss and optimizer
    criterion = MultiScaleLoss()
    
    # Only optimize Stage 2 parameters
    stage2_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(stage2_params, lr=1e-4, weight_decay=0.01)
    
    # Cosine annealing scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Stage 2 trainable parameters: {sum(p.numel() for p in stage2_params):,}\n")
    
    # Training loop
    num_epochs = 100
    best_psnr = 0.0
    best_ssim = 0.0
    best_val_loss = float('inf')
    
    log_file = checkpoint_dir / "stage2_extended_training.log"
    
    with open(log_file, 'w') as f:
        f.write(f"Extended Stage 2 Training - {datetime.now()}\n")
        f.write("="*80 + "\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Batch size: 2\n")
        f.write(f"Learning rate: 1e-4 with cosine annealing\n")
        f.write(f"Loss: L1 + SSIM + VGG + Gradient\n")
        f.write("="*80 + "\n\n")
    
    print(f"Starting training for {num_epochs} epochs...\n")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs} - LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, device)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, epoch, device)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  PSNR: {val_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {val_metrics['ssim']:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch}: Loss={val_metrics['val_loss']:.4f}, "
                   f"PSNR={val_metrics['psnr']:.2f}, SSIM={val_metrics['ssim']:.4f}\n")
        
        # Save best checkpoint (based on PSNR)
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            best_ssim = val_metrics['ssim']
            best_val_loss = val_metrics['val_loss']
            
            save_path = checkpoint_dir / "stage2_extended_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['val_loss'],
                'psnr': val_metrics['psnr'],
                'ssim': val_metrics['ssim']
            }, save_path)
            print(f"  ✓ NEW BEST! Saved: {save_path}")
        
        # Save periodic checkpoint every 20 epochs
        if epoch % 20 == 0:
            save_path = checkpoint_dir / f"stage2_extended_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['val_loss'],
                'psnr': val_metrics['psnr'],
                'ssim': val_metrics['ssim']
            }, save_path)
            print(f"  ✓ Saved checkpoint: {save_path}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best SSIM: {best_ssim:.4f}")
    print(f"Best saved: stage2_extended_best.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
