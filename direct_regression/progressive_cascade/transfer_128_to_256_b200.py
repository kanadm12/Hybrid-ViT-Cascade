"""
Transfer Learning: 128³ → 256³ on B200 GPU
Two-phase training:
  Phase 1: Train 256³ layers with frozen 128³ backbone
  Phase 2: Fine-tune all layers end-to-end
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import csv

from model_direct256_b200 import Direct256Model_B200
from loss_direct256 import Direct256Loss
from loss_multiscale import compute_psnr, compute_ssim_metric
from dataset_simple import PatientDRRDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Transfer 128³ → 256³ on B200')
    
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)  # B200: 180GB requires batch=1
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--freeze_128', action='store_true',
                       help='Freeze 128³ stages (Phase 1)')
    parser.add_argument('--checkpoint_128', type=str,
                       help='Path to 128³ checkpoint for transfer')
    parser.add_argument('--resume_256', type=str,
                       help='Path to 256³ checkpoint to resume')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='checkpoints_direct256_b200')
    
    return parser.parse_args()


def freeze_128_stages(model):
    """Freeze all layers up to 128³"""
    frozen_modules = [
        'initial_volume', 'xray_encoder', 'enc_16_32', 'enc_32_64', 'enc_64_128',
        'xray_fusion_32', 'xray_fusion_64', 'xray_fusion_128',
    ]
    
    for name, param in model.named_parameters():
        for frozen_mod in frozen_modules:
            if name.startswith(frozen_mod):
                param.requires_grad = False
                break
    
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Frozen 128³ stages:")
    print(f"  Frozen: {frozen:,} ({frozen/(frozen+trainable)*100:.1f}%)")
    print(f"  Trainable: {trainable:,} ({trainable/(frozen+trainable)*100:.1f}%)\n")
    
    return model


def train_one_epoch(model, criterion, optimizer, scaler, loader, device, epoch):
    model.train()
    
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = 0
    nan_count = 0
    
    start_time = time.time()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch in pbar:
        drr = batch["drr_stacked"].to(device, non_blocking=True)
        ct = batch["ct_volume"].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            pred = model(drr)
            loss_dict = criterion(pred, ct)
            loss = loss_dict['total_loss']
        
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            print(f"\n[WARNING] NaN loss, skipping batch")
            continue
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            psnr = compute_psnr(pred, ct)
            ssim = compute_ssim_metric(pred, ct)
        
        total_loss += loss.item()
        total_psnr += psnr
        total_ssim += ssim
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'psnr': f"{psnr:.2f}",
            'ssim': f"{ssim:.4f}",
            'nan': nan_count
        })
    
    epoch_time = time.time() - start_time
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'psnr': total_psnr / max(num_batches, 1),
        'ssim': total_ssim / max(num_batches, 1),
        'time': epoch_time
    }


@torch.no_grad()
def validate(model, criterion, loader, device):
    model.eval()
    
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Validation", leave=False)
    
    for batch in pbar:
        drr = batch["drr_stacked"].to(device, non_blocking=True)
        ct = batch["ct_volume"].to(device, non_blocking=True)
        
        with autocast():
            pred = model(drr)
            loss_dict = criterion(pred, ct)
        
        total_loss += loss_dict['total_loss'].item()
        total_psnr += compute_psnr(pred, ct)
        total_ssim += compute_ssim_metric(pred, ct)
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss'].item():.4f}",
            'psnr': f"{total_psnr/num_batches:.2f}",
            'ssim': f"{total_ssim/num_batches:.4f}"
        })
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'psnr': total_psnr / max(num_batches, 1),
        'ssim': total_ssim / max(num_batches, 1),
    }


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = PatientDRRDataset(
        dataset_path=args.dataset_path,
        split='train',
        ct_size=256,
        drr_size=512,
        vertical_flip=True
    )
    
    val_dataset = PatientDRRDataset(
        dataset_path=args.dataset_path,
        split='val',
        ct_size=256,
        drr_size=512,
        vertical_flip=True
    )
    
    print(f"[train] Found {len(train_dataset)} patients")
    print(f"[val] Found {len(val_dataset)} patients\n")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Create model
    print("Initializing 256³ model...")
    model = Direct256Model_B200().to(device)
    
    start_epoch = 1
    
    # Transfer from 128³
    if args.checkpoint_128:
        transferred, skipped = model.load_pretrained_128(args.checkpoint_128)
        print(f"✓ Loaded {transferred} layers from 128³ checkpoint\n")
        
        if args.freeze_128:
            model = freeze_128_stages(model)
            print("✓ Phase 1: Training only 256³ layers (128³ frozen)\n")
        else:
            print("✓ Phase 2: Fine-tuning all layers end-to-end\n")
    
    # Resume from 256³
    elif args.resume_256:
        print(f"Resuming from 256³ checkpoint: {args.resume_256}")
        checkpoint = torch.load(args.resume_256, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✓ Resumed from epoch {checkpoint['epoch']}\n")
    
    # Loss
    criterion = Direct256Loss().to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )
    
    # Resume optimizer/scheduler
    if args.resume_256:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    scaler = GradScaler()
    
    # Setup logging
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = checkpoint_dir / 'training_log.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'phase', 'loss', 'psnr', 'ssim', 'lr', 'time'])
    
    # Best metrics
    best_loss = float('inf')
    best_psnr = 0.0
    best_ssim = 0.0
    
    print(f"{'='*60}")
    print(f"Starting training from epoch {start_epoch} to {args.num_epochs}")
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")
        
        # Train
        train_metrics = train_one_epoch(
            model, criterion, optimizer, scaler, train_loader, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, criterion, val_loader, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print summary
        print(f"  Train - loss: {train_metrics['loss']:.4f}, "
              f"PSNR: {train_metrics['psnr']:.2f}, "
              f"SSIM: {train_metrics['ssim']:.4f}, "
              f"time: {train_metrics['time']:.1f}s")
        print(f"  Val   - loss: {val_metrics['loss']:.4f}, "
              f"PSNR: {val_metrics['psnr']:.2f}, "
              f"SSIM: {val_metrics['ssim']:.4f}")
        print(f"  LR: {current_lr:.6f}\n")
        
        # Log to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, 'train', train_metrics['loss'],
                           train_metrics['psnr'], train_metrics['ssim'],
                           current_lr, train_metrics['time']])
            writer.writerow([epoch, 'val', val_metrics['loss'],
                           val_metrics['psnr'], val_metrics['ssim'],
                           current_lr, 0.0])
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_psnr': val_metrics['psnr'],
            'val_ssim': val_metrics['ssim'],
            'args': vars(args),
        }
        
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            torch.save(checkpoint, checkpoint_dir / 'direct256_best_loss.pth')
            print(f"  ✓ Saved best loss checkpoint: {best_loss:.4f}")
        
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save(checkpoint, checkpoint_dir / 'direct256_best_psnr.pth')
            print(f"  ✓ Saved best PSNR checkpoint: {best_psnr:.2f} dB")
        
        if val_metrics['ssim'] > best_ssim:
            best_ssim = val_metrics['ssim']
            torch.save(checkpoint, checkpoint_dir / 'direct256_best_ssim.pth')
            print(f"  ✓ Saved best SSIM checkpoint: {best_ssim:.4f}")
        
        if epoch % 10 == 0:
            torch.save(checkpoint, checkpoint_dir / f'direct256_epoch_{epoch}.pth')
        
        print()
    
    print(f"{'='*60}")
    print(f"Training complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Best PSNR: {best_psnr:.2f} dB")
    print(f"  Best SSIM: {best_ssim:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
