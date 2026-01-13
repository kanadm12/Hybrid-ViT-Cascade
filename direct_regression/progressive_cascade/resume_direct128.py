"""
Resume Direct128 H200 Training from Checkpoint
Loads best checkpoint and continues training with re-enabled advanced losses
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_simple import PatientDRRDataset
from model_direct128_h200 import Direct128Model_H200
from loss_direct256 import Direct256Loss, get_loss_summary_string
from loss_multiscale import compute_psnr, compute_ssim_metric


def parse_args():
    parser = argparse.ArgumentParser(description="Resume Direct128 H200 training")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to DRR patient dataset root")
    parser.add_argument("--save_dir", type=str, default="checkpoints_direct128_h200_resumed",
                        help="Directory to save new checkpoints")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Total epochs (will continue from checkpoint epoch)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for resumed training (lower than initial)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    return parser.parse_args()


def create_datasets(args):
    """Create train/val datasets"""
    train_dataset = PatientDRRDataset(
        dataset_path=args.dataset_path,
        max_patients=None,
        split="train",
        ct_size=128,
        drr_size=512,
        vertical_flip=True,
    )

    val_dataset = PatientDRRDataset(
        dataset_path=args.dataset_path,
        max_patients=None,
        split="val",
        ct_size=128,
        drr_size=512,
        vertical_flip=True,
    )

    return train_dataset, val_dataset


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and return model, optimizer state, start epoch"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = Direct128Model_H200(
        xray_img_size=512,
        xray_feature_dim=512,
        num_rdb=5,
        use_checkpoint=True
    )
    
    # Load model state
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Get training state
    start_epoch = checkpoint.get('epoch', 0) + 1
    optimizer_state = checkpoint.get('optimizer_state', None)
    scheduler_state = checkpoint.get('scheduler_state', None)
    
    print(f"✓ Loaded checkpoint from epoch {start_epoch - 1}")
    print(f"  - Val PSNR: {checkpoint.get('val_psnr', 'N/A'):.2f} dB")
    print(f"  - Val SSIM: {checkpoint.get('val_ssim', 'N/A'):.4f}")
    print(f"  - Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model, optimizer_state, scheduler_state, start_epoch


def train_one_epoch(model, criterion, optimizer, scaler, loader, device, epoch):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    num_batches = 0
    nan_count = 0

    start_time = time.time()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        ct_volume = batch["ct_volume"].to(device, non_blocking=True)
        drr_stacked = batch["drr_stacked"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(drr_stacked)
            loss_dict = criterion(pred, ct_volume)
            loss = loss_dict["total_loss"]

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            print(f"\n[WARNING] Batch {num_batches} returned NaN loss. Skipping.")
            continue

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            running_loss += loss.item()
            running_psnr += compute_psnr(pred.detach(), ct_volume.detach())
            running_ssim += compute_ssim_metric(pred.detach(), ct_volume.detach())
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{running_psnr/num_batches:.2f}',
                'ssim': f'{running_ssim/num_batches:.4f}',
                'nan': nan_count
            })

    epoch_time = time.time() - start_time

    return {
        "loss": running_loss / max(num_batches, 1),
        "psnr": running_psnr / max(num_batches, 1),
        "ssim": running_ssim / max(num_batches, 1),
        "time": epoch_time,
    }


def eval_one_epoch(model, criterion, loader, device):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    num_batches = 0
    nan_count = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for batch in pbar:
            ct_volume = batch["ct_volume"].to(device, non_blocking=True)
            drr_stacked = batch["drr_stacked"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(drr_stacked)
                loss_dict = criterion(pred, ct_volume)
                loss = loss_dict["total_loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

            running_loss += loss.item()
            running_psnr += compute_psnr(pred.detach(), ct_volume.detach())
            running_ssim += compute_ssim_metric(pred.detach(), ct_volume.detach())
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{running_psnr/num_batches:.2f}',
                'ssim': f'{running_ssim/num_batches:.4f}'
            })

    return {
        "loss": running_loss / max(num_batches, 1),
        "psnr": running_psnr / max(num_batches, 1),
        "ssim": running_ssim / max(num_batches, 1),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    model, optimizer_state, scheduler_state, start_epoch = load_checkpoint(args.checkpoint, device)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    
    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Loss with ALL 7 components re-enabled
    criterion = Direct256Loss()
    criterion = criterion.to(device)  # Move loss modules to GPU
    print("\n✓ Re-enabled ALL 7 loss components:")
    print("  1. L1 Loss (1.0)")
    print("  2. SSIM Loss (0.5)")
    print("  3. Focal Frequency Loss (0.2)")
    print("  4. Perceptual Pyramid Loss (0.15)")
    print("  5. Total Variation Loss (0.02)")
    print("  6. Style 3D Loss (0.1)")
    print("  7. Anatomical Attention Loss (0.3)")
    print("  Total weight: 2.27\n")
    
    # Optimizer - use lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if optimizer_state is not None:
        print("Loading optimizer state from checkpoint...")
        optimizer.load_state_dict(optimizer_state)
        # Override learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    
    scaler = torch.amp.GradScaler()
    
    # Tracking
    best_loss = float('inf')
    best_psnr = 0.0
    best_ssim = 0.0
    
    print(f"\nResuming training from epoch {start_epoch} to {args.num_epochs}")
    print("="*60)
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        
        # Train
        train_metrics = train_one_epoch(model, criterion, optimizer, scaler, 
                                       train_loader, device, epoch)
        
        # Validate
        val_metrics = eval_one_epoch(model, criterion, val_loader, device)
        
        # Step scheduler
        scheduler.step()
        
        # Print metrics
        print(f"  Train - loss: {train_metrics['loss']:.4f}, "
              f"PSNR: {train_metrics['psnr']:.2f}, "
              f"SSIM: {train_metrics['ssim']:.4f}, "
              f"time: {train_metrics['time']:.1f}s")
        print(f"  Val   - loss: {val_metrics['loss']:.4f}, "
              f"PSNR: {val_metrics['psnr']:.2f}, "
              f"SSIM: {val_metrics['ssim']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_psnr': val_metrics['psnr'],
            'val_ssim': val_metrics['ssim'],
            'args': vars(args)
        }
        
        # Best loss
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            torch.save(checkpoint, save_dir / "direct128_best_loss_resumed.pth")
            print(f"  ✓ Saved best loss checkpoint: {best_loss:.4f}")
        
        # Best PSNR
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save(checkpoint, save_dir / "direct128_best_psnr_resumed.pth")
            print(f"  ✓ Saved best PSNR checkpoint: {best_psnr:.2f} dB")
        
        # Best SSIM
        if val_metrics['ssim'] > best_ssim:
            best_ssim = val_metrics['ssim']
            torch.save(checkpoint, save_dir / "direct128_best_ssim_resumed.pth")
            print(f"  ✓ Saved best SSIM checkpoint: {best_ssim:.4f}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best SSIM: {best_ssim:.4f}")


if __name__ == "__main__":
    main()
