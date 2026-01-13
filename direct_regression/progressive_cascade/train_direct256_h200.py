"""
Training script for Direct 256^3 H200 model.
Uses Direct256Model_H200 + Direct256Loss and PatientDRRDataset
with vertical flip applied to DRR images before entering the pipeline.
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset_simple import PatientDRRDataset
from model_direct256_h200 import Direct256Model_H200
from loss_direct256 import Direct256Loss, get_loss_summary_string
from loss_multiscale import compute_psnr, compute_ssim_metric


def parse_args():
    parser = argparse.ArgumentParser(description="Train Direct 256^3 H200 model")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to DRR patient dataset root")
    parser.add_argument("--save_dir", type=str, default="checkpoints_direct256_h200",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (H200: 2-3 recommended)")
    parser.add_argument("--num_epochs", type=int, default=180,
                        help="Number of training epochs (150-200 recommended)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--max_patients", type=int, default=None,
                        help="Optional limit on number of patients")
    return parser.parse_args()


def create_datasets(args):
    """Create train/val datasets with vertical DRR flip enabled."""
    train_dataset = PatientDRRDataset(
        dataset_path=args.dataset_path,
        max_patients=args.max_patients,
        split="train",
        ct_size=256,
        drr_size=512,
        vertical_flip=True,  # CRITICAL: ensure DRRs are flipped vertically
    )

    val_dataset = PatientDRRDataset(
        dataset_path=args.dataset_path,
        max_patients=args.max_patients,
        split="val",
        ct_size=256,
        drr_size=512,
        vertical_flip=True,  # keep same orientation in validation
    )

    return train_dataset, val_dataset


def train_one_epoch(model, criterion, optimizer, scaler, loader, device, epoch):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    num_batches = 0

    start_time = time.time()

    for batch in loader:
        ct_volume = batch["ct_volume"].to(device, non_blocking=True)        # (B, 1, D, H, W)
        drr_stacked = batch["drr_stacked"].to(device, non_blocking=True)    # (B, 2, 1, H, W)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(drr_stacked)
            loss_dict = criterion(pred, ct_volume)
            loss = loss_dict["total_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics (detach to CPU for safety)
        with torch.no_grad():
            running_loss += loss.item()
            running_psnr += compute_psnr(pred.detach(), ct_volume.detach())
            running_ssim += compute_ssim_metric(pred.detach(), ct_volume.detach())
            num_batches += 1

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

    with torch.no_grad():
        for batch in loader:
            ct_volume = batch["ct_volume"].to(device, non_blocking=True)
            drr_stacked = batch["drr_stacked"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(drr_stacked)
                loss_dict = criterion(pred, ct_volume)
                loss = loss_dict["total_loss"]

            running_loss += loss.item()
            running_psnr += compute_psnr(pred.detach(), ct_volume.detach())
            running_ssim += compute_ssim_metric(pred.detach(), ct_volume.detach())
            num_batches += 1

    return {
        "loss": running_loss / max(num_batches, 1),
        "psnr": running_psnr / max(num_batches, 1),
        "ssim": running_ssim / max(num_batches, 1),
    }


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    print("Creating datasets (with vertical DRR flip)...")
    train_dataset, val_dataset = create_datasets(args)
    print(f"Train patients: {len(train_dataset)}, Val patients: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Initializing model and loss...")
    model = Direct256Model_H200().to(device)
    criterion = Direct256Loss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr * 0.1,
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_loss = float("inf")
    best_psnr = 0.0
    best_ssim = 0.0

    log_path = save_dir / "training_log.txt"
    with open(log_path, "w") as log_f:
        log_f.write("epoch,phase,loss,psnr,ssim,lr,time\n")

    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")

        if scaler is None:
            # Fallback without AMP (CPU training, mostly for debugging)
            scaler_local = None
        else:
            scaler_local = scaler

        train_stats = train_one_epoch(model, criterion, optimizer, scaler_local, train_loader, device, epoch)
        scheduler.step()

        val_stats = eval_one_epoch(model, criterion, val_loader, device)

        lr_current = optimizer.param_groups[0]["lr"]

        print(f"  Train - loss: {train_stats['loss']:.4f}, PSNR: {train_stats['psnr']:.2f}, SSIM: {train_stats['ssim']:.4f}, time: {train_stats['time']:.1f}s")
        print(f"  Val   - loss: {val_stats['loss']:.4f}, PSNR: {val_stats['psnr']:.2f}, SSIM: {val_stats['ssim']:.4f}")
        print(f"  LR: {lr_current:.6f}\n")

        # Append to log
        with open(log_path, "a") as log_f:
            log_f.write(f"{epoch},train,{train_stats['loss']:.6f},{train_stats['psnr']:.3f},{train_stats['ssim']:.5f},{lr_current:.8f},{train_stats['time']:.3f}\n")
            log_f.write(f"{epoch},val,{val_stats['loss']:.6f},{val_stats['psnr']:.3f},{val_stats['ssim']:.5f},{lr_current:.8f},0.000\n")

        # Save best checkpoint by validation loss
        is_best = val_stats["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_stats["loss"]
            best_psnr = val_stats["psnr"]
            best_ssim = val_stats["ssim"]

            ckpt_path = save_dir / "direct256_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ✓ Saved new best checkpoint to {ckpt_path}")

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.4f}, PSNR: {best_psnr:.2f}, SSIM: {best_ssim:.4f}")


if __name__ == "__main__":
    main()
