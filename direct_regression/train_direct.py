"""
Training script for direct CT regression (no diffusion) - Single GPU
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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


def compute_psnr(pred, target):
    """Compute PSNR between predicted and target volumes"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
    return psnr.item()


def train_epoch(model, train_loader, optimizer, loss_fn, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_l1 = 0
    total_ssim = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
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
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'l1': f"{loss_dict['l1_loss'].item():.4f}"
        })
    
    n = len(train_loader)
    return total_loss / n, total_l1 / n, total_ssim / n


def validate(model, val_loader, loss_fn, device):
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
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("DIRECT CT REGRESSION (NO DIFFUSION) - Single GPU")
    print("=" * 80)
    
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = DirectCTRegression(**config['model']).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
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
    os.makedirs(config['checkpoints']['save_dir'], exist_ok=True)
    
    # Training loop
    best_psnr = 0
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss, train_l1, train_ssim = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device
        )
        
        # Validate
        val_loss, val_psnr = validate(model, val_loader, loss_fn, device)
        
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}:")
        print(f"  Train - Loss: {train_loss:.4f}, L1: {train_l1:.4f}, SSIM: {train_ssim:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f} dB")
        
        # Save checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': val_psnr,
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['checkpoints']['save_dir'], 'best_model.pt'))
            print(f"  ✓ Saved best model (PSNR: {val_psnr:.2f} dB)")
    
    print(f"\n{'='*80}")
    print(f"Training complete! Best PSNR: {best_psnr:.2f} dB")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
