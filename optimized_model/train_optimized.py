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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

sys.path.insert(0, '..')

from model_optimized import OptimizedCTRegression, OptimizedRegressionLoss
from utils.dataset import PatientDRRDataset


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, config):
    """Train for one epoch"""
    model.train()
    epoch_losses = {'total': 0, 'l1': 0, 'ssim': 0, 'reg': 0}
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # Handle dict format from PatientDRRDataset
        if isinstance(batch, dict):
            xrays = batch['drr_stacked'].to(device)  # (B, 2, 1, 512, 512)
            target = batch['ct_volume'].to(device)
        else:
            xrays = batch['xrays'].to(device)
            target = batch['ct_volume'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.amp.autocast('cuda', enabled=config['optimization']['use_mixed_precision']):
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
            # Handle dict format from PatientDRRDataset
            if isinstance(batch, dict):
                xrays = batch['drr_stacked'].to(device)  # (B, 2, 1, 512, 512)
                target = batch['ct_volume'].to(device)
            else:
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


def visualize_feature_maps(model, sample_batch, epoch, save_dir='visualizations'):
    """
    Visualize feature maps from different stages of the model
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        xrays = sample_batch['xrays']
        target = sample_batch['ct_volume']
        
        # Forward pass with feature extraction
        batch_size = xrays.shape[0]
        dummy_t = torch.zeros(batch_size, 256, device=xrays.device)
        
        # Get X-ray features
        xray_context, time_xray_cond, xray_features_2d = model.xray_encoder(xrays, dummy_t)
        
        # Create visualization figure
        fig = plt.figure(figsize=(20, 12))
        
        # Plot input X-rays
        ax1 = plt.subplot(3, 5, 1)
        xray_ap = xrays[0, 0, 0].cpu().numpy()
        ax1.imshow(xray_ap, cmap='gray')
        ax1.set_title(f'Epoch {epoch}: Input X-ray (AP)')
        ax1.axis('off')
        
        ax2 = plt.subplot(3, 5, 2)
        xray_lat = xrays[0, 1, 0].cpu().numpy()
        ax2.imshow(xray_lat, cmap='gray')
        ax2.set_title('Input X-ray (Lateral)')
        ax2.axis('off')
        
        # Plot X-ray encoder features (averaged across channels)
        ax3 = plt.subplot(3, 5, 3)
        xray_feat = xray_features_2d[0, 0].mean(dim=0).cpu().numpy()
        im3 = ax3.imshow(xray_feat, cmap='viridis')
        ax3.set_title('X-ray Features (AP)')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        ax4 = plt.subplot(3, 5, 4)
        xray_feat2 = xray_features_2d[0, 1].mean(dim=0).cpu().numpy()
        im4 = ax4.imshow(xray_feat2, cmap='viridis')
        ax4.set_title('X-ray Features (Lateral)')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # If using learnable priors, visualize depth weights
        if model.use_learnable_priors:
            pooled_features = F.adaptive_avg_pool1d(xray_context.unsqueeze(-1), 512).squeeze(-1)
            xray_features_avg = xray_features_2d.mean(dim=1)
            depth_weights, depth_aux = model.depth_lifter(xray_features_avg, pooled_features)
            
            # Plot depth distribution for center pixel
            ax5 = plt.subplot(3, 5, 5)
            H, W = depth_weights.shape[1:3]
            center_depth = depth_weights[0, H//2, W//2].cpu().numpy()
            ax5.plot(center_depth)
            ax5.set_title('Depth Distribution (Center)')
            ax5.set_xlabel('Depth Index')
            ax5.set_ylabel('Weight')
            ax5.grid(True, alpha=0.3)
            
            # Plot learned boundaries
            if 'boundaries' in depth_aux:
                boundaries = depth_aux['boundaries'][0].cpu().numpy() * len(center_depth)
                for b in boundaries:
                    ax5.axvline(b, color='r', linestyle='--', alpha=0.5)
            
            # Plot depth weight heatmap (middle slice)
            ax6 = plt.subplot(3, 5, 6)
            depth_mid = depth_weights.shape[-1] // 2
            depth_slice = depth_weights[0, :, :, depth_mid].cpu().numpy()
            im6 = ax6.imshow(depth_slice, cmap='hot')
            ax6.set_title(f'Depth Weights (Slice {depth_mid})')
            ax6.axis('off')
            plt.colorbar(im6, ax=ax6, fraction=0.046)
        
        # Get model prediction
        predicted, aux_info = model(xrays)
        
        # Plot predicted CT slices (axial, sagittal, coronal)
        D, H, W = predicted.shape[2:]
        
        ax7 = plt.subplot(3, 5, 7)
        pred_axial = predicted[0, 0, D//2].cpu().numpy()
        im7 = ax7.imshow(pred_axial, cmap='gray', vmin=-1, vmax=1)
        ax7.set_title('Predicted CT (Axial)')
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046)
        
        ax8 = plt.subplot(3, 5, 8)
        pred_sagittal = predicted[0, 0, :, H//2, :].cpu().numpy()
        im8 = ax8.imshow(pred_sagittal, cmap='gray', vmin=-1, vmax=1)
        ax8.set_title('Predicted CT (Sagittal)')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046)
        
        ax9 = plt.subplot(3, 5, 9)
        pred_coronal = predicted[0, 0, :, :, W//2].cpu().numpy()
        im9 = ax9.imshow(pred_coronal, cmap='gray', vmin=-1, vmax=1)
        ax9.set_title('Predicted CT (Coronal)')
        ax9.axis('off')
        plt.colorbar(im9, ax=ax9, fraction=0.046)
        
        # Plot target CT slices
        ax10 = plt.subplot(3, 5, 10)
        target_axial = target[0, 0, D//2].cpu().numpy()
        im10 = ax10.imshow(target_axial, cmap='gray', vmin=-1, vmax=1)
        ax10.set_title('Target CT (Axial)')
        ax10.axis('off')
        plt.colorbar(im10, ax=ax10, fraction=0.046)
        
        ax11 = plt.subplot(3, 5, 11)
        target_sagittal = target[0, 0, :, H//2, :].cpu().numpy()
        im11 = ax11.imshow(target_sagittal, cmap='gray', vmin=-1, vmax=1)
        ax11.set_title('Target CT (Sagittal)')
        ax11.axis('off')
        plt.colorbar(im11, ax=ax11, fraction=0.046)
        
        ax12 = plt.subplot(3, 5, 12)
        target_coronal = target[0, 0, :, :, W//2].cpu().numpy()
        im12 = ax12.imshow(target_coronal, cmap='gray', vmin=-1, vmax=1)
        ax12.set_title('Target CT (Coronal)')
        ax12.axis('off')
        plt.colorbar(im12, ax=ax12, fraction=0.046)
        
        # Plot error maps
        error = torch.abs(predicted - target)
        
        ax13 = plt.subplot(3, 5, 13)
        error_axial = error[0, 0, D//2].cpu().numpy()
        im13 = ax13.imshow(error_axial, cmap='hot', vmin=0, vmax=0.5)
        ax13.set_title('Error Map (Axial)')
        ax13.axis('off')
        plt.colorbar(im13, ax=ax13, fraction=0.046)
        
        ax14 = plt.subplot(3, 5, 14)
        error_sagittal = error[0, 0, :, H//2, :].cpu().numpy()
        im14 = ax14.imshow(error_sagittal, cmap='hot', vmin=0, vmax=0.5)
        ax14.set_title('Error Map (Sagittal)')
        ax14.axis('off')
        plt.colorbar(im14, ax=ax14, fraction=0.046)
        
        ax15 = plt.subplot(3, 5, 15)
        error_coronal = error[0, 0, :, :, W//2].cpu().numpy()
        im15 = ax15.imshow(error_coronal, cmap='hot', vmin=0, vmax=0.5)
        ax15.set_title('Error Map (Coronal)')
        ax15.axis('off')
        plt.colorbar(im15, ax=ax15, fraction=0.046)
        
        # Compute metrics
        psnr = compute_psnr(predicted, target)
        
        plt.suptitle(f'Epoch {epoch} - Feature Maps & Predictions (PSNR: {psnr:.2f} dB)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        save_path = Path(save_dir) / f'epoch_{epoch:03d}_features.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved feature maps to {save_path}")


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
    train_dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_xray_size=512,
        target_volume_size=(64, 64, 64),
        max_patients=config['data']['max_patients']
    )
    
    val_dataset = PatientDRRDataset(
        data_path=config['data']['dataset_path'],
        target_xray_size=512,
        target_volume_size=(64, 64, 64),
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
    
    # Get a sample batch for visualization
    sample_batch = next(iter(val_loader))
    sample_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in sample_batch.items()}
    
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
        
        # Visualize feature maps every epoch
        print(f"\nGenerating feature map visualizations...")
        visualize_feature_maps(model, sample_batch, epoch)
        
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
