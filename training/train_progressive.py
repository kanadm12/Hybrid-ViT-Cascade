"""
Progressive Training Script for Unified Hybrid-ViT Cascade
Train stage-by-stage: 64³ → 128³ → 256³
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import wandb
from typing import Dict, Optional

# Add parent to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir / "hybrid_vit_cascade"))

from models.unified_model import UnifiedHybridViTCascade
from utils.visualization import visualize_epoch_features


def load_config(config_path: str) -> Dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_dataloaders(config: Dict, stage_config: Dict, batch_size: int, num_workers: int = 4):
    """
    Create dataloaders for current stage using PatientDRRDataset
    """
    # Import dataset utilities
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.dataset import create_train_val_datasets
    
    # Get data config
    data_config = config.get('data', {})
    data_path = data_config.get('dataset_path', './data/drr_patient_data')
    
    # Dataset parameters
    dataset_kwargs = {
        'target_xray_size': data_config.get('target_xray_size', 512),
        'target_volume_size': tuple(stage_config['volume_size']),
        'normalize_range': tuple(data_config.get('normalize_range', [-1, 1])),
        'validate_alignment': data_config.get('validate_alignment', True),
        'augmentation': data_config.get('augmentation', False),
        'cache_in_memory': data_config.get('cache_in_memory', False),
        'flip_drrs_vertical': data_config.get('flip_drrs_vertical', False)
    }
    
    # Create datasets
    train_dataset, val_dataset, _ = create_train_val_datasets(
        data_path=data_path,
        train_split=data_config.get('train_split', 0.8),
        val_split=data_config.get('val_split', 0.1),
        **dataset_kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Print alignment report if validation was enabled
    if data_config.get('validate_alignment', True):
        print("\n" + "="*60)
        print("DRR-CT Alignment Validation Report")
        print("="*60)
        alignment_report = train_dataset.dataset.get_alignment_report()
        print(f"Total validated: {alignment_report['total_validated']}")
        print(f"Passed: {alignment_report['passed']}")
        print(f"Failed: {alignment_report['failed']}")
        print(f"Pass rate: {alignment_report['pass_rate']*100:.2f}%")
        print(f"Average error: {alignment_report['average_error']:.6f}")
        print("="*60 + "\n")
    
    return train_loader, val_loader


def train_stage(model: UnifiedHybridViTCascade,
                stage_name: str,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: optim.Optimizer,
                scheduler: Optional[optim.lr_scheduler._LRScheduler],
                num_epochs: int,
                device: torch.device,
                checkpoint_dir: Path,
                prev_stage_model: Optional[UnifiedHybridViTCascade] = None,
                use_wandb: bool = False,
                visualize_features: bool = True,
                viz_dir: Optional[Path] = None):
    """
    Train a single stage
    
    Args:
        model: Full model (all stages)
        stage_name: Name of stage to train
        train_loader: Training data
        val_loader: Validation data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs
        device: Device
        checkpoint_dir: Directory to save checkpoints
        prev_stage_model: Previous stage model (if cascading)
        use_wandb: Whether to log to Weights & Biases
        visualize_features: Whether to visualize feature maps after each epoch
        viz_dir: Directory to save visualizations (defaults to checkpoint_dir/visualizations)
    """
    if viz_dir is None:
        viz_dir = checkpoint_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    # Get a validation batch for feature visualization
    val_batch_for_viz = None
    if visualize_features:
        try:
            val_batch_for_viz = next(iter(val_loader))
        except StopIteration:
            print("Warning: Could not get validation batch for visualization")
            visualize_features = False
    
    # Freeze all stages except current
    for name, stage in model.stages.items():
        if name == stage_name:
            stage.train()
            for param in stage.parameters():
                param.requires_grad = True
        else:
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False
    
    # Keep X-ray encoder trainable
    model.xray_encoder.train()
    for param in model.xray_encoder.parameters():
        param.requires_grad = True
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = {'total': 0, 'diffusion': 0, 'physics': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, batch_data in enumerate(pbar):
            # Extract data from batch
            if isinstance(batch_data, dict):
                volumes = batch_data['ct_volume'].to(device)
                xrays = batch_data['drr_stacked'].to(device)  # (batch, 2, 1, H, W)
            else:
                volumes, xrays = batch_data
                volumes = volumes.to(device)
                xrays = xrays.to(device)
            
            # Get previous stage volume if needed
            prev_volume = None
            if prev_stage_model is not None:
                with torch.no_grad():
                    # TODO: Generate volume from previous stage
                    pass
            
            # Forward
            loss_dict = model(volumes, xrays, stage_name, prev_volume)
            
            # Backward
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Accumulate losses
            train_losses['total'] += loss_dict['loss'].item()
            train_losses['diffusion'] += loss_dict['diffusion_loss'].item()
            train_losses['physics'] += loss_dict['physics_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['loss'].item(),
                'diff': loss_dict['diffusion_loss'].item(),
                'phys': loss_dict['physics_loss'].item()
            })
        
        # Average training losses
        num_batches = len(train_loader)
        for key in train_losses:
            train_losses[key] /= num_batches
        
        # Validation
        model.eval()
        val_losses = {'total': 0, 'diffusion': 0, 'physics': 0}
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # Extract data from batch
                if isinstance(batch_data, dict):
                    volumes = batch_data['ct_volume'].to(device)
                    xrays = batch_data['drr_stacked'].to(device)
                else:
                    volumes, xrays = batch_data
                    volumes = volumes.to(device)
                    xrays = xrays.to(device)
                
                prev_volume = None  # TODO: same as training
                
                loss_dict = model(volumes, xrays, stage_name, prev_volume)
                
                val_losses['total'] += loss_dict['loss'].item()
                val_losses['diffusion'] += loss_dict['diffusion_loss'].item()
                val_losses['physics'] += loss_dict['physics_loss'].item()
        
        # Average validation losses
        num_val_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_val_batches
        
        # Calculate PSNR and SSIM every epoch (memory-efficient)
        val_metrics = {'psnr': 0, 'ssim': 0}
        try:
            torch.cuda.empty_cache()
            
            # Get one batch for metrics
            val_iter = iter(val_loader)
            batch_data = next(val_iter)
            
            if isinstance(batch_data, dict):
                volumes = batch_data['ct_volume']
                xrays = batch_data['drr_stacked']
            else:
                volumes, xrays = batch_data
            
            # Process only FIRST sample to save memory
            volumes = volumes[0:1].to(device)
            xrays = xrays[0:1].to(device)
            
            stage = model.stages[stage_name]
            
            # Do lightweight DDIM sampling (10 steps to save memory)
            num_inference_steps = 10
            timesteps = torch.linspace(model.num_timesteps - 1, 0, num_inference_steps).long().to(device)
            
            # Start from pure noise
            pred_volume = torch.randn_like(volumes)
            
            # DDIM denoising loop
            with torch.no_grad():
                for i, t in enumerate(timesteps):
                    t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                    
                    # Create timestep embeddings
                    t_normalized = t_batch.float() / model.num_timesteps
                    t_embed = model.time_embed(t_normalized.unsqueeze(-1))
                    
                    # Encode X-rays
                    xray_context, time_xray_cond, xray_features_2d = model.xray_encoder(xrays, t_embed)
                    
                    # Predict noise/velocity
                    model_output = stage(
                        noisy_volume=pred_volume,
                        xray_features=xray_features_2d,
                        xray_context=xray_features_2d,
                        time_xray_cond=time_xray_cond,
                        prev_stage_volume=None,
                        prev_stage_embed=None
                    )
                    
                    # DDIM step
                    alpha_t = model.alphas_cumprod[t_batch][:, None, None, None, None]
                    
                    if i < len(timesteps) - 1:
                        t_prev = torch.full((1,), timesteps[i+1], device=device, dtype=torch.long)
                        alpha_prev = model.alphas_cumprod[t_prev][:, None, None, None, None]
                    else:
                        alpha_prev = torch.ones_like(alpha_t)
                    
                    # Convert v-prediction to x0 prediction
                    sqrt_alpha_t = torch.sqrt(alpha_t)
                    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                    
                    pred_x0 = sqrt_alpha_t * pred_volume - sqrt_one_minus_alpha_t * model_output
                    
                    # DDIM update
                    if i < len(timesteps) - 1:
                        pred_volume = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * model_output
                    else:
                        pred_volume = pred_x0
                    
                    # Clean up
                    del model_output, xray_context, time_xray_cond, xray_features_2d, t_embed
                    torch.cuda.empty_cache()
            
            # PSNR calculation
            mse = torch.mean((pred_volume - volumes) ** 2)
            if mse > 0:
                psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
                val_metrics['psnr'] = psnr.item()
            
            # SSIM calculation
            C1 = (0.01 * 2) ** 2
            C2 = (0.03 * 2) ** 2
            
            mu1 = torch.mean(pred_volume)
            mu2 = torch.mean(volumes)
            sigma1_sq = torch.var(pred_volume)
            sigma2_sq = torch.var(volumes)
            sigma12 = torch.mean((pred_volume - mu1) * (volumes - mu2))
            
            ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
            val_metrics['ssim'] = ssim.item()
            
            # Clean up
            del pred_volume, volumes, xrays
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[WARNING] Failed to calculate PSNR/SSIM: {e}")
            val_metrics['psnr'] = 0.0
            val_metrics['ssim'] = 0.0
        
        # Logging
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Total: {train_losses['total']:.6f}, Diff: {train_losses['diffusion']:.6f}, Phys: {train_losses['physics']:.6f}")
        print(f"  Val   - Total: {val_losses['total']:.6f}, Diff: {val_losses['diffusion']:.6f}, Phys: {val_losses['physics']:.6f}")
        print(f"  Metrics - PSNR: {val_metrics['psnr']:.2f} dB, SSIM: {val_metrics['ssim']:.4f}")
        
        if use_wandb:
            wandb.log({
                f'{stage_name}/train_loss': train_losses['total'],
                f'{stage_name}/train_diffusion': train_losses['diffusion'],
                f'{stage_name}/train_physics': train_losses['physics'],
                f'{stage_name}/val_loss': val_losses['total'],
                f'{stage_name}/val_diffusion': val_losses['diffusion'],
                f'{stage_name}/val_physics': val_losses['physics'],
                f'{stage_name}/psnr': val_metrics['psnr'],
                f'{stage_name}/ssim': val_metrics['ssim'],
                'epoch': epoch
            })
        
        # Save checkpoint if best
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint_path = checkpoint_dir / f"{stage_name}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'stage_name': stage_name
            }, checkpoint_path)
            print(f"  Saved best checkpoint: {checkpoint_path}")
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step(val_losses['total'])
        
        # Visualize feature maps every 5 epochs
        if visualize_features and val_batch_for_viz is not None and (epoch + 1) % 5 == 0:
            print(f"\n  Visualizing feature maps...")
            try:
                viz_figs = visualize_epoch_features(
                    model=model,
                    val_batch=val_batch_for_viz,
                    epoch=epoch,
                    stage_name=stage_name,
                    save_dir=viz_dir,
                    device=device,
                    wandb_log=use_wandb
                )
                print(f"  Feature maps saved to {viz_dir / f'epoch_{epoch:03d}'}")
            except Exception as e:
                print(f"  Warning: Feature visualization failed: {e}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='hybrid-vit-cascade', help='W&B project name')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B
    if args.wandb:
        wandb.init(project=args.wandb_project, config=config)
    
    # Create model
    model = UnifiedHybridViTCascade(
        stage_configs=config['stages'],
        xray_img_size=config['xray_img_size'],
        xray_channels=config['xray_channels'],
        num_views=config['num_views'],
        v_parameterization=config.get('v_parameterization', True),
        num_timesteps=config.get('num_timesteps', 1000),
        extract_features=True  # Enable feature extraction for visualization
    ).to(device)
    
    print(f"\nModel created with {len(config['stages'])} stages")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Progressive training
    prev_stage_model = None
    
    for stage_idx, stage_config in enumerate(config['stages']):
        stage_name = stage_config['name']
        print(f"\n{'='*60}")
        print(f"Training Stage {stage_idx + 1}/{len(config['stages'])}: {stage_name}")
        print(f"Volume size: {stage_config['volume_size']}")
        print(f"{'='*60}\n")
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            stage_config,
            batch_size=config['training']['batch_size'],
            num_workers=config['training'].get('num_workers', 4)
        )
        
        # Optimizer (only for current stage + xray encoder)
        trainable_params = []
        trainable_params.extend(model.stages[stage_name].parameters())
        trainable_params.extend(model.xray_encoder.parameters())
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01)
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Train stage
        model = train_stage(
            model=model,
            stage_name=stage_name,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config['training']['num_epochs'],
            device=device,
            checkpoint_dir=checkpoint_dir,
            prev_stage_model=prev_stage_model,
            use_wandb=args.wandb,
            visualize_features=config['training'].get('visualize_features', True),
            viz_dir=checkpoint_dir / "visualizations"
        )
        
        # Save stage checkpoint
        stage_checkpoint = checkpoint_dir / f"{stage_name}_final.pt"
        torch.save(model.state_dict(), stage_checkpoint)
        print(f"\nStage {stage_name} training completed. Saved to {stage_checkpoint}")
        
        # Update prev_stage_model
        prev_stage_model = model
    
    print("\n" + "="*60)
    print("Progressive training completed!")
    print("="*60)
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
