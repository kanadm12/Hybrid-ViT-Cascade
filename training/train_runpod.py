"""
RunPod Training Script for Hybrid-ViT Cascade
Includes setup verification and optimized training for cloud GPUs
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import json
import argparse
from pathlib import Path
import sys
import os
from tqdm import tqdm
from typing import Dict, Optional
import time

# Add parent to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from models.unified_model import UnifiedHybridViTCascade
from utils.dataset import create_train_val_datasets
from utils.visualization import visualize_epoch_features


def verify_environment():
    """Verify RunPod environment and data availability"""
    print("\n" + "="*70)
    print("RUNPOD ENVIRONMENT VERIFICATION")
    print("="*70)
    
    # Check CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # Check data directory
    data_paths = [
        '/workspace/drr_patient_data',
        './workspace/drr_patient_data',
        '../workspace/drr_patient_data'
    ]
    
    data_found = False
    for data_path in data_paths:
        if os.path.exists(data_path):
            print(f"\n✓ Data directory found: {data_path}")
            patient_folders = [f for f in os.listdir(data_path) 
                             if os.path.isdir(os.path.join(data_path, f)) and not f.startswith('.')]
            print(f"  Found {len(patient_folders)} patient folders")
            if len(patient_folders) > 0:
                print(f"  Sample patient IDs: {patient_folders[:5]}")
            data_found = True
            break
    
    if not data_found:
        print("\n✗ WARNING: Data directory not found!")
        print("  Expected paths:")
        for path in data_paths:
            print(f"    - {path}")
        print("\n  Please ensure your data is mounted correctly.")
    
    print("="*70 + "\n")
    
    return data_found


def load_config(config_path: str) -> Dict:
    """Load configuration with environment variable substitution"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Substitute environment variables in paths
    if 'data' in config and 'dataset_path' in config['data']:
        path = config['data']['dataset_path']
        if path.startswith('$'):
            env_var = path.split('/')[0][1:]
            if env_var in os.environ:
                path = path.replace(f'${env_var}', os.environ[env_var])
                config['data']['dataset_path'] = path
    
    return config


def train_stage_with_amp(
    model: UnifiedHybridViTCascade,
    stage_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    num_epochs: int,
    device: torch.device,
    checkpoint_dir: Path,
    config: Dict,
    prev_stage_model: Optional[UnifiedHybridViTCascade] = None,
    tensorboard_writer: Optional[SummaryWriter] = None,
    visualize_features: bool = True,
    viz_dir: Optional[Path] = None
):
    """
    Train a single stage with mixed precision and gradient accumulation
    """
    
    # Setup visualization directory
    if viz_dir is None:
        viz_dir = checkpoint_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a validation batch for feature visualization
    val_batch_for_viz = None
    if visualize_features:
        try:
            val_batch_for_viz = next(iter(val_loader))
        except StopIteration:
            print("Warning: Could not get validation batch for visualization")
            visualize_features = False
    
    # Mixed precision scaler
    use_amp = config['training'].get('mixed_precision', False)
    scaler = GradScaler(enabled=use_amp)
    grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)
    
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
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        train_losses = {'total': 0, 'diffusion': 0, 'physics': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(pbar):
            # Extract data
            if isinstance(batch_data, dict):
                volumes = batch_data['ct_volume'].to(device)
                xrays = batch_data['drr_stacked'].to(device)
            else:
                volumes, xrays = batch_data
                volumes = volumes.to(device)
                xrays = xrays.to(device)
            
            # Forward with mixed precision
            with autocast(enabled=use_amp):
                prev_volume = None
                if prev_stage_model is not None:
                    with torch.no_grad():
                        # TODO: Generate volume from previous stage
                        pass
                
                loss_dict = model(volumes, xrays, stage_name, prev_volume)
                loss = loss_dict['loss'] / grad_accum_steps
            
            # NaN detection - skip batch if NaN detected
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠ Warning: NaN/Inf loss detected at batch {batch_idx}, skipping...")
                optimizer.zero_grad()
                continue
            
            # Clamp loss to prevent extreme values
            loss = torch.clamp(loss, max=100.0)
            
            # Backward with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights with gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training'].get('gradient_clip', 1.0)
                )
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                global_step += 1
            
            # Accumulate losses (skip if NaN)
            loss_val = loss_dict['loss'].item()
            if not (math.isnan(loss_val) or math.isinf(loss_val)):
                train_losses['total'] += loss_val
                train_losses['diffusion'] += loss_dict['diffusion_loss'].item()
                train_losses['physics'] += loss_dict['physics_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_val,
                'diff': loss_dict['diffusion_loss'].item(),
                'phys': loss_dict['physics_loss'].item()
            })
            
            # Log to TensorBoard
            if tensorboard_writer is not None and global_step % config['logging'].get('log_every_n_steps', 50) == 0:
                tensorboard_writer.add_scalar(f'{stage_name}/train_loss_step', loss_dict['loss'].item(), global_step)
                tensorboard_writer.add_scalar(f'{stage_name}/train_diffusion_step', loss_dict['diffusion_loss'].item(), global_step)
                tensorboard_writer.add_scalar(f'{stage_name}/train_physics_step', loss_dict['physics_loss'].item(), global_step)
        
        # Average training losses
        num_batches = len(train_loader)
        for key in train_losses:
            train_losses[key] /= num_batches
        
        # Validation
        model.eval()
        val_losses = {'total': 0, 'diffusion': 0, 'physics': 0}
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                if isinstance(batch_data, dict):
                    volumes = batch_data['ct_volume'].to(device)
                    xrays = batch_data['drr_stacked'].to(device)
                else:
                    volumes, xrays = batch_data
                    volumes = volumes.to(device)
                    xrays = xrays.to(device)
                
                with autocast(enabled=use_amp):
                    prev_volume = None
                    loss_dict = model(volumes, xrays, stage_name, prev_volume)
                
                loss_val = loss_dict['loss'].item()
                if not (math.isnan(loss_val) or math.isinf(loss_val)):
                    val_losses['total'] += loss_val
                    val_losses['diffusion'] += loss_dict['diffusion_loss'].item()
                    val_losses['physics'] += loss_dict['physics_loss'].item()
        
        # Average validation losses
        num_val_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_val_batches
        
        epoch_time = time.time() - epoch_start_time
        
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
                    with autocast(enabled=use_amp):
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
        print(f"\nEpoch {epoch+1}/{num_epochs} (time: {epoch_time:.2f}s):")
        print(f"  Train - Total: {train_losses['total']:.6f}, Diff: {train_losses['diffusion']:.6f}, Phys: {train_losses['physics']:.6f}")
        print(f"  Val   - Total: {val_losses['total']:.6f}, Diff: {val_losses['diffusion']:.6f}, Phys: {val_losses['physics']:.6f}")
        print(f"  Metrics - PSNR: {val_metrics['psnr']:.2f} dB, SSIM: {val_metrics['ssim']:.4f}")
        
        # Log to TensorBoard
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(f'{stage_name}/train_loss', train_losses['total'], epoch)
            tensorboard_writer.add_scalar(f'{stage_name}/train_diffusion', train_losses['diffusion'], epoch)
            tensorboard_writer.add_scalar(f'{stage_name}/train_physics', train_losses['physics'], epoch)
            tensorboard_writer.add_scalar(f'{stage_name}/val_loss', val_losses['total'], epoch)
            tensorboard_writer.add_scalar(f'{stage_name}/val_diffusion', val_losses['diffusion'], epoch)
            tensorboard_writer.add_scalar(f'{stage_name}/val_physics', val_losses['physics'], epoch)
            tensorboard_writer.add_scalar(f'{stage_name}/psnr', val_metrics['psnr'], epoch)
            tensorboard_writer.add_scalar(f'{stage_name}/ssim', val_metrics['ssim'], epoch)
            tensorboard_writer.add_scalar(f'{stage_name}/epoch_time', epoch_time, epoch)
            tensorboard_writer.add_scalar(f'{stage_name}/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint if best
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint_path = checkpoint_dir / f"{stage_name}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_losses['total'],
                'stage_name': stage_name,
                'config': config
            }, checkpoint_path)
            print(f"  ✓ Saved best checkpoint: {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % config['checkpointing'].get('save_every_n_epochs', 5) == 0:
            checkpoint_path = checkpoint_dir / f"{stage_name}_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_losses['total'],
                'stage_name': stage_name,
                'config': config
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")
        
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
                    wandb_log=False  # RunPod uses tensorboard
                )
                print(f"  ✓ Feature maps saved to {viz_dir / f'epoch_{epoch:03d}'}")
            except Exception as e:
                print(f"  Warning: Feature visualization failed: {e}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid-ViT Cascade on RunPod')
    parser.add_argument('--config', type=str, default='config/runpod_config.json',
                       help='Path to config JSON')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Checkpoint directory (overrides config)')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--skip_verification', action='store_true',
                       help='Skip environment verification')
    args = parser.parse_args()
    
    # Verify environment
    if not args.skip_verification:
        data_found = verify_environment()
        if not data_found:
            response = input("\nData not found. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
    
    # Load config
    config = load_config(args.config)
    print(f"\nLoaded config from: {args.config}")
    
    # Override config with command line args
    if args.checkpoint_dir:
        config['checkpointing']['checkpoint_dir'] = args.checkpoint_dir
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpointing']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Initialize TensorBoard
    tensorboard_dir = checkpoint_dir / 'tensorboard'
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"✓ TensorBoard logs will be saved to: {tensorboard_dir}")
    
    # Create model
    print("\nInitializing model...")
    model = UnifiedHybridViTCascade(
        stage_configs=config['stage_configs'],
        xray_img_size=config['xray_config']['img_size'],
        xray_channels=1,
        num_views=config['xray_config']['num_views'],
        v_parameterization=config['training'].get('v_parameterization', True),
        num_timesteps=config['training'].get('num_timesteps', 1000),
        extract_features=True  # Enable feature extraction for visualization
    ).to(device)
    
    print(f"✓ Model created with {len(config['stage_configs'])} stages")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Load data and create dataloaders for first stage to verify
    print("\nLoading data...")
    first_stage_config = config['stage_configs'][0]
    
    from utils.dataset import create_train_val_datasets
    
    data_config = config['data']
    dataset_kwargs = {
        'target_xray_size': data_config.get('target_xray_size', 512),
        'target_volume_size': tuple(first_stage_config['volume_size']),
        'normalize_range': tuple(data_config.get('normalize_range', [-1, 1])),
        'validate_alignment': data_config.get('validate_alignment', True),
        'augmentation': data_config.get('augmentation', False),
        'cache_in_memory': data_config.get('cache_in_memory', False)
    }
    
    # Just verify data loading works
    train_dataset, val_dataset, _ = create_train_val_datasets(
        data_path=data_config['dataset_path'],
        train_split=data_config.get('train_split', 0.8),
        val_split=data_config.get('val_split', 0.1),
        **dataset_kwargs
    )
    
    print(f"✓ Data loaded successfully")
    
    # Progressive training
    prev_stage_model = None
    
    for stage_idx, stage_config in enumerate(config['stage_configs']):
        stage_name = stage_config['name']
        print(f"\n{'='*70}")
        print(f"TRAINING STAGE {stage_idx + 1}/{len(config['stage_configs'])}: {stage_name}")
        print(f"Volume size: {stage_config['volume_size']}")
        print(f"{'='*70}\n")
        
        # Update dataset for this stage's volume size
        dataset_kwargs['target_volume_size'] = tuple(stage_config['volume_size'])
        
        train_dataset, val_dataset, _ = create_train_val_datasets(
            data_path=data_config['dataset_path'],
            train_split=data_config.get('train_split', 0.8),
            val_split=data_config.get('val_split', 0.1),
            **dataset_kwargs
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training'].get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training'].get('num_workers', 4),
            pin_memory=True,
            drop_last=False
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
        model = train_stage_with_amp(
            model=model,
            stage_name=stage_name,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config['training']['num_epochs_per_stage'],
            device=device,
            checkpoint_dir=checkpoint_dir,
            config=config,
            prev_stage_model=prev_stage_model,
            tensorboard_writer=tensorboard_writer,
            visualize_features=config['training'].get('visualize_features', True),
            viz_dir=checkpoint_dir / "visualizations"
        )
        
        # Save stage checkpoint
        stage_checkpoint = checkpoint_dir / f"{stage_name}_final.pt"
        torch.save(model.state_dict(), stage_checkpoint)
        print(f"\n✓ Stage {stage_name} training completed")
        print(f"  Final checkpoint saved to: {stage_checkpoint}")
        
        # Update prev_stage_model
        prev_stage_model = model
    
    print("\n" + "="*70)
    print("PROGRESSIVE TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # Close TensorBoard writer
    tensorboard_writer.close()
    print("✓ TensorBoard logs saved")


if __name__ == "__main__":
    main()
