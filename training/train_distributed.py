"""
Distributed Training Script for Hybrid-ViT Cascade
Supports multi-GPU training using PyTorch DDP

Usage:
    Single Node (4 GPUs):
        torchrun --nproc_per_node=4 training/train_distributed.py --config config/multi_view_config.json
    
    Multi-Node (2 nodes, 4 GPUs each):
        Node 0: torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" training/train_distributed.py --config config/multi_view_config.json
        Node 1: torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" training/train_distributed.py --config config/multi_view_config.json
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import json
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import wandb
import os
from typing import Dict, Optional

# Add parent to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from models.unified_model import UnifiedHybridViTCascade
from utils.visualization import visualize_epoch_features


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    return True, rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> Dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_dataloaders(config: Dict, stage_config: Dict, batch_size: int, 
                       num_workers: int = 4, is_distributed: bool = False, rank: int = 0):
    """Create dataloaders with optional distributed sampler"""
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
        'validate_alignment': data_config.get('validate_alignment', True) and rank == 0,  # Only validate on main process
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
    
    # Create samplers for distributed training
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False,
            drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create dataloaders with optimizations for multi-GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Print alignment report on main process
    if rank == 0 and data_config.get('validate_alignment', True):
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
    
    return train_loader, val_loader, train_sampler


def train_stage(model: DDP,
                stage_name: str,
                train_loader: DataLoader,
                val_loader: DataLoader,
                train_sampler: Optional[DistributedSampler],
                optimizer: optim.Optimizer,
                scheduler: Optional[optim.lr_scheduler._LRScheduler],
                num_epochs: int,
                device: torch.device,
                checkpoint_dir: Path,
                rank: int,
                world_size: int,
                use_wandb: bool = False):
    """Train a single stage with distributed training"""
    
    is_main_process = (rank == 0)
    best_val_loss = float('inf')
    
    # Freeze all stages except current (on the underlying module)
    for name, stage in model.module.stages.items():
        if name == stage_name:
            for param in stage.parameters():
                param.requires_grad = True
        else:
            for param in stage.parameters():
                param.requires_grad = False
    
    for epoch in range(num_epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Training
        model.train()
        train_losses = {'total': 0, 'diffusion': 0, 'physics': 0}
        
        # Use tqdm only on main process
        if is_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        else:
            pbar = train_loader
        
        for batch_data in pbar:
            # Extract data
            if isinstance(batch_data, dict):
                volumes = batch_data['ct_volume'].to(device)
                xrays = batch_data['drr_stacked'].to(device)
            else:
                volumes, xrays = batch_data
                volumes = volumes.to(device)
                xrays = xrays.to(device)
            
            # TODO: Add previous stage volume for cascading
            prev_volume = None
            
            # Forward pass
            loss_dict = model.module(volumes, xrays, stage_name, prev_volume)
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            train_losses['total'] += loss_dict['loss'].item()
            train_losses['diffusion'] += loss_dict['diffusion_loss'].item()
            train_losses['physics'] += loss_dict['physics_loss'].item()
            
            # Update progress bar (main process only)
            if is_main_process:
                pbar.set_postfix({
                    'loss': loss_dict['loss'].item(),
                    'diff': loss_dict['diffusion_loss'].item(),
                    'phys': loss_dict['physics_loss'].item()
                })
        
        # Average training losses across all batches
        num_batches = len(train_loader)
        for key in train_losses:
            train_losses[key] /= num_batches
        
        # Synchronize losses across GPUs
        if world_size > 1:
            for key in train_losses:
                tensor = torch.tensor(train_losses[key], device=device)
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                train_losses[key] = tensor.item()
        
        # Validation
        model.eval()
        val_losses = {'total': 0, 'diffusion': 0, 'physics': 0}
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Extract data
                if isinstance(batch_data, dict):
                    volumes = batch_data['ct_volume'].to(device)
                    xrays = batch_data['drr_stacked'].to(device)
                else:
                    volumes, xrays = batch_data
                    volumes = volumes.to(device)
                    xrays = xrays.to(device)
                
                prev_volume = None
                
                loss_dict = model.module(volumes, xrays, stage_name, prev_volume)
                
                val_losses['total'] += loss_dict['loss'].item()
                val_losses['diffusion'] += loss_dict['diffusion_loss'].item()
                val_losses['physics'] += loss_dict['physics_loss'].item()
        
        # Average validation losses
        num_val_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_val_batches
        
        # Synchronize validation losses across GPUs
        if world_size > 1:
            for key in val_losses:
                tensor = torch.tensor(val_losses[key], device=device)
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                val_losses[key] = tensor.item()
        
        # Calculate PSNR and SSIM every epoch (main process only)
        val_metrics = {'psnr': 0, 'ssim': 0}
        if is_main_process:
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
                
                stage = model.module.stages[stage_name]
                
                # Do lightweight DDIM sampling (10 steps to save memory)
                num_inference_steps = 10
                timesteps = torch.linspace(model.module.num_timesteps - 1, 0, num_inference_steps).long().to(device)
                
                # Start from pure noise
                pred_volume = torch.randn_like(volumes)
                
                # DDIM denoising loop
                with torch.no_grad():
                    for i, t in enumerate(timesteps):
                        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                        
                        # Create timestep embeddings
                        t_normalized = t_batch.float() / model.module.num_timesteps
                        t_embed = model.module.time_embed(t_normalized.unsqueeze(-1))
                        
                        # Encode X-rays
                        xray_context, time_xray_cond, xray_features_2d = model.module.xray_encoder(xrays, t_embed)
                        
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
                        alpha_t = model.module.alphas_cumprod[t_batch][:, None, None, None, None]
                        
                        if i < len(timesteps) - 1:
                            t_prev = torch.full((1,), timesteps[i+1], device=device, dtype=torch.long)
                            alpha_prev = model.module.alphas_cumprod[t_prev][:, None, None, None, None]
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
        
        # Logging (main process only)
        if is_main_process:
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
                    'epoch': epoch,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint if best
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                checkpoint_path = checkpoint_dir / f"{stage_name}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_losses['total'],
                    'stage_name': stage_name
                }, checkpoint_path)
                print(f"  Saved best checkpoint: {checkpoint_path}")
            
            # Visualize features every 5 epochs
            if (epoch + 1) % 5 == 0:
                try:
                    import matplotlib
                    matplotlib.use('Agg')  # Non-interactive backend
                    
                    # Get a validation batch for visualization
                    val_batch = next(iter(val_loader))
                    
                    # Save to workspace directory
                    viz_dir = Path('/workspace/feature_maps') / stage_name
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    
                    viz_figs = visualize_epoch_features(
                        model=model.module,
                        val_batch=val_batch,
                        epoch=epoch,
                        stage_name=stage_name,
                        save_dir=viz_dir,
                        device=device,
                        wandb_log=use_wandb
                    )
                    print(f"  Feature maps saved to {viz_dir / f'epoch_{epoch:03d}'}")
                except Exception as e:
                    print(f"  Warning: Feature visualization failed: {e}")
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step(val_losses['total'])
        
        # Synchronize all processes
        if world_size > 1:
            dist.barrier()
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='hybrid-vit-cascade', help='W&B project name')
    args = parser.parse_args()
    
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    
    if is_main_process:
        print(f"Distributed Training Setup:")
        print(f"  World Size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Local Rank: {local_rank}")
    
    # Set device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    config = load_config(args.config)
    
    # Initialize W&B (main process only)
    if args.wandb and is_main_process:
        wandb.init(project=args.wandb_project, config=config)
    
    # Create checkpoint directory (main process only)
    if is_main_process:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
    
    # Synchronize before model creation
    if is_distributed:
        dist.barrier()
    
    # Create model
    model = UnifiedHybridViTCascade(
        stage_configs=config['stage_configs'],
        xray_img_size=config['xray_config']['img_size'],
        xray_channels=1,
        num_views=config['xray_config']['num_views'],
        share_view_weights=config['xray_config'].get('share_view_weights', False),
        v_parameterization=config['training'].get('v_parameterization', True),
        num_timesteps=config['training'].get('num_timesteps', 1000),
        extract_features=False  # Disable for distributed training to save memory
    ).to(device)
    
    # Wrap model with DDP
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel created with {len(config['stage_configs'])} stages")
        print(f"Total parameters: {total_params:,}")
        print(f"Memory per GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Progressive training
    for stage_idx, stage_config in enumerate(config['stage_configs']):
        stage_name = stage_config['name']
        
        if is_main_process:
            print(f"\n{'='*60}")
            print(f"Training Stage {stage_idx + 1}/{len(config['stage_configs'])}: {stage_name}")
            print(f"Volume size: {stage_config['volume_size']}")
            print(f"{'='*60}\n")
        
        # Synchronize before creating dataloaders
        if is_distributed:
            dist.barrier()
        
        # Create dataloaders
        # Batch size per GPU - ensure at least 1
        batch_size_total = config['training'].get('batch_size', 4)
        batch_size = max(1, batch_size_total // world_size)  # At least 1 per GPU
        
        if is_main_process:
            print(f"Batch size: {batch_size} per GPU ({batch_size * world_size} total effective)")
        
        train_loader, val_loader, train_sampler = create_dataloaders(
            config=config,
            stage_config=stage_config,
            batch_size=batch_size,
            num_workers=4,  # Reduced: 4 workers total is enough for multi-GPU
            is_distributed=is_distributed,
            rank=rank
        )
        
        if is_main_process:
            print(f"Batch size per GPU: {batch_size}")
            print(f"Effective batch size: {batch_size * world_size}")
            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}\n")
        
        # Create optimizer
        learning_rate = config['training'].get('learning_rate', 1e-4)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Create scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=(rank == 0)
        )
        
        # Train stage
        num_epochs = config['training'].get('num_epochs_per_stage', 100)
        model = train_stage(
            model=model,
            stage_name=stage_name,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            checkpoint_dir=checkpoint_dir,
            rank=rank,
            world_size=world_size,
            use_wandb=(args.wandb and is_main_process)
        )
    
    # Cleanup
    if is_main_process and args.wandb:
        wandb.finish()
    
    cleanup_distributed()
    
    if is_main_process:
        print("\nTraining completed!")


if __name__ == '__main__':
    main()
