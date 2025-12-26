"""
Accelerate-based Training Script for Hybrid-ViT Cascade
Simplified multi-GPU training using HuggingFace Accelerate

Setup:
    1. Install: pip install accelerate
    2. Configure: accelerate config
    3. Run: accelerate launch training/train_accelerate.py --config config/multi_view_config.json

For quick setup with 4 GPUs:
    accelerate launch --multi_gpu --num_processes=4 training/train_accelerate.py --config config/multi_view_config.json
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed
from torch.utils.data import DataLoader
import json
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
from typing import Dict, Optional

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed, DistributedDataParallelKwargs
    ACCELERATE_AVAILABLE = True
except ImportError:
    print("ERROR: accelerate not installed. Install with: pip install accelerate")
    sys.exit(1)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from models.unified_model import UnifiedHybridViTCascade


def load_config(config_path: str) -> Dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_dataloaders(config: Dict, stage_config: Dict, batch_size: int, num_workers: int = 4):
    """Create dataloaders"""
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
    
    return train_loader, val_loader


def train_stage(model,
                stage_name: str,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: optim.Optimizer,
                scheduler: Optional[optim.lr_scheduler._LRScheduler],
                num_epochs: int,
                accelerator: Accelerator,
                checkpoint_dir: Path,
                use_wandb: bool = False):
    """Train a single stage with Accelerate"""
    
    best_val_loss = float('inf')
    
    # Get unwrapped model for accessing attributes
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Freeze all stages except current
    for name, stage in unwrapped_model.stages.items():
        if name == stage_name:
            for param in stage.parameters():
                param.requires_grad = True
        else:
            for param in stage.parameters():
                param.requires_grad = False
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = {'total': 0, 'diffusion': 0, 'physics': 0}
        
        # Progress bar on main process only
        if accelerator.is_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        else:
            pbar = train_loader
        
        for batch_data in pbar:
            # Extract data
            if isinstance(batch_data, dict):
                volumes = batch_data['ct_volume']
                xrays = batch_data['drr_stacked']
            else:
                volumes, xrays = batch_data
            
            # TODO: Add previous stage volume for cascading
            prev_volume = None
            
            # Forward pass
            loss_dict = model(volumes, xrays, stage_name, prev_volume)
            
            # Backward pass with accelerate
            accelerator.backward(loss_dict['loss'])
            
            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Accumulate losses
            train_losses['total'] += loss_dict['loss'].item()
            train_losses['diffusion'] += loss_dict['diffusion_loss'].item()
            train_losses['physics'] += loss_dict['physics_loss'].item()
            
            # Update progress bar (main process only)
            if accelerator.is_main_process:
                pbar.set_postfix({
                    'loss': loss_dict['loss'].item(),
                    'diff': loss_dict['diffusion_loss'].item(),
                    'phys': loss_dict['physics_loss'].item()
                })
        
        # Average training losses
        num_batches = len(train_loader)
        for key in train_losses:
            train_losses[key] /= num_batches
        
        # Gather losses across all processes
        train_losses = {k: accelerator.gather(torch.tensor(v).to(accelerator.device)).mean().item() 
                       for k, v in train_losses.items()}
        
        # Validation
        model.eval()
        val_losses = {'total': 0, 'diffusion': 0, 'physics': 0}
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Extract data
                if isinstance(batch_data, dict):
                    volumes = batch_data['ct_volume']
                    xrays = batch_data['drr_stacked']
                else:
                    volumes, xrays = batch_data
                
                prev_volume = None
                
                loss_dict = model(volumes, xrays, stage_name, prev_volume)
                
                val_losses['total'] += loss_dict['loss'].item()
                val_losses['diffusion'] += loss_dict['diffusion_loss'].item()
                val_losses['physics'] += loss_dict['physics_loss'].item()
        
        # Average validation losses
        num_val_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_val_batches
        
        # Gather validation losses
        val_losses = {k: accelerator.gather(torch.tensor(v).to(accelerator.device)).mean().item() 
                     for k, v in val_losses.items()}
        
        # Logging (main process only)
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train - Total: {train_losses['total']:.6f}, Diff: {train_losses['diffusion']:.6f}, Phys: {train_losses['physics']:.6f}")
            print(f"  Val   - Total: {val_losses['total']:.6f}, Diff: {val_losses['diffusion']:.6f}, Phys: {val_losses['physics']:.6f}")
            
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f'{stage_name}/train_loss': train_losses['total'],
                    f'{stage_name}/train_diffusion': train_losses['diffusion'],
                    f'{stage_name}/train_physics': train_losses['physics'],
                    f'{stage_name}/val_loss': val_losses['total'],
                    f'{stage_name}/val_diffusion': val_losses['diffusion'],
                    f'{stage_name}/val_physics': val_losses['physics'],
                    'epoch': epoch,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint if best
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                checkpoint_path = checkpoint_dir / f"{stage_name}_best.pt"
                
                # Unwrap model for saving
                unwrapped_model = accelerator.unwrap_model(model)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_losses['total'],
                    'stage_name': stage_name
                }, checkpoint_path)
                print(f"  Saved best checkpoint: {checkpoint_path}")
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step(val_losses['total'])
        
        # Wait for all processes
        accelerator.wait_for_everyone()
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='hybrid-vit-cascade', help='W&B project name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16',  # Use mixed precision for A100s
        log_with='wandb' if (args.wandb and WANDB_AVAILABLE) else None,
        project_dir=args.checkpoint_dir,
        kwargs_handlers=[
            # Enable unused parameters for stage-wise training
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ] if torch.cuda.device_count() > 1 else []
    )
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Print info on main process
    if accelerator.is_main_process:
        print(f"Accelerate Training Setup:")
        print(f"  Num processes: {accelerator.num_processes}")
        print(f"  Process index: {accelerator.process_index}")
        print(f"  Device: {accelerator.device}")
        print(f"  Mixed precision: {accelerator.mixed_precision}")
    
    # Load config
    config = load_config(args.config)
    
    # Initialize W&B (main process only)
    if args.wandb and accelerator.is_main_process and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=config)
    
    # Create checkpoint directory (main process only)
    if accelerator.is_main_process:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
    
    # Wait for checkpoint directory to be created
    accelerator.wait_for_everyone()
    
    # Create model
    model = UnifiedHybridViTCascade(
        stage_configs=config['stage_configs'],
        xray_img_size=config['xray_config']['img_size'],
        xray_channels=1,
        num_views=config['xray_config']['num_views'],
        share_view_weights=config['xray_config'].get('share_view_weights', False),
        v_parameterization=config['training'].get('v_parameterization', True),
        num_timesteps=config['training'].get('num_timesteps', 1000),
        extract_features=False  # Disable for distributed training
    )
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel created with {len(config['stage_configs'])} stages")
        print(f"Total parameters: {total_params:,}")
    
    # Progressive training
    for stage_idx, stage_config in enumerate(config['stage_configs']):
        stage_name = stage_config['name']
        
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Training Stage {stage_idx + 1}/{len(config['stage_configs'])}: {stage_name}")
            print(f"Volume size: {stage_config['volume_size']}")
            print(f"{'='*60}\n")
        
        # Create dataloaders
        batch_size = config['training'].get('batch_size', 4)
        train_loader, val_loader = create_dataloaders(
            config=config,
            stage_config=stage_config,
            batch_size=batch_size,
            num_workers=4
        )
        
        if accelerator.is_main_process:
            print(f"Batch size per GPU: {batch_size}")
            print(f"Effective batch size: {batch_size * accelerator.num_processes}")
            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}\n")
        
        # Create optimizer
        learning_rate = config['training'].get('learning_rate', 1e-4)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Create scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, 
            verbose=accelerator.is_main_process
        )
        
        # Prepare with accelerator
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
        
        # Train stage
        num_epochs = config['training'].get('num_epochs_per_stage', 100)
        model = train_stage(
            model=model,
            stage_name=stage_name,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            accelerator=accelerator,
            checkpoint_dir=checkpoint_dir,
            use_wandb=(args.wandb and accelerator.is_main_process)
        )
    
    # Cleanup
    if accelerator.is_main_process and args.wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    if accelerator.is_main_process:
        print("\nTraining completed!")


if __name__ == '__main__':
    main()
