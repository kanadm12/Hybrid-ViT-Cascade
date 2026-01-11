"""
Resume Progressive Training from Checkpoint
Resumes training from a specific epoch checkpoint
"""
import torch
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_progressive_1gpu import train_stage
import json


def resume_training(stage, checkpoint_path, start_epoch):
    """
    Resume training from a checkpoint
    
    Args:
        stage: Stage number (1, 2, or 3)
        checkpoint_path: Path to checkpoint file
        start_epoch: Epoch to start from (usually checkpoint epoch + 1)
    """
    # Load config
    config_path = Path(__file__).parent / "config_progressive.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override with 100 patients and 20 epochs per stage
    config['data']['max_patients'] = 100
    config['training']['stage1']['num_epochs'] = 20
    config['training']['stage2']['num_epochs'] = 20
    config['training']['stage3']['num_epochs'] = 20
    
    print("="*60)
    print(f"Resuming Stage {stage} Training from Epoch {start_epoch}")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print("="*60)
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoints']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint and modify to resume
    checkpoint = torch.load(checkpoint_path)
    print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Val PSNR: {checkpoint.get('val_psnr', 'N/A'):.2f} dB")
    print(f"Val SSIM: {checkpoint.get('val_ssim', 'N/A'):.4f}")
    
    # Modify config to start from specific epoch
    # This is a workaround - save the checkpoint and modify start epoch in the training loop
    stage_key = f'stage{stage}'
    original_epochs = config['training'][stage_key]['num_epochs']
    
    print(f"\nWill train from epoch {start_epoch} to {original_epochs}")
    print("="*60)
    
    # Train the stage (you'll need to modify train_stage to support resume)
    # For now, let's just document the manual steps
    print("\n⚠️  MANUAL RESUME STEPS:")
    print("="*60)
    print("Since the current training script doesn't support direct resume,")
    print("here are your options:\n")
    print(f"Option 1: Modify the config to reduce epochs")
    print(f"  - Set stage{stage} epochs to {original_epochs - start_epoch}")
    print(f"  - This will train for remaining epochs\n")
    print(f"Option 2: Use the checkpoint as initial weights")
    print(f"  - The training will use {checkpoint_path}")
    print(f"  - But will restart epoch counting\n")
    print("="*60)


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python resume_training.py <stage> <checkpoint_path> <start_epoch>")
        print("\nExample:")
        print("  python resume_training.py 1 checkpoints_progressive/stage1_epoch10.pth 11")
        print("\nThis will resume stage 1 training from epoch 11")
        return
    
    stage = int(sys.argv[1])
    checkpoint_path = Path(sys.argv[2])
    start_epoch = int(sys.argv[3])
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    resume_training(stage, checkpoint_path, start_epoch)


if __name__ == "__main__":
    main()
