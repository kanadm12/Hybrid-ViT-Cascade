"""
H200-Optimized Training Script (151GB VRAM)
Uses larger batches, deeper networks, and 32³ tokens for maximum quality

Key Differences from A100 Training:
- Stage 2 batch_size: 4 (vs 1-2)
- Stage 3 batch_size: 2 (vs 1)
- Tokens: 32³ = 32,768 (vs 16³ = 4,096)
- ViT depth: 8-12 layers (vs 4-6)

Usage:
    python train_progressive_h200.py
"""

# Replace standard model imports with H200-optimized versions
import sys
from pathlib import Path

# Temporarily redirect imports to use H200 model
original_model_path = Path(__file__).parent / "model_progressive.py"
h200_model_path = Path(__file__).parent / "model_progressive_h200.py"

# Import base training script but override model classes
from train_progressive_1gpu import *

# Override with H200 model components
sys.path.insert(0, str(Path(__file__).parent))
from model_progressive_h200 import Stage2Refiner128_H200, Stage3Refiner256_H200

# Monkey-patch the model to use H200 versions
original_progressive_model_init = ProgressiveCascadeModel.__init__

def h200_progressive_model_init(self, xray_img_size=512, xray_feature_dim=512, 
                                voxel_dim=256, use_gradient_checkpointing=True):
    """Modified init to use H200-optimized Stage 2 and Stage 3"""
    nn.Module.__init__(self)
    
    self.xray_img_size = xray_img_size
    self.xray_feature_dim = xray_feature_dim
    self.voxel_dim = voxel_dim
    
    # X-ray encoder (shared)
    self.xray_encoder = XrayEncoder(
        img_size=xray_img_size,
        feature_dim=xray_feature_dim
    )
    
    # Stage 1: Same as A100 (memory efficient already)
    self.stage1 = Stage1Base64(
        volume_size=(64, 64, 64),
        xray_img_size=xray_img_size,
        voxel_dim=voxel_dim,
        vit_depth=4,
        num_heads=4,
        xray_feature_dim=xray_feature_dim
    )
    
    # Stage 2: H200-OPTIMIZED (32³ tokens, deeper ViT, more capacity)
    self.stage2 = Stage2Refiner128_H200(
        volume_size=(128, 128, 128),
        voxel_dim=512,  # Increased from 256
        vit_depth=8,    # Increased from 6
        num_heads=16,   # Increased from 8
        xray_feature_dim=xray_feature_dim
    )
    
    # Stage 3: H200-OPTIMIZED (32³ tokens, much deeper ViT)
    self.stage3 = Stage3Refiner256_H200(
        volume_size=(256, 256, 256),
        voxel_dim=512,  # Increased from 256
        vit_depth=12,   # Increased from 8
        num_heads=16,   # Increased from 8
        xray_feature_dim=xray_feature_dim,
        use_gradient_checkpointing=use_gradient_checkpointing
    )

# Apply monkey-patch
ProgressiveCascadeModel.__init__ = h200_progressive_model_init


def main():
    """Override config with H200-optimized batch sizes"""
    # Load base config
    config_path = Path(__file__).parent / "config_progressive.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # H200 OPTIMIZATIONS
    print("\n" + "="*60)
    print("H200-OPTIMIZED TRAINING (151GB VRAM)")
    print("="*60)
    print("Stage 2: batch_size=4, vit_depth=8, num_heads=16, tokens=32³ (32,768)")
    print("Stage 3: batch_size=2, vit_depth=12, num_heads=16, tokens=32³ (32,768)")
    print("Expected Memory: Stage 2 ~60GB, Stage 3 ~100GB")
    print("Expected Quality: +2-3 dB PSNR vs A100 config")
    print("="*60 + "\n")
    
    # Override batch sizes for H200
    config['training']['stage2']['batch_size'] = 4  # A100: 1-2
    config['training']['stage3']['batch_size'] = 2  # A100: 1
    
    # Start training
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config_path}")
    print(f"Dataset: {config['data']['dataset_path']}")
    print(f"Max Patients: {config['data']['max_patients']}")
    print("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 140:  # Less than 140GB
            print("\n⚠️  WARNING: This script is optimized for H200 (151GB VRAM)")
            print("Your GPU has less memory. Consider using train_progressive_1gpu.py instead.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoints']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train each stage sequentially
    total_start = time.time()
    
    # Start from stage 1 or 2 (modify as needed)
    start_stage = 2  # Set to 1 to train all stages
    
    for stage in range(start_stage, 4):
        print(f"\n{'#'*60}")
        print(f"# STAGE {stage} TRAINING (H200-OPTIMIZED)")
        print(f"{'#'*60}\n")
        
        stage_start = time.time()
        train_stage(config, stage, checkpoint_dir)
        stage_time = time.time() - stage_start
        
        print(f"Stage {stage} completed in {stage_time/3600:.2f} hours\n")
    
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print("ALL STAGES TRAINING COMPLETE!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\nCheckpoints saved in:", checkpoint_dir)
    print("="*60)


if __name__ == "__main__":
    main()
