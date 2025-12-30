"""
Test forward pass with minimal memory usage
"""
import torch
import gc

# Clear any existing GPU memory
torch.cuda.empty_cache()
gc.collect()

print("Loading model...")
from model_optimized import OptimizedCTRegression

# Create model
model = OptimizedCTRegression(
    volume_size=(64, 64, 64),
    xray_feature_dim=512,
    voxel_dim=256,
    use_learnable_priors=True
).cuda()

print(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Test with batch_size=1 and no gradients
with torch.no_grad():
    xrays = torch.randn(1, 2, 1, 512, 512).cuda()
    print(f"Input shape: {xrays.shape}")
    
    output, aux = model(xrays)
    print(f"✅ Success! Output shape: {output.shape}")
    
    # Check memory usage
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

print("\nForward pass successful! Ready to train.")
