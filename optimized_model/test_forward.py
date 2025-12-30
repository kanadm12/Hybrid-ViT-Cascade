import torch
from model_optimized import OptimizedCTRegression

print("Creating model...")
model = OptimizedCTRegression(
    volume_size=(64, 64, 64),
    xray_feature_dim=512,
    voxel_dim=256,
    use_learnable_priors=True
)

print("Creating input...")
xrays = torch.randn(2, 2, 1, 512, 512)

print("Running forward pass...")
try:
    output, aux = model(xrays)
    print(f"Success! Output shape: {output.shape}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
