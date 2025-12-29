"""
Deep diagnostic of ViT backbone to identify why it's not learning
"""
import torch
import sys
import json
sys.path.insert(0, '/workspace/Hybrid-ViT-Cascade')

from models.unified_model import UnifiedHybridViTCascade
from utils.dataset import PatientDRRDataset

print("=" * 80)
print("DEEP ViT BACKBONE DIAGNOSIS")
print("=" * 80)

# Load config and model
with open('config/runpod_config.json') as f:
    config = json.load(f)

device = torch.device('cuda:0')
model = UnifiedHybridViTCascade(
    stage_configs=config['stage_configs'],
    xray_img_size=config['xray_config']['img_size'],
    xray_channels=1,
    num_views=config['xray_config']['num_views'],
    share_view_weights=config['xray_config'].get('share_view_weights', False),
    v_parameterization=config['training'].get('v_parameterization', True),
    num_timesteps=config['training'].get('num_timesteps', 1000),
    extract_features=False
).to(device)

# Load checkpoint
ckpt = torch.load('checkpoints/stage1_best.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
print(f"\nLoaded checkpoint: Epoch {ckpt.get('epoch', '?')}, PSNR {ckpt.get('psnr', '?'):.2f} dB\n")

# Load data
dataset = PatientDRRDataset(
    data_path='/workspace/drr_patient_data',
    target_volume_size=(64, 64, 64),
    max_patients=1,
    validate_alignment=False
)
sample = dataset[0]
volume = sample['ct_volume'].unsqueeze(0).to(device)
xrays = sample['drr_stacked'].unsqueeze(0).to(device)

print("=" * 80)
print("1. X-RAY ENCODER ANALYSIS")
print("=" * 80)

model.eval()
with torch.no_grad():
    t = torch.tensor([500], device=device).long()
    t_normalized = t.float() / 1000
    t_embed = model.time_embed(t_normalized.unsqueeze(-1))
    
    xray_context, time_xray_cond, xray_features_2d = model.xray_encoder(xrays, t_embed)
    
    print(f"X-ray input: shape={xrays.shape}, range=[{xrays.min():.3f}, {xrays.max():.3f}]")
    print(f"X-ray features 2D: shape={xray_features_2d.shape}, range=[{xray_features_2d.min():.3f}, {xray_features_2d.max():.3f}]")
    print(f"X-ray features std: {xray_features_2d.std():.3f}")
    print(f"X-ray context: shape={xray_context.shape}, range=[{xray_context.min():.3f}, {xray_context.max():.3f}]")
    print(f"Time+XRay cond: shape={time_xray_cond.shape}, range=[{time_xray_cond.min():.3f}, {time_xray_cond.max():.3f}]")
    
    # Check if features are meaningful
    if xray_features_2d.std() < 0.1:
        print("⚠️  WARNING: X-ray features have very low variance - encoder may have collapsed!")

print("\n" + "=" * 80)
print("2. VIT BACKBONE INPUT/OUTPUT ANALYSIS")
print("=" * 80)

with torch.no_grad():
    noise = torch.randn_like(volume)
    sqrt_alphas_t = model.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alphas_t = model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    x_noisy = sqrt_alphas_t * volume + sqrt_one_minus_alphas_t * noise
    
    print(f"Noisy input: shape={x_noisy.shape}, range=[{x_noisy.min():.3f}, {x_noisy.max():.3f}], std={x_noisy.std():.3f}")
    
    # Get stage
    stage = model.stages['stage1']
    
    # Hook to capture intermediate activations
    activations = {}
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook
    
    # Register hooks on ViT blocks
    for i, block in enumerate(stage.blocks):
        block.self_attn.register_forward_hook(get_activation(f'block{i}_self_attn'))
        block.cross_attn.register_forward_hook(get_activation(f'block{i}_cross_attn'))
        block.mlp.register_forward_hook(get_activation(f'block{i}_mlp'))
    
    # Forward pass
    predicted = stage(
        noisy_volume=x_noisy,
        xray_features=xray_features_2d,
        xray_context=xray_features_2d,
        time_xray_cond=time_xray_cond,
        prev_stage_volume=None,
        prev_stage_embed=None
    )
    
    print(f"Predicted output: shape={predicted.shape}, range=[{predicted.min():.3f}, {predicted.max():.3f}], std={predicted.std():.3f}")
    
    # Analyze each block
    print("\n" + "=" * 80)
    print("3. PER-BLOCK ACTIVATION ANALYSIS")
    print("=" * 80)
    
    num_blocks = len(stage.blocks)
    for i in range(num_blocks):
        print(f"\n--- Block {i} ---")
        
        if f'block{i}_self_attn' in activations:
            sa = activations[f'block{i}_self_attn']
            print(f"  Self-Attention: std={sa.std():.4f}, range=[{sa.min():.3f}, {sa.max():.3f}]")
            if sa.std() < 0.01:
                print(f"    ⚠️  WARNING: Self-attention output is nearly constant!")
        
        if f'block{i}_cross_attn' in activations:
            ca = activations[f'block{i}_cross_attn']
            print(f"  Cross-Attention: std={ca.std():.4f}, range=[{ca.min():.3f}, {ca.max():.3f}]")
            if ca.std() < 0.01:
                print(f"    ⚠️  WARNING: Cross-attention output is nearly constant!")
        
        if f'block{i}_mlp' in activations:
            mlp = activations[f'block{i}_mlp']
            print(f"  MLP: std={mlp.std():.4f}, range=[{mlp.min():.3f}, {mlp.max():.3f}]")
            if mlp.std() < 0.01:
                print(f"    ⚠️  WARNING: MLP output is nearly constant!")

print("\n" + "=" * 80)
print("4. GRADIENT FLOW ANALYSIS")
print("=" * 80)

model.train()
volume.requires_grad = True
xrays.requires_grad = True

loss_dict = model(volume, xrays, stage_name='stage1')
loss = loss_dict['total_loss']
loss.backward()

print(f"Loss: {loss.item():.6f}")

# Check gradients
grad_stats = []
for name, param in stage.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_stats.append((name, grad_norm))

# Sort by gradient norm
grad_stats.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 largest gradients:")
for name, grad_norm in grad_stats[:10]:
    print(f"  {name}: {grad_norm:.6f}")

print("\nBottom 10 smallest gradients:")
for name, grad_norm in grad_stats[-10:]:
    print(f"  {name}: {grad_norm:.6f}")
    if grad_norm < 1e-8:
        print(f"    ⚠️  WARNING: Gradient is essentially zero - dead neuron!")

print("\n" + "=" * 80)
print("5. WEIGHT STATISTICS")
print("=" * 80)

weight_stats = []
for name, param in stage.named_parameters():
    if 'weight' in name:
        weight_stats.append((name, param.std().item(), param.abs().max().item()))

print("\nWeight statistics (std, max_abs):")
for name, std, max_abs in weight_stats[:15]:
    print(f"  {name}: std={std:.4f}, max={max_abs:.4f}")
    if std < 0.01:
        print(f"    ⚠️  WARNING: Weights have very low variance!")

print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)
print("\nPotential issues found:")

issues = []
if xray_features_2d.std() < 0.1:
    issues.append("X-ray encoder producing low-variance features")
if predicted.std() < 0.2:
    issues.append("Model output has collapsed to low variance")

dead_grads = sum(1 for _, grad in grad_stats if grad < 1e-8)
if dead_grads > len(grad_stats) * 0.1:
    issues.append(f"{dead_grads} parameters have near-zero gradients (dead neurons)")

if not issues:
    print("No obvious issues detected - problem may be more subtle")
else:
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")

print("\nRecommended fixes:")
print("1. Reduce learning rate to 1e-5 or 5e-6")
print("2. Add gradient clipping (max_norm=1.0)")
print("3. Initialize ViT blocks with smaller weights (std=0.01)")
print("4. Add batch normalization before ViT blocks")
print("5. Increase warmup steps to 1000")
