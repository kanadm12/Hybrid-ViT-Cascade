"""
Inspect checkpoint keys to debug loading issues
"""
import torch

checkpoint_path = "checkpoints_progressive/stage1_best.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  {key}")

print("\n\nModel state dict keys (first 50):")
state_dict = checkpoint['model_state_dict']
for i, key in enumerate(sorted(state_dict.keys())):
    if i < 50:
        print(f"  {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'}")
    
print(f"\n\nTotal keys in model_state_dict: {len(state_dict)}")

# Check for stage2 keys
stage2_keys = [k for k in state_dict.keys() if 'stage2' in k]
print(f"\n\nStage 2 keys ({len(stage2_keys)}):")
for key in stage2_keys[:20]:
    print(f"  {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'}")

# Check for pos_embed in stage2
pos_embed_keys = [k for k in state_dict.keys() if 'pos_embed' in k and 'stage2' in k]
print(f"\n\nStage 2 pos_embed keys:")
for key in pos_embed_keys:
    print(f"  {key}: {state_dict[key].shape}")
