"""
Test Enhanced Model - Verify architecture works before training
"""
import torch
import sys
sys.path.insert(0, '..')

from model_enhanced import EnhancedDirectModel, EnhancedLoss

def test_enhanced_model():
    print("="*60)
    print("Testing Enhanced Direct Model")
    print("="*60)
    
    # Model config
    volume_size = (64, 64, 64)
    base_channels = 64
    
    # Create model
    print("\n1. Creating model...")
    model = EnhancedDirectModel(
        volume_size=volume_size,
        base_channels=base_channels
    ).cuda()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Create dummy input
    print("\n2. Creating dummy input...")
    batch_size = 2
    xrays = torch.randn(batch_size, 2, 512, 512).cuda()
    print(f"   Input shape: {xrays.shape}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    model.eval()
    with torch.no_grad():
        predicted, aux_outputs = model(xrays)
    
    print(f"   Output shape: {predicted.shape}")
    print(f"   Auxiliary outputs: {len(aux_outputs)}")
    for i, aux in enumerate(aux_outputs):
        print(f"     Scale {i}: {aux.shape}")
    
    # Test loss
    print("\n4. Testing loss functions...")
    target = torch.randn_like(predicted)
    
    criterion = EnhancedLoss(
        l1_weight=1.0,
        ssim_weight=0.5,
        perceptual_weight=0.1,
        edge_weight=0.1,
        multiscale_weight=0.3
    )
    
    loss_dict = criterion(predicted, target, aux_outputs)
    
    print(f"   Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"   L1 loss: {loss_dict['l1_loss'].item():.4f}")
    print(f"   SSIM loss: {loss_dict['ssim_loss'].item():.4f}")
    print(f"   Perceptual loss: {loss_dict['perceptual_loss'].item():.4f}")
    print(f"   Edge loss: {loss_dict['edge_loss'].item():.4f}")
    print(f"   Multiscale loss: {loss_dict['multiscale_loss'].item():.4f}")
    
    # Test memory usage
    print("\n5. Estimating memory usage...")
    torch.cuda.synchronize()
    memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
    print(f"   Peak memory: {memory_used:.2f} GB")
    print(f"   Per GPU (4 GPUs): ~{memory_used:.2f} GB")
    print(f"   Batch size 4: ~{memory_used * 2:.2f} GB per GPU (safe for A100 80GB)")
    
    # Test backward
    print("\n6. Testing backward pass...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    xrays = torch.randn(batch_size, 2, 512, 512).cuda()
    target = torch.randn(batch_size, 64, 64, 64).cuda()
    
    predicted, aux_outputs = model(xrays)
    loss_dict = criterion(predicted, target, aux_outputs)
    loss = loss_dict['total_loss']
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("   ✓ Backward pass successful!")
    
    # Test mixed precision
    print("\n7. Testing mixed precision training...")
    scaler = torch.amp.GradScaler('cuda')
    
    with torch.amp.autocast('cuda'):
        predicted, aux_outputs = model(xrays)
        loss_dict = criterion(predicted, target, aux_outputs)
        loss = loss_dict['total_loss']
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print("   ✓ Mixed precision successful!")
    
    print("\n" + "="*60)
    print("✓ All tests passed! Model is ready for training.")
    print("="*60)


if __name__ == '__main__':
    test_enhanced_model()
