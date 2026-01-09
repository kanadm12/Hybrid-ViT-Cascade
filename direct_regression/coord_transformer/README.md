# Coordinate-Based Transformer for CT Reconstruction 🎯

**Revolutionary set-based architecture** that processes 3D volumes as coordinate sets with transformer attention.

## 🌟 Architecture Overview

This approach treats the 3D CT volume as a **set of 3D coordinates** rather than a rigid grid, enabling:

1. **Flexible querying** - Query any arbitrary points in 3D space
2. **Multi-resolution inference** - Train once, infer at any resolution
3. **Attention mechanisms** - Explicit cross-attention and self-attention
4. **Interpretability** - Can visualize attention maps

### Pipeline

```
X-ray Images (512×512)
    ↓
Vision Transformer Encoder (patch-based)
    → 256 patch features (each 512-D)
    ↓
3D Coordinates (x,y,z) in [-1, 1]
    ↓
Fourier Feature Embedding (learnable frequencies)
    → High-dimensional coordinate features
    ↓
Transformer Blocks (6 layers):
  1. Self-Attention: coordinates ↔ coordinates
  2. Cross-Attention: coordinates ← X-ray patches
  3. Feed-Forward Network
    ↓
Density Prediction Head
    → CT Volume (any resolution!)
```

## 🎯 Key Components

### 1. Fourier Feature Embedding
Maps 3D coordinates through learnable frequencies:
```python
(x, y, z) → [sin(2πB₁·x), cos(2πB₁·x), ..., sin(2πBₙ·x), cos(2πBₙ·x)]
```
where B is a learnable frequency matrix.

### 2. Vision Transformer X-ray Encoder
- Splits X-rays into 32×32 patches
- 6-layer transformer with positional encoding
- Produces 256 patch embeddings per view

### 3. Cross-Attention Mechanism
```
Query: Coordinate features (what we want to know)
Key/Value: X-ray patch features (what we observe)
→ Each coordinate attends to relevant X-ray regions
```

### 4. Self-Attention Among Coordinates
- Coordinates exchange information
- Captures spatial coherence
- Enables smooth anatomical structures

## 🚀 Quick Start

### Training (4 GPU)

```bash
# Use the launcher
cd ..
launch_coord_transformer_4gpu.bat

# Or direct command
python coord_transformer/train_coord_transformer_4gpu.py \
    --config coord_transformer/config_coord_transformer.json
```

**Training Configuration:**
- **Patients**: 100 (80 train / 10 val / 10 test)
- **Epochs**: 100
- **Batch size**: 2 per GPU (8 total)
- **Learning rate**: 3e-5 with 10-epoch warmup
- **Transformer**: 6 blocks, 8 heads, 512-D
- **X-ray encoder**: 6-layer ViT with 32px patches

### Inference

#### Standard Resolution (64³)
```bash
python coord_transformer/inference_coord_transformer.py \
    --checkpoint checkpoints_coord_transformer/coord_transformer_best.pt \
    --num_samples 10
```

#### Multi-Resolution (96³)
```bash
python coord_transformer/inference_coord_transformer.py \
    --checkpoint checkpoints_coord_transformer/coord_transformer_best.pt \
    --output_resolution 96,96,96 \
    --num_samples 10
```

#### High Resolution (128³)
```bash
python coord_transformer/inference_coord_transformer.py \
    --checkpoint checkpoints_coord_transformer/coord_transformer_best.pt \
    --output_resolution 128,128,128 \
    --num_samples 5
```

## 💡 Advantages Over Other Methods

| Feature | CNNs | NeRF | **Coord Transformer** |
|---------|------|------|----------------------|
| Attention mechanism | ❌ | ❌ | ✅ Cross + Self |
| Interpretable | ❌ | ❌ | ✅ Attention maps |
| Flexible queries | ❌ | ✅ | ✅ Set-based |
| Multi-resolution | ❌ | ✅ | ✅ |
| X-ray integration | Global | Latent | ✅ Per-patch |
| Spatial coherence | Convolution | MLP | ✅ Self-attention |

## 🔬 Technical Details

### Memory Management
Processes coordinates in chunks (4096 at a time) to handle large volumes efficiently.

### Loss Function
- **L1 reconstruction**: Pixel-wise accuracy
- **Gradient matching**: Preserves edges and boundaries
- **Smoothness regularization**: Prevents noise

### Optimizer
- AdamW with weight decay 0.05
- Cosine learning rate schedule
- 10-epoch linear warmup

## 📊 Expected Performance

Based on transformer architectures in medical imaging:

- **PSNR**: 29-33 dB
- **Edge preservation**: Excellent (explicit gradient loss)
- **Detail capture**: Superior to pure CNNs
- **Attention visualization**: Can identify important X-ray regions

## 🎓 Unique Capabilities

### 1. Arbitrary Point Querying
```python
# Query 100 random points
random_coords = torch.randn(batch_size, 100, 3)
densities = model.query_points(xrays, random_coords)
```

### 2. Sparse Reconstruction
Reconstruct only regions of interest (e.g., heart, spine).

### 3. Attention Visualization
Analyze which X-ray patches each 3D coordinate attends to.

### 4. Progressive Refinement
Start with coarse grid, progressively query finer details.

## 📁 Files in This Folder

- **[model_coord_transformer.py](model_coord_transformer.py)** - Complete model architecture
- **[config_coord_transformer.json](config_coord_transformer.json)** - Training configuration
- **[train_coord_transformer_4gpu.py](train_coord_transformer_4gpu.py)** - 4-GPU distributed training
- **[inference_coord_transformer.py](inference_coord_transformer.py)** - Multi-resolution inference

## 🔧 Customization

### Increase Model Capacity
Edit `config_coord_transformer.json`:
```json
{
  "model": {
    "coord_embed_dim": 768,        // Larger embeddings
    "num_transformer_blocks": 8,   // More layers
    "num_heads": 12                // More attention heads
  }
}
```

### Adjust Loss Weights
```json
{
  "training": {
    "loss_weights": {
      "l1_weight": 1.0,
      "gradient_weight": 0.5,     // Higher for sharper edges
      "smoothness_weight": 0.02   // Lower for more detail
    }
  }
}
```

## 🆚 Comparison with NeRF CT

Both are revolutionary, but different:

**NeRF CT**:
- Simpler architecture
- Implicit MLP decoder
- More memory-efficient
- Good for smooth representations

**Coordinate Transformer** (this):
- More sophisticated
- Explicit attention mechanisms
- Better for complex relationships
- Interpretable attention maps

**Recommendation**: Train both and compare!

## 🎯 Research Foundations

This architecture combines ideas from:
- **Perceiver** (Jaegle et al., 2021) - Cross-attention to inputs
- **Set Transformers** (Lee et al., 2019) - Set-based processing
- **Vision Transformer** (Dosovitskiy et al., 2020) - Patch-based encoding
- **Fourier Features** (Tancik et al., 2020) - Coordinate embeddings

## 🐛 Troubleshooting

**Out of memory?**
- Reduce batch size in config
- Reduce `num_transformer_blocks`
- The model processes in 4096-coordinate chunks automatically

**Training unstable?**
- Increase warmup epochs
- Reduce learning rate
- Check gradient clipping is enabled

**Poor detail capture?**
- Increase `gradient_weight` in loss
- Reduce `smoothness_weight`
- Increase model capacity

## 📈 Monitoring Training

Watch these metrics:
- **Train Loss**: Should steadily decrease
- **Gradient Loss**: Indicates edge preservation quality
- **Val PSNR**: Higher is better (>30 dB is excellent)
- **Val SSIM**: Structural similarity (>0.9 is good)

Best models are automatically saved based on validation loss.

## 🎨 Visualization

Inference generates:
1. **NIfTI files** - 3D volumes for medical viewers
2. **Slice images** - Axial, coronal, sagittal views
3. **Metrics JSON** - Quantitative results

## 🚀 Next Steps

1. **Train the model** - Run for 100 epochs
2. **Test multi-resolution** - Evaluate at 64³, 96³, 128³
3. **Analyze attention** - Visualize what X-ray regions are important
4. **Compare with baselines** - See improvement over direct regression
5. **Fine-tune** - Adjust hyperparameters based on results

---

**This is cutting-edge research!** 🎓 The coordinate-based transformer represents the frontier of CT reconstruction methods.
