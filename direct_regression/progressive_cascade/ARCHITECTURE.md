# Progressive Multi-Scale CT Reconstruction - Architecture Diagram

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PROGRESSIVE CASCADE SYSTEM                         │
│                  64³ → 128³ → 256³ Multi-Scale                       │
└──────────────────────────────────────────────────────────────────────┘

INPUT: 2 X-rays (AP + Lateral) @ 512×512
  │
  └─► Multi-Scale X-ray Encoder (Shared)
        │
        ├─► Full Resolution:    512×512 features ──┐
        ├─► Mid Resolution:     256×256 features ──┤
        └─► Low Resolution:     128×128 features ──┤
                                                     │
        ┌────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Base Reconstruction (64³)                                  │
│ ┌─────────────────────────────────────────────────────────────┐    │
│ │ • ViT Backbone: 4 layers, 4 heads                           │    │
│ │ • Cross-Attention: 128×128 X-ray features                   │    │
│ │ • Loss: L1 + SSIM                                           │    │
│ │ • Training: 50 epochs, LR=1e-4, batch_size=8               │    │
│ │ • Output: 64×64×64 volume                                   │    │
│ └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
        │
        ├─► Volume 64³ (PSNR: 28-30 dB, SSIM: 0.85-0.90)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Texture Refinement (128³)                                 │
│ ┌─────────────────────────────────────────────────────────────┐    │
│ │ • Upsample: 64³ → 128³ (trilinear)                          │    │
│ │ • ViT Backbone: 6 layers, 8 heads                           │    │
│ │ • Cross-Attention: 256×256 X-ray features                   │    │
│ │ • Residual Connection: with upsampled 64³                   │    │
│ │ • Loss: L1 + SSIM + VGG Perceptual                         │    │
│ │ • Training: 30 epochs, LR=5e-5, batch_size=4               │    │
│ │ • Output: 128×128×128 volume                                │    │
│ └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
        │
        ├─► Volume 128³ (PSNR: 32-35 dB, SSIM: 0.92-0.95)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Detail Enhancement (256³)                                  │
│ ┌─────────────────────────────────────────────────────────────┐    │
│ │ • Upsample: 128³ → 256³ (trilinear)                         │    │
│ │ • ViT Backbone: 8 layers, 8 heads                           │    │
│ │ • Cross-Attention: 512×512 X-ray features (full res)        │    │
│ │ • Detail Enhancer: High-frequency CNN branch                │    │
│ │ • Residual Connections: base + refinement + details         │    │
│ │ • Gradient Checkpointing: Enabled for memory                │    │
│ │ • Loss: L1 + SSIM + VGG + Gradient + DRR                   │    │
│ │ • Training: 20 epochs, LR=2e-5, batch_size=2               │    │
│ │ • Output: 256×256×256 volume                                │    │
│ └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
        │
        └─► Final Volume 256³ (PSNR: 35-38 dB, SSIM: 0.95-0.97)
```

## Loss Function Progression

```
┌──────────────────────────────────────────────────────────────────┐
│                    FREQUENCY-AWARE LOSSES                         │
└──────────────────────────────────────────────────────────────────┘

STAGE 1 (64³) - Focus: Coarse Structure
┌────────────────────────────────────┐
│ L1 Loss           (weight: 1.0)    │  Intensity matching
│ SSIM Loss         (weight: 0.5)    │  Structural similarity
│                                     │
│ Total = 1.0×L1 + 0.5×SSIM          │
└────────────────────────────────────┘
         │
         ▼
STAGE 2 (128³) - Focus: Add Texture
┌────────────────────────────────────┐
│ L1 Loss           (weight: 1.0)    │  Base intensity
│ SSIM Loss         (weight: 0.5)    │  Structure
│ VGG Perceptual    (weight: 0.1)    │  ← NEW: Texture
│                                     │
│ Total = 1.0×L1 + 0.5×SSIM          │
│         + 0.1×VGG                   │
└────────────────────────────────────┘
         │
         ▼
STAGE 3 (256³) - Focus: Fine Details & Consistency
┌────────────────────────────────────┐
│ L1 Loss           (weight: 1.0)    │  Base intensity
│ SSIM Loss         (weight: 0.5)    │  Structure
│ VGG Perceptual    (weight: 0.1)    │  Texture
│ Gradient Mag      (weight: 0.2)    │  ← NEW: Edges
│ DRR Reprojection  (weight: 0.3)    │  ← NEW: X-ray consistency
│                                     │
│ Total = 1.0×L1 + 0.5×SSIM          │
│         + 0.1×VGG + 0.2×Grad        │
│         + 0.3×DRR                   │
└────────────────────────────────────┘
```

## DRR Reprojection Loss (Stage 3)

```
Predicted CT Volume (256³)
        │
        ├─► DRR Generator (AP view)     ─┐
        │        │                        │
        │        └─► Projection: AP      │
        │                                 │
        └─► DRR Generator (Lateral view) ┤
                 │                        │  L1 Loss
                 └─► Projection: Lateral │
                                          │
Input X-rays                              │
        ├─► AP X-ray        ─────────────┤
        └─► Lateral X-ray   ─────────────┘

DRR Loss = L1(DRR_AP, XRay_AP) + L1(DRR_Lateral, XRay_Lateral)
```

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│               PROGRESSIVE TRAINING SCHEDULE                  │
└─────────────────────────────────────────────────────────────┘

PHASE 1: Train Stage 1
├─ Initialize: Random weights
├─ Train: 50 epochs @ LR=1e-4, batch_size=8
├─ Loss: L1 + SSIM
├─ Save: stage1_best.pth
└─ Time: ~6 hours (4×A100)

PHASE 2: Train Stage 2
├─ Load: stage1_best.pth
├─ Freeze: Stage 1 weights ❄️
├─ Train: Stage 2 only, 30 epochs @ LR=5e-5, batch_size=4
├─ Loss: L1 + SSIM + VGG
├─ Save: stage2_best.pth
└─ Time: ~12 hours

PHASE 3: Train Stage 3
├─ Load: stage2_best.pth
├─ Freeze: Stage 1 & 2 weights ❄️❄️
├─ Train: Stage 3 only, 20 epochs @ LR=2e-5, batch_size=2
├─ Gradient Checkpointing: Enabled ✓
├─ Loss: L1 + SSIM + VGG + Gradient + DRR
├─ Save: stage3_best.pth
└─ Time: ~24 hours

TOTAL TIME: ~42 hours
```

## Memory Usage Profile

```
┌──────────────────────────────────────────────────────────────┐
│                   MEMORY REQUIREMENTS                         │
└──────────────────────────────────────────────────────────────┘

Stage 1 (64³)
├─ Batch Size: 8
├─ Volume Memory: 8 × 64³ × 4 bytes ≈ 8 MB
├─ Features: ~5 GB
├─ Gradients: ~5 GB
├─ Optimizer: ~5 GB
└─ Total: ~10 GB per GPU ✓ Fits easily

Stage 2 (128³)
├─ Batch Size: 4
├─ Volume Memory: 4 × 128³ × 4 bytes ≈ 32 MB
├─ Features: ~8 GB
├─ Gradients: ~8 GB
├─ Optimizer: ~8 GB
└─ Total: ~16 GB per GPU ✓ Fits comfortably

Stage 3 (256³)
├─ Batch Size: 2
├─ Volume Memory: 2 × 256³ × 4 bytes ≈ 128 MB
├─ Features: ~15 GB
├─ Gradients: ~15 GB (with checkpointing)
├─ Optimizer: ~15 GB
└─ Total: ~35 GB per GPU ✓ Requires gradient checkpointing
```

## Multi-Scale Cross-Attention Matching

```
┌────────────────────────────────────────────────────────────┐
│           X-RAY FEATURES ↔ CT RESOLUTION MATCHING          │
└────────────────────────────────────────────────────────────┘

Stage 1: 64³ CT Volume
   │
   └─► Cross-Attention with 128×128 X-ray features
       (4× downsampled from 512×512)
       
       CT:    64 × 64 × 64
       X-ray: 128 × 128 → flatten to 16,384 tokens
       Ratio: ~4 X-ray pixels per CT voxel

Stage 2: 128³ CT Volume
   │
   └─► Cross-Attention with 256×256 X-ray features
       (2× downsampled from 512×512)
       
       CT:    128 × 128 × 128
       X-ray: 256 × 256 → flatten to 65,536 tokens
       Ratio: ~4 X-ray pixels per CT voxel

Stage 3: 256³ CT Volume
   │
   └─► Cross-Attention with 512×512 X-ray features
       (Full resolution, no downsampling)
       
       CT:    256 × 256 × 256
       X-ray: 512 × 512 → flatten to 262,144 tokens
       Ratio: ~4 X-ray pixels per CT voxel

Key Insight: Consistent information density across stages!
```

## Performance Progression

```
┌────────────────────────────────────────────────────────────┐
│              EXPECTED QUALITY IMPROVEMENT                   │
└────────────────────────────────────────────────────────────┘

Metric: PSNR (Peak Signal-to-Noise Ratio)
40 dB │
      │                                        ╔════╗
      │                                        ║ S3 ║ 35-38 dB
35 dB │                        ╔════╗          ╚════╝
      │                        ║ S2 ║ 32-35 dB
30 dB │        ╔════╗          ╚════╝
      │        ║ S1 ║ 28-30 dB
25 dB │        ╚════╝
      │
      └────────────────────────────────────────────────►
            Stage 1    Stage 2    Stage 3    Resolution
             64³        128³       256³

Metric: SSIM (Structural Similarity Index)
1.0   │                                        ╔════╗
      │                                        ║ S3 ║ 0.95-0.97
0.9   │                        ╔════╗          ╚════╝
      │        ╔════╗          ║ S2 ║ 0.92-0.95
0.8   │        ║ S1 ║          ╚════╝
      │        ╚════╝ 0.85-0.90
      │
      └────────────────────────────────────────────────►
            Stage 1    Stage 2    Stage 3
```

## File Organization

```
progressive_cascade/
│
├── Core Model Components
│   ├── model_progressive.py
│   │   ├── MultiScaleXrayEncoder     (X-ray feature extraction)
│   │   ├── Stage1Base64              (64³ base)
│   │   ├── Stage2Refiner128          (128³ refinement)
│   │   ├── Stage3Refiner256          (256³ refinement)
│   │   └── ProgressiveCascadeModel   (full cascade)
│   │
│   └── loss_multiscale.py
│       ├── SSIMLoss                  (3D structure)
│       ├── TriPlanarVGGLoss          (2D texture)
│       ├── GradientMagnitudeLoss     (edges)
│       ├── DRRReprojectionLoss       (consistency)
│       ├── Stage1Loss / Stage2Loss / Stage3Loss
│       └── MultiScaleLoss            (unified)
│
├── Training & Inference
│   ├── train_progressive_4gpu.py     (4-GPU DDP training)
│   └── inference_progressive.py      (inference & evaluation)
│
├── Configuration
│   ├── config_progressive.json       (all hyperparameters)
│   └── __init__.py                   (package exports)
│
├── Documentation
│   ├── README.md                     (full docs)
│   ├── QUICKSTART.md                 (5-min guide)
│   ├── IMPLEMENTATION_SUMMARY.md     (technical)
│   └── ARCHITECTURE.md               (this file)
│
└── Tools
    ├── utils.py                      (helpers)
    ├── test_implementation.py        (validation)
    └── launch_progressive.bat        (Windows launcher)
```

## Key Design Principles

1. **Progressive Refinement**: Each stage builds on previous stage output
2. **Frequency Awareness**: Low → Mid → High frequency losses
3. **Geometric Consistency**: DRR reprojection ensures X-ray alignment
4. **Memory Efficiency**: Gradient checkpointing + stage-wise training
5. **Proven Foundation**: Leverages existing validated components
6. **Multi-Scale Matching**: X-ray features scaled to CT resolution
7. **Residual Learning**: Refine rather than reconstruct from scratch

---

**Ready to achieve high-quality CT reconstruction! 🚀**
