"""
Hybrid-ViT Cascade Package

Combines the best of both approaches:
- Physics constraints (DRR loss) from Hybrid approach
- Progressive cascade + AdaLN from ViT approach
- Multi-resolution depth priors for anatomical guidance
"""

from .models.cascaded_depth_lifting import (
    ResolutionDepthPriors,
    CascadedDepthWeightNetwork,
    CascadedDepthLifting
)

from .models.hybrid_vit_backbone import (
    HybridViTBlock3D,
    HybridViT3D
)

from .models.unified_model import (
    UnifiedCascadeStage,
    UnifiedHybridViTCascade
)

__version__ = "0.1.0"

__all__ = [
    # Depth lifting
    'ResolutionDepthPriors',
    'CascadedDepthWeightNetwork',
    'CascadedDepthLifting',
    
    # ViT backbone
    'HybridViTBlock3D',
    'HybridViT3D',
    
    # Unified model
    'UnifiedCascadeStage',
    'UnifiedHybridViTCascade',
]
