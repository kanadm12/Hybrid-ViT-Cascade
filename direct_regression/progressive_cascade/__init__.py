"""
Progressive Multi-Scale CT Reconstruction Package

A progressive 64³→128³→256³ cascade system for X-ray to CT reconstruction
with frequency-aware training and geometric consistency.
"""

__version__ = "1.0.0"
__author__ = "CT Reconstruction Team"

from .model_progressive import (
    ProgressiveCascadeModel,
    Stage1Base64,
    Stage2Refiner128,
    Stage3Refiner256,
    MultiScaleXrayEncoder
)

from .loss_multiscale import (
    MultiScaleLoss,
    Stage1Loss,
    Stage2Loss,
    Stage3Loss,
    compute_psnr,
    compute_ssim_metric
)

__all__ = [
    'ProgressiveCascadeModel',
    'Stage1Base64',
    'Stage2Refiner128',
    'Stage3Refiner256',
    'MultiScaleXrayEncoder',
    'MultiScaleLoss',
    'Stage1Loss',
    'Stage2Loss',
    'Stage3Loss',
    'compute_psnr',
    'compute_ssim_metric'
]
