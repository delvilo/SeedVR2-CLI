"""
Performance optimization module for SeedVR2
Contains optimized tensor operations and video processing functions

Extracted from: seedvr2.py (lines 1633-1730)
"""

import torch
from typing import List


def optimized_channels_to_last(tensor):
    """🚀 Optimized replacement for rearrange(tensor, 'b c ... -> b ... c')
    Moves channels from position 1 to last position using PyTorch native operations.
    """
    if tensor.ndim == 3:  # [batch, channels, spatial]
        return tensor.permute(0, 2, 1)
    elif tensor.ndim == 4:  # [batch, channels, height, width]
        return tensor.permute(0, 2, 3, 1)
    elif tensor.ndim == 5:  # [batch, channels, depth, height, width]
        return tensor.permute(0, 2, 3, 4, 1)
    else:
        # Fallback for other dimensions - move channel (dim=1) to last
        dims = list(range(tensor.ndim))
        dims = [dims[0]] + dims[2:] + [dims[1]]  # [0, 2, 3, ..., 1]
        return tensor.permute(*dims)


def optimized_channels_to_second(tensor):
    """🚀 Optimized replacement for rearrange(tensor, 'b ... c -> b c ...')
    Moves channels from last position to position 1 using PyTorch native operations.
    """
    if tensor.ndim == 3:  # [batch, spatial, channels]
        return tensor.permute(0, 2, 1)
    elif tensor.ndim == 4:  # [batch, height, width, channels]
        return tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 5:  # [batch, depth, height, width, channels]
        return tensor.permute(0, 4, 1, 2, 3)
    else:
        # Fallback for other dimensions - move last dim to position 1
        dims = list(range(tensor.ndim))
        dims = [dims[0], dims[-1]] + dims[1:-1]  # [0, -1, 1, 2, ..., -2]
        return tensor.permute(*dims)


def optimized_video_rearrange(video_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """Return CTHW -> TCHW (or CHW -> 1CHW) views without stacking samples."""
    # Each pipeline call contains one sample; permute/unsqueeze are views.
    # Stacking equally shaped samples allocated another full decoded batch.
    return [optimized_single_video_rearrange(video) for video in video_tensors]


def optimized_single_video_rearrange(video: torch.Tensor) -> torch.Tensor:
    """
    🚀 OPTIMIZED version for single video tensor
    Replaces rearrange() with native PyTorch operations
    
    Transforms:
    - 3D: c h w -> 1 c h w (add temporal dimension)
    - 4D: c t h w -> t c h w (permute dimensions)
    
    Expected gains: 2-5x faster than rearrange()
    
    Args:
        video: Input video tensor
        
    Returns:
        Rearranged tensor with temporal dimension first
    """
    if video.ndim == 3:
        # c h w -> 1 c h w (add temporal dimension t=1)
        return video.unsqueeze(0)
    else:  # ndim == 4
        # c t h w -> t c h w (permute channels and temporal)
        return video.permute(1, 0, 2, 3)


def optimized_sample_to_image_format(sample: torch.Tensor) -> torch.Tensor:
    """
    🚀 OPTIMIZED version to convert sample to image format
    Replaces rearrange() with native PyTorch operations
    
    Transforms:
    - 3D: c h w -> 1 h w c (add temporal dimension + permute to image format)
    - 4D: t c h w -> t h w c (permute to image format)
    
    Expected gains: 2-5x faster than rearrange()
    
    Args:
        sample: Input sample tensor
        
    Returns:
        Tensor in image format (channels last)
    """
    if sample.ndim == 3:
        # c h w -> 1 h w c (add temporal dimension then permute)
        return sample.unsqueeze(0).permute(0, 2, 3, 1)
    else:  # ndim == 4
        # t c h w -> t h w c (permute channels to last)
        return sample.permute(0, 2, 3, 1)



