"""Select the CLI's single GPU before importing any CUDA-aware module."""

import os
import sys


def configure_environment(index: int) -> str:
    if index < 0:
        raise ValueError('--cuda_device must be a non-negative CUDA device index')
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized() and not os.environ.get('PYTEST_CURRENT_TEST'):
        raise RuntimeError('Start the CLI in a fresh process to select its CUDA device')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        selected = str(index)
    else:
        devices = [item.strip() for item in visible.split(',') if item.strip()]
        if not devices:
            selected = ''
        elif index >= len(devices):
            raise ValueError(f'--cuda_device {index} requested, but only {len(devices)} CUDA device(s) are visible')
        else:
            selected = devices[index]
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'backend:cudaMallocAsync')
    os.environ['CUDA_VISIBLE_DEVICES'] = selected
    return selected
