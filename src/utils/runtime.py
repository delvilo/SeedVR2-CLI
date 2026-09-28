"""Checks for the supported Colab Linux/CUDA runtime."""

import platform
import sys

import torch


def validate_runtime() -> None:
    """Fail before downloading model weights on unsupported runtimes."""
    if platform.system() != "Linux":
        raise RuntimeError("SeedVR2 CLI requires Linux with NVIDIA CUDA")
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("SeedVR2 CLI requires Python 3.13")
    if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError(
            "SeedVR2 CLI targets Colab PyTorch 2.11.0+cu128 (CUDA 12.8); "
            f"found torch {torch.__version__}, CUDA {torch.version.cuda}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("No NVIDIA CUDA GPU available; select a GPU runtime in Colab")
