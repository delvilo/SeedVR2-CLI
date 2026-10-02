"""Checks for the supported Colab Linux/CUDA runtime."""

import platform
import sys

import torch


def validate_runtime() -> None:
    """Fail before downloading model weights on unsupported runtimes."""
    if platform.system() != "Linux":
        raise RuntimeError("SeedVR2 CLI requires Linux with NVIDIA CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("No NVIDIA CUDA GPU available; select a GPU runtime in Colab")
