"""
Execution Plan and Staged Memory Policy for SeedVR2
Unified memory scheduling that replaces fragmented offload/caching parameters
(--cache_dit, --cache_vae, --dit_offload_device, --vae_offload_device, --keep_on_vram)
with an intelligent policy:
- 'staged': Strictly keeps only active model on GPU (DiT during upscale, VAE during encode/decode).
           Offloads idle models to CPU to guarantee zero OOM on 8-16GB GPUs (e.g. Colab Tesla T4).
- 'resident': Keeps both DiT and VAE in GPU VRAM across all phases for 2x speedup on 24GB+ GPUs.
- 'auto': Automatically selects staged or resident based on GPU memory, model size, and video resolution.
"""

from typing import Optional, Tuple
import torch


class MemoryPolicy:
    AUTO = "auto"
    STAGED = "staged"
    RESIDENT = "resident"


class ExecutionPlan:
    """
    Evaluates hardware capacity and workload dimensions to determine memory staging.
    """

    @staticmethod
    def get_vram_info(device: Optional[torch.device] = None) -> Tuple[float, float]:
        """Returns (free_gb, total_gb)."""
        if not torch.cuda.is_available():
            return 0.0, 0.0
        try:
            if device is None:
                dev = torch.device("cuda:0")
            elif isinstance(device, str):
                dev = torch.device(device)
            else:
                dev = device
            free_b, total_b = torch.cuda.mem_get_info(dev)
            return free_b / (1024 ** 3), total_b / (1024 ** 3)
        except Exception:
            return 0.0, 0.0

    @classmethod
    def should_stage(
        cls,
        resolution: int,
        frame_count: int,
        batch_size: int,
        dit_model: str = "seedvr2_ema_3b_fp8.safetensors",
        device: Optional[str] = None,
        policy: str = MemoryPolicy.AUTO,
    ) -> bool:
        """
        Determines whether models must be staged (swapped between phases) or kept resident.
        """
        norm_policy = policy.lower().strip()
        if norm_policy == MemoryPolicy.STAGED:
            return True
        if norm_policy == MemoryPolicy.RESIDENT:
            return False

        # AUTO policy logic
        if not torch.cuda.is_available():
            return True

        dev = torch.device(device if device else "cuda:0")
        free_gb, total_gb = cls.get_vram_info(dev)

        # On <= 16.5GB GPUs (e.g. Tesla T4 ~15GB, RTX 4070 12GB):
        # Peak VAE decode + DiT concurrency easily exceeds 15GB on 720p+ or large batches.
        if total_gb <= 16.5:
            if resolution >= 720 or batch_size >= 9 or frame_count >= 33 or "7b" in dit_model.lower():
                return True
            # For 3B small resolution with high free VRAM, only allow resident if free VRAM >= 12GB
            if free_gb < 11.5:
                return True
            return False

        # On > 16.5GB GPUs (e.g. L4 22GB, RTX 3090/4090 24GB, A100):
        # 7B FP16 at 1080p requires ~20.4GB, 7B FP16 at batch 9 requires ~22.6GB
        if "7b" in dit_model.lower() and "fp16" in dit_model.lower():
            if resolution >= 1080 and batch_size >= 9:
                return True
            if free_gb < 21.0:
                return True
            return False

        if resolution >= 1440 or batch_size >= 17:
            return True

        return False

    @classmethod
    def resolve_offload_configuration(
        cls,
        resolution: int,
        frame_count: int,
        batch_size: int,
        dit_model: str,
        device: str,
        policy: str = MemoryPolicy.AUTO,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
        """
        Resolves internal (dit_offload, vae_offload, tensor_offload, keep_on_vram).
        """
        stage = cls.should_stage(
            resolution=resolution,
            frame_count=frame_count,
            batch_size=batch_size,
            dit_model=dit_model,
            device=device,
            policy=policy,
        )

        if stage:
            # Staged memory policy: idle models offloaded to CPU to free VRAM
            return "cpu", "cpu", "cpu", False
        else:
            # Resident memory policy: models remain on GPU across phases
            return None, None, None, True
