"""Conservative admission estimates and bounded OOM recovery for CLI jobs.

Estimates are planning hints, not allocator limits. CUDA kernels and codecs have
workspace costs that must also be measured on the target runtime.
"""

from dataclasses import asdict, dataclass, replace
from ..config import InferenceConfig, VideoConfig, TileConfig, OffloadConfig
from pathlib import Path
import math

GIB = 1024 ** 3


def available_ram_bytes():
    import psutil
    available = psutil.virtual_memory().available
    # Colab may expose host RAM through /proc; respect its container allowance.
    try:
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit != 'max':
            used = int(Path('/sys/fs/cgroup/memory.current').read_text())
            available = min(available, max(0, int(limit) - used))
    except (OSError, ValueError):
        pass
    return available


@dataclass(frozen=True)
class MemoryPlan:
    chunk_size: int
    batch_size: int
    tile_size: int
    ram_bytes: int
    vram_bytes: int
    estimated_working_bytes: int

    def as_dict(self):
        return asdict(self)


def plan_memory(width, height, resolution, max_resolution, requested_chunk,
                requested_batch, overlap, ram_bytes, vram_bytes=0,
                model_bytes=0, prepend=0):
    """Bound input/latent storage and allow two decoded batches plus postprocess.

Temporal and spatial VAE padding are included conservatively. Model bytes are
accounted separately on the GPU; RAM is the free working allowance after any
resident models. User frame limits are ceilings, never silently increased.
    """
    scale = resolution / min(width, height)
    if max_resolution > 0:
        scale = min(scale, max_resolution / max(width, height))
    out_w = math.ceil(width * scale / 64) * 64
    out_h = math.ceil(height * scale / 64) * 64
    raw_frame = width * height * 4  # also admits RGBA
    decoded_frame = out_w * out_h * 4 * 2
    latent_frame = math.ceil(out_w / 8) * math.ceil(out_h / 8) * 16 * 2
    batch = max(1, ((requested_batch - 1) // 4) * 4 + 1)
    # Decoded pair, color reference and FP32 color workspaces.
    ram_batch_limit = int(ram_bytes * 0.65) // max(1, decoded_frame * 12)
    if vram_bytes:
        activation_room = max(0, vram_bytes - model_bytes)
        gpu_batch_limit = activation_room // max(1, decoded_frame * 48)
        ram_batch_limit = min(ram_batch_limit, gpu_batch_limit)
    if ram_batch_limit < 1:
        raise MemoryError('Memory budget cannot admit one output frame; increase the budget or lower resolution')
    batch = min(batch, max(1, ((ram_batch_limit - 1) // 4) * 4 + 1))
    effective_overlap = min(overlap, batch - 1)
    duplication = batch / (batch - effective_overlap)
    # Overlapping batches and 4n+1 padding duplicate small latent/RGBA references.
    per_input_frame = math.ceil(raw_frame * (3 + 2 * duplication) + latent_frame * 4 * duplication)
    fixed = decoded_frame * batch * 12
    capacity = (ram_bytes - fixed) // max(1, per_input_frame) - effective_overlap - prepend
    if capacity < 1:
        raise MemoryError('Memory budget cannot admit the input and overlap buffers')
    # A finite default even for tiny inputs: no whole-video allocation.
    chunk = min(requested_chunk if requested_chunk > 0 else 33, int(capacity))
    tile = 512 if vram_bytes and vram_bytes < 24 * GIB else 1024
    return MemoryPlan(chunk, batch, tile, ram_bytes, vram_bytes,
                      fixed + (chunk + effective_overlap + prepend) * per_input_frame)


def is_oom(error):
    if isinstance(error, MemoryError):
        return True
    import torch
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and any(text in str(error).lower()
        for text in ('out of memory', 'cannot allocate memory', 'allocation on device')))


def reduce_workload(config: InferenceConfig, video: VideoConfig) -> tuple[InferenceConfig, VideoConfig] | None:
    """Return a reduced execution plan; never mutate the user's settings."""
    def reduce_tile(tile: TileConfig) -> TileConfig:
        size = max(128, tile.size // 2 // 64 * 64)
        return replace(tile, enabled=True, size=size, overlap=min(tile.overlap, size // 4))

    batch = max(1, ((config.sampling.batch_size - 1) // 8) * 4 + 1)
    blocks = 36 if '7b' in config.model.dit_model else 32
    reduced = replace(config,
        memory_policy="staged",
        sampling=replace(config.sampling, batch_size=batch,
                         temporal_overlap=min(config.sampling.temporal_overlap, batch - 1)),
        vae=replace(config.vae, encode=reduce_tile(config.vae.encode), decode=reduce_tile(config.vae.decode)),
        offload=OffloadConfig(), cache=config.cache,
        swap=replace(config.swap, blocks_to_swap=min(blocks, max(config.swap.blocks_to_swap + 4, blocks // 2))))
    window = replace(video, chunk_size=max(1, (video.chunk_size or 33) // 2))
    return (reduced, window) if (reduced, window) != (config, video) else None
