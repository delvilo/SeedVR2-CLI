"""Conservative admission estimates and bounded OOM recovery for CLI jobs.

Estimates are planning hints, not allocator limits. CUDA kernels and codecs have
workspace costs that must also be measured on the target runtime.
"""

from dataclasses import asdict, dataclass
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


def reduce_workload(args):
    """Change execution granularity only; preserve resolution and random seed."""
    old = (args.batch_size, args.chunk_size, args.vae_encode_tiled,
           args.vae_decode_tiled, args.vae_encode_tile_size, args.vae_decode_tile_size,
           args.dit_offload_device, args.vae_offload_device, args.tensor_offload_device,
           args.cache_dit, args.cache_vae, args.blocks_to_swap)
    args.vae_encode_tiled = args.vae_decode_tiled = True
    args.vae_encode_tile_size = max(128, args.vae_encode_tile_size // 2 // 64 * 64)
    args.vae_decode_tile_size = max(128, args.vae_decode_tile_size // 2 // 64 * 64)
    args.vae_encode_tile_overlap = min(args.vae_encode_tile_overlap, args.vae_encode_tile_size // 4)
    args.vae_decode_tile_overlap = min(args.vae_decode_tile_overlap, args.vae_decode_tile_size // 4)
    args.batch_size = max(1, ((args.batch_size - 1) // 8) * 4 + 1)
    args.chunk_size = max(1, (args.chunk_size or 33) // 2)
    args.temporal_overlap = min(args.temporal_overlap, args.batch_size - 1)
    args.dit_offload_device = args.vae_offload_device = 'cpu'
    args.tensor_offload_device = 'cpu'
    args.cache_dit = args.cache_vae = False
    blocks = 36 if '7b' in args.dit_model else 32
    args.blocks_to_swap = min(blocks, max(args.blocks_to_swap + 4, blocks // 2))
    new = (args.batch_size, args.chunk_size, args.vae_encode_tiled,
           args.vae_decode_tiled, args.vae_encode_tile_size, args.vae_decode_tile_size,
           args.dit_offload_device, args.vae_offload_device, args.tensor_offload_device,
           args.cache_dit, args.cache_vae, args.blocks_to_swap)
    return old != new
