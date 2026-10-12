"""Resolve per-file memory settings without mutating the user's configuration."""

from dataclasses import replace
from pathlib import Path
import math
import torch

from ..config import AppConfig, TileConfig
from ..optimization.memory_budget import GIB, available_ram_bytes, plan_memory
from ..optimization.memory_profile import MemoryRecorder
from ..utils.constants import get_base_cache_dir
from ..utils.debug import Debug
from ..utils.model_registry import DEFAULT_VAE
from .session import InferenceSession


def plan_file(config: AppConfig, width: int, height: int, device: str,
              session: InferenceSession, debug: Debug) -> AppConfig:
    inference, memory = config.inference, config.memory
    sampling, cache = inference.sampling, inference.cache
    available = available_ram_bytes()
    ram = min(int(memory.ram_budget_gb * GIB), available) if memory.ram_budget_gb else int(available * .8)
    free_vram = torch.cuda.mem_get_info(torch.device(device))[0] if (torch.cuda.is_available() and torch.device(device).type == 'cuda') else 0
    vram = min(int(memory.vram_budget_gb * GIB), free_vram) if memory.vram_budget_gb else int(free_vram * .9)
    model_dir = Path(inference.model.model_dir or get_base_cache_dir())
    weights = [(model_dir / name).stat().st_size if (model_dir / name).is_file() else 0
               for name in (inference.model.dit_model, DEFAULT_VAE)]
    reserve = int(sum(weights) * 1.25)
    blocks = 36 if '7b' in inference.model.dit_model else 32
    swap, offload = inference.swap, inference.offload
    from ..optimization.execution_plan import ExecutionPlan, MemoryPolicy
    is_resident = (inference.memory_policy == MemoryPolicy.RESIDENT) or (
        inference.memory_policy == MemoryPolicy.AUTO and not ExecutionPlan.should_stage(
            resolution=inference.sampling.resolution,
            frame_count=33,
            batch_size=inference.sampling.batch_size,
            dit_model=inference.model.dit_model,
            policy=inference.memory_policy
        )
    )
    if is_resident:
        swap = replace(swap, blocks_to_swap=0)
        offload = replace(offload, dit='none', vae='none', tensor='none')
        dit_resident = weights[0]
        session.vram_cache_bytes = int(vram * 0.95)
        cache = replace(cache, vram_gb=session.vram_cache_bytes / GIB)
    else:
        if vram and weights[0]:
            needed = math.ceil(blocks * max(0, 1 - .45 * vram / weights[0]))
            swap = replace(swap, blocks_to_swap=min(blocks, max(swap.blocks_to_swap, needed)))
            if swap.blocks_to_swap:
                offload = replace(offload, dit='cpu')
        dit_resident = int(weights[0] * max(.1, 1 - swap.blocks_to_swap / blocks))
        cache_ram = int(cache.ram_gb * GIB) if cache.ram_gb is not None else ram // 4
        cache_vram = int(cache.vram_gb * GIB)
        session.ram_cache_bytes = min(cache_ram, ram // 2) if is_resident else 0
        session.vram_cache_bytes = min(cache_vram, vram // 2) if is_resident else 0
        cache = replace(cache, ram_gb=session.ram_cache_bytes / GIB, vram_gb=session.vram_cache_bytes / GIB)
    session.finish(is_resident, is_resident)
    plan = plan_memory(width, height, sampling.resolution, sampling.max_resolution,
                       config.video.chunk_size, sampling.batch_size, sampling.temporal_overlap,
                       ram_bytes=max(0, ram - reserve - session.ram_cache_bytes),
                       vram_bytes=vram, model_bytes=max(dit_resident, weights[1]), prepend=sampling.prepend_frames)

    def bound_tile(tile: TileConfig) -> TileConfig:
        size = min(tile.size, plan.tile_size)
        return replace(tile, size=size, overlap=min(tile.overlap, size // 4))

    inference = replace(inference, swap=swap, offload=offload, cache=cache,
        sampling=replace(sampling, batch_size=plan.batch_size,
                         temporal_overlap=min(sampling.temporal_overlap, plan.batch_size - 1)),
        vae=replace(inference.vae, encode=bound_tile(inference.vae.encode), decode=bound_tile(inference.vae.decode)))
    planned = replace(config, inference=inference, video=replace(config.video, chunk_size=plan.chunk_size))
    debug.log(f'Memory plan: chunk={plan.chunk_size}, batch={plan.batch_size}, '
              f'tile={plan.tile_size}, RAM working allowance={plan.ram_bytes / GIB:.2f} GiB',
              category='memory', force=True)
    MemoryRecorder(memory.report).record('plan', **plan.as_dict(), resolution=sampling.resolution,
        seed=sampling.seed, model_reserve_bytes=reserve, cache_ram_bytes=session.ram_cache_bytes,
        cache_vram_bytes=session.vram_cache_bytes)
    return planned
