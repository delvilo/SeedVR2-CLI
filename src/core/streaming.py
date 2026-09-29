"""Decode with a bounded pending tail; emit only finalized temporal overlaps."""

from collections import deque
from dataclasses import replace
from contextlib import closing
from .context import GenerationContext

import sys
import torch

from .generation_utils import blend_overlapping_frames, ensure_precision_initialized
from .generation_phases import postprocess_all_batches
from .model_loader import materialize_model
from ..optimization.memory_manager import cleanup_vae, manage_model_device


def blend_decoded_batches(batches, overlap):
    """Emit finalized batch slices, retaining a bounded tail across boundaries.

    Large overlaps can span several already trimmed batches. Delay each slice's
    color correction until no later blend can touch it; compact trimmed storage
    so small strides do not retain multiple full decoded batches.
    """
    pending = deque()
    retained = 0
    total = 0
    for index, current in batches:
        skipped = 0
        if overlap and len(current) > overlap and total >= overlap:
            parts = []
            needed = overlap
            for _, frames, _ in reversed(pending):
                take = min(needed, len(frames))
                parts.append(frames[-take:])
                needed -= take
                if not needed:
                    break
            parts.reverse()
            blended = blend_overlapping_frames(torch.cat(parts), current[:overlap], overlap)
            offset = 0
            for part in parts:
                part.copy_(blended[offset:offset + len(part)])
                offset += len(part)
            del parts, part, frames, blended
            current = current[overlap:].clone()
            skipped = overlap
        pending.append((index, current, skipped))
        retained += len(current)
        total += len(current)
        del current
        while pending and retained - len(pending[0][1]) >= overlap:
            item = pending.popleft()
            retained -= len(item[1])
            yield item
            del item
    while pending:
        yield pending.popleft()


def iter_decoded(runner, ctx: GenerationContext, debug, cache_model=False):
    """Decode onto the offload device, releasing each consumed latent."""
    try:
        if next(runner.vae.parameters()).device.type == 'meta':
            materialize_model(runner, 'vae', ctx.vae_device, runner.config, debug)
        ensure_precision_initialized(ctx, runner, debug)
        manage_model_device(runner.vae, ctx.vae_device, 'VAE', debug, runner=runner)
        height, width = ctx.true_target_dims
        if runner.tile_debug == 'decode' and runner.decode_tiled:
            debug.decode_tile_boundaries = []
        for index in range(len(ctx.all_upscaled_latents)):
            latent = ctx.all_upscaled_latents[index]
            if latent is None:
                continue
            latent = latent.to(device=ctx.vae_device, dtype=ctx.compute_dtype)
            decoded = runner.vae_decode([latent])[0]
            # CTHW -> THWC, or CHW -> 1HWC; these are views.
            sample = decoded.permute(1, 2, 3, 0) if decoded.ndim == 4 else decoded.permute(1, 2, 0).unsqueeze(0)
            sample = sample[:ctx.all_ori_lengths[index], :height, :width]
            sample = sample.to(device='cpu', dtype=ctx.compute_dtype)
            ctx.all_upscaled_latents[index] = None
            del latent, decoded
            yield index, sample
            del sample
    finally:
        cleanup_vae(runner, debug, cache_model=cache_model and sys.exception() is None)
        ctx.all_upscaled_latents = None


def iter_output_batches(runner, ctx: GenerationContext, debug, color_correction, cache_model=False):
    """Postprocess finalized decoded batches without a whole-chunk output."""
    with closing(iter_decoded(runner, ctx, debug, cache_model)) as decoded:
        with closing(blend_decoded_batches(decoded, (ctx.actual_temporal_overlap if ctx.actual_temporal_overlap is not None else 0))) as assembled:
            for index, sample, skipped in assembled:
                local = replace(ctx)
                if ctx.is_rgba:
                    rgba = torch.empty((*sample.shape[:-1], 4), dtype=sample.dtype, device=sample.device)
                    rgba[..., :3] = sample
                    rgba[..., 3] = 1
                    sample = rgba
                    del rgba
                    local.all_alpha_channels = list(ctx.all_alpha_channels)
                    local.all_input_rgb = list(ctx.all_input_rgb)
                    local.all_alpha_channels[index] = local.all_alpha_channels[index][skipped:skipped + sample.shape[0]]
                    local.all_input_rgb[index] = local.all_input_rgb[index][skipped:skipped + sample.shape[0]]
                local.final_video = sample
                local.decode_batch_info = [(0, sample.shape[0], index, sample.shape[0])]
                local.total_frames = sample.shape[0]
                local.postprocess_skip_head = skipped
                postprocess_all_batches(local, debug, color_correction=color_correction, cleanup=False)
                result = local.final_video
                local.final_video = None
                if ctx.all_alpha_channels is not None:
                    ctx.all_alpha_channels[index] = None
                if ctx.all_input_rgb is not None:
                    ctx.all_input_rgb[index] = None
                local.clear()
                yield result
                del result, sample, local
