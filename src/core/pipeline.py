"""Four inference phases for one bounded input window, independent of the CLI."""

from __future__ import annotations
from collections.abc import Generator
from contextlib import closing
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from ..config import InferenceConfig
from ..utils.constants import get_base_cache_dir
from ..utils.model_registry import DEFAULT_VAE
from ..utils.debug import Debug
from ..optimization.memory_profile import MemoryRecorder
from ..optimization.memory_manager import cleanup_text_embeddings
from .generation_utils import (setup_generation_context, prepare_runner,
    compute_generation_info, log_generation_start, load_text_embeddings, script_directory)
from .generation_phases import encode_all_batches, upscale_all_batches
from .streaming import iter_output_batches

if TYPE_CHECKING:
    from .session import InferenceSession


def parse_offload_device(value: str | None, cache_enabled: bool = False, keep_on_vram: bool = False) -> str | None:
    if keep_on_vram:
        return None
    if value is None or value == "none":
        return "cpu" if cache_enabled else None
    if value in ("cpu", "cuda") or ":" in value:
        return value
    if value.isdigit():
        return f"cuda:{value}"
    return value


def iter_processed_batches(
    frames_tensor: torch.Tensor, config: InferenceConfig, device: str,
    debug: Debug, session: InferenceSession, memory_report: str | None = None,
) -> Generator[torch.Tensor, None, None]:
    inference_device = device
    keep_on_vram = config.keep_on_vram
    cache_dit = config.cache.dit or keep_on_vram
    cache_vae = config.cache.vae or keep_on_vram
    dit_offload = parse_offload_device(config.offload.dit, cache_dit, keep_on_vram)
    vae_offload = parse_offload_device(config.offload.vae, cache_vae, keep_on_vram)
    tensor_offload = parse_offload_device(config.offload.tensor, False, keep_on_vram)
    ctx = setup_generation_context(dit_device=inference_device, vae_device=inference_device,
        dit_offload_device=dit_offload, vae_offload_device=vae_offload,
        tensor_offload_device=tensor_offload, debug=debug, keep_on_vram=keep_on_vram)
    recorder = MemoryRecorder(memory_report, inference_device)
    # Prepare runner with caching support
    model_dir = config.model.model_dir if config.model.model_dir is not None else get_base_cache_dir()

    success = False
    previous_recorder = getattr(debug, 'external_memory_recorder', False)
    debug.external_memory_recorder = bool(memory_report)
    try:
        with recorder.phase("setup"):
            runner = prepare_runner(
                dit_model=config.model.dit_model,
                vae_model=DEFAULT_VAE,
                model_dir=model_dir,
                debug=debug,
                ctx=ctx,
                dit_cache=cache_dit,
                vae_cache=cache_vae,
                session=session,
                block_swap_config=replace(config.swap, offload_device=dit_offload or device),
                encode_tiled=config.vae.encode.enabled,
                encode_tile_size=(config.vae.encode.size, config.vae.encode.size),
                encode_tile_overlap=(config.vae.encode.overlap, config.vae.encode.overlap),
                decode_tiled=config.vae.decode.enabled,
                decode_tile_size=(config.vae.decode.size, config.vae.decode.size),
                decode_tile_overlap=(config.vae.decode.overlap, config.vae.decode.overlap),
                tile_debug=config.vae.debug.lower() if config.vae.debug else "false",
                attention_mode=config.attention_mode,
                torch_compile_args_dit=config.compile if config.compile.dit else None,
                torch_compile_args_vae=config.compile if config.compile.vae else None
            )
            runner._keep_on_vram = keep_on_vram

            # Preload text embeddings before Phase 1 to avoid sync stall in Phase 2
            ctx.text_embeds = load_text_embeddings(script_directory, ctx.dit_device, ctx.compute_dtype, debug)
            debug.log("Loaded text embeddings for DiT", category="dit")

        # Compute generation info and log start (handles prepending internally)
        frames_tensor, gen_info = compute_generation_info(
            ctx=ctx,
            images=frames_tensor,
            resolution=config.sampling.resolution,
            max_resolution=config.sampling.max_resolution,
            batch_size=config.sampling.batch_size,
            uniform_batch_size=config.sampling.uniform_batch_size,
            seed=config.sampling.seed,
            prepend_frames=config.sampling.prepend_frames,
            temporal_overlap=config.sampling.temporal_overlap,
            debug=debug
        )
        log_generation_start(gen_info, debug)

        # Phase 1: Encode
        with recorder.phase("encode"):
            ctx = encode_all_batches(
                runner, ctx=ctx, images=frames_tensor,
                debug=debug,
                batch_size=config.sampling.batch_size,
                uniform_batch_size=config.sampling.uniform_batch_size,
                seed=config.sampling.seed,
                progress_callback=None,
                temporal_overlap=config.sampling.temporal_overlap,
                resolution=config.sampling.resolution,
                max_resolution=config.sampling.max_resolution,
                input_noise_scale=config.sampling.input_noise_scale,
                color_correction=config.sampling.color_correction
            )

        # Phase 2: Upscale
        with recorder.phase("upscale"):
            ctx = upscale_all_batches(
                runner, ctx=ctx, debug=debug, progress_callback=None,
                seed=config.sampling.seed,
                latent_noise_scale=config.sampling.latent_noise_scale,
                cache_model=cache_dit
            )

        # CPU postprocessing avoids co-residency with the active VAE on CUDA.
        ctx.postprocess_device = torch.device('cpu')
        drop = config.sampling.prepend_frames
        with recorder.phase('decode_postprocess_write'):
            with closing(iter_output_batches(runner, ctx, debug, config.sampling.color_correction, cache_vae)) as outputs:
                for result in outputs:
                    skip = min(drop, len(result))
                    drop -= skip
                    if skip < len(result):
                        yield result[skip:]
                    del result
        success = True
    finally:
        cleanup_text_embeddings(ctx, debug)
        ctx.clear()
        debug.external_memory_recorder = previous_recorder
        debug.clear_history()
        if success:
            session.finish(cache_dit, cache_vae)
        else:
            session.close()
