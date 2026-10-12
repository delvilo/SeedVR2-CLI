"""CLI lifecycle: parse, select CUDA, initialize dependencies and report errors."""

from collections.abc import Sequence
from dataclasses import replace
import os
from pathlib import Path
import shutil
import time

from .arguments import parse_arguments
from .bootstrap import configure_environment


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_arguments(argv)
    try:
        selected = configure_environment(config.cuda_device)
        # All CUDA-aware imports happen after selecting the single visible GPU.
        from ..utils.runtime import validate_runtime
        validate_runtime()
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    from ..core.scheduler import run_jobs
    from ..utils.constants import get_base_cache_dir
    from ..utils.debug import Debug
    from ..utils.downloads import download_weight
    from ..utils.model_registry import DEFAULT_VAE

    model = config.inference.model
    model = replace(model, model_dir=str(Path(model.model_dir or get_base_cache_dir()).expanduser().resolve()))
    config = replace(config, inference=replace(config.inference, model=model))
    debug = Debug(config.debug)
    debug.print_header()
    try:
        debug.log(f'Configuration: {config}', category='setup')
        debug.log(f'Using CUDA device: {selected} (visible as cuda:0)', category='device')
        if config.output.video_backend == 'ffmpeg' and shutil.which('ffmpeg') is None:
            raise RuntimeError('--video_backend ffmpeg requires ffmpeg in PATH')
        started = time.monotonic()
        import torch
        from ..utils.model_registry import get_min_vram
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Pre-flight VRAM capacity check
        if device.startswith('cuda'):
            free_vram_gb = torch.cuda.mem_get_info(torch.device(device))[0] / (1024**3)
            min_vram_gb = get_min_vram(model.dit_model)
            blocks_swapped = config.inference.swap.blocks_to_swap
            effective_min = min_vram_gb * max(0.4, 1.0 - (blocks_swapped / 32) * 0.6) if blocks_swapped > 0 else min_vram_gb
            debug.log(f'VRAM pre-flight check: {free_vram_gb:.2f}GB free, model requires ~{effective_min:.1f}GB', category='memory')
            if free_vram_gb < effective_min:
                raise RuntimeError(
                    f"Insufficient VRAM on {device}: Model '{model.dit_model}' requires at least {effective_min:.1f}GB VRAM, "
                    f"but only {free_vram_gb:.2f}GB is available.\n"
                    f"Solutions:\n"
                    f"  1. Use a more compact quantized model: --dit_model seedvr2_ema_3b-Q4_K_M.gguf\n"
                    f"  2. Enable BlockSwap offload: --blocks_to_swap 32 --dit_offload_device cpu\n"
                    f"  3. Free GPU memory from other processes."
                )

        if not download_weight(dit_model=model.dit_model, vae_model=DEFAULT_VAE,
                               model_dir=model.model_dir, debug=debug):
            raise RuntimeError('Failed to download required models. Check console output above.')
        count = run_jobs(config, device, debug)
        elapsed = time.monotonic() - started
        debug.log(f'All upscaling processes completed successfully in {elapsed:.2f}s', category='success', force=True)
        if count and elapsed > 0:
            debug.log(f'Average FPS: {count / elapsed:.2f} frames/sec', category='timing', force=True)
        return 0
    except Exception as error:
        debug.log(f'Error during processing: {error}', level='ERROR', category='generation', force=True)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        debug.log(f'Process {os.getpid()} terminating - VRAM will be automatically freed', category='cleanup', force=True)
        debug.print_footer()
