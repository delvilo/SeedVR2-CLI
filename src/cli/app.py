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
        if not download_weight(dit_model=model.dit_model, vae_model=DEFAULT_VAE,
                               model_dir=model.model_dir, debug=debug):
            raise RuntimeError('Failed to download required models. Check console output above.')
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
