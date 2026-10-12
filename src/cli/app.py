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
            debug.log('FFmpeg not found in PATH; falling back to OpenCV video writer', level='WARNING', category='setup')
            config = replace(config, output=replace(config.output, video_backend='opencv'))
        started = time.monotonic()
        import torch
        from ..utils.model_registry import get_min_vram, estimate_keep_on_vram_gb
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Pre-flight VRAM capacity check
        if device.startswith('cuda'):
            free_vram_gb = torch.cuda.mem_get_info(torch.device(device))[0] / (1024**3)
            total_vram_gb = torch.cuda.get_device_properties(torch.device(device)).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(torch.device(device))

            from ..optimization.execution_plan import ExecutionPlan, MemoryPolicy
            is_resident = (config.inference.memory_policy == MemoryPolicy.RESIDENT) or (
                config.inference.memory_policy == MemoryPolicy.AUTO and not ExecutionPlan.should_stage(
                    resolution=config.inference.sampling.resolution,
                    frame_count=33,
                    batch_size=config.inference.sampling.batch_size,
                    dit_model=model.dit_model,
                    device=device,
                    policy=config.inference.memory_policy,
                )
            )
            if is_resident:
                est_keep_vram_gb = estimate_keep_on_vram_gb(
                    dit_model=model.dit_model,
                    resolution=config.inference.sampling.resolution,
                    batch_size=config.inference.sampling.batch_size
                )
                debug.log(f'VRAM pre-flight check (resident memory): {free_vram_gb:.2f}GB free, estimated concurrent required ~{est_keep_vram_gb:.1f}GB', category='memory')
                if est_keep_vram_gb > free_vram_gb:
                    rec_model = "seedvr2_ema_7b-Q4_K_M.gguf" if "7b" in model.dit_model.lower() and free_vram_gb >= 12.0 else "seedvr2_ema_3b-Q4_K_M.gguf"
                    rec_batch = max(1, config.inference.sampling.batch_size // 2) if config.inference.sampling.batch_size > 1 else 1
                    rec_res = 720 if config.inference.sampling.resolution > 720 else 360
                    error_msg = (
                        f"\n================================================================================\n"
                        f"[OOM PRE-FLIGHT ERROR] Resident memory policy cannot run on {gpu_name} ({device}):\n"
                        f"  Estimated concurrent VRAM required: ~{est_keep_vram_gb:.1f} GB\n"
                        f"  Available VRAM: {free_vram_gb:.2f} GB (Total: {total_vram_gb:.2f} GB)\n"
                        f"  Model: {model.dit_model} | Resolution: {config.inference.sampling.resolution}p | Batch: {config.inference.sampling.batch_size}\n\n"
                        f"Keeping both DiT and VAE concurrently resident in VRAM with these settings will trigger OOM.\n"
                        f"Auto-fallback is disabled per safety specification. Process is aborted.\n\n"
                        f"Recommended adjustments to run successfully:\n"
                        f"  1. Use --memory_policy staged (swaps DiT and VAE between phases):\n"
                        f"     Add '--memory_policy staged' to your command\n"
                        f"  2. Use a smaller quantized model that fits resident in VRAM:\n"
                        f"     --dit_model {rec_model}\n"
                        f"  3. Reduce batch size:\n"
                        f"     --batch_size {rec_batch}\n"
                        f"  4. Reduce target resolution:\n"
                        f"     --resolution {rec_res}\n"
                        f"================================================================================"
                    )
                    debug.log(error_msg, category="error", level="ERROR", force=True)
                    raise RuntimeError(error_msg)
            else:
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
                        f"  2. Enable BlockSwap offload: --blocks_to_swap 32\n"
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
