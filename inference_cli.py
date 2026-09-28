#!/usr/bin/env python3
"""
SeedVR2 Video Upscaler - Standalone CLI Interface

Command-line interface for high-quality upscaling using SeedVR2 diffusion models.
Runs inference on one selected CUDA device.

Key Features:
    • Streaming Mode: Memory-efficient processing of long videos in chunks, avoiding
      full video loading into RAM while maintaining temporal consistency
    • Memory Optimization: BlockSwap for limited VRAM, VAE tiling for large resolutions,
      intelligent tensor offloading between processing phases
    • Performance: Torch.compile integration, BFloat16 compute pipeline,
      efficient model caching for batch and streaming processing
    • Flexibility: Multiple output formats (MP4/PNG), advanced color correction methods,
      directory batch processing with auto-format detection
    • Quality Control: Temporal overlap blending, frame prepending for artifact reduction,
      configurable noise scales for detail preservation

Architecture:
    The CLI implements a 4-phase processing pipeline:
    1. Encode: VAE encoding with optional input noise and tiling
    2. Upscale: DiT transformer upscaling with latent space diffusion
    3. Decode: tiled VAE decoding into bounded output batches
    4. Postprocess/write: finalize overlaps and write immediately on the CPU

Usage:
    python inference_cli.py video.mp4 --resolution 1080
    For complete usage examples, run: python inference_cli.py --help

Requirements:
    • Ubuntu 24.04, Python 3.13, PyTorch 2.11.0+cu128
    • 16GB+ VRAM recommended (8GB minimum with BlockSwap)
    • OpenCV, NumPy for video I/O

Model Support:
    • 3B models: seedvr2_ema_3b_fp16.safetensors (default), _fp8_e4m3fn/GGUF variants
    • 7B models: seedvr2_ema_7b_fp16.safetensors, _fp8_e4m3fn/GGUF variants
    • VAE: ema_vae_fp16.safetensors (shared across all models)
    • Auto-downloads from HuggingFace on first run with SHA256 validation
"""

# Standard library imports
import sys
import os
import argparse
import time
import math
from typing import Dict, Any, List, Optional, Tuple, Literal
from contextlib import closing, ExitStack
from pathlib import Path

# Set up path before any other imports to fix module resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Set CUDA visibility before importing torch so a selected GPU maps to cuda:0.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--cuda_device", type=int, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
_requested_cuda_device = _pre_args.cuda_device if _pre_args.cuda_device is not None else 0
if _requested_cuda_device < 0:
    raise SystemExit("--cuda_device must be a non-negative CUDA device index")
_visible_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
if _visible_cuda_devices is not None:
    _visible_cuda_devices = [item.strip() for item in _visible_cuda_devices.split(",") if item.strip()]
    if not _visible_cuda_devices:
        _selected_cuda_device = ""
    elif _requested_cuda_device >= len(_visible_cuda_devices):
        raise SystemExit(
            f"--cuda_device {_requested_cuda_device} requested, but only {len(_visible_cuda_devices)} CUDA device(s) are visible"
        )
    else:
        _selected_cuda_device = _visible_cuda_devices[_requested_cuda_device]
else:
    _selected_cuda_device = str(_requested_cuda_device)
os.environ["CUDA_VISIBLE_DEVICES"] = _selected_cuda_device

# Heavy dependency imports after environment configuration
import torch
import cv2
import numpy as np
import shutil

# Project imports
from src.utils.downloads import download_weight
from src.utils.model_registry import get_available_dit_models, DEFAULT_DIT, DEFAULT_VAE
from src.utils.constants import get_base_cache_dir
from src.utils.runtime import validate_runtime
from src.core.generation_utils import (
    setup_generation_context, 
    prepare_runner, 
    compute_generation_info, 
    log_generation_start,
    load_text_embeddings,
    script_directory
)
from src.core.generation_phases import (
    encode_all_batches, 
    upscale_all_batches
)
from src.utils.debug import Debug
from src.optimization.memory_manager import clear_memory, cleanup_text_embeddings
from src.core.session import InferenceSession
from src.core.streaming import iter_output_batches
from src.optimization.memory_budget import GIB, available_ram_bytes, plan_memory, is_oom, reduce_workload
from src.optimization.memory_profile import MemoryRecorder
from src.utils.media import FFMPEGVideoWriter, OutputTransaction, read_frames, frame_pixels
debug = Debug(enabled=False)  # Will be enabled via --debug CLI flag


# =============================================================================
# Device Management Helpers
# =============================================================================

def _device_id_to_name(device_id: str, platform_type: str = "cuda") -> str:
    """
    Convert device ID to full device name.
    
    Args:
        device_id: Device ID ("0", "1") or special value ("cpu", "none")
        platform_type: CUDA backend
    
    Returns:
        Full device name ("cuda:0", "cpu", "none")
    """
    if device_id in ("cpu", "none"):
        return device_id
    
    return f"{platform_type}:{device_id}"


def _parse_offload_device(offload_arg: str, platform_type: str = None, cache_enabled: bool = False) -> Optional[str]:
    """
    Parse offload device argument to full device name.
    
    Args:
        offload_arg: Offload device argument ("none", "cpu", "0", "1", or "cuda:1")
        platform_type: Override platform type
        cache_enabled: If True and offload_arg is "none", default to "cpu"
    
    Returns:
        Full device name or None
    """
    if offload_arg == "none":
        # If caching enabled but no offload device specified, default to CPU
        return "cpu" if cache_enabled else None
    
    if offload_arg == "cpu":
        return "cpu"
    
    # If already a full device name (cuda:1), return as-is
    if ":" in offload_arg:
        return offload_arg
    
    # Otherwise treat as device ID
    return _device_id_to_name(offload_arg, platform_type)


# =============================================================================
# Constants
# =============================================================================

# Supported file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


# =============================================================================
# Video I/O Functions
# =============================================================================

def get_media_files(directory: str) -> List[str]:
    """
    Get all video and image files from directory, sorted alphabetically.
    
    Args:
        directory: Path to directory to scan
        
    Returns:
        Sorted list of file paths (strings) matching video or image extensions
    """
    valid_extensions = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
    path = Path(directory)
    
    # Get all files and filter by extension (case-insensitive)
    files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    return sorted([str(f) for f in files])


def extract_frames_from_image(image_path: str) -> Tuple[torch.Tensor, float]:
    """
    Extract single frame from image file and convert to tensor format.
    
    Reads image using OpenCV, converts BGR to RGB, normalizes to [0,1] range,
    and formats as single-frame video tensor for consistent processing.
    
    Args:
        image_path: Path to input image file
        
    Returns:
        Tuple containing:
            - frames_tensor: Single frame as tensor [1, H, W, C], Float16, range [0,1] (C=3 for RGB, C=4 for RGBA)
            - fps: Default FPS value (30.0) for image-to-video conversion
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be opened
    """
    debug.log(f"Loading image: {image_path}", category="file")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read image with alpha channel preserved
    frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ValueError(f"Cannot open image file: {image_path}")
    
    # Convert BGR(A) to RGB(A) based on channel count
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        debug.log(f"Detected RGBA image (alpha channel preserved)", category="file")
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if frame.dtype == np.uint16:
        frame = frame.astype(np.float32) / 65535.0
    frames_tensor = torch.from_numpy(frame[None, ...])

    debug.log(f"Image tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}", category="memory")
    
    return frames_tensor, 30.0  # Default FPS for images


def get_input_type(input_path: str) -> Literal['video', 'image', 'directory', 'unknown']:
    """
    Determine input type from file path.
    
    Args:
        input_path: Path to input file or directory
        
    Returns:
        Input type: 'video', 'image', 'directory', or 'unknown'
        
    Raises:
        FileNotFoundError: If input path doesn't exist
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    if path.is_dir():
        return 'directory'
    
    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in IMAGE_EXTENSIONS:
        return "image"
    else:
        return "unknown"


def generate_output_path(input_path: str, output_format: str, output_dir: Optional[str] = None, 
                        input_type: Optional[str] = None, from_directory: bool = False) -> str:
    """
    Generate output path based on input path and format.
    
    Args:
        input_path: Source file path
        output_format: "mp4" or "png"
        output_dir: Optional output directory (overrides default behavior)
        input_type: Optional input type ("image", "video", "directory")
        from_directory: True if processing files from a directory (batch mode)
    
    Returns:
        Absolute output path (file for single image/video, directory for sequences)
    """
    input_path_obj = Path(input_path)
    input_name = input_path_obj.stem
    
    # Determine base directory and whether to add suffix
    if output_dir:
        # User specified output directory - use as-is, no suffix
        base_dir = Path(output_dir)
        add_suffix = False
    elif from_directory:
        # Batch mode: create sibling folder with _upscaled, keep original filenames
        original_dir = input_path_obj.parent
        base_dir = original_dir.parent / f"{original_dir.name}_upscaled"
        add_suffix = False
    else:
        # Single file mode: output to same directory with _upscaled suffix
        base_dir = input_path_obj.parent
        add_suffix = True
    
    # Build filename with optional suffix
    file_suffix = "_upscaled" if add_suffix else ""
    
    # Generate output path based on format
    if output_format == "png":
        if input_type == "image":
            output_path = base_dir / f"{input_name}{file_suffix}.png"
        else:
            output_path = base_dir / f"{input_name}{file_suffix}"
    else:
        output_path = base_dir / f"{input_name}{file_suffix}.mp4"
    
    return str(output_path.resolve())


def _plan_file(args, width, height, device_id, session):
    """Reserve model/cache headroom before admitting bounded pixel buffers."""
    planned = argparse.Namespace(**vars(args))
    available = available_ram_bytes()
    ram = min(int(args.ram_budget_gb * GIB), available) if args.ram_budget_gb else int(available * .8)
    free_vram = torch.cuda.mem_get_info(int(device_id))[0] if torch.cuda.is_available() else 0
    vram = min(int(args.vram_budget_gb * GIB), free_vram) if args.vram_budget_gb else int(free_vram * .9)
    model_dir = Path(args.model_dir or get_base_cache_dir())
    weights = [(model_dir / name).stat().st_size if (model_dir / name).is_file() else 0
               for name in (args.dit_model, DEFAULT_VAE)]
    # Checkpoint size is a planning estimate, not the live tensor footprint.
    reserve = int(sum(weights) * 1.25)
    blocks = 36 if '7b' in args.dit_model else 32
    if vram and weights[0]:
        needed = math.ceil(blocks * max(0, 1 - .45 * vram / weights[0]))
        planned.blocks_to_swap = min(blocks, max(args.blocks_to_swap, needed))
        if planned.blocks_to_swap:
            planned.dit_offload_device = 'cpu'
    dit_resident = int(weights[0] * max(.1, 1 - planned.blocks_to_swap / blocks))
    cache_ram = int(args.cache_ram_gb * GIB) if args.cache_ram_gb is not None else ram // 4
    cache_vram = int(args.cache_vram_gb * GIB)
    session.ram_cache_bytes = min(cache_ram, ram // 2) if args.cache_dit or args.cache_vae else 0
    session.vram_cache_bytes = min(cache_vram, vram // 2) if args.cache_dit or args.cache_vae else 0
    planned.cache_ram_gb = session.ram_cache_bytes / GIB
    planned.cache_vram_gb = session.vram_cache_bytes / GIB
    session.finish(args.cache_dit, args.cache_vae)
    plan = plan_memory(width, height, args.resolution, args.max_resolution,
                       args.chunk_size, args.batch_size,
                       args.temporal_overlap,
                       ram_bytes=max(0, ram - reserve - session.ram_cache_bytes),
                       vram_bytes=vram, model_bytes=max(dit_resident, weights[1]), prepend=args.prepend_frames)
    planned.chunk_size, planned.batch_size = plan.chunk_size, plan.batch_size
    planned.temporal_overlap = min(args.temporal_overlap, plan.batch_size - 1)
    for phase in ('encode', 'decode'):
        size = min(getattr(args, f'vae_{phase}_tile_size'), plan.tile_size)
        setattr(planned, f'vae_{phase}_tile_size', size)
        setattr(planned, f'vae_{phase}_tile_overlap', min(getattr(args, f'vae_{phase}_tile_overlap'), size // 4))
    debug.log(f'Memory plan: chunk={planned.chunk_size}, batch={planned.batch_size}, '
              f'tile={plan.tile_size}, RAM working allowance={plan.ram_bytes / GIB:.2f} GiB',
              category='memory', force=True)
    MemoryRecorder(args.memory_report).record('plan', **plan.as_dict(), resolution=args.resolution,
        seed=args.seed, model_reserve_bytes=reserve, cache_ram_bytes=session.ram_cache_bytes,
        cache_vram_bytes=session.vram_cache_bytes)
    return planned


def process_single_file(input_path: str, args: argparse.Namespace, device_id: str,
                       output_path: Optional[str] = None, format_auto_detected: bool = False,
                       runner_cache: Optional[InferenceSession] = None, output_is_resolved: bool = False) -> int:
    """Restart an OOM attempt from the beginning, publishing only complete output."""
    input_type = get_input_type(input_path)
    if input_type not in ('video', 'image'):
        return 0
    is_sequence = input_type == 'video' and args.output_format == 'png'
    if output_path is None:
        output_path = generate_output_path(input_path, args.output_format, input_type=input_type)
    elif not output_is_resolved and (not Path(output_path).suffix or is_sequence):
        output_path = generate_output_path(input_path, args.output_format, output_dir=output_path, input_type=input_type)
    if input_type == 'video':
        cap = cv2.VideoCapture(input_path)
        try:
            if not cap.isOpened():
                raise ValueError(f'Cannot open video: {input_path}')
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
    else:
        probe, _ = extract_frames_from_image(input_path)
        height, width = probe.shape[1:3]
        del probe
    if min(width, height) <= 0:
        raise ValueError(f'Invalid media dimensions: {width}x{height}')
    session = runner_cache or InferenceSession(debug=debug)
    recorder = MemoryRecorder(args.memory_report, _device_id_to_name(device_id))
    try:
        planned = _plan_file(args, width, height, device_id, session)
        for attempt in range(args.oom_retries + 1):
            recorder.record('attempt', input=str(input_path), attempt=attempt,
                            chunk_size=planned.chunk_size, batch_size=planned.batch_size)
            try:
                with OutputTransaction(output_path, image_sequence=is_sequence, sequence_name=Path(input_path).stem) as output:
                    count = _write_file_attempt(input_path, str(output.path), input_type, planned, device_id, session)
                    if count:
                        output.commit()
                recorder.record('complete', input=str(input_path), attempt=attempt, frames=count)
                debug.log(f'Output saved to: {output_path} ({count} frames)', category='file', force=True)
                return count
            except Exception as error:
                if not is_oom(error) or attempt >= args.oom_retries:
                    raise
                reason = str(error)
                if not reduce_workload(planned):
                    raise
            # Leave the except scope first: traceback frames may own GPU tensors.
            session.close()
            clear_memory(debug=debug, deep=True, force=True)
            recorder.record('oom_retry', attempt=attempt + 1, reason=reason,
                batch_size=planned.batch_size, chunk_size=planned.chunk_size,
                tile_size=planned.vae_decode_tile_size, seed=planned.seed, resolution=planned.resolution)
            debug.log(f'OOM: restarting file with chunk={planned.chunk_size}, batch={planned.batch_size}, '
                      f'tile={planned.vae_decode_tile_size}', level='WARNING', category='memory', force=True)
    finally:
        if runner_cache is None:
            session.close()


def _write_file_attempt(input_path, output_path, input_type, args, device_id, session):
    """Own the capture, generator and writer together, including cancellation."""
    if input_type == 'image':
        frames, fps = extract_frames_from_image(input_path)
        result = _process_frames_core(frames, args, device_id, debug, session)
        if args.output_format == 'mp4':
            writer = save_frames_to_video(result, output_path, fps, video_backend=args.video_backend, use_10bit=args.use_10bit)
            writer.release()
        else:
            _save_image_bgr(frame_pixels(result[0]), output_path)
        return result.shape[0]
    cap = cv2.VideoCapture(input_path)
    writer = None
    complete = False
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {input_path}')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - args.skip_first_frames)
        if args.load_cap:
            count = min(count, args.load_cap)
        if not count:
            return 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.skip_first_frames)
        batches = _stream_video_chunks(cap, count, args.chunk_size, args.temporal_overlap,
            args, device_id, debug, session, log_progress=True,
            total_chunks=(count + args.chunk_size - 1) // args.chunk_size)
        written = 0
        with ExitStack() as stack:
            if hasattr(batches, 'close'):
                stack.callback(batches.close)
            for batch in batches:
                if args.output_format == 'png':
                    save_frames_to_image(batch, output_path, Path(input_path).stem, start_index=written)
                else:
                    writer = save_frames_to_video(batch, output_path, fps, writer=writer,
                        video_backend=args.video_backend, use_10bit=args.use_10bit)
                written += batch.shape[0]
                del batch
        if writer is not None:
            writer.release()
            writer = None
        complete = True
        return written
    finally:
        cap.release()
        if writer is not None:
            if not complete and hasattr(writer, 'abort'):
                writer.abort()
            else:
                writer.release()


def _read_frames_from_cap(cap, max_frames):
    return read_frames(cap, max_frames)


def _stream_video_chunks(cap, frames_to_process, chunk_size, overlap, args, device_id,
                         debug, runner_cache, log_progress=False, total_chunks=0,
                         cleanup_timer_name=None, log_prefix=''):
    """Read a uint8 window; yield finalized output batches directly to the sink."""
    chunk_args = argparse.Namespace(**vars(args))
    frames_read = 0
    prev_raw_tail = None
    chunk_idx = 0
    recorder = MemoryRecorder(args.memory_report, _device_id_to_name(device_id))
    while frames_read < frames_to_process:
        with recorder.phase('read'):
            new_frames = read_frames(cap, min(chunk_size, frames_to_process - frames_read))
        if new_frames is None:
            break
        frames_read += len(new_frames)
        chunk_idx += 1
        if chunk_idx > 1:
            chunk_args.prepend_frames = 0
        context_count = min(overlap, len(prev_raw_tail)) if prev_raw_tail is not None else 0
        frames = torch.cat((prev_raw_tail[-context_count:], new_frames)) if context_count else new_frames
        prev_raw_tail = frames[-overlap:].clone() if overlap else None
        del new_frames
        if log_progress:
            debug.log(f'{log_prefix}Chunk {chunk_idx}/{total_chunks}: {len(frames) - context_count} new frames',
                      category='generation', force=True)
        drop = context_count
        with closing(_iter_processed_batches(frames, chunk_args, device_id, debug, runner_cache)) as outputs:
            for result in outputs:
                skip = min(drop, len(result))
                drop -= skip
                if skip < len(result):
                    yield result[skip:]
                del result
        del frames
        recorder.record('chunk_complete', chunk=chunk_idx, input_frames=frames_read)


def _save_image_bgr(frame_np: np.ndarray, file_path: str) -> None:
    """
    Save a single RGB(A) uint8 frame to disk, converting to BGR(A) for OpenCV.
    
    Args:
        frame_np: Frame as uint8 numpy array [H, W, C] where C is 3 (RGB) or 4 (RGBA)
        file_path: Output file path
    """
    if frame_np.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGRA)
    else:
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(file_path), frame_bgr):
        raise OSError(f'Cannot write image: {file_path}')


def save_frames_to_video(
    frames_tensor: torch.Tensor, 
    output_path: str, 
    fps: float = 30.0,
    writer: Optional[cv2.VideoWriter] = None,
    video_backend: str = "opencv",
    use_10bit: bool = False
) -> Optional[cv2.VideoWriter]:
    """
    Save frames tensor to MP4 video file.
    
    Converts one frame at a time from normalized floating point to uint8
    (or uint16 RGB transport for FFmpeg 10-bit), using BGR only for OpenCV. Supports streaming mode where
    an existing writer is passed and kept open for subsequent chunks.
    
    Args:
        frames_tensor: Frames in format [T, H, W, C], Float32, range [0,1]
        output_path: Output video file path (directory created if doesn't exist)
        fps: Frames per second for output video (default: 30.0)
        writer: Existing VideoWriter for streaming (if None, creates new one)
    
    Returns:
        VideoWriter if streaming mode (caller must close), None if standalone mode
    
    Raises:
        ValueError: If video writer cannot be initialized
    """
    T, H, W, C = frames_tensor.shape
    
    if writer is None:
        debug.log(f"Saving {T} frames to video: {output_path} (backend={video_backend})", category="file")
        os.makedirs(Path(output_path).parent, exist_ok=True)
        if video_backend == "ffmpeg":
            writer = FFMPEGVideoWriter(output_path, W, H, fps, use_10bit)
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        if not writer.isOpened():
            writer.release()
            raise ValueError(f"Cannot create video writer for: {output_path}")
    
    try:
        for frame in frames_tensor:
            pixels = frame_pixels(frame, high_bit_depth=use_10bit and video_backend == 'ffmpeg')
            if video_backend == 'ffmpeg':
                writer.write_rgb(pixels)
            else:
                writer.write(cv2.cvtColor(pixels[..., :3], cv2.COLOR_RGB2BGR))
    except BaseException:
        if hasattr(writer, 'abort'):
            writer.abort()
        else:
            writer.release()
        raise

    return writer  # Caller always closes


def save_frames_to_image(
    frames_tensor: torch.Tensor, 
    output_dir: str, 
    base_name: str,
    start_index: int = 0
) -> int:
    """
    Save frames tensor as sequential PNG image files.
    
    Each frame saved as {base_name}_{index:0Nd}.png with zero-padded indices.
    Converts Float32 [0,1] to uint8 [0,255] and RGB(A) to BGR(A) for OpenCV.
    
    Args:
        frames_tensor: Frames in format [T, H, W, C], Float32, range [0,1]
        output_dir: Directory to save PNG files (created if doesn't exist)
        base_name: Base name for output files (e.g., "frame" → "frame_00000.png")
        start_index: Starting index for filenames (for streaming continuation)
    
    Returns:
        Number of frames saved
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total = frames_tensor.shape[0]
    
    if start_index == 0:
        debug.log(f"Saving {total} frames as PNGs to directory: {output_dir}", category="file")
    digits = 6  # Supports up to 999,999 frames (~11.5 hours at 24fps)

    for idx, frame in enumerate(frames_tensor):
        filename = f"{base_name}_{start_index + idx:0{digits}d}.png"
        file_path = os.path.join(output_dir, filename)
        _save_image_bgr(frame_pixels(frame), file_path)
        if debug.enabled and (idx + 1) % 100 == 0:
            debug.log(f"Saved {idx + 1}/{total} images", category="file")

    debug.log(f"Saved {total} images to '{output_dir}'", category="success")
    return total


# =============================================================================
# Core Processing Logic
# =============================================================================

def _process_frames_core(frames_tensor, args, device_id, debug, runner_cache=None):
    """Collect batches for a single image into its final image tensor."""
    with closing(_iter_processed_batches(frames_tensor, args, device_id, debug, runner_cache)) as outputs:
        batches = list(outputs)
    return torch.cat(batches) if batches else torch.empty(0, dtype=torch.bfloat16)


def _iter_processed_batches(frames_tensor, args, device_id, debug, runner_cache=None):
    inference_device = _device_id_to_name(device_id)
    session = runner_cache or InferenceSession(debug=debug)
    cache_dit, cache_vae = args.cache_dit, args.cache_vae
    dit_offload = _parse_offload_device(args.dit_offload_device, 'cuda', cache_dit)
    vae_offload = _parse_offload_device(args.vae_offload_device, 'cuda', cache_vae)
    tensor_offload = _parse_offload_device(args.tensor_offload_device, 'cuda')
    ctx = setup_generation_context(dit_device=inference_device, vae_device=inference_device,
        dit_offload_device=dit_offload, vae_offload_device=vae_offload,
        tensor_offload_device=tensor_offload, debug=debug)
    recorder = MemoryRecorder(args.memory_report, inference_device)
    # Build torch compile args
    torch_compile_args_dit = None
    torch_compile_args_vae = None
    if args.compile_dit:
        torch_compile_args_dit = {
            "backend": args.compile_backend,
            "mode": args.compile_mode,
            "fullgraph": args.compile_fullgraph,
            "dynamic": args.compile_dynamic,
            "dynamo_cache_size_limit": args.compile_dynamo_cache_size_limit,
            "dynamo_recompile_limit": args.compile_dynamo_recompile_limit,
        }
    if args.compile_vae:
        torch_compile_args_vae = {
            "backend": args.compile_backend,
            "mode": args.compile_mode,
            "fullgraph": args.compile_fullgraph,
            "dynamic": args.compile_dynamic,
            "dynamo_cache_size_limit": args.compile_dynamo_cache_size_limit,
            "dynamo_recompile_limit": args.compile_dynamo_recompile_limit,
        }
    
    # Prepare runner with caching support
    model_dir = args.model_dir if args.model_dir is not None else get_base_cache_dir()
    
    success = False
    previous_recorder = getattr(debug, 'external_memory_recorder', False)
    debug.external_memory_recorder = bool(args.memory_report)
    try:
        with recorder.phase("setup"):
            runner, cache_context = prepare_runner(
                dit_model=args.dit_model,
                vae_model=DEFAULT_VAE,
                model_dir=model_dir,
                debug=debug,
                ctx=ctx,
                dit_cache=cache_dit,
                vae_cache=cache_vae,
                session=session,
                block_swap_config={
                    'blocks_to_swap': args.blocks_to_swap,
                    'swap_io_components': args.swap_io_components,
                    'offload_device': dit_offload,
                },
                encode_tiled=args.vae_encode_tiled,
                encode_tile_size=(args.vae_encode_tile_size, args.vae_encode_tile_size),
                encode_tile_overlap=(args.vae_encode_tile_overlap, args.vae_encode_tile_overlap),
                decode_tiled=args.vae_decode_tiled,
                decode_tile_size=(args.vae_decode_tile_size, args.vae_decode_tile_size),
                decode_tile_overlap=(args.vae_decode_tile_overlap, args.vae_decode_tile_overlap),
                tile_debug=args.tile_debug.lower() if args.tile_debug else "false",
                attention_mode=args.attention_mode,
                torch_compile_args_dit=torch_compile_args_dit,
                torch_compile_args_vae=torch_compile_args_vae
            )
    
            cache_context.clear()

            # Preload text embeddings before Phase 1 to avoid sync stall in Phase 2
            ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], ctx['compute_dtype'], debug)
            debug.log("Loaded text embeddings for DiT", category="dit")

        # Compute generation info and log start (handles prepending internally)
        frames_tensor, gen_info = compute_generation_info(
            ctx=ctx,
            images=frames_tensor,
            resolution=args.resolution,
            max_resolution=args.max_resolution,
            batch_size=args.batch_size,
            uniform_batch_size=args.uniform_batch_size,
            seed=args.seed,
            prepend_frames=args.prepend_frames,
            temporal_overlap=args.temporal_overlap,
            debug=debug
        )
        log_generation_start(gen_info, debug)
    
        # Phase 1: Encode
        with recorder.phase("encode"):
            ctx = encode_all_batches(
                runner, ctx=ctx, images=frames_tensor,
                debug=debug,
                batch_size=args.batch_size,
                uniform_batch_size=args.uniform_batch_size,
                seed=args.seed,
                progress_callback=None,
                temporal_overlap=args.temporal_overlap,
                resolution=args.resolution,
                max_resolution=args.max_resolution,
                input_noise_scale=args.input_noise_scale,
                color_correction=args.color_correction
            )
    
        # Phase 2: Upscale
        with recorder.phase("upscale"):
            ctx = upscale_all_batches(
                runner, ctx=ctx, debug=debug, progress_callback=None,
                seed=args.seed,
                latent_noise_scale=args.latent_noise_scale,
                cache_model=cache_dit
            )
    
        # CPU postprocessing avoids co-residency with the active VAE on CUDA.
        ctx['postprocess_device'] = torch.device('cpu')
        drop = args.prepend_frames
        with recorder.phase('decode_postprocess_write'):
            with closing(iter_output_batches(runner, ctx, debug, args.color_correction, cache_vae)) as outputs:
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
        if success and runner_cache is not None:
            session.finish(cache_dit, cache_vae)
        else:
            session.close()










# =============================================================================
# Argument Parsing
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for SeedVR2 CLI.
    
    Configures all available options including model selection, processing parameters,
    memory optimization settings, and output configuration.
    
    Returns:
        Parsed arguments namespace with all CLI parameters
    
    Note:
        - Default model directory resolves to the repo's models/SEEDVR2 if not specified
    """
    
    # Get the actual invocation path for usage examples
    invocation = sys.argv[0]
    
    # Multi-line usage examples for --help
    usage_examples = f"""
Examples:

  Basic image upscaling:
    python {invocation} image.jpg

  Basic video upscaling with temporal consistency:
    python {invocation} video.mp4 --resolution 720 --batch_size 33
    
  Streaming mode for long videos with 10-bit video output (requires FFMPEG):
    python {invocation} long_video.mp4 --resolution 1080 --batch_size 33 --chunk_size 330 --temporal_overlap 3 --video_backend ffmpeg --10bit

  Memory-optimized for low VRAM (8GB):
    python {invocation} image.png --dit_model seedvr2_ema_3b-Q8_0.gguf --blocks_to_swap 32 --swap_io_components --dit_offload_device cpu --vae_offload_device cpu
    
  High resolution with VAE tiling:
    python {invocation} video.mp4 --resolution 1440 --batch_size 31 --uniform_batch_size --temporal_overlap 3 --vae_encode_tiled --vae_decode_tiled
    
  Batch directory processing:
    python {invocation} media_folder/ --output processed/ --cache_dit --cache_vae --dit_offload_device cpu --vae_offload_device cpu --resolution 1080 --max_resolution 1920
"""
    
    parser = argparse.ArgumentParser(
        description="SeedVR2 Video Upscaler - CLI for high-quality image/video upscaling and batch processing",
        epilog=usage_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False
    )
    
    # Input/Output
    io_group = parser.add_argument_group('Input/Output options')
    io_group.add_argument("input", type=str,
                        help="Input: video file (.mp4, .avi, etc.), image file (.png, .jpg, etc.), or directory")
    io_group.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto-generated in 'output/' directory)")
    io_group.add_argument("--output_format", type=str, default=None, choices=["mp4", "png", None],
                        help="Output format: 'mp4' (video) or 'png' (image sequence). Default: auto-detect from input type")
    io_group.add_argument("--video_backend", type=str, default="opencv", choices=["opencv", "ffmpeg"],
                        help="Video encoder backend: 'opencv' (default) or 'ffmpeg' (requires ffmpeg in PATH)")
    io_group.add_argument("--10bit", dest="use_10bit", action="store_true",
                        help="Save 10-bit video with x265 codec (reduces banding). Without this flag, "
                         "ffmpeg uses x264 for maximum compatibility. Requires --video_backend ffmpeg")
    io_group.add_argument("--model_dir", type=str, default=None,
                        help="Model directory (default: $SEEDVR2_MODEL_DIR or repo/models/SEEDVR2)")
    
    # Model Selection
    model_group = parser.add_argument_group('Model selection')
    model_group.add_argument("--dit_model", type=str, default=DEFAULT_DIT,
                        choices=get_available_dit_models(),
                        help="DiT model to use. Options: 3B (fp16/fp8/GGUF) or 7B (fp16/fp8/GGUF). Default: 3B FP8")
    
    # Processing Parameters
    process_group = parser.add_argument_group('Processing parameters')
    process_group.add_argument("--resolution", type=int, default=1080,
                        help="Target short-side resolution in pixels (default: 1080)")
    process_group.add_argument("--max_resolution", type=int, default=0,
                        help="Maximum resolution for any edge. Scales down if exceeded. 0 = no limit (default: 0)")
    process_group.add_argument("--batch_size", type=int, default=5,
                        help="Frames per batch (must follow 4n+1: 1, 5, 9, 13, 17, 21,...). "
                         "Ideally matches shot length for best temporal consistency. Higher values improve "
                         "quality and speed but require more VRAM. Default: 5")
    process_group.add_argument("--uniform_batch_size", action="store_true",
                        help="Pad final batch to match batch_size. Prevents temporal artifacts caused by small "
                         "final batches. Add extra compute but recommended for optimal quality.")
    process_group.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    process_group.add_argument("--skip_first_frames", type=int, default=0,
                        help="Skip N initial frames (default: 0)")
    process_group.add_argument("--load_cap", type=int, default=0,
                        help="Load maximum N frames from video. 0 = load all (default: 0)")
    process_group.add_argument("--chunk_size", type=int, default=0,
                        help="Maximum input frames per chunk. 0 = automatic finite window (at most 33); memory planning may reduce it")
    process_group.add_argument("--prepend_frames", type=int, default=0,
                        help="Prepend N reversed frames to reduce start artifacts (auto-removed). Default: 0")
    process_group.add_argument("--temporal_overlap", type=int, default=0,
                        help="Frames to overlap between batches/GPUs for smooth blending (default: 0)")
    
    # Quality Control
    quality_group = parser.add_argument_group('Quality control')
    quality_group.add_argument("--color_correction", type=str, default="lab", 
                    choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                    help="Color correction method: 'lab' (perceptual color matching, recommended), 'wavelet' (frequency-based), "
                    "'wavelet_adaptive' (wavelet + saturation correction), 'hsv' (hue-conditional), 'adain' (statistical transfer), "
                    "'none' (disabled) (default: lab)")
    quality_group.add_argument("--input_noise_scale", type=float, default=0.0,
                        help="Input noise injection scale (0.0-1.0). Adds variation to input images (default: 0.0)")
    quality_group.add_argument("--latent_noise_scale", type=float, default=0.0,
                        help="Latent noise injection scale (0.0-1.0). Adds variation to latent space (default: 0.0)")
    
    # Device Management
    device_group = parser.add_argument_group('Device management')
    device_group.add_argument("--cuda_device", type=int, default=None,
                    help="One visible CUDA device index; default: device 0")
    device_group.add_argument("--dit_offload_device", type=str, default="cpu",
                        help="DiT offload device when idle: 'none' (keep on selected GPU) or 'cpu' (offload to RAM). "
                             "Frees VRAM between phases. Required for BlockSwap. Default: cpu")
    device_group.add_argument("--vae_offload_device", type=str, default="cpu",
                        help="VAE offload device when idle: 'none' or 'cpu'. Frees VRAM between phases. Default: cpu")
    device_group.add_argument("--tensor_offload_device", type=str, default="cpu",
                        help="Intermediate tensor storage: 'cpu' (recommended) or 'none' (keep on selected GPU). Default: cpu")
    
    memory_group = parser.add_argument_group('Memory budgets (GiB; planning estimates, not hard limits)')
    memory_group.add_argument('--ram_budget_gb', type=float, default=0, help='Total free RAM allowance; 0 = 80%% of available RAM')
    memory_group.add_argument('--vram_budget_gb', type=float, default=0, help='Free VRAM allowance; 0 = 90%% of available VRAM')
    memory_group.add_argument('--cache_ram_gb', type=float, default=None, help='CPU model cache cap; default = one quarter of RAM allowance, 0 disables retention')
    memory_group.add_argument('--cache_vram_gb', type=float, default=0, help='GPU model cache cap; default 0')
    memory_group.add_argument('--memory_report', type=str, default=None, help='Append phase RSS/CUDA peaks and retry events to JSONL')
    memory_group.add_argument('--oom_retries', type=int, default=3, help='File restarts with smaller chunk/batch/tile sizes after OOM; 0 disables retries')

    # Memory Optimization (BlockSwap)
    blockswap_group = parser.add_argument_group('Memory optimization (BlockSwap)')
    blockswap_group.add_argument("--blocks_to_swap", type=int, default=0,
                        help="Requested minimum blocks to swap for VRAM savings. 0-32 (3B) or 0-36 (7B). "
                             "The planner/retry policy may increase it. Default: 0")
    blockswap_group.add_argument("--swap_io_components", action="store_true",
                        help="Offload DiT I/O layers for extra VRAM savings. Requires --dit_offload_device")
    
    # VAE Tiling
    vae_group = parser.add_argument_group('VAE tiling (for high resolution upscale)')
    vae_group.add_argument("--vae_encode_tiled", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable VAE encode tiling to reduce VRAM during encoding (default: enabled)")
    vae_group.add_argument("--vae_encode_tile_size", type=int, default=1024,
                        help="VAE encode tile size in pixels (default: 1024). Applied to both height and width. Only used if --vae_encode_tiled is set")
    vae_group.add_argument("--vae_encode_tile_overlap", type=int, default=128,
                        help="VAE encode tile overlap in pixels (default: 128). Reduces visible seams between tiles. Only used if --vae_encode_tiled is set")
    vae_group.add_argument("--vae_decode_tiled", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable VAE decode tiling to reduce VRAM during decoding (default: enabled)")
    vae_group.add_argument("--vae_decode_tile_size", type=int, default=1024,
                        help="VAE decode tile size in pixels (default: 1024). Applied to both height and width. Only used if --vae_decode_tiled is set")
    vae_group.add_argument("--vae_decode_tile_overlap", type=int, default=128,
                        help="VAE decode tile overlap in pixels (default: 128). Reduces visible seams between tiles. Only used if --vae_decode_tiled is set")
    vae_group.add_argument("--tile_debug", type=str, default="false", choices=["false", "encode", "decode"],
                        help="Visualize tiles: 'false' (default), 'encode', or 'decode'")
    
    # Performance
    perf_group = parser.add_argument_group('Performance optimization')
    perf_group.add_argument("--attention_mode", type=str, default="sdpa",
                        choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"],
                        help="Attention backend: 'sdpa' (default), 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3' (Blackwell GPUs)")
    perf_group.add_argument("--compile_dit", action="store_true", 
                        help="Enable torch.compile for DiT model (20-40%% speedup, requires PyTorch 2.0+ and Triton)")
    perf_group.add_argument("--compile_vae", action="store_true",
                        help="Enable torch.compile for VAE model (15-25%% speedup, requires PyTorch 2.0+ and Triton)")
    perf_group.add_argument("--compile_backend", type=str, default="inductor", choices=["inductor", "cudagraphs"],
                        help="Compilation backend: 'inductor' (full optimization with Triton) or 'cudagraphs' (lightweight, no kernel optimization) (default: inductor)")
    perf_group.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="Optimization level: 'default' (fast compilation), 'reduce-overhead' (lower overhead), 'max-autotune' (best runtime, slow compilation), "
                        "'max-autotune-no-cudagraphs' (like max-autotune without cudagraphs) (default: default)")
    perf_group.add_argument("--compile_fullgraph", action="store_true",
                        help="Compile entire model as single graph (faster but less flexible). May fail with dynamic shapes (default: False)")
    perf_group.add_argument("--compile_dynamic", action="store_true",
                        help="Handle varying input shapes without recompilation. Useful for different resolutions/batch sizes (default: False)")
    perf_group.add_argument("--compile_dynamo_cache_size_limit", type=int, default=64,
                        help="Max cached compiled versions per function. Increase when using many different input shapes. Higher uses more memory (default: 64)")
    perf_group.add_argument("--compile_dynamo_recompile_limit", type=int, default=128,
                        help="Max recompilation attempts before fallback to eager mode. Safety limit to prevent compilation loops (default: 128)")
    
    # Model Caching (for batch processing)
    cache_group = parser.add_argument_group('Model caching (batch processing)')
    cache_group.add_argument("--cache_dit", action="store_true",
                        help="Keep DiT model in memory between files. Subject to cache memory caps; idle models default to CPU")
    cache_group.add_argument("--cache_vae", action="store_true",
                        help="Keep VAE model in memory between files. Subject to cache memory caps; idle models default to CPU")
    
    # Debugging
    debug_group = parser.add_argument_group('Debugging')
    debug_group.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging")
    
    # Auto-show help if no arguments provided
    if len(sys.argv) == 1:
        sys.argv.append('--help')

    args = parser.parse_args()
    for name in ('chunk_size', 'temporal_overlap', 'prepend_frames', 'skip_first_frames', 'load_cap',
                 'oom_retries', 'ram_budget_gb', 'vram_budget_gb', 'cache_vram_gb', 'cache_ram_gb'):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0):
            parser.error(f'--{name} must be finite and nonnegative')
    if args.batch_size < 1 or args.batch_size % 4 != 1:
        parser.error('--batch_size must be 4n+1 (1, 5, 9, ...)')
    if args.resolution < 1 or args.max_resolution < 0:
        parser.error('Resolution must be positive and max_resolution nonnegative')
    for phase in ('encode', 'decode'):
        size, overlap = getattr(args, f'vae_{phase}_tile_size'), getattr(args, f'vae_{phase}_tile_overlap')
        if size < 128 or size % 64 or overlap < 0 or overlap >= size:
            parser.error(f'VAE {phase} tile size must be a multiple of 64 >= 128; overlap must be in [0, size)')
    if args.use_10bit and args.video_backend != 'ffmpeg':
        parser.error('--10bit requires --video_backend ffmpeg')
    return args


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """
    Main entry point for SeedVR2 Video Upscaler CLI.
    
    Orchestrates the complete upscaling workflow:
        1. Parse and validate command-line arguments
        2. Extract frames from input video/image(s)
        3. Download required models if not cached
        4. Process frames on one selected GPU
        5. Save results as video or PNG sequence
        6. Report timing and FPS (calculated from total wall-clock time)
    
    Error handling:
        - Validates tile configuration before processing
        - Provides detailed error messages with traceback
        - Ensures proper cleanup on exit (VRAM automatically freed)
    
    Raises:
        SystemExit: On argument validation failure or processing error
    """
    # Parse arguments
    args = parse_arguments()
    try:
        validate_runtime()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    args.model_dir = os.path.abspath(os.path.expanduser(args.model_dir or get_base_cache_dir()))

    # Update debug instance with --debug flag
    debug.enabled = args.debug

    # print header
    debug.print_header()
    
    debug.log("Arguments:", category="setup")
    for key, value in vars(args).items():
        debug.log(f"{key}: {value}", category="none", indent_level=1)

    if args.vae_encode_tiled and args.vae_encode_tile_overlap >= args.vae_encode_tile_size:
        debug.log(f"VAE encode tile overlap ({args.vae_encode_tile_overlap}) must be smaller than tile size ({args.vae_encode_tile_size})", level="ERROR", category="vae", force=True)
        sys.exit(1)
    
    if args.vae_decode_tiled and args.vae_decode_tile_overlap >= args.vae_decode_tile_size:
        debug.log(f"VAE decode tile overlap ({args.vae_decode_tile_overlap}) must be smaller than tile size ({args.vae_decode_tile_size})", level="ERROR", category="vae", force=True)
        sys.exit(1)
    
    # Validate ffmpeg availability if selected
    if args.video_backend == "ffmpeg" and shutil.which("ffmpeg") is None:
        debug.log("--video_backend ffmpeg requires ffmpeg in PATH. Install ffmpeg or use --video_backend opencv", 
                 level="ERROR", category="setup", force=True)
        sys.exit(1)
    
    # Inform about caching defaults
    if args.cache_dit and args.dit_offload_device == "none":
        debug.log(
            "DiT caching enabled: Using default system memory (CPU) for offload. "
            "Set --dit_offload_device explicitly to use a different device.",
            category="cache", force=True
        )
    
    if args.cache_vae and args.vae_offload_device == "none":
        debug.log(
            "VAE caching enabled: Using default system memory (CPU) for offload. "
            "Set --vae_offload_device explicitly to use a different device.",
            category="cache", force=True
        )

    if args.debug:
        debug.log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'none')}", category="device")
        debug.log(f"torch.cuda.device_count(): {torch.cuda.device_count()}", category="device")
    
    runner_cache = InferenceSession(debug=debug)
    try:
        start_time = time.time()
        
        if torch.cuda.device_count() < 1:
            raise RuntimeError("No CUDA device is visible to the CLI")
        device_id = "0"
        if args.debug:
            debug.log(f"Using CUDA device: {_selected_cuda_device} (visible as cuda:0)", category="device")
        
        # Download models once before processing
        if not download_weight(dit_model=args.dit_model, vae_model=DEFAULT_VAE, model_dir=args.model_dir, debug=debug):
            debug.log("Failed to download required models. Check console output above.", level="ERROR", category="download", force=True)
            sys.exit(1)
        
        # Determine input type and process accordingly
        input_type = get_input_type(args.input)

        # Track total frames for FPS calculation (time tracked via start_time)
        total_frames_processed = 0
        
        # Track if output format was user-specified or auto-detected
        format_auto_detected = args.output_format is None
        
        if input_type == 'directory':
            media_files = get_media_files(args.input)
            if not media_files:
                debug.log(f"No video or image files found in directory: {args.input}", 
                        level="ERROR", category="file", force=True)
                sys.exit(1)
            
            debug.log(f"Found {len(media_files)} media files to process", category="file", force=True)
            
            for idx, file_path in enumerate(media_files, 1):
                # Visual separation between files (except before first file)
                if idx > 1:
                    debug.log("", category="none", force=True)
                    debug.log("━" * 60, category="none", force=True)
                    debug.log("", category="none", force=True)
                
                debug.log(f"Processing file {idx}/{len(media_files)}", category="generation", force=True)
                
                # Auto-detect format per file if not user-specified
                if format_auto_detected:
                    file_type = get_input_type(file_path)
                    file_output_format = "mp4" if file_type == "video" else "png"
                else:
                    file_output_format = args.output_format
                
                # Temporarily override args.output_format for this file
                original_format = args.output_format
                args.output_format = file_output_format
                
                # generate_output_path handles None gracefully with "outputs" default
                output_path = generate_output_path(file_path, file_output_format, args.output, 
                                   input_type=get_input_type(file_path), from_directory=True)
                
                # Process with explicit output path and runner cache
                frames = process_single_file(file_path, args, device_id, output_path,
                                            format_auto_detected=format_auto_detected,
                                            runner_cache=runner_cache, output_is_resolved=True)
                total_frames_processed += frames
                
                # Restore original format
                args.output_format = original_format

        elif input_type in ("video", "image"):
            # Auto-detect output format for single file if not specified
            if format_auto_detected:
                args.output_format = "mp4" if input_type == "video" else "png"
            
            frames = process_single_file(args.input, args, device_id, args.output,
                                        format_auto_detected=format_auto_detected,
                                        runner_cache=runner_cache)
            total_frames_processed += frames
        
        else:
            debug.log(f"Unsupported input type: {args.input}", level="ERROR", category="file", force=True)
            sys.exit(1)
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        debug.log("", category="none", force=True)
        debug.log(f"All upscaling processes completed successfully in {total_time:.2f}s", category="success", force=True)
        
        # Calculate and display FPS based on overall wall-clock time
        if total_time > 0 and total_frames_processed > 0:
            fps = total_frames_processed / total_time
            debug.log(f"Average FPS: {fps:.2f} frames/sec", category="timing", force=True)
        
    except Exception as e:
        debug.log(f"Error during processing: {e}", level="ERROR", category="generation", force=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        runner_cache.close()
        debug.log(f"Process {os.getpid()} terminating - VRAM will be automatically freed", category="cleanup", force=True)

        # print footer
        debug.print_footer()

if __name__ == "__main__":
    main()
