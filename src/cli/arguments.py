"""Argument parsing and validation, with no inference dependencies."""

import argparse
import math
import sys
from collections.abc import Sequence

from ..config import (
    AppConfig, BlockSwapConfig, CacheConfig, InferenceConfig,
    MemoryConfig, ModelConfig, OffloadConfig, OutputConfig, SamplingConfig,
    TileConfig, VAEConfig, VideoConfig,
)
from ..utils.model_registry import DEFAULT_DIT, get_available_dit_models


def create_parser(invocation: str | None = None) -> argparse.ArgumentParser:
    """Create and configure the CLI ArgumentParser."""
    if invocation is None:
        invocation = sys.argv[0]

    usage_examples = f"""
Examples:

  Basic image upscaling:
    python {invocation} image.jpg

  Basic video upscaling with temporal consistency:
    python {invocation} video.mp4 --resolution 720 --batch_size 33

  Streaming mode for long videos with 10-bit video output (requires FFMPEG):
    python {invocation} long_video.mp4 --resolution 1080 --batch_size 33 --chunk_size 330 --temporal_overlap 3 --video_backend ffmpeg --10bit

  Memory-optimized for low VRAM (8GB / Colab T4):
    python {invocation} image.png --dit_model seedvr2_ema_3b-Q8_0.gguf --blocks_to_swap 32 --swap_io_components --memory_policy staged
  High resolution with VAE tiling:
    python {invocation} video.mp4 --resolution 1440 --batch_size 33 --uniform_batch_size --temporal_overlap 3 --vae_encode_tiled --vae_decode_tiled

  Batch directory processing:
    python {invocation} media_folder/ --output processed/ --memory_policy auto --resolution 1080 --max_resolution 1920
"""

    parser = argparse.ArgumentParser(
        prog="seedvr2-cli",
        description="SeedVR2-cli - High-performance command-line image/video upscaler",
        epilog=usage_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False
    )

    _add_io_group(parser)
    _add_model_group(parser)
    _add_processing_group(parser)
    _add_quality_group(parser)
    _add_device_group(parser)
    _add_memory_group(parser)
    _add_blockswap_group(parser)
    _add_vae_group(parser)
    _add_perf_group(parser)
    _add_debug_group(parser)

    return parser


def _add_io_group(parser: argparse.ArgumentParser) -> None:
    io_group = parser.add_argument_group('Input/Output options')
    io_group.add_argument("input", type=str,
                        help="Input: video file (.mp4, .avi, etc.), image file (.png, .jpg, etc.), or directory")
    io_group.add_argument("--output", type=str, default=None,
                        help="Output path (default: beside the input with an _upscaled suffix)")
    io_group.add_argument("--output_format", type=str, default=None, choices=["mp4", "png", None],
                        help="Output format: 'mp4' (video) or 'png' (image sequence). Default: auto-detect from input type")
    io_group.add_argument("--video_backend", type=str, default="ffmpeg", choices=["opencv", "ffmpeg"],
                        help="Video encoder backend: 'ffmpeg' (default, recommended) or 'opencv'")
    io_group.add_argument("--10bit", dest="use_10bit", action="store_true",
                        help="Save 10-bit video with x265 codec (reduces banding). Without this flag, "
                             "ffmpeg uses x264 for maximum compatibility. Requires --video_backend ffmpeg")
    io_group.add_argument("--nvenc", action="store_true",
                        help="Use NVIDIA NVENC hardware-accelerated H.265 encoding (hevc_nvenc) with P7 HQ preset for high-speed, near-lossless output")
    io_group.add_argument("--model_dir", type=str, default=None,
                        help="Model directory (default: $SEEDVR2_MODEL_DIR or repo/models/SEEDVR2)")


def _add_model_group(parser: argparse.ArgumentParser) -> None:
    model_group = parser.add_argument_group('Model selection')
    model_group.add_argument("--dit_model", type=str, default=DEFAULT_DIT,
                        choices=get_available_dit_models(),
                        help="DiT model to use. Options: 3B (fp16/fp8/GGUF) or 7B (fp16/fp8/GGUF). Default: 3B FP8")


def _add_processing_group(parser: argparse.ArgumentParser) -> None:
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
                        help="Frames to overlap between batches/chunks for smooth blending (default: 0)")
    process_group.add_argument("--resume", action="store_true",
                        help="Resume upscaling a long video from existing chunk checkpoints (saves time on Colab T4 disconnects)")


def _add_quality_group(parser: argparse.ArgumentParser) -> None:
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


def _add_device_group(parser: argparse.ArgumentParser) -> None:
    device_group = parser.add_argument_group('Device management')
    device_group.add_argument("--cuda_device", type=int, default=None,
                    help="One visible CUDA device index; default: device 0")
    device_group.add_argument("--tensor_offload_device", type=str, default="cpu",
                        help="Intermediate tensor storage: 'cpu' (recommended) or 'none' (keep on selected GPU). Default: cpu")
    device_group.add_argument("--memory_policy", type=str, default="auto",
                        choices=["auto", "staged", "resident"],
                        help="Memory management policy: 'auto' (automatically chooses staged on T4/16GB or resident on 24GB+), "
                             "'staged' (swaps DiT and VAE between phases to eliminate OOM on <=16GB GPUs), "
                             "or 'resident' (keeps both models in VRAM for maximum speed). Default: auto")

def _add_memory_group(parser: argparse.ArgumentParser) -> None:
    memory_group = parser.add_argument_group('Memory diagnostics & recovery')
    memory_group.add_argument('--memory_report', type=str, default=None, help='Append phase RSS/CUDA peaks and retry events to JSONL')
    memory_group.add_argument('--oom_retries', type=int, default=3, help='File restarts with smaller chunk/batch/tile sizes after OOM; 0 disables retries')

def _add_blockswap_group(parser: argparse.ArgumentParser) -> None:
    blockswap_group = parser.add_argument_group('Memory optimization (BlockSwap)')
    blockswap_group.add_argument("--blocks_to_swap", type=int, default=0,
                        help="Requested minimum blocks to swap for VRAM savings. 0-32 (3B) or 0-36 (7B). "
                             "The planner/retry policy may increase it. Default: 0")
    blockswap_group.add_argument("--swap_io_components", action="store_true",
                        help="Offload DiT I/O layers for extra VRAM savings. Requires --dit_offload_device")


def _add_vae_group(parser: argparse.ArgumentParser) -> None:
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


def _add_perf_group(parser: argparse.ArgumentParser) -> None:
    perf_group = parser.add_argument_group('Performance optimization')
    perf_group.add_argument("--attention_mode", type=str, default="sparse_sage_attention",
                        choices=["sdpa", "flash_attn_2", "flash_attn_3", "sparse_sage_attention", "sageattn_2", "sageattn_3"],
                        help="Attention backend: 'sparse_sage_attention' (default, fast INT8 Triton kernel, zero compilation), 'flash_attn_2' (Ampere/Ada/Hopper/L4), 'flash_attn_3', or 'sdpa'")
    perf_group.add_argument("--quality_profile", type=str, default="balanced",
                        choices=["detail", "balanced", "throughput"],
                        help="Motion-aware dynamic budget profile: 'detail' (max fidelity), 'balanced' (default), or 'throughput' (fastest)")
    perf_group.add_argument("--motion_intensity", type=float, default=1.0,
                        help="Motion sensitivity multiplier for dynamic budget (default: 1.0)")
def _add_debug_group(parser: argparse.ArgumentParser) -> None:
    debug_group = parser.add_argument_group('Debugging')
    debug_group.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging")


def _validate_parsed_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate parsed command-line arguments, raising parser errors if invalid."""
    for name in ('chunk_size', 'temporal_overlap', 'prepend_frames', 'skip_first_frames', 'load_cap', 'oom_retries'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
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
    if args.cuda_device is not None and args.cuda_device < 0:
        parser.error("--cuda_device must be a non-negative CUDA device index")


def parse_arguments(argv: Sequence[str] | None = None) -> AppConfig:
    """
    Parse and validate command-line arguments for SeedVR2 CLI.

    Configures all available options including model selection, processing parameters,
    memory optimization settings, and output configuration.

    Returns:
        Validated, immutable application settings

    Note:
        - Default model directory resolves to the repo's models/SEEDVR2 if not specified
    """
    parser = create_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv or ['--help'])
    _validate_parsed_arguments(parser, args)
    return config_from_namespace(args)


def config_from_namespace(a: argparse.Namespace) -> AppConfig:
    """Translate validated CLI values once; downstream code only sees dataclasses."""
    return AppConfig(
        inference=InferenceConfig(
            model=ModelConfig(
                dit_model=a.dit_model,
                model_dir=a.model_dir,
            ),
            sampling=SamplingConfig(
                resolution=a.resolution,
                max_resolution=a.max_resolution,
                batch_size=a.batch_size,
                uniform_batch_size=a.uniform_batch_size,
                seed=a.seed,
                prepend_frames=a.prepend_frames,
                temporal_overlap=a.temporal_overlap,
                color_correction=a.color_correction,
                input_noise_scale=a.input_noise_scale,
                latent_noise_scale=a.latent_noise_scale,
            ),
            offload=OffloadConfig(
                dit="none" if a.memory_policy == "resident" else "cpu",
                vae="none" if a.memory_policy == "resident" else "cpu",
                tensor=a.tensor_offload_device,
            ),
            vae=VAEConfig(
                encode=TileConfig(
                    enabled=a.vae_encode_tiled,
                    size=a.vae_encode_tile_size,
                    overlap=a.vae_encode_tile_overlap,
                ),
                decode=TileConfig(
                    enabled=a.vae_decode_tiled,
                    size=a.vae_decode_tile_size,
                    overlap=a.vae_decode_tile_overlap,
                ),
                debug=a.tile_debug,
            ),

            cache=CacheConfig(),
            swap=BlockSwapConfig(
                blocks_to_swap=0 if a.memory_policy == "resident" else a.blocks_to_swap,
                swap_io_components=False if a.memory_policy == "resident" else a.swap_io_components,
            ),
            attention_mode=a.attention_mode,
            memory_policy=a.memory_policy,
            quality_profile=a.quality_profile,
            motion_intensity=a.motion_intensity,
        ),
        video=VideoConfig(
            chunk_size=a.chunk_size,
            skip_first_frames=a.skip_first_frames,
            load_cap=a.load_cap,
            resume=a.resume,
        ),
        memory=MemoryConfig(
            oom_retries=a.oom_retries,
            report=a.memory_report,
        ),
        output=OutputConfig(
            path=a.output,
            format=a.output_format,
            video_backend=a.video_backend,
            use_10bit=a.use_10bit,
            nvenc=a.nvenc,
        ),
        input=a.input,
        cuda_device=a.cuda_device if a.cuda_device is not None else 0,
        debug=a.debug,
    )
