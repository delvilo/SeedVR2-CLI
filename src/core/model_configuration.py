"""
Model Configuration and Setup for SeedVR2

This module orchestrates model configuration, caching, and runtime settings:
- Runner configuration and lifecycle management
- Model caching with validation and updates
- DiT and VAE model setup with meta device initialization
- Configuration change detection and updates
- torch.compile integration
- BlockSwap configuration
- VAE tiling configuration

Key Features:
- Unified runner configuration via configure_runner()
- Session-owned model caching with checkpoint identity validation
- Configuration change detection to avoid unnecessary reloads
- Meta device initialization for memory efficiency
- Dynamic configuration updates (BlockSwap, torch.compile, attention mode)
- Automatic cleanup of stale cache entries

Main Functions:
- configure_runner: Main entry point for runner setup
- apply_model_specific_config: Apply BlockSwap and torch.compile to models
- _setup_models: Setup DiT and VAE models (cached or new)
- _update_dit_config: Update DiT configuration on cached models
- _update_vae_config: Update VAE configuration on cached models

Cache Management:
- _create_new_runner: Create new runner with config
- _configure_runner_settings: Configure runner settings for tiling/compile/BlockSwap

Configuration Helpers:
- _configs_equal: Compare configurations for equality
- _describe_blockswap_config: Human-readable BlockSwap description
- _describe_compile_config: Human-readable torch.compile description
- _describe_attention_mode: Human-readable attention mode description
- _describe_tiling_config: Human-readable VAE tiling description
- _update_model_config: Generic config update handler with change detection

Model Setup:
- _setup_dit_model: Setup DiT model from cache or create new structure
- _setup_vae_model: Setup VAE model from cache or create new structure

torch.compile Support:
- _configure_torch_compile: Configure torch.compile settings
- _apply_torch_compile: Apply torch.compile to full model
- _apply_vae_submodule_compile: Apply torch.compile to VAE submodules
- _disable_compile_for_dynamic_modules: Mark dynamic modules no-compile
- _propagate_debug_to_modules: Propagate debug instance to submodules

BlockSwap Management:
- _handle_blockswap_change: Handle BlockSwap configuration changes

This module uses model_loader for actual weight loading operations.
"""

from .context import GenerationContext

from ..config import BlockSwapConfig

import os
import torch
from omegaconf import OmegaConf
from typing import Dict, Any, Optional, Tuple, Union, Callable, List

from .model_loader import (
    prepare_model_structure,
    script_directory
)
from .infer import VideoDiffusionInfer
from .session import InferenceSession, checkpoint_identity
from ..common.config import load_config
from ..models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from ..optimization.compatibility import (
    CompatibleDiT,
    TRITON_AVAILABLE,
    validate_attention_mode
)
from ..optimization.blockswap import is_blockswap_enabled, validate_blockswap_config, apply_block_swap_to_dit, cleanup_blockswap
from ..optimization.memory_manager import cleanup_dit, cleanup_vae
from ..utils.constants import find_model_file


def _configs_equal(config1: object, config2: object) -> bool:
    """
    Compare typed configuration objects or local metadata for equality.
    Handles None values properly.
    
    Args:
        config1: First configuration object (can be None)
        config2: Second configuration object (can be None)
        
    Returns:
        True if configs are equivalent, False otherwise
    """
    # Both None = equal
    if config1 is None and config2 is None:
        return True
    
    # One None, one not = different
    if (config1 is None) != (config2 is None):
        return False
    
    # Compare typed settings or local metadata
    return config1 == config2


def _describe_blockswap_config(config: Optional[BlockSwapConfig]) -> str:
    """
    Generate human-readable description of BlockSwap configuration.
    
    Args:
        config: BlockSwap configuration object
        
    Returns:
        Human-readable description string
    """
    if not is_blockswap_enabled(config):
        return "disabled"
    
    blocks_to_swap = config.blocks_to_swap
    swap_io_components = config.swap_io_components
    
    block_text = "block" if blocks_to_swap <= 1 else "blocks"
    parts = [f"{blocks_to_swap} {block_text}"]
    if swap_io_components:
        parts.append("I/O offload")
    
    return f"enabled ({', '.join(parts)})"



def _describe_attention_mode(attention_mode: Optional[str]) -> str:
    """
    Generate human-readable description of attention mode configuration.
    
    Args:
        attention_mode: Attention mode string ('sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3')
        
    Returns:
        Human-readable description string
    """
    if attention_mode is None:
        return "sdpa (default)"
    
    mode_descriptions = {
        'sparse_sage_attention': 'Sparse SageAttention (INT8 Triton)',
        'sdpa': 'PyTorch SDPA',
        'flash_attn_2': 'Flash Attention 2',
        'flash_attn_3': 'Flash Attention 3',
    }
    
    return mode_descriptions.get(attention_mode, attention_mode)
    

def _describe_tiling_config(encode_tiled: bool, encode_tile_size: Optional[Tuple[int, int]], 
                           encode_tile_overlap: Optional[Tuple[int, int]],
                           decode_tiled: bool, decode_tile_size: Optional[Tuple[int, int]], 
                           decode_tile_overlap: Optional[Tuple[int, int]]) -> str:
    """
    Generate human-readable description of VAE tiling configuration.
    
    Args:
        encode_tiled: Whether encode tiling is enabled
        encode_tile_size: Tile size for encoding
        encode_tile_overlap: Tile overlap for encoding
        decode_tiled: Whether decode tiling is enabled
        decode_tile_size: Tile size for decoding
        decode_tile_overlap: Tile overlap for decoding
        
    Returns:
        Human-readable description string
    """
    if not encode_tiled and not decode_tiled:
        return "disabled"
    
    parts = []
    if encode_tiled:
        parts.append(f"encode Tile: {encode_tile_size}, Overlap: {encode_tile_overlap}")
    if decode_tiled:
        parts.append(f"decode Tile: {decode_tile_size}, Overlap: {decode_tile_overlap}")
    
    return "; ".join(parts)

    
def _is_meta_device_model(model: torch.nn.Module) -> bool:
    """Check if model parameters are on meta device (not materialized yet)."""
    try:
        param_device = next(model.parameters()).device
        return param_device.type == 'meta'
    except StopIteration:
        return False


def _sync_model_config_attributes(
    runner: 'VideoDiffusionInfer',
    model: torch.nn.Module,
    new_configs: Dict[str, Any],
    cached_config_attrs: Dict[str, str],
    model_config_attrs: Optional[Dict[str, str]] = None
) -> None:
    """Sync configuration attributes to both runner and model objects."""
    for config_name, attr_name in cached_config_attrs.items():
        setattr(runner, attr_name, new_configs.get(config_name))

        if model_config_attrs and config_name in model_config_attrs:
            model_attr_name = model_config_attrs[config_name]
        else:
            model_attr_name = f'_config_{config_name.split("_")[-1]}'
        setattr(model, model_attr_name, new_configs.get(config_name))


def _format_config_display_name(config_name: str) -> str:
    """Format config property name for display in logs."""
    if config_name == 'torch_compile':
        return 'torch.compile'
    elif config_name == 'block_swap':
        return 'BlockSwap'
    elif config_name == 'attention_mode':
        return 'Attention Mode'
    return config_name.replace('_', ' ').title()


def _describe_config_change(
    config_name: str,
    cached_value: Any,
    new_value: Any,
    desc_func: Callable
) -> str:
    """Generate human-readable change string for a specific config attribute."""
    if config_name == 'tiling':
        old_cfg = cached_value or {}
        new_cfg = new_value or {}
        old_desc = desc_func(
            old_cfg.get('encode_tiled', False),
            old_cfg.get('encode_tile_size'),
            old_cfg.get('encode_tile_overlap'),
            old_cfg.get('decode_tiled', False),
            old_cfg.get('decode_tile_size'),
            old_cfg.get('decode_tile_overlap')
        )
        new_desc = desc_func(
            new_cfg.get('encode_tiled', False),
            new_cfg.get('encode_tile_size'),
            new_cfg.get('encode_tile_overlap'),
            new_cfg.get('decode_tiled', False),
            new_cfg.get('decode_tile_size'),
            new_cfg.get('decode_tile_overlap')
        )
    else:
        old_desc = desc_func(cached_value)
        new_desc = desc_func(new_value)

    display_name = _format_config_display_name(config_name)
    return f"{display_name}: {old_desc} → {new_desc}"


def _detect_config_changes(
    runner: 'VideoDiffusionInfer',
    new_configs: Dict[str, Any],
    cached_config_attrs: Dict[str, str],
    config_describers: Dict[str, Callable]
) -> Tuple[Dict[str, bool], List[str]]:
    """
    Detect configuration changes between runner's cached attributes and new configs.

    Returns:
        Tuple of (changes_detected dict, config_changes description list)
    """
    config_changes = []
    changes_detected = {}

    for config_name, attr_name in cached_config_attrs.items():
        cached_value = getattr(runner, attr_name, None)
        new_value = new_configs.get(config_name)
        changed = not _configs_equal(cached_value, new_value)
        changes_detected[config_name] = changed

        if changed:
            desc_func = config_describers.get(config_name)
            if desc_func:
                change_desc = _describe_config_change(config_name, cached_value, new_value, desc_func)
                config_changes.append(change_desc)

    return changes_detected, config_changes



def _update_model_config(
    runner: 'VideoDiffusionInfer',
    model_attr: str,
    model_type: str,
    new_configs: Dict[str, Any],
    cached_config_attrs: Dict[str, str],
    model_config_attrs: Dict[str, str],
    config_describers: Dict[str, Callable],
    special_handlers: Optional[Dict[str, Callable]] = None,
    debug: Optional['Debug'] = None
) -> bool:
    """
    Update model configuration uniformly for DiT or VAE.
    
    This generic function handles config comparison, logging, and attribute updates
    for both DiT and VAE models, reducing code duplication.
    
    Args:
        runner: VideoDiffusionInfer instance
        model_attr: Attribute name for the model ('dit' or 'vae')
        model_type: Model type string for logging ('DiT' or 'VAE')
        new_configs: Dict mapping config names to new values
                    e.g., {'torch_compile': ..., 'block_swap': ...}
        cached_config_attrs: Dict mapping config names to runner attribute names
                           e.g., {'torch_compile': '_dit_compile_args', ...}
        model_config_attrs: Dict mapping config names to model storage attribute names.
                            e.g., {'torch_compile': '_config_compile', ...}
        config_describers: Dict mapping config names to description functions
                          e.g., {'torch_compile': _describe_compile_config, ...}
        special_handlers: Optional dict of config-specific handlers
                         e.g., {'block_swap': handle_blockswap_change}
        debug: Debug instance
        
    Returns:
        True if successful
    """
    model = getattr(runner, model_attr)
    
    # Check if model is on meta device (not materialized yet)
    if _is_meta_device_model(model):
        _sync_model_config_attributes(runner, model, new_configs, cached_config_attrs, model_config_attrs)
        return True
    
    # Get cached configurations and check what changed
    changes_detected, config_changes = _detect_config_changes(
        runner, new_configs, cached_config_attrs, config_describers
    )
    
    # If nothing changed, reuse model as-is
    if not any(changes_detected.values()):
        if debug:
            debug.log(f"{model_type} configuration unchanged, reusing cached model", category=model_type.lower())
        _sync_model_config_attributes(runner, model, new_configs, cached_config_attrs, model_config_attrs)
        return True
    
    # Log configuration changes
    if debug:
        debug.log(f"{model_type} configuration changed:", category=model_type.lower(), force=True)
        for change in config_changes:
            debug.log(f"{change}", category=model_type.lower(), force=True, indent_level=1)
    

    # Execute special handlers for config-specific logic
    if special_handlers:
        for config_name, handler in special_handlers.items():
            if changes_detected.get(config_name, False):
                handler(runner, cached_config_attrs[config_name], 
                       new_configs[config_name], debug)
    
    # Update config attributes on runner and model
    setattr(runner, model_attr, model)
    _sync_model_config_attributes(runner, model, new_configs, cached_config_attrs, model_config_attrs)
    
    # Mark that configs need to be applied
    config_needs_app_attr = f'_{model_attr}_config_needs_application'
    setattr(runner, config_needs_app_attr, True)
    
    return True


def _handle_blockswap_change(
    runner: 'VideoDiffusionInfer',
    attr_name: str,
    new_config: Optional[BlockSwapConfig],
    debug: Optional['Debug'] = None
) -> None:
    """
    Handle BlockSwap-specific configuration changes with proper cleanup.
    
    Called by _update_model_config when BlockSwap configuration changes are detected.
    Manages transition between BlockSwap states (enabled/disabled/reconfigured) with
    proper cleanup to avoid memory leaks and state corruption.
    
    Args:
        runner: VideoDiffusionInfer instance with DiT model
        attr_name: Runner attribute name storing cached config (e.g., '_dit_block_swap_config')
        new_config: New BlockSwap configuration dict or None to disable
        debug: Debug instance for logging
    """
    cached_config = getattr(runner, attr_name, None)
    
    # Determine BlockSwap status from configs
    had_blockswap = is_blockswap_enabled(cached_config)
    has_blockswap = is_blockswap_enabled(new_config)
    
    # If old config had BlockSwap features, clean them up if disabling or changing configuration
    if had_blockswap:
        if not has_blockswap or cached_config != new_config:
            if debug:
                debug.log("Cleaning up previous BlockSwap configuration", category="blockswap")
            cleanup_blockswap(runner=runner, keep_state_for_cache=False)
    
    # Mark as inactive so the new config can be applied
    runner._blockswap_active = False


def _update_dit_config(
    runner: 'VideoDiffusionInfer',
    block_swap_config: Optional[BlockSwapConfig],
    attention_mode: Optional[str],
    debug: Optional['Debug'] = None
) -> bool:
    """Update DiT model configuration when reusing cached model."""
    return _update_model_config(
        runner=runner,
        model_attr='dit',
        model_type='DiT',
        new_configs={
            'block_swap': block_swap_config,
            'attention_mode': attention_mode
        },
        cached_config_attrs={
            'block_swap': '_dit_block_swap_config',
            'attention_mode': '_dit_attention_mode'
        },
        model_config_attrs={
            'block_swap': '_config_swap',
            'attention_mode': '_config_attn'
        },
        config_describers={
            'block_swap': _describe_blockswap_config,
            'attention_mode': _describe_attention_mode
        },
        special_handlers={
            'block_swap': _handle_blockswap_change
        },
        debug=debug
    )


def _update_vae_config(
    runner: 'VideoDiffusionInfer',
    debug: Optional['Debug'] = None
) -> bool:
    """Update VAE model configuration when reusing cached model."""
    new_tiling_config = getattr(runner, '_new_vae_tiling_config', None)
    return _update_model_config(
        runner=runner,
        model_attr='vae',
        model_type='VAE',
        new_configs={
            'tiling': new_tiling_config
        },
        cached_config_attrs={
            'tiling': '_vae_tiling_config'
        },
        model_config_attrs={
            'tiling': '_config_tiling'
        },
        config_describers={
            'tiling': _describe_tiling_config
        },
        special_handlers=None,
        debug=debug
    )


def _create_new_runner(
    dit_model: str,
    vae_model: str,
    base_cache_dir: str,
    debug: Optional['Debug'] = None
) -> VideoDiffusionInfer:
    """
    Create a new VideoDiffusionInfer runner instance from scratch.
    
    Loads appropriate configuration file based on model size (3B or 7B), creates
    runner instance, and initializes with default settings. Called when no cached
    session runner is available or when checkpoint identity changes.
    
    Args:
        dit_model: DiT model filename (determines config selection)
                  - Contains "7b" → loads configs_7b/main.yaml
                  - Otherwise → loads configs_3b/main.yaml
        vae_model: VAE model filename (stored for reference, not used in config selection)
        base_cache_dir: Base directory for model files (not used directly but passed for context)
        debug: Debug instance for logging and timing
        
    Returns:
        VideoDiffusionInfer: Newly created runner with:
            - Loaded OmegaConf configuration
            - Initialized diffusion sampler and schedule
            - Config set to mutable (readonly=False)
            - No models loaded (structure only)
    """
    debug.log(f"Creating new runner: DiT={dit_model}, VAE={vae_model}", 
             category="runner", force=True)
    
    debug.start_timer("config_load")
    config_path = os.path.join(script_directory, 
                              './configs_7b' if "7b" in dit_model else './configs_3b', 
                              'main.yaml')
    config = load_config(config_path)
    debug.end_timer("config_load", "Config loading")
    
    debug.start_timer("runner_video_infer")
    runner = VideoDiffusionInfer(config, debug)
    OmegaConf.set_readonly(runner.config, False)
    debug.end_timer("runner_video_infer", "Video diffusion inference runner initialization")
    
    return runner


def configure_runner(
    dit_model: str,
    vae_model: str,
    base_cache_dir: str,
    debug: 'Debug',
    ctx: GenerationContext,
    dit_cache: bool = False,
    vae_cache: bool = False,
    session: Optional[InferenceSession] = None,
    block_swap_config: Optional[BlockSwapConfig] = None,
    encode_tiled: bool = False,
    encode_tile_size: Optional[Tuple[int, int]] = None,
    encode_tile_overlap: Optional[Tuple[int, int]] = None,
    decode_tiled: bool = False,
    decode_tile_size: Optional[Tuple[int, int]] = None,
    decode_tile_overlap: Optional[Tuple[int, int]] = None,
    tile_debug: str = "false",
    attention_mode: str = 'sparse_sage_attention',

) -> VideoDiffusionInfer:
    """
    Configure VideoDiffusionInfer runner with model loading and settings.
    
    Handles model changes and caching logic with independent DiT/VAE caching support.
    
    Args:
        dit_model: DiT model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
        vae_model: VAE model filename (e.g., "ema_vae_fp16.safetensors")
        base_cache_dir: Base directory containing model files
        debug: Debug instance for logging (required)
        ctx: Generation context from setup_generation_context
        dit_cache: Whether to cache DiT model between runs
        vae_cache: Whether to cache VAE model between runs
        session: InferenceSession owning the runner and retained models
        block_swap_config: Optional BlockSwap configuration for DiT memory optimization
        encode_tiled: Enable tiled encoding to reduce VRAM during VAE encoding
        encode_tile_size: Tile size for encoding (height, width)
        encode_tile_overlap: Tile overlap for encoding (height, width)
        decode_tiled: Enable tiled decoding to reduce VRAM during VAE decoding
        decode_tile_size: Tile size for decoding (height, width)
        decode_tile_overlap: Tile overlap for decoding (height, width)
        tile_debug: Tile visualization mode (false/encode/decode)
        attention_mode: Attention computation backend ('sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3')
        torch_compile_args_dit: Optional torch.compile configuration for DiT model
        torch_compile_args_vae: Optional torch.compile configuration for VAE model
        
    Returns:
        VideoDiffusionInfer: Configured runner
        
    Features:
        - Independent DiT and VAE caching for flexible memory management
        - Dynamic model reloading when models change
        - Optional torch.compile optimization for inference speedup
        - Separate encode/decode tiling configuration for optimal performance
        - Memory optimization and BlockSwap integration
        
    Raises:
        ValueError: If debug instance is not provided
    """
    
    if debug is None:
        raise ValueError("Debug instance must be provided to configure_runner")
    
    # Validate BlockSwap configuration early (before any model loading)
    block_swap_config = validate_blockswap_config(
        block_swap_config=block_swap_config,
        dit_device=ctx.dit_device,
        dit_offload_device=ctx.dit_offload_device,
        debug=debug
    )
    
    key = (checkpoint_identity(os.path.join(base_cache_dir, dit_model)),
           checkpoint_identity(os.path.join(base_cache_dir, vae_model)),
           str(ctx.dit_device), str(ctx.vae_device), str(ctx.compute_dtype))
    factory = lambda: _create_new_runner(dit_model, vae_model, base_cache_dir, debug)
    runner = session.acquire(key, factory) if session is not None else factory()
    cached_dit = getattr(runner, 'dit', None) if dit_cache else None
    cached_vae = getattr(runner, 'vae', None) if vae_cache else None

    # Phase 3: Configure runner settings
    _configure_runner_settings(
        runner, ctx,
        encode_tiled, encode_tile_size, encode_tile_overlap,
        decode_tiled, decode_tile_size, decode_tile_overlap,
        tile_debug, attention_mode,
        block_swap_config, debug
    )
    
    # Phase 4: Setup models (load from cache or create new)
    _setup_models(
        runner, cached_dit, cached_vae, dit_model, vae_model,
        base_cache_dir, block_swap_config, debug
    )
    
    return runner


def _configure_runner_settings(
    runner: VideoDiffusionInfer,
    ctx: GenerationContext,
    encode_tiled: bool,
    encode_tile_size: Optional[Tuple[int, int]],
    encode_tile_overlap: Optional[Tuple[int, int]],
    decode_tiled: bool,
    decode_tile_size: Optional[Tuple[int, int]],
    decode_tile_overlap: Optional[Tuple[int, int]],
    tile_debug: str,
    attention_mode: str,

    block_swap_config: Optional[BlockSwapConfig],
    debug: Optional['Debug'] = None
) -> None:
    """
    Configure runner settings for VAE tiling, torch.compile, and BlockSwap.
    
    Stores configuration settings on runner for later comparison and application.
    Settings are stored in temporary "_new_*" attributes and later validated/applied
    in _setup_models phase. This separation allows configuration change detection
    when reusing cached models.
    
    Args:
        runner: VideoDiffusionInfer instance to configure
        ctx: Generation context from setup_generation_context
        encode_tiled: Enable tiled VAE encoding to reduce VRAM during encoding
        encode_tile_size: Tile dimensions (height, width) for encoding in pixels
        encode_tile_overlap: Overlap dimensions (height, width) between encoding tiles
        decode_tiled: Enable tiled VAE decoding to reduce VRAM during decoding
        decode_tile_size: Tile dimensions (height, width) for decoding in pixels
        decode_tile_overlap: Overlap dimensions (height, width) between decoding tiles
        tile_debug: Tile visualization mode (false/encode/decode)
        attention_mode: Attention computation backend ('sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3')

        block_swap_config: BlockSwap configuration for DiT model or None
        debug: Debug instance (stored on runner for model access)
    """
    # VAE tiling settings
    runner.encode_tiled = encode_tiled
    runner.encode_tile_size = encode_tile_size
    runner.encode_tile_overlap = encode_tile_overlap
    runner.decode_tiled = decode_tiled
    runner.decode_tile_size = decode_tile_size
    runner.decode_tile_overlap = decode_tile_overlap
    runner.tile_debug = tile_debug
    
    # Store the new configs temporarily for later comparison
    # Don't set them as attributes yet - let the update functions handle that

    runner._new_dit_block_swap_config = block_swap_config
    runner._new_dit_attention_mode = attention_mode
    runner._new_vae_tiling_config = {
        'encode_tiled': encode_tiled,
        'encode_tile_size': encode_tile_size,
        'encode_tile_overlap': encode_tile_overlap,
        'decode_tiled': decode_tiled,
        'decode_tile_size': decode_tile_size,
        'decode_tile_overlap': decode_tile_overlap
    }
    
    # Store device configuration on runner for submodule access (e.g., BlockSwap, Cleanup)
    runner._dit_device = ctx.dit_device
    runner._vae_device = ctx.vae_device
    runner._dit_offload_device = ctx.dit_offload_device
    runner._vae_offload_device = ctx.vae_offload_device
    runner._tensor_offload_device = ctx.tensor_offload_device
    runner._compute_dtype = ctx.compute_dtype

    runner.debug = debug


def _setup_models(
    runner: VideoDiffusionInfer,
    cached_dit: Optional[torch.nn.Module],
    cached_vae: Optional[torch.nn.Module],
    dit_model: str,
    vae_model: str,
    base_cache_dir: str,
    block_swap_config: Optional[BlockSwapConfig],
    debug: Optional['Debug'] = None
) -> None:
    """
    Setup DiT and VAE models from cache or create new structures.
    
    Central orchestration function that:
    1. Sets up DiT model (cached or new structure)
    2. Sets up VAE model (cached or new structure)
    3. Validates and updates configurations for cached models
    4. Stores initial configurations for new models
    5. Cleans up temporary configuration attributes
    
    This function coordinates the complex interaction between model caching,
    configuration management, and the meta device initialization strategy.
    
    Args:
        runner: VideoDiffusionInfer instance to setup
        cached_dit: Invocation-local DiT model reference, if retained
        cached_vae: Invocation-local VAE model reference, if retained
        dit_model: DiT model filename for loading/validation
        vae_model: VAE model filename for loading/validation
        base_cache_dir: Base directory containing model files
        block_swap_config: BlockSwap configuration (passed to DiT setup)
        debug: Debug instance for logging and timing
    """
    debug.start_timer("model_structures")
    
    # Setup DiT
    dit_created = _setup_dit_model(runner, cached_dit, dit_model, base_cache_dir,
                                   block_swap_config, debug)
    
    # Only update DiT config if model was cached/reused (not newly created)
    if not dit_created and hasattr(runner, 'dit') and runner.dit is not None:
        _update_dit_config(runner, runner._new_dit_block_swap_config, runner._new_dit_attention_mode, debug)
    elif dit_created:
        runner._dit_block_swap_config = runner._new_dit_block_swap_config
        runner._dit_attention_mode = runner._new_dit_attention_mode
        if hasattr(runner, 'dit') and runner.dit:
            runner.dit._config_swap = runner._new_dit_block_swap_config
            runner.dit._config_attn = runner._new_dit_attention_mode
    # Setup VAE
    vae_created = _setup_vae_model(runner, cached_vae, vae_model, base_cache_dir, debug)
    
    # Only update VAE config if model was cached/reused (not newly created)
    if not vae_created and hasattr(runner, 'vae') and runner.vae is not None:
        _update_vae_config(runner, debug)
    elif vae_created:
        runner._vae_tiling_config = runner._new_vae_tiling_config
        if hasattr(runner, 'vae') and runner.vae:
            runner.vae._config_tiling = runner._new_vae_tiling_config
    # Clean up temporary attributes
    for attr in ['_new_dit_block_swap_config', '_new_dit_attention_mode', '_new_vae_tiling_config']:
        if hasattr(runner, attr):
            delattr(runner, attr)
    
    debug.end_timer("model_structures", "Model structures prepared")


def _setup_dit_model(
    runner: VideoDiffusionInfer,
    cached_dit: Optional[torch.nn.Module],
    dit_model: str,
    base_cache_dir: str,
    block_swap_config: Optional[BlockSwapConfig],
    debug: Optional['Debug'] = None
) -> bool:
    """
    Setup DiT model from cache or create new meta device structure.
    
    Handles three scenarios:
    1. Model changed: Cleanup old model, create new structure
    2. Cached model available: Reuse cached model, restore config
    3. No model exists: Create new meta device structure
    
    Args:
        runner: VideoDiffusionInfer instance to setup
        cached_dit: Invocation-local DiT model reference, if retained
        dit_model: DiT model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
        base_cache_dir: Base directory containing model files
        block_swap_config: BlockSwap configuration to store (not applied here)
        debug: Debug instance for logging
        
    Returns:
        bool: True if new model structure was created, False if cached model reused
    """
    
    # Check if model changed - clean up old model if different
    current_dit_name = getattr(runner, '_dit_model_name', None)
    if current_dit_name and current_dit_name != dit_model:
        if hasattr(runner, 'dit') and runner.dit is not None:
            debug.log(f"DiT model changed ({current_dit_name} → {dit_model}), cleaning old model", 
                     category="cache", force=True)
            cleanup_dit(runner=runner, debug=debug, cache_model=False)
    
    if cached_dit is not None:
        # Reuse cached DiT model
        debug.log(f"Reusing cached DiT (session): {dit_model}",
                category="reuse", force=True)
        runner.dit = cached_dit
        runner._dit_checkpoint = find_model_file(dit_model, base_cache_dir)
        runner._dit_model_name = dit_model
        
        # Restore config attributes from model to runner (config travels with model)
        runner._dit_compile_args = getattr(runner.dit, '_config_compile', None)
        runner._dit_block_swap_config = getattr(runner.dit, '_config_swap', None)
        runner._dit_attention_mode = getattr(runner.dit, '_config_attn', None)
        
        # blockswap_active will be set by apply_block_swap_to_dit
        # when the model is materialized to the inference device
        runner._blockswap_active = False
        
        return False
    elif not hasattr(runner, 'dit') or runner.dit is None:
        # Create new DiT model
        # Set DiT dtype from runner's compute_dtype
        # compute_dtype = getattr(runner, '_compute_dtype', torch.bfloat16)
        # dit_dtype_str = str(compute_dtype).split('.')[-1]
        # runner.config.dit.dtype = dit_dtype_str
        # runner._dit_dtype_override = compute_dtype
        
        dit_checkpoint_path = find_model_file(dit_model, base_cache_dir)
        runner = prepare_model_structure(runner, "dit", dit_checkpoint_path, 
                                        runner.config, debug, block_swap_config)
        runner._dit_model_name = dit_model
        return True
    else:
        # Model already exists from previous run with same runner
        runner._dit_model_name = dit_model
        return False


def _setup_vae_model(
    runner: VideoDiffusionInfer,
    cached_vae: Optional[torch.nn.Module],
    vae_model: str,
    base_cache_dir: str,
    debug: Optional['Debug'] = None
) -> bool:
    """
    Setup VAE model from cache or create new meta device structure.
    
    Handles three scenarios:
    1. Model changed: Cleanup old model, configure and create new structure
    2. Cached model available: Reuse cached model, restore config
    3. No model exists: Configure VAE settings, create new meta device structure
    
    Args:
        runner: VideoDiffusionInfer instance to setup
        cached_vae: Invocation-local VAE model reference, if retained
        vae_model: VAE model filename (e.g., "ema_vae_fp16.safetensors")
        base_cache_dir: Base directory containing model files
        debug: Debug instance for logging
        
    Returns:
        bool: True if new model structure was created, False if cached model reused
    """
    
    # Check if model changed - clean up old model if different
    current_vae_name = getattr(runner, '_vae_model_name', None)
    if current_vae_name and current_vae_name != vae_model:
        if hasattr(runner, 'vae') and runner.vae is not None:
            debug.log(f"VAE model changed ({current_vae_name} → {vae_model}), cleaning old model", 
                     category="cache", force=True)
            cleanup_vae(runner=runner, debug=debug, cache_model=False)
    
    if cached_vae is not None:
        # Reuse cached VAE model
        debug.log(f"Reusing cached VAE (session): {vae_model}",
                category="reuse", force=True)
        runner.vae = cached_vae
        runner._vae_checkpoint = find_model_file(vae_model, base_cache_dir)
        runner._vae_model_name = vae_model
        
        # Restore config attributes from model to runner (config travels with model)
        runner._vae_compile_args = getattr(runner.vae, '_config_compile', None)
        runner._vae_tiling_config = getattr(runner.vae, '_config_tiling', None)
        
        return False
    elif not hasattr(runner, 'vae') or runner.vae is None:
        # Create new VAE model
        # Configure VAE
        vae_config_path = os.path.join(script_directory, 
                                      'src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml')
        vae_config = load_config(vae_config_path)
        
        spatial_downsample_factor = vae_config.get('spatial_downsample_factor', 8)
        temporal_downsample_factor = vae_config.get('temporal_downsample_factor', 4)
        vae_config.spatial_downsample_factor = spatial_downsample_factor
        vae_config.temporal_downsample_factor = temporal_downsample_factor

        runner.config.vae.model = OmegaConf.merge(runner.config.vae.model, vae_config)
        
        # Set VAE dtype from runner's compute_dtype
        compute_dtype = getattr(runner, '_compute_dtype', torch.bfloat16)
        vae_dtype_str = str(compute_dtype).split('.')[-1]
        runner.config.vae.dtype = vae_dtype_str
        runner._vae_dtype_override = compute_dtype
        
        vae_checkpoint_path = find_model_file(vae_model, base_cache_dir)
        runner = prepare_model_structure(runner, "vae", vae_checkpoint_path, 
                                        runner.config, debug, None)
        
        debug.log(
            f"VAE downsample factors configured "
            f"(spatial: {spatial_downsample_factor}x, "
            f"temporal: {temporal_downsample_factor}x)",
            category="vae"
        )

        runner._vae_model_name = vae_model
        return True
    else:
        # Model already exists from previous run with same runner
        runner._vae_model_name = vae_model
        return False


def apply_model_specific_config(model: torch.nn.Module, runner: VideoDiffusionInfer, 
                                config: OmegaConf, is_dit: bool, 
                                debug: Optional['Debug'] = None) -> torch.nn.Module:
    """
    Apply model-specific configurations (FP8, BlockSwap, torch.compile).
    
    This function is idempotent and can be safely called on both newly materialized
    and already-configured models. It checks state flags to determine what needs
    to be applied.
    
    Critical: For DiT, BlockSwap must be applied BEFORE torch.compile.
    torch.compile captures the computational graph, so any wrapping done
    after compilation (like BlockSwap's forward wrapping) won't work.
    
    Args:
        model: Loaded model instance
        runner: Runner to attach model to
        config: Full configuration object
        is_dit: Whether this is a DiT model (vs VAE)
        debug: Debug instance
        
    Returns:
        Configured model with BlockSwap and torch.compile applied if configured
    """
    if is_dit:
        # DiT-specific
        # Apply compatibility wrapper with compute_dtype
        if not isinstance(model, CompatibleDiT):
            debug.log("Applying DiT compatibility wrapper", category="setup")
            debug.start_timer("CompatibleDiT")
            # Get compute_dtype from runner if available, fallback to bfloat16
            compute_dtype = getattr(runner, '_compute_dtype', torch.bfloat16)
            model = CompatibleDiT(model, debug, compute_dtype=compute_dtype, skip_conversion=False)
            debug.end_timer("CompatibleDiT", "Compatibility wrapper application")
        else:
            debug.log("Reusing existing DiT compatibility wrapper", category="reuse")
        
        # Apply attention mode and compute_dtype to all FlashAttentionVarlen modules
        if hasattr(runner, '_dit_attention_mode'):
            requested_attention_mode = runner._dit_attention_mode or 'sdpa'
            
            # Validate and get final attention_mode (with warning if fallback needed)
            attention_mode = validate_attention_mode(requested_attention_mode, debug)
            
            # Get compute_dtype from runner
            compute_dtype = getattr(runner, '_compute_dtype', torch.bfloat16)            
            debug.log(f"Applying {attention_mode} attention mode and {compute_dtype} compute dtype to model", category="setup")
            
            # Get the actual model (unwrap if needed)
            actual_model = model.dit_model if hasattr(model, 'dit_model') else model
            
            # Update all FlashAttentionVarlen instances
            updated_count = 0
            for module in actual_model.modules():
                if type(module).__name__ == 'FlashAttentionVarlen':
                    module.attention_mode = attention_mode
                    module.compute_dtype = compute_dtype
                    updated_count += 1
            
            if updated_count > 0:
                debug.log(f"Applied {attention_mode} and compute_dtype={compute_dtype} to {updated_count} modules", category="success")

        # Apply BlockSwap before torch.compile (only if not already active)
        # BlockSwap wraps forward methods, and torch.compile needs to capture the wrapped version
        if hasattr(runner, '_dit_block_swap_config') and runner._dit_block_swap_config:
            # Check if BlockSwap needs to be applied
            needs_blockswap = not (hasattr(runner, '_blockswap_active') and runner._blockswap_active)
            if needs_blockswap:
                runner.dit = model
                apply_block_swap_to_dit(runner, runner._dit_block_swap_config, debug)
                # Mark as active after successful application
                runner._blockswap_active = True
        

        runner.dit = model

        # Clear the config application flag after successful application
        if hasattr(runner, '_dit_config_needs_application'):
            runner._dit_config_needs_application = False
        
    else:
        # VAE-specific configurations
        # Set to eval mode (no gradients needed for inference)
        if model.training:
            debug.log("VAE model set to eval mode (gradients disabled)", category="vae")
            debug.start_timer("model_requires_grad")
            model.requires_grad_(False).eval()
            debug.end_timer("model_requires_grad", "VAE model set to eval mode")
        
        # Configure causal slicing if available - always apply as it's lightweight
        if hasattr(model, "set_causal_slicing") and hasattr(config.vae, "slicing"):
            debug.log("Configuring VAE causal slicing for temporal processing", category="vae")
            debug.start_timer("vae_set_causal_slicing")
            model.set_causal_slicing(**config.vae.slicing)
            debug.end_timer("vae_set_causal_slicing", "VAE causal slicing configuration")
        
        # Set memory limits if available - always apply to ensure limits are set
        if hasattr(model, "set_memory_limit") and hasattr(config.vae, "memory_limit"):
            debug.log("Configuring VAE memory limits for causal convolutions", category="vae")
            debug.start_timer("vae_set_memory_limit")
            model.set_memory_limit(**config.vae.memory_limit)
            debug.end_timer("vae_set_memory_limit", "VAE memory limits configured")


        # Propagate debug and tensor_offload_device to submodules
        model.debug = debug
        model.tensor_offload_device = runner._tensor_offload_device
        _propagate_debug_to_modules(model, debug)
        runner.vae = model
        
        # Clear the config application flag after successful application
        if hasattr(runner, '_vae_config_needs_application'):
            runner._vae_config_needs_application = False
    
    return model


def _propagate_debug_to_modules(module: torch.nn.Module, debug: 'Debug') -> None:
    """
    Propagate debug instance to specific submodules that need it.
    Only targets modules that actually use debug to avoid unnecessary memory overhead.
    
    Args:
        module: Parent module to propagate through
        debug: Debug instance to attach
    """
    if debug is None:
        return  # Early exit if no debug instance
        
    target_modules = {'ResnetBlock3D', 'Upsample3D', 'InflatedCausalConv3d', 'GroupNorm'}
    
    for name, submodule in module.named_modules():
        if submodule.__class__.__name__ in target_modules:
            if not hasattr(submodule, 'debug'):  # Only set if not already present
                submodule.debug = debug
