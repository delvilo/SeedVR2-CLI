"""
Memory management module for SeedVR2
Handles VRAM usage, cache management, and memory optimization

Extracted from: seedvr2.py (lines 373-405, 607-626, 1016-1044)
"""

from ..core.context import GenerationContext

import torch
import gc
import time
import psutil
from typing import Tuple, Dict, Any, Optional, List, Union


def _device_str(device: Union[torch.device, str]) -> str:
    """Normalized uppercase device string for comparison and logging."""
    return str(device).upper()


def is_cuda_available() -> bool:
    """Check if CUDA backend is available."""
    return torch.cuda.is_available()


def get_gpu_backend() -> str:
    """Get the active GPU backend type.
    
    Returns:
        'cuda': NVIDIA CUDA
        'cpu': No GPU backend available
    """
    if is_cuda_available():
        return 'cuda'
    return 'cpu'




def get_basic_vram_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Get basic VRAM availability info (free and total memory).
    Used for capacity planning and initial checks.
    
    Args:
        device: Optional device to query. If None, uses cuda:0
    
    Returns:
        dict: {"free_gb": float, "total_gb": float} or {"error": str}
    """
    try:
        if is_cuda_available():
            if device is None:
                device = torch.device("cuda:0")
            elif not isinstance(device, torch.device):
                device = torch.device(device)
            free_memory, total_memory = torch.cuda.mem_get_info(device)
        else:
            return {"error": "No CUDA GPU available"}
        
        return {
            "free_gb": free_memory / (1024**3),
            "total_gb": total_memory / (1024**3)
        }
    except Exception as e:
        return {"error": f"Failed to get memory info: {str(e)}"}


# Initial VRAM check at module load
vram_info = get_basic_vram_info(device=None)
if "error" not in vram_info:
    backend = "CUDA"
    print(f"📊 Initial {backend} memory: {vram_info['free_gb']:.2f}GB free / {vram_info['total_gb']:.2f}GB total")
else:
    print(f"⚠️ Memory check failed: {vram_info['error']} - No available backend!")


def get_vram_usage(device: Optional[torch.device] = None, debug: Optional['Debug'] = None) -> Tuple[float, float, float, float]:
    """
    Get current VRAM usage metrics for monitoring.
    Used for tracking memory consumption during processing.

    Args:
        device: Optional device to query. If None, uses cuda:0
        debug: Optional debug instance for logging
    
    Returns:
        tuple: (allocated_gb, reserved_gb, peak_allocated_gb, peak_reserved_gb)
               Returns (0, 0, 0, 0) if no GPU available
    """
    try:
        if is_cuda_available():
            if device is None:
                device = torch.device("cuda:0")
            elif not isinstance(device, torch.device):
                device = torch.device(device)
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
            return allocated, reserved, peak_allocated, peak_reserved
    except Exception as e:
        if debug:
            debug.log(f"Failed to get VRAM usage: {e}", level="WARNING", category="memory", force=True)
    return 0.0, 0.0, 0.0, 0.0


def get_ram_usage(debug: Optional['Debug'] = None) -> Tuple[float, float, float, float]:
    """
    Get current RAM usage metrics for the current process.
    Provides accurate tracking of process-specific memory consumption.
    
    Args:
        debug: Optional debug instance for logging
    
    Returns:
        tuple: (process_gb, available_gb, total_gb, used_by_others_gb)
               Returns (0, 0, 0, 0) if psutil not available or on error
    """
    try:
        if not psutil:
            return 0.0, 0.0, 0.0, 0.0
            
        # Get current process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        process_gb = process_memory.rss / (1024**3)
        
        # Get system memory
        sys_memory = psutil.virtual_memory()
        total_gb = sys_memory.total / (1024**3)
        available_gb = sys_memory.available / (1024**3)
        
        # Calculate memory used by other processes
        # This is the CORRECT calculation:
        total_used_gb = total_gb - available_gb  # Total memory used by ALL processes
        used_by_others_gb = max(0, total_used_gb - process_gb)  # Subtract current process
        
        return process_gb, available_gb, total_gb, used_by_others_gb
        
    except Exception as e:
        if debug:
            debug.log(f"Failed to get RAM usage: {e}", level="WARNING", category="memory", force=True)
        return 0.0, 0.0, 0.0, 0.0
    
    
# Global cache for OS libraries (initialized once)
_os_memory_lib = None


def clear_memory(debug: Optional['Debug'] = None, deep: bool = False, force: bool = True, 
                timer_name: Optional[str] = None) -> None:
    """
    Clear memory caches with two-tier approach for optimal performance.
    
    Args:
        debug: Debug instance for logging (optional)
        force: If True, always clear. If False, only clear when <5% free
        deep: If True, perform deep cleanup including GC and OS operations.
              If False (default), only perform minimal GPU cache clearing.
        timer_name: Optional suffix for timer names to make them unique per invocation
    
    Two-tier approach:
        - Minimal mode (deep=False): GPU cache operations (~1-5ms)
          Used for frequent calls during batch processing
        - Deep mode (deep=True): Complete cleanup with GC and OS operations (~10-50ms)
          Used at key points like model switches or final cleanup
    """
    global _os_memory_lib
    
    # Create unique timer names if suffix provided
    if timer_name:
        main_timer = f"memory_clear_{timer_name}"
        gpu_timer = f"gpu_cache_clear_{timer_name}"
        gc_timer = f"garbage_collection_{timer_name}"
        os_timer = f"os_memory_release_{timer_name}"
        completion_msg = f"clear_memory() completion ({timer_name})"
    else:
        main_timer = "memory_clear"
        gpu_timer = "gpu_cache_clear"
        gc_timer = "garbage_collection"
        os_timer = "os_memory_release"
        completion_msg = "clear_memory() completion"
    
    # Start timer for entire operation
    if debug:
        debug.start_timer(main_timer)

    # Check if we should clear based on memory pressure
    if not force:
        should_clear = False
        
        # Use existing function for memory info
        mem_info = get_basic_vram_info(device=None)
        
        if "error" not in mem_info and mem_info["total_gb"] > 0:
            # Check VRAM pressure (5% free threshold)
            free_ratio = mem_info["free_gb"] / mem_info["total_gb"]
            if free_ratio < 0.05:
                should_clear = True
                if debug:
                    debug.log(f"VRAM pressure: {mem_info['free_gb']:.2f}GB free of {mem_info['total_gb']:.2f}GB", category="memory")
        
        # Check system RAM separately.
        if not should_clear:
            mem = psutil.virtual_memory()
            if mem.available < mem.total * 0.05:
                should_clear = True
                if debug:
                    debug.log(f"RAM pressure: {mem.available/(1024**3):.2f}GB free of {mem.total/(1024**3):.2f}GB", category="memory")
        
        if not should_clear:
            # End timer before early return to keep stack clean
            if debug:
                debug.end_timer(main_timer)
            return
    
    # Determine cleanup level
    cleanup_mode = "deep" if deep else "minimal"
    if debug:
        debug.log(f"Clearing memory caches ({cleanup_mode})...", category="cleanup")
    
    # ===== MINIMAL OPERATIONS (Always performed) =====
    # Step 1: Clear GPU caches - Fast operations (~1-5ms)
    if debug:
        debug.start_timer(gpu_timer)
    
    if is_cuda_available():
        torch.cuda.empty_cache()
    
    if debug:
        debug.end_timer(gpu_timer, "GPU cache clearing")

    # ===== DEEP OPERATIONS (Only when deep=True) =====
    if deep:
        # Step 2: Deep garbage collection (expensive ~5-20ms)
        if debug:
            debug.start_timer(gc_timer)

        gc.collect(2)

        if debug:
            debug.end_timer(gc_timer, "Garbage collection")

        # Step 3: Return memory to Linux (~5-30ms)
        if debug:
            debug.start_timer(os_timer)

        try:
            import ctypes  # Import only when needed
            if _os_memory_lib is None:
                _os_memory_lib = ctypes.CDLL("libc.so.6")
            _os_memory_lib.malloc_trim(0)
        except Exception as e:
            if debug:
                debug.log(f"Failed to perform OS memory operations: {e}", level="WARNING", category="memory", force=True)

        if debug:
            debug.end_timer(os_timer, "OS memory release")
    
    # End overall timer
    if debug:
        debug.end_timer(main_timer, completion_msg)


def retry_on_oom(func, *args, debug=None, operation_name="operation", **kwargs):
    """Run once; the CLI retries the file transaction with a smaller workload.

    Retrying here would retain the failed call's traceback/tensors and repeat the
    same allocation. The outer retry unwinds those references before recovery.
    """
    return func(*args, **kwargs)


def reset_vram_peak(device: Optional[torch.device] = None, debug: Optional['Debug'] = None) -> None:
    """
    Reset VRAM peak memory statistics for fresh tracking.
    
    Args:
        device: Optional device to reset stats for. If None, uses cuda:0
        debug: Optional debug instance for logging
    """
    if debug and debug.enabled:
        debug.log("Resetting VRAM peak memory statistics", category="memory")
    try:
        if is_cuda_available():
            if device is None:
                device = torch.device("cuda:0")
            elif not isinstance(device, torch.device):
                device = torch.device(device)
            torch.cuda.reset_peak_memory_stats(device)
    except Exception as e:
        if debug and debug.enabled:
            debug.log(f"Failed to reset peak memory stats: {e}", level="WARNING", category="memory", force=True)


def clear_rope_lru_caches(model: Optional[torch.nn.Module], debug: Optional['Debug'] = None) -> int:
    """
    Clear ALL LRU caches from RoPE modules.
    
    Args:
        model: PyTorch model to clear caches from
        debug: Optional debug instance for logging
        
    Returns:
        Number of caches cleared
    """
    if model is None:
        return 0
    
    cleared_count = 0
    try:
        for name, module in model.named_modules():
            if hasattr(module, 'get_axial_freqs') and hasattr(module.get_axial_freqs, 'cache_clear'):
                try:
                    module.get_axial_freqs.cache_clear()
                    cleared_count += 1
                except Exception as e:
                    if debug:
                        debug.log(f"Failed to clear RoPE LRU cache for module {name}: {e}", level="WARNING", category="memory", force=True)
    except (AttributeError, RuntimeError) as e:
        if debug:
            debug.log(f"Failed to iterate model modules for RoPE LRU cache clearing: {e}", level="WARNING", category="memory", force=True)
    
    return cleared_count


def release_tensor_memory(tensor: Optional[torch.Tensor]) -> None:
    """Clear gradients only. Owners release storage by dropping references.

    Never mutate storage through .data: views and other owners may still need it.
    """
    if tensor is not None and torch.is_tensor(tensor) and tensor.is_leaf:
        tensor.grad = None


def release_tensor_collection(collection: Any, recursive: bool = True) -> None:
    """
    Release GPU memory from tensors in any collection (list, tuple, dict, or single tensor).
    
    Args:
        collection: Tensor, list, tuple, dict, or nested structure to release
        recursive: If True, handle nested structures recursively
        
    Examples:
        release_tensor_collection(tensor)                    # Single tensor
        release_tensor_collection([tensor1, tensor2])        # List of tensors
        release_tensor_collection([[t1, t2], [t3, t4]])     # Nested lists
        release_tensor_collection({'a': tensor})             # Dict values
    """
    if collection is None:
        return
    
    if torch.is_tensor(collection):
        release_tensor_memory(collection)
    elif isinstance(collection, dict):
        if recursive:
            for value in collection.values():
                release_tensor_collection(value, recursive=True)
        collection.clear()
    elif isinstance(collection, list):
        if recursive:
            for value in collection:
                release_tensor_collection(value, recursive=True)
        collection.clear()
    elif isinstance(collection, tuple) and recursive:
        for value in collection:
            release_tensor_collection(value, recursive=True)


def release_text_embeddings(*embeddings: torch.Tensor, debug: Optional['Debug'] = None, names: Optional[List[str]] = None) -> None:
    """
    Release memory for text embeddings
    
    Args:
        *embeddings: Variable number of embedding tensors to release
        debug: Optional debug instance for logging
        names: Optional list of names for logging
    """
    for i, embedding in enumerate(embeddings):
        if embedding is not None:
            release_tensor_memory(embedding)
            if debug and names and i < len(names):
                debug.log(f"Cleaned up {names[i]}", category="cleanup")


def cleanup_text_embeddings(ctx: GenerationContext, debug: Optional['Debug'] = None) -> None:
    """
    Clean up text embeddings from invocation state.
    Extracts embeddings, releases memory, and clears the context entry.
    
    Args:
        ctx: GenerationContext potentially containing 'text_embeds'
        debug: Optional debug instance for logging
    """
    if not ctx or not ctx.text_embeds:
        return
    
    embeddings = []
    names = []
    for key, embeds_list in ctx.text_embeds.items():
        if embeds_list:
            embeddings.extend(embeds_list)
            names.append(key)
    
    if embeddings:
        release_text_embeddings(*embeddings, debug=debug)
        
        if debug:
            debug.log(f"Cleaned up text embeddings: {', '.join(names)}", category="cleanup")
    
    ctx.text_embeds = None

    
def release_model_memory(model: Optional[torch.nn.Module], debug=None) -> None:
    """Clear model-owned temporary state; the session drops the model reference."""
    if model is None:
        return
    model.zero_grad(set_to_none=True)
    clear_rope_lru_caches(model, debug)
    for module in model.modules():
        if hasattr(module, 'memory'):
            module.memory = None


def manage_tensor(
    tensor: torch.Tensor,
    target_device: torch.device,
    tensor_name: str = "tensor",
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = False,
    debug: Optional['Debug'] = None,
    reason: Optional[str] = None,
    indent_level: int = 0
) -> torch.Tensor:
    """
    Unified tensor management for device movement and dtype conversion.
    
    Handles both device transfers (CPU ↔ GPU) and dtype conversions (e.g., float16 → bfloat16)
    with intelligent early-exit optimization and comprehensive logging.
    
    Args:
        tensor: Tensor to manage
        target_device: Target device (torch.device object)
        tensor_name: Descriptive name for logging (e.g., "latent", "sample", "alpha_channel")
        dtype: Optional target dtype to cast to (if None, keeps original dtype)
        non_blocking: Whether to use non-blocking transfer
        debug: Debug instance for logging
        reason: Optional reason for the operation (e.g., "inference", "offload", "dtype alignment")
        indent_level: Indentation level for debug logging (0=no indent, 1=2 spaces, etc.)
        
    Returns:
        Tensor on target device with optional dtype conversion
        
    Note:
        - Skips operation if tensor already has target device and dtype (zero-copy)
        - Uses PyTorch's optimized .to() for efficient device/dtype handling
        - Logs all operations consistently for tracking and debugging
    """
    if tensor is None:
        return tensor
    
    # Get current state
    current_device = tensor.device
    current_dtype = tensor.dtype
    target_dtype = dtype if dtype is not None else current_dtype
    
    # Check if movement is actually needed
    needs_device_move = _device_str(current_device) != _device_str(target_device)
    needs_dtype_change = dtype is not None and current_dtype != target_dtype
    
    if not needs_device_move and not needs_dtype_change:
        # Already on target device and dtype - skip
        return tensor
    
    # Determine reason for movement
    if reason is None:
        if needs_device_move and needs_dtype_change:
            reason = "device and dtype conversion"
        elif needs_device_move:
            reason = "device movement"
        else:
            reason = "dtype conversion"
    
    # Log the movement
    if debug:
        current_device_str = _device_str(current_device)
        target_device_str = _device_str(target_device)
        
        dtype_info = ""
        if needs_dtype_change:
            dtype_info = f", {current_dtype} → {target_dtype}"
        
        debug.log(
            f"Moving {tensor_name} from {current_device_str} to {target_device_str}{dtype_info} ({reason})",
            category="general",
            indent_level=indent_level
        )
    
    # Perform the operation based on what needs to change
    if needs_device_move and needs_dtype_change:
        # Both device and dtype need to change
        return tensor.to(target_device, dtype=target_dtype, non_blocking=non_blocking)
    elif needs_device_move:
        # Only device needs to change
        return tensor.to(target_device, non_blocking=non_blocking)
    else:
        # Only dtype needs to change
        return tensor.to(dtype=target_dtype)


def manage_model_device(model: torch.nn.Module, target_device: torch.device, model_name: str,
                       debug: Optional['Debug'] = None, reason: Optional[str] = None,
                       runner: Optional[Any] = None) -> bool:
    """
    Move model to target device with optimizations.
    Handles BlockSwap-enabled models transparently.
    
    Args:
        model: The model to move
        target_device: Target device (torch.device object, e.g., torch.device('cuda:0'))
        model_name: Name for logging (e.g., "VAE", "DiT")
        debug: Debug instance for logging
        reason: Optional custom reason for the movement
        runner: Optional runner instance for BlockSwap detection
        
    Returns:
        bool: True if model was moved, False if already on target device
    """
    if model is None or target_device is None:
        return False
    if isinstance(target_device, str):
        target_device = torch.device(target_device)
    # Check if this is a BlockSwap-enabled DiT model
    is_blockswap_model = False
    actual_model = model
    if runner and model_name == "DiT":
        # Import here to avoid circular dependency
        from .blockswap import is_blockswap_enabled
        # Check if BlockSwap config exists and is enabled
        has_blockswap_config = (
            hasattr(runner, '_dit_block_swap_config') and 
            is_blockswap_enabled(runner._dit_block_swap_config)
        )
        
        if has_blockswap_config:
            is_blockswap_model = True
            # Get the actual model (handle CompatibleDiT wrapper)
            if hasattr(model, "dit_model"):
                actual_model = model.dit_model

    # Get current device
    try:
        current_device = next(model.parameters()).device
    except StopIteration:
        return False
    
    # Extract device type for comparison (both are torch.device objects)
    target_type = target_device.type
    current_device_upper = _device_str(current_device)
    target_device_upper = _device_str(target_device)

    # Compare normalized device types
    if current_device_upper == target_device_upper and not is_blockswap_model:
        # Already on target device type, no movement needed
        if debug:
            debug.log(f"{model_name} already on {current_device_upper}, skipping movement", category="general")
        return False
        
    # Handle BlockSwap models specially
    if is_blockswap_model:
        return _handle_blockswap_model_movement(
            runner, actual_model, current_device, target_device, target_type,
            model_name, debug, reason
        )
    
    # Standard model movement (non-BlockSwap)
    return _standard_model_movement(
        model, current_device, target_device, target_type, model_name,
        debug, reason
    )


def _handle_blockswap_model_movement(runner: Any, model: torch.nn.Module, 
                                    current_device: torch.device, target_device: torch.device, 
                                    target_type: str, model_name: str,
                                    debug: Optional['Debug'] = None, reason: Optional[str] = None) -> bool:
    """
    Handle device movement for BlockSwap-enabled models.
    
    Args:
        runner: Runner instance with BlockSwap configuration
        model: Model to move (actual unwrapped model)
        current_device: Current device of the model
        target_device: Target device (torch.device object)
        target_type: Target device type (cpu/cuda)
        model_name: Model name for logging
        debug: Debug instance
        reason: Movement reason
        
    Returns:
        bool: True if model was moved
    """
    # Import here to avoid circular dependency
    from .blockswap import set_blockswap_bypass

    if target_type == "cpu":
        # Moving to offload device (typically CPU)
        # Check if any parameter is on GPU (for accurate logging)
        actual_source_device = None
        for param in model.parameters():
            if param.device.type == 'cuda':
                actual_source_device = param.device
                break
        
        source_device_desc = _device_str(actual_source_device) if actual_source_device else _device_str(target_device)
        
        if debug:
            debug.log(f"Moving {model_name} from {source_device_desc} to {_device_str(target_device)} ({reason or 'model caching'})", category="general")
        
        # Enable bypass to allow movement
        set_blockswap_bypass(runner=runner, bypass=True, debug=debug)
        
        # Start timer
        timer_name = f"{model_name.lower()}_to_{target_type}"
        if debug:
            debug.start_timer(timer_name)
        
        # Move entire model to target offload device
        model.to(target_device)
        model.zero_grad(set_to_none=True)
        
        if debug:
            debug.end_timer(timer_name, f"BlockSwap model offloaded to {_device_str(target_device)}")
        
        return True
        
    else:
        # Moving to GPU (reload)
        # Check if we're in bypass mode (coming from offload)
        if not getattr(model, "_blockswap_bypass_protection", False):
            # Not in bypass mode, blocks are already configured
            if debug:
                debug.log(f"{model_name} with BlockSwap active - blocks already distributed across devices, skipping movement", category="general")
            return False
        
        # Get actual current device for accurate logging
        actual_current_device = None
        for param in model.parameters():
            if param.device.type != 'meta':
                actual_current_device = param.device
                break
        
        current_device_desc = _device_str(actual_current_device) if actual_current_device else "OFFLOAD"
        
        if debug:
            debug.log(f"Moving {model_name} from {current_device_desc} to {_device_str(target_device)} ({reason or 'inference requirement'})", category="general")
        
        timer_name = f"{model_name.lower()}_to_gpu"
        if debug:
            debug.start_timer(timer_name)
        
        # Restore blocks to their configured devices
        if hasattr(model, "blocks") and hasattr(model, "blocks_to_swap"):
            # Use configured offload_device from BlockSwap config
            offload_device = model._block_swap_config.get("offload_device")
            if not offload_device:
                raise ValueError("BlockSwap config missing offload_device")
            
            # Move blocks according to BlockSwap configuration
            for b, block in enumerate(model.blocks):
                if b > model.blocks_to_swap:
                    # This block should be on GPU
                    block.to(target_device)
                else:
                    # This block stays on offload device (will be swapped during forward)
                    block.to(offload_device)
            
            # Handle I/O components
            if not model._block_swap_config.get("swap_io_components", False):
                # I/O components should be on GPU if not offloaded
                for name, module in model.named_children():
                    if name != "blocks":
                        module.to(target_device)
            else:
                # I/O components stay on offload device
                for name, module in model.named_children():
                    if name != "blocks":
                        module.to(offload_device)
            
            if debug:
                # Get actual configuration from runner
                if hasattr(model, '_block_swap_config'):
                    blocks_on_gpu = model._block_swap_config.get('total_blocks', 32) - model._block_swap_config.get('blocks_swapped', 16)
                    total_blocks = model._block_swap_config.get('total_blocks', 32)
                    main_device = model._block_swap_config.get('main_device', 'GPU')
                    debug.log(f"BlockSwap blocks restored to configured devices ({blocks_on_gpu}/{total_blocks} blocks on {_device_str(main_device)})", category="success")
                else:
                    debug.log("BlockSwap blocks restored to configured devices", category="success")

        
        # Reactivate BlockSwap now that blocks are restored to their configured devices
        runner._blockswap_active = True
        
        # Disable bypass, re-enable protection
        set_blockswap_bypass(runner=runner, bypass=False, debug=debug)
        
        if debug:
            debug.end_timer(timer_name, "BlockSwap model restored")
        
        return True


def _standard_model_movement(model: torch.nn.Module, current_device: torch.device,
                            target_device: torch.device, target_type: str, model_name: str,
                            debug: Optional['Debug'] = None, reason: Optional[str] = None) -> bool:
    """
    Handle standard (non-BlockSwap) model movement.
    
    Args:
        model: Model to move
        current_device: Current device of the model
        target_device: Target device (torch.device object)
        target_type: Target device type
        model_name: Model name for logging
        debug: Debug instance
        reason: Movement reason
        
    Returns:
        bool: True if model was moved
    """
    # Check if model is on meta device - can't move meta tensors
    if current_device.type == 'meta':
        if debug:
            debug.log(f"{model_name} is on meta device - skipping movement (will materialize when needed)", 
                     category=model_name.lower())
        return False
    
    # Determine reason for movement
    reason = reason or "inference requirement"
    
    # Log the movement with full device strings
    if debug:
        current_device_str = _device_str(current_device)
        target_device_str = _device_str(target_device)
        debug.log(f"Moving {model_name} from {current_device_str} to {target_device_str} ({reason})", category="general")

    # Start timer based on direction
    timer_name = f"{model_name.lower()}_to_{'gpu' if target_type != 'cpu' else 'cpu'}"
    if debug:
        debug.start_timer(timer_name)
    
    # Move model and clear gradients
    model.to(target_device)
    model.zero_grad(set_to_none=True)
    
    # Clear VAE memory buffers when moving to CPU
    if target_type == 'cpu' and model_name == "VAE":
        cleared_count = 0
        for module in model.modules():
            if hasattr(module, 'memory') and module.memory is not None:
                if torch.is_tensor(module.memory) and module.memory.is_cuda:
                    module.memory = None
                    cleared_count += 1
        if cleared_count > 0 and debug:
            debug.log(f"Cleared {cleared_count} VAE memory buffers", category="success")
    
    # End timer
    if debug:
        debug.end_timer(timer_name, f"{model_name} moved to {_device_str(target_device)}")
    
    return True


def clear_runtime_caches(runner: Any, debug: Optional['Debug'] = None) -> int:
    """
    Clear all runtime caches and temporary attributes.
    """
    if not runner:
        return 0
    
    if debug:
        debug.start_timer("runtime_cache_clear")
    
    cleaned_items = 0
    
    # 1. Clear main runner cache
    if hasattr(runner, 'cache') and hasattr(runner.cache, 'cache'):
        if debug:
            debug.start_timer("runner_cache_clear")

        cache_entries = len(runner.cache.cache)
        
        # Properly release tensor memory and delete as we go
        for key in list(runner.cache.cache.keys()):
            value = runner.cache.cache[key]
            if torch.is_tensor(value):
                release_tensor_memory(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if torch.is_tensor(item):
                        release_tensor_memory(item)
            # Delete immediately to release reference
            del runner.cache.cache[key]

        # Final clear for safety
        runner.cache.cache.clear()
        cleaned_items += cache_entries

        if debug:
            debug.end_timer("runner_cache_clear", f"Clearing main runner cache entries")

        if cache_entries > 0 and debug:
            debug.log(f"Cleared {cache_entries} runtime cache entries", category="success")
    
    # 2. Clear RoPE caches
    if hasattr(runner, 'dit'):
        if debug:
            debug.start_timer("rope_cache_clear")

        model = runner.dit
        if hasattr(model, 'dit_model'):  # Handle wrapper
            model = model.dit_model
        
        rope_cleared = clear_rope_lru_caches(model=model, debug=debug)
        cleaned_items += rope_cleared
        if debug:
            debug.end_timer("rope_cache_clear", "Clearing RoPE LRU caches")

        if rope_cleared > 0 and debug:
            debug.log(f"Cleared {rope_cleared} RoPE LRU caches", category="success")
    
    # 3. Clear temporary attributes
    temp_attrs = ['_temp_cache', '_block_cache', '_swap_cache', '_generation_cache',
                  '_rope_cache', '_intermediate_cache', '_backward_cache']
    
    for obj in [runner, getattr(runner, 'dit', None), getattr(runner, 'vae', None)]:
        if obj is None:
            continue
            
        actual_obj = obj.dit_model if hasattr(obj, 'dit_model') else obj
        
        for attr in temp_attrs:
            if hasattr(actual_obj, attr):
                delattr(actual_obj, attr)
                cleaned_items += 1

    if debug:
        debug.end_timer("runtime_cache_clear", f"clear_runtime_caches() completion")

    return cleaned_items


def cleanup_dit(runner: Any, debug: Optional['Debug'] = None, cache_model: bool = False) -> None:
    """
    Cleanup DiT model and BlockSwap state after upscaling phase.
    Called at the end of upscale_all_batches when DiT is no longer needed.
    
    Args:
        runner: Runner instance containing DiT model
        debug: Debug instance for logging
        cache_model: If True, move DiT to offload_device; if False, delete completely
    """
    if runner is None or getattr(runner, 'dit', None) is None:
        return
    
    if debug:
        debug.log("Cleaning up DiT components", category="cleanup")
    
    # 1. Clear DiT-specific runtime caches first
    if hasattr(runner, 'dit'):
        model = runner.dit
        if hasattr(model, 'dit_model'):  # Handle wrapper
            model = model.dit_model
        
        # Clear RoPE caches
        rope_cleared = clear_rope_lru_caches(model=model, debug=debug)
        if rope_cleared > 0 and debug:
            debug.log(f"Cleared {rope_cleared} RoPE LRU caches", category="success")
        
        # Clear DiT temporary attributes
        temp_attrs = ['_temp_cache', '_block_cache', '_swap_cache', '_generation_cache',
                      '_rope_cache', '_intermediate_cache', '_backward_cache']
        
        actual_obj = model.dit_model if hasattr(model, 'dit_model') else model
        for attr in temp_attrs:
            if hasattr(actual_obj, attr):
                delattr(actual_obj, attr)
    
    # 2. Handle model offloading (for caching or before deletion)
    try:
        param_device = next(runner.dit.parameters()).device
        
        # Move model off GPU if needed
        if cache_model and param_device.type not in ['meta', 'cpu']:
            offload_target = getattr(runner, '_dit_offload_device', None)
            if offload_target is None or offload_target == 'none':
                offload_target = torch.device('cpu')
            reason = "model caching" if cache_model else "releasing GPU memory"
            manage_model_device(model=runner.dit, target_device=offload_target, model_name="DiT",
                               debug=debug, reason=reason, runner=runner)
        elif param_device.type == 'meta' and debug:
            debug.log("DiT on meta device - keeping structure for cache", category="cleanup")
    except StopIteration:
        pass
    
    # 3. Clean BlockSwap after model movement
    if hasattr(runner, "_blockswap_active") and runner._blockswap_active:
        # Import here to avoid circular dependency
        from .blockswap import cleanup_blockswap
        cleanup_blockswap(runner=runner, keep_state_for_cache=cache_model)
    
    # 4. Complete cleanup if not caching
    if not cache_model:
        release_model_memory(model=runner.dit, debug=debug)
        runner.dit = None
        if debug:
            debug.log("DiT model deleted", category="cleanup")
        
        # Clear DiT config attributes - not needed when model is not cached (will be recreated)
        if hasattr(runner, '_dit_compile_args'):
            delattr(runner, '_dit_compile_args')
        if hasattr(runner, '_dit_block_swap_config'):
            delattr(runner, '_dit_block_swap_config')
        if hasattr(runner, '_dit_attention_mode'):
            delattr(runner, '_dit_attention_mode')
    
    # 5. Clear DiT temporary attributes (should be already cleared in materialize_model)
    runner._dit_checkpoint = None
    runner._dit_dtype_override = None
    
    # 6. Clear DiT-related components and temporary attributes
    runner.sampler = None
    runner.sampling_timesteps = None
    runner.schedule = None


def cleanup_vae(runner: Any, debug: Optional['Debug'] = None, cache_model: bool = False) -> None:
    """
    Cleanup VAE model after decoding phase.
    Called at the end of decode_all_batches when VAE is no longer needed.
    
    Args:
        runner: Runner instance containing VAE model
        debug: Debug instance for logging
        cache_model: If True, move VAE to offload_device; if False, delete completely
    """
    if runner is None or getattr(runner, 'vae', None) is None:
        return
    
    if debug:
        debug.log("Cleaning up VAE components", category="cleanup")
    
    # 1. Clear VAE-specific temporary attributes
    if hasattr(runner, 'vae'):
        temp_attrs = ['_temp_cache', '_block_cache', '_swap_cache', '_generation_cache',
                      '_rope_cache', '_intermediate_cache', '_backward_cache']
        
        for attr in temp_attrs:
            if hasattr(runner.vae, attr):
                delattr(runner.vae, attr)
    
    # 2. Handle model offloading (for caching or before deletion)
    try:
        param_device = next(runner.vae.parameters()).device
        
        # Move model off GPU if needed
        if cache_model and param_device.type not in ['meta', 'cpu']:
            offload_target = getattr(runner, '_vae_offload_device', None)
            if offload_target is None or offload_target == 'none':
                offload_target = torch.device('cpu')
            reason = "model caching" if cache_model else "releasing GPU memory"
            manage_model_device(model=runner.vae, target_device=offload_target, model_name="VAE",
                               debug=debug, reason=reason, runner=runner)
        elif param_device.type == 'meta' and debug:
            debug.log("VAE on meta device - keeping structure for cache", category="cleanup")
    except StopIteration:
        pass
    
    # 3. Complete cleanup if not caching
    if not cache_model:
        release_model_memory(model=runner.vae, debug=debug)
        runner.vae = None
        if debug:
            debug.log("VAE model deleted", category="cleanup")
        
        # Clear VAE config attributes - not needed when model is not cached (will be recreated)
        if hasattr(runner, '_vae_compile_args'):
            delattr(runner, '_vae_compile_args')
        if hasattr(runner, '_vae_tiling_config'):
            delattr(runner, '_vae_tiling_config')
    
    # 3. Clear VAE temporary attributes (should be already cleared in materialize_model)
    runner._vae_checkpoint = None
    runner._vae_dtype_override = None


def complete_cleanup(runner: Any, debug: Optional['Debug'] = None, dit_cache: bool = False, vae_cache: bool = False) -> None:
    """
    Complete cleanup of runner and remaining components with independent model caching support.
    This is a lightweight cleanup for final stage, as model-specific cleanup
    happens in their respective phases (cleanup_dit, cleanup_vae).
    
    Args:
        runner: Runner instance to clean up
        debug: Debug instance for logging
        dit_cache: If True, preserve DiT model on offload_device for future runs
        vae_cache: If True, preserve VAE model on offload_device for future runs
        
    Behavior:
        - Can cache DiT and VAE independently for flexible memory management
        - Preserves _dit_model_name and _vae_model_name when either model is cached for change detection
        - Clears all temporary attributes and runtime caches
        - Performs deep memory cleanup only when both models are fully released
        
    Note:
        Model name tracking (_dit_model_name, _vae_model_name) is only cleared if neither
        model is cached, enabling proper model change detection on subsequent runs.
    """
    if not runner:
        return
    
    if debug:
        cleanup_type = "partial cleanup" if (dit_cache or vae_cache) else "full cleanup"
        debug.log(f"Starting {cleanup_type}", category="cleanup")
    
    # 1. Cleanup any remaining models if they still exist
    # (This handles cases where phases were skipped or errored)
    if hasattr(runner, 'dit') and runner.dit is not None:
        cleanup_dit(runner=runner, debug=debug, cache_model=dit_cache)
    
    if hasattr(runner, 'vae') and runner.vae is not None:
        cleanup_vae(runner=runner, debug=debug, cache_model=vae_cache)
    
    # 2. Clear remaining runtime caches
    clear_runtime_caches(runner=runner, debug=debug)
    
    # 3. Clear config and other non-model components when fully releasing runner
    if not (dit_cache or vae_cache):
        # Full cleanup - clear config and model tracking
        runner.config = None
        runner._dit_model_name = None
        runner._vae_model_name = None
    
    # 4. Final memory cleanup
    clear_memory(debug=debug, deep=True, force=True, timer_name="complete_cleanup")
    
    # 5. Clear cuBLAS workspaces
    torch._C._cuda_clearCublasWorkspaces() if hasattr(torch._C, '_cuda_clearCublasWorkspaces') else None
    
    # Log what models are cached for next run
    if dit_cache or vae_cache:
        cached_models = []
        if dit_cache and hasattr(runner, '_dit_model_name'):
            cached_models.append(f"DiT ({runner._dit_model_name})")
        if vae_cache and hasattr(runner, '_vae_model_name'):
            cached_models.append(f"VAE ({runner._vae_model_name})")
        
        if cached_models:
            models_str = " and ".join(cached_models)
            debug.log(f"Models cached for next run: {models_str}", category="cache", force=True)
    
    if debug:
        debug.log(f"Completed {cleanup_type}", category="success")
