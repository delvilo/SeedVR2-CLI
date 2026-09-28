"""One CLI session owns one runner and its bounded model cache."""

from pathlib import Path


def checkpoint_identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat() if path.is_file() else None
    return (str(path), stat.st_size if stat else None, stat.st_mtime_ns if stat else None)


def resident_bytes(model):
    """Count distinct storage, including quantized buffers and tied weights."""
    totals = {'cpu': 0, 'cuda': 0}
    seen = set()
    if model is not None:
        from itertools import chain
        for tensor in chain(model.parameters(), model.buffers()):
            if tensor.device.type not in totals:
                continue
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            if key not in seen:
                totals[tensor.device.type] += storage.nbytes()
                seen.add(key)
    return totals


class InferenceSession:
    def __init__(self, ram_cache_bytes=0, vram_cache_bytes=0, debug=None):
        self.runner = None
        self.key = None
        self.ram_cache_bytes = ram_cache_bytes
        self.vram_cache_bytes = vram_cache_bytes
        self.debug = debug

    def acquire(self, key, factory):
        if key != self.key:
            self.close()
            self.key = key
        if self.runner is None:
            self.runner = factory()
        return self.runner

    def finish(self, keep_dit=False, keep_vae=False):
        if self.runner is None:
            return
        from ..optimization.memory_manager import cleanup_dit, cleanup_vae, clear_runtime_caches
        if not keep_dit:
            cleanup_dit(self.runner, self.debug, cache_model=False)
        if not keep_vae:
            cleanup_vae(self.runner, self.debug, cache_model=False)
        clear_runtime_caches(self.runner, self.debug)
        sizes = {name: resident_bytes(getattr(self.runner, name, None)) for name in ('dit', 'vae')}
        # Evict the largest model first. A zero budget disables that cache tier.
        for name in sorted(sizes, key=lambda n: sum(sizes[n].values()), reverse=True):
            if (sum(v['cpu'] for v in sizes.values()) <= self.ram_cache_bytes and
                    sum(v['cuda'] for v in sizes.values()) <= self.vram_cache_bytes):
                break
            if self.debug:
                self.debug.log(f'Evicting {name} to respect model cache budget', category='cache', force=True)
            (cleanup_dit if name == 'dit' else cleanup_vae)(self.runner, self.debug, cache_model=False)
            sizes[name] = {'cpu': 0, 'cuda': 0}

    def close(self):
        if self.runner is not None:
            from ..optimization.memory_manager import complete_cleanup
            runner, self.runner = self.runner, None
            complete_cleanup(runner, self.debug)
        self.key = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
