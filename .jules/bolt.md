## 2025-05-20 - AdaSingle Modulation Loop Optimization
**Learning:** `AdaSingle.forward` is invoked thousands of times per generation run across every DiT transformer block and timestep. Calling `self.layers.index(layer)` does an O(N) linear list search on string comparisons every call, and `einops.rearrange` incurs string parsing overhead.
**Action:** Pre-compute `self.layer_indices` dictionary in `__init__` for O(1) index lookup and replace `einops.rearrange` with native PyTorch `.reshape()` in hot modulation paths.
