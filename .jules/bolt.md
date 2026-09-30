## 2025-05-20 - AdaSingle Modulation Loop Optimization
**Learning:** `AdaSingle.forward` is invoked thousands of times per generation run across every DiT transformer block and timestep. Calling `self.layers.index(layer)` does an O(N) linear list search on string comparisons every call, and `einops.rearrange` incurs string parsing overhead.
**Action:** Pre-compute `self.layer_indices` dictionary in `__init__` for O(1) index lookup and replace `einops.rearrange` with native PyTorch `.reshape()` in hot modulation paths.

## 2025-05-21 - RoPE Rotary Embedding Hot Path Optimization
**Learning:** `NaMMRotaryEmbedding3d` and `NaRotaryEmbedding3d` forward passes execute `einops.rearrange` on every query and key tensor thousands of times across diffusion timesteps and attention blocks. The string pattern parsing in `einops` creates significant overhead in hot tensor operations.
**Action:** Replace `einops.rearrange` in RoPE forward paths with native PyTorch `.transpose(0, 1)`, `.view()`, and `.flatten()` operations.
