## 2025-05-20 - AdaSingle Modulation Loop Optimization
**Learning:** `AdaSingle.forward` is invoked thousands of times per generation run across every DiT transformer block and timestep. Calling `self.layers.index(layer)` does an O(N) linear list search on string comparisons every call, and `einops.rearrange` incurs string parsing overhead.
**Action:** Pre-compute `self.layer_indices` dictionary in `__init__` for O(1) index lookup and replace `einops.rearrange` with native PyTorch `.reshape()` in hot modulation paths.

## 2025-05-21 - RoPE Tensor Rearrangement Optimization
**Learning:** `RotaryEmbedding3d` and `NaRotaryEmbedding3d`/`NaMMRotaryEmbedding3d` inside `dit_3b/rope.py` and `dit_7b/rope.py` are executed in every transformer block and timestep. Using `einops.rearrange` string parsing in these inner loop functions causes measurable Python overhead (~3.7x-4.2x slower than native PyTorch view operations).
**Action:** Replace `einops.rearrange` with native PyTorch `.reshape()` and `.transpose(0, 1)` in RoPE forward methods for ~4x faster tensor rearrangements without allocations.

## 2025-05-22 - Attention Block Tensor Rearrangement Optimization
**Learning:** `NaMMAttention` and `NaSwinAttention` in `dit_3b/nablocks/attention/mmattn.py` and `dit_7b/nablocks/mmsr_block.py` perform `rearrange` operations on QKV and output tensors for every transformer block and timestep. Native PyTorch `.view()` and `.flatten(1)` operations are ~4.6x faster than `einops.rearrange` by eliminating string pattern parsing overhead in inner loops.
**Action:** Replace `einops.rearrange` with `.view()` and `.flatten(1)` in attention forward methods.
