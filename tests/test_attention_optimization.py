import time
import torch
import pytest
from einops import rearrange


def test_qkv_rearrange_equivalence():
    """Verify native PyTorch view produces exact same result as einops rearrange for QKV tensors."""
    seq_len = 1024
    num_heads = 16
    head_dim = 64
    inner_dim = num_heads * head_dim
    qkv_dim = inner_dim * 3

    vid_qkv = torch.randn(seq_len, qkv_dim)

    res_einops = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=head_dim)
    res_native = vid_qkv.view(vid_qkv.shape[0], 3, -1, head_dim)

    assert torch.equal(res_einops, res_native), "QKV rearrange outputs do not match!"


def test_attn_flatten_equivalence():
    """Verify native PyTorch flatten produces exact same result as einops rearrange for attn output."""
    seq_len = 1024
    num_heads = 16
    head_dim = 64

    attn = torch.randn(seq_len, num_heads, head_dim)

    res_einops = rearrange(attn, "l h d -> l (h d)")
    res_native = attn.flatten(1)

    assert torch.equal(res_einops, res_native), "Attn output rearrange outputs do not match!"


def test_repeated_text_attn_rearrange_equivalence():
    """Verify native PyTorch view/flatten produces exact same result as einops for repeated text Q/K."""
    seq_len = 256
    num_heads = 16
    head_dim = 64

    txt_q = torch.randn(seq_len, num_heads, head_dim)

    # Step 1: l h d -> l (h d)
    flat_einops = rearrange(txt_q, "l h d -> l (h d)")
    flat_native = txt_q.flatten(1)
    assert torch.equal(flat_einops, flat_native)

    # Step 2: l (h d) -> l h d
    unflat_einops = rearrange(flat_einops, "l (h d) -> l h d", h=num_heads)
    unflat_native = flat_native.view(flat_native.shape[0], num_heads, -1)
    assert torch.equal(unflat_einops, unflat_native)


def test_rearrange_performance():
    """Verify native PyTorch view is measurably faster than einops rearrange."""
    seq_len = 2048
    num_heads = 16
    head_dim = 64
    qkv_dim = num_heads * head_dim * 3

    vid_qkv = torch.randn(seq_len, qkv_dim)
    iters = 2000

    # Benchmark einops
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=head_dim)
    t_einops = time.perf_counter() - t0

    # Benchmark native
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = vid_qkv.view(vid_qkv.shape[0], 3, -1, head_dim)
    t_native = time.perf_counter() - t0

    assert t_native < t_einops, f"Native view ({t_native:.4f}s) should be faster than einops ({t_einops:.4f}s)"
