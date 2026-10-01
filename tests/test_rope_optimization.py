import pytest
import torch

from src.common.cache import Cache
from src.models.dit_3b.rope import (
    RotaryEmbedding3d as RotaryEmbedding3d_3b,
    NaMMRotaryEmbedding3d as NaMMRotaryEmbedding3d_3b,
)
from src.models.dit_7b.rope import (
    RotaryEmbedding3d as RotaryEmbedding3d_7b,
    NaRotaryEmbedding3d as NaRotaryEmbedding3d_7b,
)


def test_dit_3b_rotary_embedding_3d():
    dim = 64
    rope = RotaryEmbedding3d_3b(dim=dim)

    b, h, T, H, W, d = 1, 4, 2, 4, 4, dim
    L = T * H * W

    q = torch.randn(b, h, L, d)
    k = torch.randn(b, h, L, d)

    out_q, out_k = rope(q, k, size=(T, H, W))

    assert out_q.shape == q.shape
    assert out_k.shape == k.shape
    assert not torch.isnan(out_q).any()
    assert not torch.isnan(out_k).any()


def test_dit_3b_na_mm_rotary_embedding_3d():
    dim = 60
    rope = NaMMRotaryEmbedding3d_3b(dim=dim)
    cache = Cache()

    L_vid = 16
    L_txt = 8
    h = 4
    d = dim

    vid_q = torch.randn(L_vid, h, d)
    vid_k = torch.randn(L_vid, h, d)
    txt_q = torch.randn(L_txt, h, d)
    txt_k = torch.randn(L_txt, h, d)

    vid_shape = torch.tensor([[2, 2, 4]], dtype=torch.long)  # 2*2*4 = 16
    txt_shape = torch.tensor([[8]], dtype=torch.long)

    out_vid_q, out_vid_k, out_txt_q, out_txt_k = rope(
        vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache
    )

    assert out_vid_q.shape == vid_q.shape
    assert out_vid_k.shape == vid_k.shape
    assert out_txt_q.shape == txt_q.shape
    assert out_txt_k.shape == txt_k.shape
    assert not torch.isnan(out_vid_q).any()
    assert not torch.isnan(out_txt_q).any()


def test_dit_7b_rotary_embedding_3d():
    dim = 64
    rope = RotaryEmbedding3d_7b(dim=dim)

    b, h, T, H, W, d = 1, 4, 2, 4, 4, dim
    L = T * H * W

    q = torch.randn(b, h, L, d)
    k = torch.randn(b, h, L, d)

    out_q, out_k = rope(q, k, size=(T, H, W))

    assert out_q.shape == q.shape
    assert out_k.shape == k.shape
    assert not torch.isnan(out_q).any()
    assert not torch.isnan(out_k).any()


def test_dit_7b_na_rotary_embedding_3d():
    dim = 64
    rope = NaRotaryEmbedding3d_7b(dim=dim)
    cache = Cache()

    L = 16
    h = 4
    d = dim

    q = torch.randn(L, h, d)
    k = torch.randn(L, h, d)
    shape = torch.tensor([[2, 2, 4]], dtype=torch.long)

    out_q, out_k = rope(q, k, shape, cache)

    assert out_q.shape == q.shape
    assert out_k.shape == k.shape
    assert not torch.isnan(out_q).any()
    assert not torch.isnan(out_k).any()
