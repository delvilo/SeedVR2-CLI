import pytest
import torch

from src.common.cache import Cache
from src.models.dit_3b.normalization import get_norm_layer as get_norm_layer_3b
from src.models.dit_7b.normalization import get_norm_layer as get_norm_layer_7b
from src.models.dit_3b.nablocks.attention.mmattn import NaMMAttention, NaSwinAttention as NaSwinAttention_3b
from src.models.dit_7b.nablocks.mmsr_block import NaSwinAttention as NaSwinAttention_7b


def test_dit_3b_na_mm_attention():
    cache = Cache()
    rms_norm = get_norm_layer_3b("rms")

    attn = NaMMAttention(
        vid_dim=1152,
        txt_dim=1152,
        heads=16,
        head_dim=72,
        qk_bias=True,
        qk_norm=rms_norm,
        qk_norm_eps=1e-6,
        rope_type="mmrope3d",
        rope_dim=36,
        shared_weights=False,
        attention_mode='sdpa',
    )

    L_vid = 1024
    L_txt = 256
    vid = torch.randn(L_vid, 1152)
    txt = torch.randn(L_txt, 1152)
    vid_shape = torch.tensor([[4, 16, 16]], dtype=torch.long)
    txt_shape = torch.tensor([[256]], dtype=torch.long)

    out_v, out_t = attn(vid, txt, vid_shape, txt_shape, cache)

    assert out_v.shape == vid.shape
    assert out_t.shape == txt.shape
    assert not torch.isnan(out_v).any()
    assert not torch.isnan(out_t).any()


def test_dit_3b_na_swin_attention():
    rms_norm = get_norm_layer_3b("rms")

    attn = NaSwinAttention_3b(
        vid_dim=1152,
        txt_dim=1152,
        heads=16,
        head_dim=72,
        qk_bias=True,
        qk_norm=rms_norm,
        qk_norm_eps=1e-6,
        rope_type="mmrope3d",
        rope_dim=36,
        shared_weights=False,
        window=(2, 4, 4),
        window_method="720pwin_by_size_bysize",
        attention_mode='sdpa',
    )

    L_vid = 1024
    L_txt = 256
    vid = torch.randn(L_vid, 1152)
    txt = torch.randn(L_txt, 1152)
    vid_shape = torch.tensor([[4, 16, 16]], dtype=torch.long)
    txt_shape = torch.tensor([[256]], dtype=torch.long)

    out_v, out_t = attn(vid, txt, vid_shape, txt_shape, Cache())

    assert out_v.shape == vid.shape
    assert out_t.shape == txt.shape
    assert not torch.isnan(out_v).any()
    assert not torch.isnan(out_t).any()


def test_dit_7b_na_swin_attention():
    rms_norm = get_norm_layer_7b("rms")

    attn = NaSwinAttention_7b(
        vid_dim=2048,
        txt_dim=2048,
        heads=16,
        head_dim=128,
        qk_bias=True,
        qk_rope=True,
        qk_norm=rms_norm,
        qk_norm_eps=1e-6,
        window=(2, 4, 4),
        window_method="720pwin_by_size_bysize",
        shared_qkv=False,
        attention_mode='sdpa',
    )

    L_vid = 1024
    L_txt = 256
    vid = torch.randn(L_vid, 2048)
    txt = torch.randn(L_txt, 2048)
    vid_shape = torch.tensor([[4, 16, 16]], dtype=torch.long)
    txt_shape = torch.tensor([[256]], dtype=torch.long)

    out_v, out_t = attn(vid, txt, vid_shape, txt_shape, Cache())

    assert out_v.shape == vid.shape
    assert out_t.shape == txt.shape
    assert not torch.isnan(out_v).any()
    assert not torch.isnan(out_t).any()
