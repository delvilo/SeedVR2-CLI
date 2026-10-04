import torch
from src.common.cache import Cache
from src.models.dit_3b.normalization import get_norm_layer
from src.models.dit_3b.nablocks.attention.mmattn import NaMMAttention, NaSwinAttention as NaSwinAttention3b
from src.models.dit_7b.nablocks.mmsr_block import NaSwinAttention as NaSwinAttention7b


def test_nam_attention_3b_forward():
    torch.manual_seed(42)
    vid_dim = 64
    txt_dim = 64
    heads = 4
    head_dim = 16

    attn = NaMMAttention(
        vid_dim=vid_dim,
        txt_dim=txt_dim,
        heads=heads,
        head_dim=head_dim,
        qk_bias=True,
        qk_norm=get_norm_layer("fusedrms"),
        qk_norm_eps=1e-6,
        rope_type=None,
        rope_dim=16,
        shared_weights=False,
        attention_mode='sdpa',
    )

    L_vid = 32
    L_txt = 16
    vid = torch.randn(L_vid, vid_dim)
    txt = torch.randn(L_txt, txt_dim)
    vid_shape = torch.tensor([[1, 4, 8]], dtype=torch.long)
    txt_shape = torch.tensor([[16]], dtype=torch.long)
    cache = Cache()

    vid_out, txt_out = attn(vid, txt, vid_shape, txt_shape, cache)
    assert vid_out.shape == (L_vid, vid_dim)
    assert txt_out.shape == (L_txt, txt_dim)


def test_naswin_attention_7b_forward():
    torch.manual_seed(42)
    vid_dim = 64
    txt_dim = 64
    heads = 4
    head_dim = 16

    attn = NaSwinAttention7b(
        vid_dim=vid_dim,
        txt_dim=txt_dim,
        heads=heads,
        head_dim=head_dim,
        qk_bias=True,
        qk_rope=False,
        qk_norm=get_norm_layer("fusedrms"),
        qk_norm_eps=1e-6,
        window=(1, 4, 4),
        window_method='720pwin_by_size_bysize',
        shared_qkv=False,
        attention_mode='sdpa',
    )

    L_vid = 32
    L_txt = 16
    vid = torch.randn(L_vid, vid_dim)
    txt = torch.randn(L_txt, txt_dim)
    vid_shape = torch.tensor([[1, 4, 8]], dtype=torch.long)
    txt_shape = torch.tensor([[16]], dtype=torch.long)
    cache = Cache()

    vid_out, txt_out = attn(vid, txt, vid_shape, txt_shape, cache)
    assert vid_out.shape == (L_vid, vid_dim)
    assert txt_out.shape == (L_txt, txt_dim)
