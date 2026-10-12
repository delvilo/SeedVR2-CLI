import torch
import torch.nn as nn

from src.core.pipeline import parse_offload_device
from src.core.generation_utils import setup_generation_context
from src.core.alpha_upscaling import guided_filter_pytorch
from src.models.dit_3b.rope import RotaryEmbedding3d as RotaryEmbedding3d_3b
from src.models.dit_7b.rope import RotaryEmbedding3d as RotaryEmbedding3d_7b
from src.models.dit_3b.modulation import AdaSingle as AdaSingle_3b
from src.models.dit_7b.modulation import AdaSingle as AdaSingle_7b
from src.config import BlockSwapConfig
from src.optimization.blockswap import apply_block_swap_to_dit
from src.optimization.memory_manager import manage_model_device
from src.utils.debug import Debug


def test_parse_offload_device():
    assert parse_offload_device(None) is None
    assert parse_offload_device(None, cache_enabled=True) == "cpu"
    assert parse_offload_device("none") is None
    assert parse_offload_device("none", cache_enabled=True) == "cpu"
    assert parse_offload_device("cuda") == "cuda"
    assert parse_offload_device("cuda:0") == "cuda:0"
    assert parse_offload_device("0") == "cuda:0"
    assert parse_offload_device("cpu") == "cpu"


def test_normalize_device_none():
    ctx = setup_generation_context(
        vae_offload_device="none",
        dit_offload_device="none",
        tensor_offload_device="none",
    )
    assert ctx.vae_offload_device is None
    assert ctx.dit_offload_device is None
    assert ctx.tensor_offload_device is None


def test_manage_model_device_string():
    m = nn.Linear(4, 4)
    # Target device passed as str should be normalized without AttributeError
    result = manage_model_device(m, "cpu", "LinearTest")
    assert result is False  # already on cpu


def test_guided_filter_pytorch_half_precision():
    guide = torch.rand(1, 3, 16, 16, dtype=torch.bfloat16)
    src = torch.rand(1, 1, 16, 16, dtype=torch.bfloat16)
    # Should not raise NameError: name 'debug' is not defined
    res = guided_filter_pytorch(guide, src)
    assert res.shape == (1, 1, 16, 16)
    assert res.dtype == torch.bfloat16


def test_rope_device_handling():
    dim = 16
    rope_3b = RotaryEmbedding3d_3b(dim=dim)
    rope_7b = RotaryEmbedding3d_7b(dim=dim)

    # Pre-cache freqs
    _ = rope_3b.get_axial_freqs(2, 4, 4)
    _ = rope_7b.get_axial_freqs(2, 4, 4)

    b, h, T, H, W, d = 1, 2, 2, 4, 4, dim
    L = T * H * W

    q = torch.randn(b, h, L, d, dtype=torch.bfloat16)
    k = torch.randn(b, h, L, d, dtype=torch.bfloat16)

    out_q_3b, out_k_3b = rope_3b(q, k, size=(T, H, W))
    assert out_q_3b.shape == q.shape
    assert out_q_3b.dtype == torch.bfloat16

    out_q_7b, out_k_7b = rope_7b(q, k, size=(T, H, W))
    assert out_q_7b.shape == q.shape
    assert out_q_7b.dtype == torch.bfloat16


def test_adasingle_layer_indices_fallback():
    ada_3b = AdaSingle_3b(dim=10, emb_dim=60, layers=["in", "out"])
    del ada_3b.layer_indices  # simulate unpickled / deserialized instance
    hid = torch.randn(2, 10)
    emb = torch.randn(2, 60)
    out = ada_3b(hid, emb, layer="in", mode="in")
    assert out.shape == hid.shape

    ada_7b = AdaSingle_7b(dim=10, emb_dim=60, layers=["in", "out"])
    del ada_7b.layer_indices
    out_7b = ada_7b(hid, emb, layer="in", mode="in")
    assert out_7b.shape == hid.shape


def test_blockswap_buffer_only_and_dtype_to():
    class ModelWithBufferOnlyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(24, 24)
            self.blocks = nn.ModuleList([nn.Linear(24, 24)])
            self.rope = RotaryEmbedding3d_3b(dim=24)

        def forward(self, x):
            x = self.in_proj(x)
            for b in self.blocks:
                x = b(x)
            return x

    debug = Debug(False)
    model = ModelWithBufferOnlyModule()

    class DummyRunner:
        def __init__(self, dit, debug):
            self.dit = dit
            self.debug = debug
            self._dit_device = torch.device("cpu")
    runner = DummyRunner(dit=model, debug=debug)
    cfg = BlockSwapConfig(
        blocks_to_swap=1, swap_io_components=True, offload_device=torch.device("cpu")
    )
    apply_block_swap_to_dit(runner, cfg, debug)

    # Test 1: model.to(dtype=...) must not raise TypeError
    model.to(dtype=torch.float16)
    assert model.in_proj.weight.dtype == torch.float16

    # Test 2: rope forward with swap_io_components must not raise StopIteration
    q = torch.randn(1, 2, 8, 24, dtype=torch.float16)
    k = torch.randn(1, 2, 8, 24, dtype=torch.float16)
    out_q, out_k = model.rope(q, k, size=(1, 4, 2))
    assert out_q.shape == q.shape

def test_estimate_keep_on_vram_gb():
    from src.utils.model_registry import estimate_keep_on_vram_gb
    # 3B Q4 at 1080p, batch 5
    vram_3b = estimate_keep_on_vram_gb("seedvr2_ema_3b-Q4_K_M.gguf", 1080, 5)
    assert 6.0 <= vram_3b <= 8.5
    # 7B Q8 at 1080p, batch 5
    vram_7b_q8 = estimate_keep_on_vram_gb("seedvr2_ema_7b-Q8_0.gguf", 1080, 5)
    assert 13.0 <= vram_7b_q8 <= 17.0
    # 7B fp16 at 1080p, batch 5
    vram_7b_fp16 = estimate_keep_on_vram_gb("seedvr2_ema_7b_fp16.safetensors", 1080, 5)
    assert vram_7b_fp16 >= 20.0


def test_cli_keep_on_vram_flags():
    from src.cli.arguments import parse_arguments
    cfg = parse_arguments(["dummy.mp4", "--keep_on_vram"])
    assert cfg.inference.keep_on_vram is True
    assert cfg.inference.offload.dit == "none"
    assert cfg.inference.offload.vae == "none"
    assert cfg.inference.offload.tensor == "none"
    assert cfg.inference.cache.dit is True
    assert cfg.inference.cache.vae is True
    assert cfg.inference.swap.blocks_to_swap == 0
