import pytest
import torch

from src.config import BlockSwapConfig
from src.optimization.blockswap import apply_block_swap_to_dit, is_blockswap_enabled, cleanup_blockswap
from src.utils.debug import Debug


class DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


class DummyModel(torch.nn.Module):
    def __init__(self, num_blocks=4):
        super().__init__()
        self.in_proj = torch.nn.Linear(10, 10)
        self.blocks = torch.nn.ModuleList([DummyBlock() for _ in range(num_blocks)])
        self.out_proj = torch.nn.Linear(10, 10)


class DummyRunner:
    def __init__(self, dit, debug=None, dit_device=None):
        self.dit = dit
        self.debug = debug
        if dit_device is not None:
            self._dit_device = dit_device


def test_is_blockswap_enabled():
    assert not is_blockswap_enabled(BlockSwapConfig(blocks_to_swap=0, swap_io_components=False))
    assert is_blockswap_enabled(BlockSwapConfig(blocks_to_swap=2, swap_io_components=False))
    assert is_blockswap_enabled(BlockSwapConfig(blocks_to_swap=0, swap_io_components=True))


def test_apply_block_swap_early_returns():
    debug = Debug(False)
    model = DummyModel()
    runner = DummyRunner(dit=model, debug=debug)

    # Disabled config
    cfg_disabled = BlockSwapConfig(blocks_to_swap=0, swap_io_components=False)
    apply_block_swap_to_dit(runner, cfg_disabled, debug)
    assert not getattr(runner, "_blockswap_active", False)

    # Invalid debug argument raises error if runner has no debug
    runner_no_debug = DummyRunner(dit=model)
    cfg = BlockSwapConfig(blocks_to_swap=2, swap_io_components=False)
    with pytest.raises(ValueError, match="Debug instance must be provided"):
        apply_block_swap_to_dit(runner_no_debug, cfg, None)


def test_apply_block_swap_and_cleanup():
    debug = Debug(False)
    model = DummyModel(num_blocks=4)
    runner = DummyRunner(dit=model, debug=debug, dit_device=torch.device('cpu'))

    cfg = BlockSwapConfig(blocks_to_swap=2, swap_io_components=True, offload_device=torch.device('cpu'))
    apply_block_swap_to_dit(runner, cfg, debug)

    assert runner._blockswap_active
    assert model.blocks_to_swap == 1  # 2 blocks -> 0-indexed limit is 1
    assert hasattr(model, "_block_swap_config")
    assert model._block_swap_config["blocks_swapped"] == 2
    assert model._block_swap_config["swap_io_components"] is True

    # Test cleanup
    cleanup_blockswap(runner, keep_state_for_cache=False)
    assert not runner._blockswap_active

def test_automated_swap_io_components_in_planning():
    from src.cli.arguments import parse_arguments
    from src.core.planning import plan_file
    from src.core.session import InferenceSession
    # Passing high blocks_to_swap=32 automatically sets swap_io_components=True in planning
    cfg = parse_arguments(["dummy.mp4", "--blocks_to_swap", "32", "--memory_policy", "staged"])
    with InferenceSession() as session:
        planned = plan_file(cfg, 640, 360, "cpu", session, Debug(False))
    assert planned.inference.swap.blocks_to_swap >= 30
    assert planned.inference.swap.swap_io_components is True
