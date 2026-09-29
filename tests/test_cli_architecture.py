"""CLI isolation, immutable job settings, and typed inference integration."""

from contextlib import closing
from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import cv2
import numpy as np
import pytest
import torch

from src.cli.arguments import parse_arguments
from src.cli.bootstrap import configure_environment
from src.config import BlockSwapConfig, CompileConfig, InferenceConfig, SamplingConfig
from src.core import generation_utils, model_configuration, pipeline, scheduler
from src.core.context import GenerationContext
from src.core.session import InferenceSession
from src.media import output, reader
from src.optimization.blockswap import is_blockswap_enabled, validate_blockswap_config
from src.optimization.memory_budget import GIB
from src.utils.debug import Debug


def test_cli_import_and_help_do_not_load_inference_dependencies():
    code = '''
import importlib.abc, os, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'cv2', 'numpy', 'diffusers'}:
            raise AssertionError('Heavy import: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
before = dict(os.environ)
sys.argv = ['unrelated-host', '--not-a-seedvr-option']
import inference_cli
assert os.environ == before
from src.cli.arguments import parse_arguments
config = parse_arguments(['input.mp4'])
assert config.inference.sampling.batch_size == 5
inference_cli.main(['--help'])
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--cuda_device' in result.stdout


def test_parser_returns_nested_immutable_settings():
    cfg = parse_arguments(['in.mov', '--batch_size', '9', '--chunk_size', '17',
        '--compile_dit', '--compile_dynamic', '--compile_mode', 'reduce-overhead',
        '--vae_decode_tile_size', '512', '--cache_dit', '--cache_ram_gb', '2',
        '--blocks_to_swap', '8', '--video_backend', 'ffmpeg', '--10bit',
        '--output', 'out.mp4', '--skip_first_frames', '3', '--load_cap', '11'])
    assert cfg.inference.sampling.batch_size == 9
    assert cfg.inference.compile.dit and cfg.inference.compile.dynamic
    assert cfg.inference.compile.mode == 'reduce-overhead'
    assert cfg.inference.vae.decode.size == 512
    assert cfg.inference.cache.dit and cfg.inference.cache.ram_gb == 2
    assert cfg.inference.swap.blocks_to_swap == 8
    assert cfg.output.path == 'out.mp4' and cfg.output.use_10bit
    assert (cfg.video.chunk_size, cfg.video.skip_first_frames, cfg.video.load_cap) == (17, 3, 11)
    with pytest.raises(FrozenInstanceError):
        cfg.inference.sampling.batch_size = 13
    with pytest.raises(FrozenInstanceError):
        cfg.output.format = 'png'


@pytest.mark.parametrize('argv', [
    ['--cuda_device', '0,1'], ['--cuda_device', '-1'], ['--batch_size', '4'],
    ['--ram_budget_gb', 'nan'], ['--chunk_size', '-1'], ['--10bit'],
])
def test_parser_rejects_invalid_requests_before_loading_models(argv):
    with pytest.raises(SystemExit) as error:
        parse_arguments(['input.mp4', *argv])
    assert error.value.code == 2


def test_bootstrap_maps_visible_device_and_preserves_cpu_controls(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-a,GPU-b')
    monkeypatch.setenv('OMP_NUM_THREADS', '7')
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    assert configure_environment(1) == 'GPU-b'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-b'
    assert os.environ['OMP_NUM_THREADS'] == '7'
    assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True'
    with pytest.raises(ValueError):
        configure_environment(1)


def test_directory_jobs_do_not_mutate_auto_format_or_sampling(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'a.png').touch()
    (source / 'b.mp4').touch()
    cfg = parse_arguments([str(source)])
    jobs = list(scheduler.iter_file_jobs(cfg))
    assert [j.config.output.format for j in jobs] == ['png', 'mp4']
    assert jobs[0].destination == str(tmp_path / 'input_upscaled' / 'a.png')
    assert jobs[1].destination == str(tmp_path / 'input_upscaled' / 'b.mp4')
    assert cfg.output.format is None
    assert cfg.inference.sampling.prepend_frames == 0


def test_planner_does_not_change_next_files_settings(monkeypatch):
    from src.core import planning
    cfg = parse_arguments(['input.mp4', '--batch_size', '33', '--chunk_size', '330',
                           '--temporal_overlap', '3', '--ram_budget_gb', '2'])
    monkeypatch.setattr(planning, 'available_ram_bytes', lambda: 2 * GIB)
    with InferenceSession() as session:
        plan = planning.plan_file(cfg, 1920, 1080, 'cpu', session, Debug(False))
    assert plan.video.chunk_size < 330
    assert cfg.video.chunk_size == 330
    assert cfg.inference.sampling.batch_size == 33
    assert plan.inference.sampling.seed == cfg.inference.sampling.seed


def test_generation_context_releases_owners_without_copying_storage():
    tensor = torch.ones(4)
    ref = weakref.ref(tensor)
    ctx = GenerationContext(input_images=tensor, all_latents=[tensor])
    local = replace(ctx)
    del tensor
    ctx.clear()
    assert ref() is not None  # The other invocation view still owns its reference.
    local.clear()
    assert ref() is None
    with pytest.raises(AttributeError):
        ctx.misspelled_field = 1


def test_typed_blockswap_validation_and_compile_adapter(monkeypatch):
    swap = BlockSwapConfig(blocks_to_swap=4)
    assert is_blockswap_enabled(swap)
    assert validate_blockswap_config(swap, torch.device('cuda:0'), torch.device('cpu'), Debug(False)) is swap
    with pytest.raises(ValueError, match='invalid'):
        validate_blockswap_config(swap, torch.device('cuda:0'), torch.device('cuda:0'), Debug(False))
    compile_config = CompileConfig(backend='cudagraphs', dynamic=True, dynamo_cache_size_limit=71)
    # Keep the test's Dynamo settings local.
    monkeypatch.setattr(torch._dynamo.config, 'cache_size_limit', torch._dynamo.config.cache_size_limit)
    monkeypatch.setattr(torch._dynamo.config, 'recompile_limit', torch._dynamo.config.recompile_limit)
    settings, ok = model_configuration._configure_torch_compile(compile_config, 'DiT', Debug(False))
    assert ok and settings == dict(backend='cudagraphs', mode='default', fullgraph=False, dynamic=True)


def test_typed_runner_configuration_reuses_models(tmp_path, monkeypatch):
    runner = SimpleNamespace(dit=torch.nn.Linear(1, 1), vae=torch.nn.Linear(1, 1))
    monkeypatch.setattr(model_configuration, '_create_new_runner', lambda *a: runner)
    seen = []
    monkeypatch.setattr(model_configuration, '_setup_models', lambda r, dit, vae, *a: seen.append((dit, vae)))
    cfg = CompileConfig(dit=True)
    swap = BlockSwapConfig()
    ctx = generation_utils.setup_generation_context(dit_device='cpu', vae_device='cpu')
    with InferenceSession() as session:
        for _ in range(2):
            actual = generation_utils.prepare_runner('dit', 'vae', str(tmp_path), Debug(False), ctx,
                dit_cache=True, vae_cache=True, session=session,
                block_swap_config=swap, torch_compile_args_dit=cfg)
            assert actual is runner
            assert runner._new_dit_compile_args is cfg
            assert runner._new_dit_block_swap_config is swap
    assert len(seen) == 2 and all(dit is not None and vae is not None for dit, vae in seen)


def test_pipeline_real_phases_use_typed_context_and_preserve_pixels(monkeypatch):
    """Run actual encode/upscale/decode/postprocess plumbing with tiny identity models."""
    from omegaconf import OmegaConf
    class Runner:
        def __init__(self):
            self.dit = torch.nn.Linear(1, 1).bfloat16()
            self.vae = torch.nn.Linear(1, 1).bfloat16()
            self.config = OmegaConf.create({'diffusion': {'cfg': {}, 'timesteps': {'sampling': {}}}})
            self.tile_debug, self.encode_tiled, self.decode_tiled = 'false', False, False
        def vae_encode(self, videos):
            return videos
        def vae_decode(self, latents):
            return latents
        def configure_diffusion(self, **kwargs):
            pass
        def get_condition(self, noise, *, task, latent_blur):
            return latent_blur
        def inference(self, *, conditions, **kwargs):
            return conditions
    captured = []
    def prepare(**kwargs):
        captured.append(kwargs['ctx'])
        return kwargs['session'].acquire(('identity',), Runner)
    monkeypatch.setattr(pipeline, 'prepare_runner', prepare)
    monkeypatch.setattr(pipeline, 'load_text_embeddings', lambda *a: {})
    raw = torch.full((5, 64, 64, 3), 128, dtype=torch.uint8)
    cfg = InferenceConfig(sampling=SamplingConfig(resolution=64, batch_size=5, color_correction='none'))
    with InferenceSession() as session:
        with closing(session.iter_batches(raw, cfg, 'cpu')) as batches:
            result = torch.cat(list(batches))
    assert result.shape == raw.shape
    torch.testing.assert_close(result.float(), raw.float() / 255, atol=.005, rtol=0)
    assert captured[0].input_images is None and captured[0].all_latents is None
    assert captured[0].all_upscaled_latents is None and captured[0].text_embeds is None


def test_scheduler_streams_video_range_to_png_with_one_session(tmp_path, monkeypatch):
    video = tmp_path / 'input.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 24, (16, 16))
    if not writer.isOpened():
        pytest.skip('MJPG encoder unavailable')
    for i in range(8):
        writer.write(np.full((16, 16, 3), i * 20, dtype=np.uint8))
    writer.release()
    cfg = parse_arguments([str(video), '--output_format', 'png', '--output', str(tmp_path / 'frames'),
                           '--skip_first_frames', '2', '--load_cap', '5', '--chunk_size', '2'])
    monkeypatch.setattr(scheduler, 'plan_file', lambda cfg, *a: cfg)
    identities = []
    def identity(self, frames, config, *args):
        identities.append(id(self))
        yield frames.float() / 255
    monkeypatch.setattr(InferenceSession, 'iter_batches', identity)
    assert scheduler.run_jobs(cfg, 'cpu', Debug(False)) == 5
    files = sorted((tmp_path / 'frames' / 'input').glob('*.png'))
    assert len(files) == 5 and len(set(identities)) == 1
    with reader.VideoReader(str(video), skip=2, limit=5) as source:
        expected = source.read(5).numpy()
    for path, frame in zip(files, expected, strict=True):
        np.testing.assert_array_equal(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), frame)


def test_image_to_opencv_video_closes_writer_before_publication(tmp_path, monkeypatch):
    image = tmp_path / 'input.png'
    cv2.imwrite(str(image), np.full((32, 32, 3), 128, dtype=np.uint8))
    destination = tmp_path / 'output.mp4'
    cfg = parse_arguments([str(image), '--output_format', 'mp4', '--output', str(destination)])
    monkeypatch.setattr(scheduler, 'plan_file', lambda cfg, *a: cfg)
    def identity(self, frames, config, *args):
        yield frames.float() / 255
    monkeypatch.setattr(InferenceSession, 'iter_batches', identity)
    assert scheduler.run_jobs(cfg, 'cpu', Debug(False)) == 1
    with reader.VideoReader(str(destination)) as source:
        assert source.count == 1
        assert source.read(1).shape == (1, 32, 32, 3)
    assert not list(tmp_path.glob('.seedvr-*'))
