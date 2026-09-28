"""CPU regression coverage for ownership, media precision, and bounded streaming.

The diffusion weights are deliberately replaced by tiny runners. These tests do
not claim GPU quality or performance coverage; see docs/memory.md for that gate.
"""
from argparse import Namespace
from contextlib import closing
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import weakref

import cv2
import numpy as np
import pytest
import torch

import inference_cli as cli
from src.core.generation_phases import decode_all_batches, postprocess_all_batches, _prepare_video_batch
from src.core.generation_utils import blend_overlapping_frames
from src.core.session import InferenceSession, resident_bytes
from src.core.streaming import blend_decoded_batches, iter_output_batches
from src.optimization.memory_budget import GIB, plan_memory, reduce_workload, is_oom
from src.optimization.memory_manager import release_tensor_collection
from src.optimization.memory_profile import MemoryRecorder
from src.utils.debug import Debug
from src.utils.media import read_frames, frame_pixels, FFMPEGVideoWriter, OutputTransaction

# Small CPU test tensors should not start a large OpenMP thread pool.
torch.set_num_threads(1)


class Capture:
    def __init__(self, count, shape=(8, 10, 3)):
        self.count, self.pos, self.shape = count, 0, shape
        self.closed = False

    def read(self):
        if self.pos == self.count:
            return False, None
        result = np.full(self.shape, self.pos, dtype=np.uint8)
        self.pos += 1
        return True, result

    def release(self):
        self.closed = True


def args(monkeypatch, **values):
    monkeypatch.setattr(sys, 'argv', ['inference_cli.py', 'input.mp4'])
    result = cli.parse_arguments()
    result.output_format = 'mp4'
    vars(result).update(values)
    return result


def test_uint8_input_and_per_batch_normalization():
    cap = Capture(7)
    raw = read_frames(cap, 5)
    assert raw.dtype == torch.uint8 and raw.shape == (5, 8, 10, 3)
    expected = raw[1:4].float().div(255).half().permute(0, 3, 1, 2)
    torch.testing.assert_close(_prepare_video_batch(raw, 1, 4), expected)
    assert read_frames(cap, 5).shape[0] == 2
    assert read_frames(cap, 5) is None


def test_frame_quantization_preserves_input_and_10bit_transport():
    frame = torch.linspace(0, 1, 1024).view(1, 1024, 1).expand(-1, -1, 3)
    original = frame.clone()
    eight, sixteen = frame_pixels(frame), frame_pixels(frame, True)
    assert eight.dtype == np.uint8 and sixteen.dtype == np.dtype('<u2')
    assert len(np.unique(eight)) == 256
    assert len(np.unique(sixteen)) == 1024
    torch.testing.assert_close(frame, original, rtol=0, atol=0)
    assert frame_pixels(frame.bfloat16()).shape == frame.shape


@pytest.mark.parametrize('overlap', [0, 1, 2, 3, 4])
@pytest.mark.parametrize('lengths', [[5], [5, 5, 4], [5] * 10 + [1]])
def test_blend_matches_whole_array_reference(overlap, lengths):
    torch.manual_seed(13)
    batches = [torch.rand(n, 4, 6, 3, dtype=torch.bfloat16) for n in lengths]
    result = torch.empty(0, 4, 6, 3, dtype=torch.bfloat16)
    metadata = []
    for i, batch in enumerate(batches):
        skipped = 0
        if i and overlap and len(batch) > overlap and len(result) >= overlap:
            result[-overlap:] = blend_overlapping_frames(result[-overlap:], batch[:overlap], overlap)
            batch = batch[overlap:]
            skipped = overlap
        metadata.append((i, skipped))
        result = torch.cat((result, batch))
    actual = list(blend_decoded_batches(enumerate([b.clone() for b in batches]), overlap))
    assert [(i, skipped) for i, _, skipped in actual] == metadata
    torch.testing.assert_close(torch.cat([batch for _, batch, _ in actual]), result, atol=0, rtol=0)


def test_blend_does_not_hold_a_growing_history():
    refs = []
    def source():
        for index in range(200):
            tensor = torch.full((5, 8, 8, 3), index / 200)
            refs.append(weakref.ref(tensor))
            assert sum(ref() is not None for ref in refs) <= 4
            yield index, tensor
    for _, output, _ in blend_decoded_batches(source(), 4):
        assert len(output) <= 5
        del output


def fake_runner(decoded):
    runner = SimpleNamespace(vae=torch.nn.Linear(1, 1), dit=None, tile_debug='false', decode_tiled=True)
    iterator = iter(decoded)
    runner.vae_decode = lambda _: [next(iterator).clone()]
    return runner


def context(lengths=(5, 5, 4), overlap=3):
    total = lengths[0] + sum(n - overlap for n in lengths[1:])
    starts = [i * (lengths[0] - overlap) for i in range(len(lengths))]
    return dict(all_upscaled_latents=[torch.zeros(1) for _ in lengths], all_ori_lengths=list(lengths),
                true_target_dims=(8, 8), total_frames=total, tensor_offload_device=torch.device('cpu'),
                vae_device=torch.device('cpu'), postprocess_device=torch.device('cpu'), compute_dtype=torch.bfloat16,
                actual_temporal_overlap=overlap, is_rgba=False,
                input_images=torch.rand(total, 8, 8, 3).half(),
                batch_metadata=[(start, start + n, 0) for start, n in zip(starts, lengths)],
                video_transform=lambda x: (x * 2 - 1).permute(1, 0, 2, 3))


@pytest.mark.parametrize('color', ['none', 'lab', 'adain', 'hsv', 'wavelet', 'wavelet_adaptive'])
def test_streamed_postprocess_matches_four_phase_reference(color):
    torch.manual_seed(7)
    ctx = context()
    decoded = [torch.rand(3, n, 8, 8).bfloat16().mul(2).sub(1) for n in ctx['all_ori_lengths']]
    reference = copy.deepcopy(ctx)
    debug = Debug(enabled=False)
    decode_all_batches(fake_runner(decoded), reference, debug)
    postprocess_all_batches(reference, debug, color_correction=color)
    result = torch.cat(list(iter_output_batches(fake_runner(decoded), ctx, debug, color)))
    torch.testing.assert_close(result, reference['final_video'], atol=0, rtol=0)
    assert 'final_video' not in ctx
    assert 'all_upscaled_latents' not in ctx


def test_rgba_alpha_matches_existing_processing():
    ctx = context(lengths=(1,), overlap=0)
    ctx.update(is_rgba=True, all_alpha_channels=[torch.full((1, 1, 8, 8), .25)],
               all_input_rgb=[torch.full((1, 3, 8, 8), .5)])
    decoded = [torch.zeros(3, 1, 8, 8, dtype=torch.bfloat16)]
    reference = copy.deepcopy(ctx)
    decode_all_batches(fake_runner(decoded), reference, Debug(False))
    postprocess_all_batches(reference, Debug(False), color_correction='none')
    result = torch.cat(list(iter_output_batches(fake_runner(decoded), ctx, Debug(False), 'none')))
    torch.testing.assert_close(result, reference['final_video'], rtol=0, atol=0)
    assert result.shape == (1, 8, 8, 4)
    torch.testing.assert_close(result[..., :3], torch.full_like(result[..., :3], .5))
    assert torch.isfinite(result).all()
    assert (result[..., 3] < 1).all()
    assert ctx['all_alpha_channels'] == [None]


def test_generator_close_releases_models_and_latents():
    ctx = context()
    decoded = [torch.zeros(3, n, 8, 8, dtype=torch.bfloat16) for n in ctx['all_ori_lengths']]
    runner = fake_runner(decoded)
    with closing(iter_output_batches(runner, ctx, Debug(False), 'none', cache_model=True)) as outputs:
        next(outputs)
    assert runner.vae is None
    assert 'all_upscaled_latents' not in ctx


def test_tiled_vae_accumulation_preserves_pixels_on_cpu():
    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKL
    fake = SimpleNamespace(spatial_downsample_factor=1, tensor_offload_device='cpu', debug=None)
    fake.slicing_decode = lambda z: z.clone()
    value = torch.rand(1, 3, 5, 8, 10)
    result = VideoAutoencoderKL.tiled_decode(fake, value, tile_size=(6, 6), tile_overlap=(2, 2))
    assert result.device.type == 'cpu'
    torch.testing.assert_close(result, value)


def test_release_collection_drops_owners_without_corrupting_aliases():
    tensor = torch.arange(10.)
    alias = tensor[2:5]
    refs = {'nested': [tensor]}
    release_tensor_collection(refs)
    assert refs == {}
    torch.testing.assert_close(alias, torch.tensor([2., 3., 4.]))


def test_session_budgets_identity_and_no_result_context():
    session = InferenceSession(ram_cache_bytes=16)
    runner = session.acquire(('weights-a',), lambda: SimpleNamespace(dit=torch.nn.Linear(4, 4), vae=torch.nn.Linear(1, 1)))
    assert session.acquire(('weights-a',), lambda: None) is runner
    session.finish(True, True)
    assert runner.dit is None and runner.vae is not None
    assert not hasattr(session, 'ctx')
    replacement = session.acquire(('weights-b',), lambda: SimpleNamespace(dit=None, vae=None))
    assert replacement is not runner and runner.vae is None
    session.close()
    assert session.runner is None


def test_resident_bytes_deduplicates_tied_storage():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.ones(16))
    model.register_buffer('tied', model.a.detach())
    model.register_buffer('quant_scale', torch.ones(2))
    assert resident_bytes(model) == {'cpu': 18 * 4, 'cuda': 0}


def test_budget_and_oom_ladder(monkeypatch):
    plan = plan_memory(1920, 1080, 1080, 0, 330, 33, 3, 2 * GIB, 10 * GIB, 3 * GIB)
    assert 0 < plan.chunk_size <= 330
    assert plan.batch_size % 4 == 1 and plan.batch_size <= 33
    assert plan.estimated_working_bytes <= plan.ram_bytes
    assert plan_memory(640, 360, 360, 0, 0, 5, 0, 2 * GIB).chunk_size <= 33
    with pytest.raises(MemoryError):
        plan_memory(1920, 1080, 2160, 0, 33, 33, 3, 1)
    options = args(monkeypatch, batch_size=33, chunk_size=100, temporal_overlap=3)
    assert reduce_workload(options)
    assert options.seed == 42 and options.resolution == 1080
    assert options.batch_size < 33 and options.batch_size % 4 == 1
    assert options.temporal_overlap < options.batch_size
    assert is_oom(torch.cuda.OutOfMemoryError('allocation'))
    assert is_oom(MemoryError())
    assert not is_oom(ValueError('bad input'))


def test_transaction_keeps_old_file_on_error(tmp_path):
    dest = tmp_path / 'output.mp4'
    dest.write_bytes(b'old')
    with pytest.raises(MemoryError):
        with OutputTransaction(dest) as output:
            output.path.write_bytes(b'partial')
            raise MemoryError()
    assert dest.read_bytes() == b'old'
    with OutputTransaction(dest) as output:
        output.path.write_bytes(b'new')
        output.commit()
    assert dest.read_bytes() == b'new'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['output.mp4']


def test_png_sequence_removes_stale_frames_only(tmp_path):
    dest = tmp_path / 'sequence'
    dest.mkdir()
    (dest / 'clip_000002.png').write_bytes(b'old tail')
    (dest / 'notes.txt').write_text('keep')
    with OutputTransaction(dest, True, 'clip') as output:
        (output.path / 'clip_000000.png').write_bytes(b'new')
        output.commit()
    assert sorted(p.name for p in dest.iterdir()) == ['clip_000000.png', 'notes.txt']


def test_memory_recorder_records_failed_phases_without_history(tmp_path):
    path = tmp_path / 'memory.jsonl'
    recorder = MemoryRecorder(path)
    with pytest.raises(RuntimeError):
        with recorder.phase('failure'):
            raise RuntimeError('test')
    record = json.loads(path.read_text())
    assert record['phase'] == 'failure' and record['ok'] is False
    assert record['rss_peak'] >= record['rss_before'] > 0
    assert set(vars(recorder)) == {'path', 'device'}


def test_chunk_stream_keeps_uint8_and_drops_only_context(monkeypatch):
    options = args(monkeypatch, chunk_size=5, temporal_overlap=3, prepend_frames=2)
    observed = []
    def process(frames, opts, *unused):
        assert frames.dtype == torch.uint8
        observed.append((len(frames), opts.prepend_frames))
        for start in range(0, len(frames), 2):
            yield frames[start:start + 2].float().div(255)
    monkeypatch.setattr(cli, '_iter_processed_batches', process)
    outputs = list(cli._stream_video_chunks(Capture(13), 13, 5, 3, options, 'cpu', Debug(False), None))
    actual = torch.cat(outputs)[:, 0, 0, 0]
    torch.testing.assert_close(actual, torch.arange(13).float() / 255)
    assert observed == [(5, 2), (8, 0), (6, 0)]


def test_oom_restarts_file_without_publishing_partial_results(tmp_path, monkeypatch):
    input_path = tmp_path / 'input.png'
    cv2.imwrite(str(input_path), np.zeros((8, 8, 3), dtype=np.uint8))
    dest = tmp_path / 'output.png'
    dest.write_bytes(b'old')
    options = args(monkeypatch, output_format='png', chunk_size=33, batch_size=9)
    monkeypatch.setattr(cli, '_plan_file', lambda options, *_: copy.copy(options))
    attempts = []
    def write(source, output, kind, settings, *_):
        attempts.append((settings.chunk_size, settings.seed, settings.resolution))
        assert dest.read_bytes() == b'old'
        Path(output).write_bytes(b'partial' if len(attempts) == 1 else b'complete')
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError('test OOM after writing')
        return 1
    monkeypatch.setattr(cli, '_write_file_attempt', write)
    assert cli.process_single_file(str(input_path), options, ['cpu'], str(dest)) == 1
    assert attempts == [(33, 42, 1080), (16, 42, 1080)]
    assert dest.read_bytes() == b'complete'


@pytest.mark.parametrize('high_bit_depth', [False, True])
def test_ffmpeg_real_round_trip(tmp_path, high_bit_depth):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg/ffprobe not installed')
    path = tmp_path / 'sample.mp4'
    writer = FFMPEGVideoWriter(path, 64, 32, 24, high_bit_depth)
    for i in range(3):
        frame = torch.linspace(0, 1, 64).view(1, 64, 1).expand(32, -1, 3)
        writer.write_rgb(frame_pixels(frame, high_bit_depth))
    writer.release()
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=pix_fmt,nb_frames', '-of', 'json', str(path)]))['streams'][0]
    assert data['nb_frames'] == '3'
    assert data['pix_fmt'] == ('yuv420p10le' if high_bit_depth else 'yuv420p')


def test_ffmpeg_failure_is_not_silent(tmp_path):
    if not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg not installed')
    writer = FFMPEGVideoWriter(tmp_path / 'bad.unknown', 8, 8, 24)
    with pytest.raises(RuntimeError, match='FFmpeg'):
        writer.release()


def test_sliced_vae_decodes_into_one_cpu_buffer_and_clears_causal_state():
    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKL, MemoryState
    fake = SimpleNamespace(use_slicing=True, slicing_latent_min_size=2,
                           temporal_downsample_factor=2, tensor_offload_device='cpu', modules=lambda: [])
    calls = []
    def decode(z, memory_state):
        calls.append(memory_state)
        return (torch.cat((z[:, :, :1], z[:, :, 1:].repeat_interleave(2, dim=2)), dim=2)
                if memory_state == MemoryState.INITIALIZING else z.repeat_interleave(2, dim=2))
    fake._decode = decode
    value = torch.arange(6.).reshape(1, 1, 6, 1, 1)
    output = VideoAutoencoderKL.slicing_decode(fake, value)
    expected = torch.cat((value[:, :, :1], value[:, :, 1:].repeat_interleave(2, dim=2)), dim=2)
    torch.testing.assert_close(output, expected)
    assert calls == [MemoryState.INITIALIZING, MemoryState.ACTIVE, MemoryState.ACTIVE]


def test_failed_file_writer_closes_capture_and_generator(tmp_path, monkeypatch):
    options = args(monkeypatch, chunk_size=5)
    cap = Capture(10)
    cap.isOpened = lambda: True
    cap.get = lambda key: 24 if key == cv2.CAP_PROP_FPS else 10
    cap.set = lambda *_: True
    monkeypatch.setattr(cli.cv2, 'VideoCapture', lambda _: cap)
    closed = []
    def stream(*unused, **unused_kwargs):
        try:
            yield torch.zeros(1, 8, 8, 3)
        finally:
            closed.append(True)
    monkeypatch.setattr(cli, '_stream_video_chunks', stream)
    def write(*unused, **unused_kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(cli, 'save_frames_to_video', write)
    with pytest.raises(OSError, match='disk full'):
        cli._write_file_attempt('input.mp4', str(tmp_path / 'out.mp4'), 'video', options, ['cpu'], None)
    assert cap.closed and closed == [True]


def test_non_cached_cleanup_does_not_offload_deleted_model(monkeypatch):
    from src.optimization import memory_manager as manager
    runner = SimpleNamespace(vae=SimpleNamespace(parameters=lambda: iter([
        SimpleNamespace(device=SimpleNamespace(type='cuda'))])))
    calls = []
    monkeypatch.setattr(manager, 'manage_model_device', lambda *a, **k: calls.append(True))
    monkeypatch.setattr(manager, 'release_model_memory', lambda *a, **k: None)
    manager.cleanup_vae(runner, cache_model=False)
    assert runner.vae is None and calls == []


def test_full_core_stream_clears_session_context_and_prepend(tmp_path, monkeypatch):
    options = args(monkeypatch, prepend_frames=2, color_correction='none', memory_report=str(tmp_path / 'events.jsonl'))
    debug = Debug(False)
    session = InferenceSession(debug=debug)
    contexts = []
    def prepare(**kwargs):
        return session.acquire(('fake',), lambda: SimpleNamespace(dit=None, vae=None)), {}
    def compute(ctx, images, **kwargs):
        return torch.cat((images[:2], images)), {}
    def encode(runner, ctx, images, **kwargs):
        contexts.append(ctx)
        ctx['pixels'] = images.float().div(255)
        return ctx
    def output(runner, ctx, *unused):
        for start in range(0, len(ctx['pixels']), 2):
            yield ctx['pixels'][start:start + 2]
    monkeypatch.setattr(cli, 'prepare_runner', prepare)
    monkeypatch.setattr(cli, 'load_text_embeddings', lambda *a: {})
    monkeypatch.setattr(cli, 'compute_generation_info', compute)
    monkeypatch.setattr(cli, 'log_generation_start', lambda *a: None)
    monkeypatch.setattr(cli, 'encode_all_batches', encode)
    monkeypatch.setattr(cli, 'upscale_all_batches', lambda runner, ctx, **kwargs: ctx)
    monkeypatch.setattr(cli, 'iter_output_batches', output)
    raw = torch.arange(5, dtype=torch.uint8).view(5, 1, 1, 1).expand(-1, 8, 8, 3)
    for _ in range(2):
        result = torch.cat(list(cli._iter_processed_batches(raw, options, 'cpu', debug, session)))
        torch.testing.assert_close(result, raw.float() / 255)
    assert contexts == [{}, {}]
    assert not hasattr(session, 'ctx')
    session.close()
    records = [json.loads(line) for line in Path(options.memory_report).read_text().splitlines()]
    assert [r['phase'] for r in records] == ['setup', 'encode', 'upscale', 'decode_postprocess_write'] * 2
    assert all(r['ok'] for r in records)


def test_worker_errors_reach_parent_as_oom(monkeypatch):
    from queue import Queue
    queue = Queue()
    def worker(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError('worker allocation')
    monkeypatch.setattr(cli, '_worker_run', worker)
    cli._worker_process(2, '0', None, {}, queue, None)
    index, error = queue.get_nowait()
    assert index == 2 and error['oom'] and error['error'] == 'worker allocation'
