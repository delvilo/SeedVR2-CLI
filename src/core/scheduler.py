"""Sequential file/chunk scheduling and whole-file OOM retries."""

from collections.abc import Generator, Iterator
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ..config import AppConfig, InferenceConfig, VideoConfig
from ..media.reader import (VideoReader, extract_frames_from_image, get_input_type,
                            get_media_files, probe_dimensions)
from ..media.output import OutputSink, OutputTransaction, generate_output_path
from ..optimization.memory_budget import is_oom, reduce_workload
from ..optimization.memory_manager import clear_memory
from ..optimization.memory_profile import MemoryRecorder
from ..utils.debug import Debug
from .planning import plan_file
from .session import InferenceSession


@dataclass(frozen=True, slots=True)
class FileJob:
    source: str
    destination: str
    kind: str
    config: AppConfig


def iter_file_jobs(config: AppConfig) -> Iterator[FileJob]:
    kind = get_input_type(config.input)
    directory = kind == 'directory'
    files = get_media_files(config.input) if directory else [config.input]
    if not files:
        raise ValueError(f'No video or image files found in directory: {config.input}')
    for source in files:
        kind = get_input_type(source)
        if kind not in ('video', 'image'):
            raise ValueError(f'Unsupported input type: {source}')
        fmt = config.output.format or ('mp4' if kind == 'video' else 'png')
        destination = config.output.path
        sequence = kind == 'video' and fmt == 'png'
        if directory:
            destination = generate_output_path(source, fmt, destination, kind, from_directory=True)
        elif destination is None:
            destination = generate_output_path(source, fmt, input_type=kind)
        elif not Path(destination).suffix or sequence:
            destination = generate_output_path(source, fmt, destination, kind)
        yield FileJob(source, destination, kind,
                      replace(config, output=replace(config.output, path=destination, format=fmt)))


def stream_video_chunks(source: VideoReader, config: InferenceConfig, video: VideoConfig,
                        device: str, debug: Debug, session: InferenceSession,
                        memory_report: str | None = None) -> Generator[torch.Tensor, None, None]:
    """Keep one uint8 input window and its raw overlap; emit finalized batches."""
    if video.chunk_size < 1:
        raise ValueError('Resolve the finite chunk size before streaming')
    frames_read = 0
    prev_raw_tail: torch.Tensor | None = None
    overlap = config.sampling.temporal_overlap
    recorder = MemoryRecorder(memory_report, device)
    total_chunks = (source.count + video.chunk_size - 1) // video.chunk_size
    chunk_index = 0
    while frames_read < source.count:
        with recorder.phase('read'):
            new_frames = source.read(min(video.chunk_size, source.count - frames_read))
        if new_frames is None:
            break
        frames_read += len(new_frames)
        chunk_index += 1
        if chunk_index > 1:
            config = replace(config, sampling=replace(config.sampling, prepend_frames=0))
        context_count = min(overlap, len(prev_raw_tail)) if prev_raw_tail is not None else 0
        frames = torch.cat((prev_raw_tail[-context_count:], new_frames)) if context_count and prev_raw_tail is not None else new_frames
        prev_raw_tail = frames[-overlap:].clone() if overlap else None
        del new_frames
        debug.log(f'Chunk {chunk_index}/{total_chunks}: {len(frames) - context_count} new frames',
                  category='generation', force=True)
        drop = context_count
        with closing(session.iter_batches(frames, config, device, memory_report)) as outputs:
            for result in outputs:
                skip = min(drop, len(result))
                drop -= skip
                if skip < len(result):
                    yield result[skip:]
                del result
        del frames
        recorder.record('chunk_complete', chunk=chunk_index, input_frames=frames_read)


def write_file_attempt(job: FileJob, temporary_path: str, device: str,
                       session: InferenceSession, debug: Debug) -> int:
    """Own the source, generator and sink together, including cancellation."""
    config = job.config
    if job.kind == 'image':
        frames, fps = extract_frames_from_image(job.source, debug)
        with OutputSink(temporary_path, config.output, job.source, job.kind, fps, debug) as sink:
            with closing(session.iter_batches(frames, config.inference, device, config.memory.report)) as batches:
                for batch in batches:
                    sink.write(batch)
                    del batch
        return sink.written
    with VideoReader(job.source, config.video.skip_first_frames, config.video.load_cap) as source:
        with OutputSink(temporary_path, config.output, job.source, job.kind, source.fps, debug) as sink:
            with closing(stream_video_chunks(source, config.inference, config.video, device,
                                              debug, session, config.memory.report)) as batches:
                for batch in batches:
                    sink.write(batch)
                    del batch
        return sink.written


def process_file(job: FileJob, device: str, session: InferenceSession, debug: Debug) -> int:
    width, height = probe_dimensions(job.source, job.kind, debug)
    if min(width, height) <= 0:
        raise ValueError(f'Invalid media dimensions: {width}x{height}')
    planned = plan_file(job.config, width, height, device, session, debug)
    recorder = MemoryRecorder(planned.memory.report, device)
    sequence = job.kind == 'video' and planned.output.format == 'png'
    for attempt in range(planned.memory.oom_retries + 1):
        recorder.record('attempt', input=job.source, attempt=attempt,
                        chunk_size=planned.video.chunk_size, batch_size=planned.inference.sampling.batch_size)
        try:
            with OutputTransaction(job.destination, image_sequence=sequence, sequence_name=Path(job.source).stem) as output:
                count = write_file_attempt(replace(job, config=planned), str(output.path), device, session, debug)
                if count:
                    output.commit()
            recorder.record('complete', input=job.source, attempt=attempt, frames=count)
            debug.log(f'Output saved to: {job.destination} ({count} frames)', category='file', force=True)
            return count
        except Exception as error:
            if not is_oom(error) or attempt >= planned.memory.oom_retries:
                raise
            reason = str(error)
            reduced = reduce_workload(planned.inference, planned.video)
            if reduced is None:
                raise
            planned = replace(planned, inference=reduced[0], video=reduced[1])
        # Leave the exception scope before releasing traceback-owned tensors.
        session.close()
        clear_memory(debug=debug, deep=True, force=True)
        sampling = planned.inference.sampling
        recorder.record('oom_retry', attempt=attempt + 1, reason=reason,
            batch_size=sampling.batch_size, chunk_size=planned.video.chunk_size,
            tile_size=planned.inference.vae.decode.size, seed=sampling.seed, resolution=sampling.resolution)
        debug.log(f'OOM: restarting file with chunk={planned.video.chunk_size}, batch={sampling.batch_size}, '
                  f'tile={planned.inference.vae.decode.size}', level='WARNING', category='memory', force=True)
    raise RuntimeError('No processing attempt was scheduled')


def run_jobs(config: AppConfig, device: str, debug: Debug) -> int:
    total = 0
    with InferenceSession(debug=debug) as session:
        for index, job in enumerate(iter_file_jobs(config), 1):
            debug.log(f'Processing file {index}: {job.source}', category='generation', force=True)
            total += process_file(job, device, session, debug)
    return total
