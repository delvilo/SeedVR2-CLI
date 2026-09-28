#!/usr/bin/env python3
"""Reproducible CPU buffer baseline, plus summaries of real CLI JSONL profiles.

Synthetic mode compares the previous whole-output allocation pattern with the
new streaming assembler in isolated processes. It does not run diffusion/VAE
weights, assess image quality, or predict Colab GPU memory.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def measure(mode, frames, batch_size, width, height):
    import resource
    import time
    import torch
    # Import the same runtime modules before measuring either mode.
    from src.core.streaming import blend_decoded_batches
    from src.utils.media import frame_pixels
    from src.optimization.memory_profile import process_rss_bytes
    torch.set_num_threads(1)
    torch.zeros(1).add_(1)  # initialize CPU allocator
    before = process_rss_bytes()
    start = time.monotonic()
    written = 0
    checksum = 0

    def decoded_batches():
        for index, offset in enumerate(range(0, frames, batch_size)):
            yield index, torch.full((min(batch_size, frames - offset), height, width, 3),
                                    .5, dtype=torch.bfloat16)

    if mode == 'legacy':
        # adc0480: a complete BF16 output, then whole FP32 and uint8 conversions.
        output = torch.empty((frames, height, width, 3), dtype=torch.bfloat16)
        offset = 0
        for _, batch in decoded_batches():
            output[offset:offset + len(batch)].copy_(batch)
            offset += len(batch)
        converted = output.float()
        pixels = (converted.numpy() * 255).astype('uint8')
        for frame in pixels:
            checksum += int(frame[0, 0, 0])
            written += 1
    else:
        for _, batch, _ in blend_decoded_batches(decoded_batches(), 0):
            for frame in batch:
                pixels = frame_pixels(frame)
                checksum += int(pixels[0, 0, 0])
                written += 1
            del batch, frame, pixels
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return dict(kind='synthetic_output_buffers', mode=mode, frames=frames, written=written,
                batch_size=batch_size, width=width, height=height, rss_before=before,
                rss_peak=peak, rss_peak_growth=max(0, peak - before), checksum=checksum,
                seconds=time.monotonic() - start, python=sys.version.split()[0], torch=torch.__version__)


def summarize(path):
    phases = {}
    counts = {'oom_retries': 0, 'completed_files': 0}
    with Path(path).open() as source:
        for line in source:
            event = json.loads(line)
            if event['event'] == 'phase':
                entry = phases.setdefault(event['phase'], {'calls': 0, 'seconds': 0})
                entry['calls'] += 1
                entry['seconds'] += event['seconds']
                for key in ('rss_peak', 'cuda_peak_allocated', 'cuda_peak_reserved'):
                    if key in event:
                        entry[key] = max(entry.get(key, 0), event[key])
            elif event['event'] == 'oom_retry':
                counts['oom_retries'] += 1
            elif event['event'] == 'complete':
                counts['completed_files'] += 1
    return {'profile': str(path), 'phases': phases, **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', help='Summarize a real --memory_report JSONL file')
    parser.add_argument('--output', help='Save JSON results')
    parser.add_argument('--frames', type=int, nargs='+', default=[33, 330])
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=180)
    parser.add_argument('--mode', choices=['legacy', 'stream'], help=argparse.SUPPRESS)
    options = parser.parse_args()
    if min(*options.frames, options.batch_size, options.width, options.height) < 1:
        parser.error('Frame counts and dimensions must be positive')
    if options.profile:
        result = summarize(options.profile)
    elif options.mode:
        result = measure(options.mode, options.frames[0], options.batch_size, options.width, options.height)
    else:
        result = []
        for mode in ('legacy', 'stream'):
            for frames in options.frames:
                command = [sys.executable, __file__, '--mode', mode, '--frames', str(frames),
                    '--batch_size', str(options.batch_size), '--width', str(options.width), '--height', str(options.height)]
                text = subprocess.check_output(command, text=True)
                result.append(json.loads(text.splitlines()[-1]))
        # A generous fixed allowance covers allocator warmup; it does not scale with duration.
        streamed = [run for run in result if run['mode'] == 'stream']
        if max(run['rss_peak'] for run in streamed) - min(run['rss_peak'] for run in streamed) > 64 * 1024**2:
            raise SystemExit('Streaming RSS grew by more than 64 MiB between clip lengths')
        for run in result:
            if run['written'] != run['frames'] or run['checksum'] != run['frames'] * 127:
                raise SystemExit('Frame count/pixel regression in synthetic baseline')
    encoded = json.dumps(result, indent=None if options.mode else 2)
    if options.output:
        path = Path(options.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
