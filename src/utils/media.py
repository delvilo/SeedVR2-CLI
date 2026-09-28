"""Bounded media buffers and transactional file output."""

import os
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
import torch


def read_frames(cap, count):
    """Keep decoded input as uint8; allocate one contiguous input window."""
    buffer = None
    loaded = 0
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        if buffer is None:
            buffer = np.empty((count, *frame.shape), dtype=np.uint8)
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=buffer[loaded])
        loaded += 1
    return torch.from_numpy(buffer[:loaded]) if loaded else None


def frame_pixels(frame, high_bit_depth=False):
    """Quantize a single frame, never a complete output chunk."""
    value = frame.detach().to(device='cpu', dtype=torch.float32).clamp(0, 1)
    # Keep previous 8-bit truncation; 16-bit transport preserves 10-bit precision.
    if high_bit_depth:
        return value.mul_(65535).round_().numpy().astype('<u2')
    return value.mul_(255).numpy().astype(np.uint8)


class FFMPEGVideoWriter:
    def __init__(self, path, width, height, fps, use_10bit=False):
        self.high_bit_depth = use_10bit
        self.errors = tempfile.TemporaryFile()
        self.proc = None
        try:
            self.proc = subprocess.Popen([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'rgb48le' if use_10bit else 'rgb24',
                '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
                '-c:v', 'libx265' if use_10bit else 'libx264',
                '-pix_fmt', 'yuv420p10le' if use_10bit else 'yuv420p',
                '-preset', 'medium', '-crf', '12', str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.errors)
        except BaseException:
            self.errors.close()
            raise

    def isOpened(self):
        return self.proc is not None and self.proc.poll() is None

    def _error(self):
        self.errors.seek(0, os.SEEK_END)
        self.errors.seek(max(0, self.errors.tell() - 8192))
        return self.errors.read().decode(errors='replace').strip()

    def write_rgb(self, frame):
        if self.proc is None:
            raise RuntimeError('FFmpeg writer is closed')
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame[..., :3]).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(f'FFmpeg stopped: {self._error()}') from exc

    def release(self):
        if self.proc is None:
            return
        try:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            code = self.proc.wait()
            if code:
                raise RuntimeError(f'FFmpeg failed ({code}): {self._error()}')
        except BaseException:
            self.abort()
            raise
        finally:
            self.proc = None
            self.errors.close()

    def abort(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            self.proc = None
        self.errors.close()


class OutputTransaction:
    """A failed/OOM attempt never appends to an already published result."""
    def __init__(self, destination, image_sequence=False, sequence_name=None):
        self.destination = Path(destination)
        self.image_sequence = image_sequence
        self.sequence_name = sequence_name

    def __enter__(self):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='.seedvr-', dir=self.destination.parent)
        self.path = Path(self.temp.name) / self.destination.name
        if self.image_sequence:
            self.path.mkdir()
        return self

    def commit(self):
        if self.image_sequence:
            if not self.sequence_name:
                raise ValueError('A sequence name is required to replace generated frames')
            self.destination.mkdir(parents=True, exist_ok=True)
            backup = Path(self.temp.name) / 'previous'
            backup.mkdir()
            import glob
            pattern = f'{glob.escape(self.sequence_name)}_*.png'
            # Preserve unrelated files; remove stale frames from a longer old run.
            # Backup and roll back on filesystem errors. MP4 uses one atomic rename.
            try:
                for frame in self.destination.glob(pattern):
                    if frame.stem[len(self.sequence_name) + 1:].isdigit():
                        os.replace(frame, backup / frame.name)
            except BaseException:
                for frame in backup.iterdir():
                    os.replace(frame, self.destination / frame.name)
                raise
            try:
                for frame in self.path.iterdir():
                    os.replace(frame, self.destination / frame.name)
            except BaseException:
                for frame in self.destination.glob(pattern):
                    if frame.stem[len(self.sequence_name) + 1:].isdigit():
                        frame.unlink()
                for frame in backup.iterdir():
                    os.replace(frame, self.destination / frame.name)
                raise
        else:
            os.replace(self.path, self.destination)

    def __exit__(self, *_):
        self.temp.cleanup()
