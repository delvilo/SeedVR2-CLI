"""Per-frame quantization, writers and atomic output publication."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from ..config import OutputConfig
from ..utils.debug import Debug


def frame_pixels(frame, high_bit_depth=False):
    """Quantize a single frame, never a complete output chunk."""
    value = frame.detach().to(device='cpu', dtype=torch.float32).clamp(0, 1)
    # Keep previous 8-bit truncation; 16-bit transport preserves 10-bit precision.
    if high_bit_depth:
        return value.mul_(65535).round_().numpy().astype('<u2')
    return value.mul_(255).numpy().astype(np.uint8)


class FFMPEGVideoWriter:
    def __init__(self, path, width, height, fps, use_10bit=False, use_nvenc=False):
        self.high_bit_depth = use_10bit
        self.errors = tempfile.TemporaryFile()
        self.proc = None
        if use_nvenc:
            cmd_nvenc = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'rgb48le' if use_10bit else 'rgb24',
                '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
                '-c:v', 'hevc_nvenc',
                '-preset', 'p7', '-tune', 'hq', '-rc', 'vbr', '-cq', '20', '-b:v', '0',
                '-multipass', 'fullres', '-bf', '3', '-b_ref_mode', 'middle',
                '-rc-lookahead', '32', '-spatial-aq', '1', '-temporal-aq', '1',
                '-pix_fmt', 'p010le' if use_10bit else 'yuv420p', str(path)
            ]
            try:
                self.proc = subprocess.Popen(cmd_nvenc, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.errors)
            except Exception:
                self.proc = None
        if self.proc is None:
            cmd_cpu = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'rgb48le' if use_10bit else 'rgb24',
                '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
                '-c:v', 'libx265' if use_10bit else 'libx264',
                '-pix_fmt', 'yuv420p10le' if use_10bit else 'yuv420p',
                '-preset', 'medium', '-crf', '12', str(path)
            ]
            try:
                self.proc = subprocess.Popen(cmd_cpu, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.errors)
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


def generate_output_path(input_path: str, output_format: str, output_dir: Optional[str] = None,
                        input_type: Optional[str] = None, from_directory: bool = False) -> str:
    """
    Generate output path based on input path and format.

    Args:
        input_path: Source file path
        output_format: "mp4" or "png"
        output_dir: Optional output directory (overrides default behavior)
        input_type: Optional input type ("image", "video", "directory")
        from_directory: True if processing files from a directory (batch mode)

    Returns:
        Absolute output path (file for single image/video, directory for sequences)
    """
    input_path_obj = Path(input_path)
    input_name = input_path_obj.stem

    # Determine base directory and whether to add suffix
    if output_dir:
        # User specified output directory - use as-is, no suffix
        base_dir = Path(output_dir)
        add_suffix = False
    elif from_directory:
        # Batch mode: create sibling folder with _upscaled, keep original filenames
        original_dir = input_path_obj.parent
        base_dir = original_dir.parent / f"{original_dir.name}_upscaled"
        add_suffix = False
    else:
        # Single file mode: output to same directory with _upscaled suffix
        base_dir = input_path_obj.parent
        add_suffix = True

    # Build filename with optional suffix
    file_suffix = "_upscaled" if add_suffix else ""

    # Generate output path based on format
    if output_format == "png":
        if input_type == "image":
            output_path = base_dir / f"{input_name}{file_suffix}.png"
        else:
            output_path = base_dir / f"{input_name}{file_suffix}"
    else:
        output_path = base_dir / f"{input_name}{file_suffix}.mp4"

    return str(output_path.resolve())


def _save_image_bgr(frame_np: np.ndarray, file_path: str) -> None:
    """
    Save a single RGB(A) uint8 frame to disk, converting to BGR(A) for OpenCV.

    Args:
        frame_np: Frame as uint8 numpy array [H, W, C] where C is 3 (RGB) or 4 (RGBA)
        file_path: Output file path
    """
    if frame_np.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGRA)
    else:
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(file_path), frame_bgr):
        raise OSError(f'Cannot write image: {file_path}')


def save_frames_to_video(
    frames_tensor: torch.Tensor,
    output_path: str,
    fps: float = 30.0,
    writer: cv2.VideoWriter | FFMPEGVideoWriter | None = None,
    video_backend: str = "ffmpeg",
    use_10bit: bool = False,
    use_nvenc: bool = False,
    *, debug: Debug
) -> cv2.VideoWriter | FFMPEGVideoWriter:
    """
    Save frames tensor to MP4 video file.

    Converts one frame at a time from normalized floating point to uint8
    (or uint16 RGB transport for FFmpeg 10-bit), using BGR only for OpenCV. Supports streaming mode where
    an existing writer is passed and kept open for subsequent chunks.

    Args:
        frames_tensor: Frames in format [T, H, W, C], Float32, range [0,1]
        output_path: Output video file path (directory created if doesn't exist)
        fps: Frames per second for output video (default: 30.0)
        writer: Existing VideoWriter for streaming (if None, creates new one)

    Returns:
        Open writer; the caller must close it after the final batch

    Raises:
        ValueError: If video writer cannot be initialized
    """
    T, H, W, C = frames_tensor.shape

    if writer is None:
        debug.log(f"Saving {T} frames to video: {output_path} (backend={video_backend})", category="file")
        os.makedirs(Path(output_path).parent, exist_ok=True)
        if video_backend == "ffmpeg":
            try:
                writer = FFMPEGVideoWriter(output_path, W, H, fps, use_10bit, use_nvenc)
            except Exception as exc:
                debug.log(f"FFmpeg writer failed ({exc}), falling back to OpenCV VideoWriter", level="WARNING", category="file")
                fourcc = cv2.VideoWriter.fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        else:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        if not writer.isOpened():
            writer.release()
            raise ValueError(f"Cannot create video writer for: {output_path}")

    try:
        for frame in frames_tensor:
            pixels = frame_pixels(frame, high_bit_depth=use_10bit and video_backend == 'ffmpeg')
            if isinstance(writer, FFMPEGVideoWriter):
                writer.write_rgb(pixels)
            else:
                writer.write(cv2.cvtColor(pixels[..., :3], cv2.COLOR_RGB2BGR))
    except BaseException:
        if hasattr(writer, 'abort'):
            writer.abort()
        else:
            writer.release()
        raise

    return writer


def save_frames_to_image(
    frames_tensor: torch.Tensor,
    output_dir: str,
    base_name: str,
    start_index: int = 0,
    *, debug: Debug
) -> int:
    """
    Save frames tensor as sequential PNG image files.

    Each frame saved as {base_name}_{index:0Nd}.png with zero-padded indices.
    Converts Float32 [0,1] to uint8 [0,255] and RGB(A) to BGR(A) for OpenCV.

    Args:
        frames_tensor: Frames in format [T, H, W, C], Float32, range [0,1]
        output_dir: Directory to save PNG files (created if doesn't exist)
        base_name: Base name for output files (e.g., "frame" → "frame_00000.png")
        start_index: Starting index for filenames (for streaming continuation)

    Returns:
        Number of frames saved
    """
    os.makedirs(output_dir, exist_ok=True)

    total = frames_tensor.shape[0]

    if start_index == 0:
        debug.log(f"Saving {total} frames as PNGs to directory: {output_dir}", category="file")
    digits = 6  # Supports up to 999,999 frames (~11.5 hours at 24fps)

    for idx, frame in enumerate(frames_tensor):
        filename = f"{base_name}_{start_index + idx:0{digits}d}.png"
        file_path = os.path.join(output_dir, filename)
        _save_image_bgr(frame_pixels(frame), file_path)
        if debug.enabled and (idx + 1) % 100 == 0:
            debug.log(f"Saved {idx + 1}/{total} images", category="file")

    debug.log(f"Saved {total} images to '{output_dir}'", category="success")
    return total


class OutputSink:
    """Stream finalized batches to a writer; close or abort it with the attempt."""
    def __init__(self, path: str, config: OutputConfig, source: str,
                 kind: str, fps: float, debug: Debug):
        if config.format is None:
            raise ValueError('Resolve the output format before opening a sink')
        self.path, self.config, self.debug = path, config, debug
        self.base_name, self.kind, self.fps = Path(source).stem, kind, fps
        self.writer: cv2.VideoWriter | FFMPEGVideoWriter | None = None
        self.written = 0

    def write(self, batch: torch.Tensor) -> None:
        if self.config.format == 'mp4':
            self.writer = save_frames_to_video(batch, self.path, self.fps,
                writer=self.writer, video_backend=self.config.video_backend,
                use_10bit=self.config.use_10bit, use_nvenc=getattr(self.config, 'nvenc', False), debug=self.debug)
        elif self.kind == 'image':
            _save_image_bgr(frame_pixels(batch[0]), self.path)
        else:
            save_frames_to_image(batch, self.path, self.base_name,
                                 start_index=self.written, debug=self.debug)
        self.written += len(batch)

    def __enter__(self) -> 'OutputSink':
        return self

    def __exit__(self, exc_type, *_):
        writer, self.writer = self.writer, None
        if writer is not None:
            if exc_type is not None and hasattr(writer, 'abort'):
                writer.abort()
            else:
                writer.release()
