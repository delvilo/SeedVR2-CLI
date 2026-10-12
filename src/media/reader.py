"""Media discovery, uint8 input windows, and capture ownership."""

import os
from pathlib import Path
from typing import Literal, Tuple, List

import cv2
import numpy as np
import torch
from ..utils.debug import Debug

try:
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_AVAILABLE = True
except (ImportError, OSError):
    TORCHCODEC_AVAILABLE = False
VIDEO_EXTENSIONS = {'.mkv', '.avi', '.mp4', '.flv', '.mov', '.wmv', '.m4v', '.webm'}
IMAGE_EXTENSIONS = {'.jpg', '.tiff', '.tif', '.png', '.webp', '.bmp', '.jpeg'}


def get_media_files(directory: str) -> List[str]:
    """
    Get all video and image files from directory, sorted alphabetically.

    Args:
        directory: Path to directory to scan

    Returns:
        Sorted list of file paths (strings) matching video or image extensions
    """
    valid_extensions = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
    path = Path(directory)

    # Get all files and filter by extension (case-insensitive)
    files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]

    return sorted([str(f) for f in files])


def extract_frames_from_image(image_path: str, debug: Debug) -> Tuple[torch.Tensor, float]:
    """
    Extract single frame from image file and convert to tensor format.

    Reads image using OpenCV, converts BGR to RGB, preserves uint8 (normalizes 16-bit images to float32),
    and formats as single-frame video tensor for consistent processing.

    Args:
        image_path: Path to input image file

    Returns:
        Tuple containing:
            - frames_tensor: Single frame as tensor [1, H, W, C], uint8 or normalized float32 (C=3 for RGB, C=4 for RGBA)
            - fps: Default FPS value (30.0) for image-to-video conversion

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be opened
    """
    debug.log(f"Loading image: {image_path}", category="file")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Read image with alpha channel preserved
    frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ValueError(f"Cannot open image file: {image_path}")

    # Convert BGR(A) to RGB(A) based on channel count
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        debug.log(f"Detected RGBA image (alpha channel preserved)", category="file")
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame.dtype == np.uint16:
        frame = frame.astype(np.float32) / 65535.0
    frames_tensor = torch.from_numpy(frame[None, ...])

    debug.log(f"Image tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}", category="memory")

    return frames_tensor, 30.0


def get_input_type(input_path: str) -> Literal['video', 'image', 'directory', 'unknown']:
    """
    Determine input type from file path.

    Args:
        input_path: Path to input file or directory

    Returns:
        Input type: 'video', 'image', 'directory', or 'unknown'

    Raises:
        FileNotFoundError: If input path doesn't exist
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if path.is_dir():
        return 'directory'

    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in IMAGE_EXTENSIONS:
        return "image"
    else:
        return "unknown"

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


class VideoReader:
    """Own one video capture or TorchCodec decoder, exposing requested frame ranges."""
    def __init__(self, path: str, skip: int = 0, limit: int = 0):
        self.path = path
        self.skip = skip
        self.decoder = None
        self.cap = None

        if TORCHCODEC_AVAILABLE:
            try:
                self.decoder = VideoDecoder(path)
                meta = self.decoder.metadata
                self.width = int(meta.width)
                self.height = int(meta.height)
                self.fps = float(meta.average_fps_from_header or 30.0)
                total = int(meta.num_frames_from_header or len(self.decoder))
                self.count = max(0, total - skip)
                if limit:
                    self.count = min(self.count, limit)
                self.current_idx = skip
                return
            except Exception:
                self.decoder = None

        self.cap = cv2.VideoCapture(path)
        try:
            if not self.cap.isOpened():
                raise ValueError(f'Cannot open video: {path}')
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.count = max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - skip)
            if limit:
                self.count = min(self.count, limit)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
        except BaseException:
            self.cap.release()
            raise

    def read(self, count: int) -> torch.Tensor | None:
        if self.decoder is not None:
            end_idx = min(self.current_idx + count, self.skip + self.count)
            if self.current_idx >= end_idx:
                return None
            frames = self.decoder.get_frames_in_range(self.current_idx, end_idx).data
            self.current_idx = end_idx
            return frames.permute(0, 2, 3, 1).contiguous()
        return read_frames(self.cap, count)

    def __enter__(self) -> 'VideoReader':
        return self

    def __exit__(self, *_):
        if self.cap is not None:
            self.cap.release()
        self.decoder = None

def probe_dimensions(path: str, kind: str, debug: Debug) -> tuple[int, int]:
    if kind == 'video':
        with VideoReader(path) as source:
            return source.width, source.height
    frames, _ = extract_frames_from_image(path, debug)
    return frames.shape[2], frames.shape[1]
