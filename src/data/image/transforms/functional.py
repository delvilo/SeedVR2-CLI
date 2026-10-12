"""
Native PyTorch functional image and tensor transformations.
Zero external dependency on torchvision; optimized for performance and compile compatibility.
"""

from typing import Tuple, Union, Sequence, Callable
import torch
import torch.nn.functional as F
from PIL import Image


class InterpolationMode:
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


def resize(
    img: Union[torch.Tensor, Image.Image],
    size: Union[int, Sequence[int]],
    interpolation: str = InterpolationMode.BICUBIC,
    antialias: bool = True,
) -> Union[torch.Tensor, Image.Image]:
    """Resize image tensor or PIL Image using native PyTorch or PIL."""
    if isinstance(size, int):
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            w, h = img.size
        if h < w:
            new_h = size
            new_w = round(w * size / h)
        else:
            new_w = size
            new_h = round(h * size / w)
        target_size = (new_h, new_w)
    else:
        target_size = (size[0], size[1])

    if isinstance(img, torch.Tensor):
        mode_str = str(interpolation).lower()
        if "bicubic" in mode_str:
            mode = "bicubic"
            align_corners = False
        elif "bilinear" in mode_str or "linear" in mode_str:
            mode = "bilinear"
            align_corners = False
        elif "nearest" in mode_str:
            mode = "nearest"
            antialias = False
            align_corners = None
        else:
            mode = "bicubic"
            align_corners = False

        orig_ndim = img.ndim
        if orig_ndim == 3:
            img = img.unsqueeze(0)

        orig_dtype = img.dtype
        needs_float = (img.dtype in (torch.float16, torch.bfloat16)) or (not img.is_floating_point())
        img_in = img.float() if needs_float else img

        out = F.interpolate(
            img_in,
            size=target_size,
            mode=mode,
            antialias=antialias,
            align_corners=align_corners,
        )

        if needs_float:
            if not orig_dtype.is_floating_point:
                out = out.round().clamp(0, 255).to(orig_dtype)
            else:
                out = out.to(orig_dtype)
        if orig_ndim == 3:
            out = out.squeeze(0)
        return out

    elif isinstance(img, Image.Image):
        resample = Image.Resampling.BICUBIC
        mode_str = str(interpolation).lower()
        if "nearest" in mode_str:
            resample = Image.Resampling.NEAREST
        elif "bilinear" in mode_str:
            resample = Image.Resampling.BILINEAR
        return img.resize((target_size[1], target_size[0]), resample=resample)

    raise TypeError(f"Unsupported image type: {type(img)}")


def center_crop(
    img: Union[torch.Tensor, Image.Image],
    output_size: Union[int, Sequence[int]],
) -> Union[torch.Tensor, Image.Image]:
    """Center crop image tensor or PIL Image."""
    if isinstance(output_size, int):
        target_h, target_w = output_size, output_size
    else:
        target_h, target_w = output_size[0], output_size[1]

    if isinstance(img, torch.Tensor):
        h, w = img.shape[-2:]
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return img[..., top : top + target_h, left : left + target_w]

    elif isinstance(img, Image.Image):
        w, h = img.size
        left = (w - target_w) // 2
        top = (h - target_h) // 2
        return img.crop((left, top, left + target_w, top + target_h))

    raise TypeError(f"Unsupported image type: {type(img)}")


def crop(
    img: Union[torch.Tensor, Image.Image],
    top: int,
    left: int,
    height: int,
    width: int,
) -> Union[torch.Tensor, Image.Image]:
    """Crop image tensor or PIL Image."""
    if isinstance(img, torch.Tensor):
        return img[..., top : top + height, left : left + width]
    elif isinstance(img, Image.Image):
        return img.crop((left, top, left + width, top + height))
    raise TypeError(f"Unsupported image type: {type(img)}")


class Compose:
    """Chain multiple transform callables sequentially."""

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Lambda:
    """Apply a user-defined function as a transform."""

    def __init__(self, lambd: Callable):
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class Normalize:
    """In-place or broadcasting tensor normalization."""

    def __init__(self, mean: Union[float, Sequence[float]], std: Union[float, Sequence[float]]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(self.mean, (int, float)) and isinstance(self.std, (int, float)):
            return tensor.sub(self.mean).div_(self.std)
        mean_t = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std_t = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device)
        while mean_t.ndim < tensor.ndim - 1:
            mean_t = mean_t.view(-1, *([1] * (tensor.ndim - 2)))
            std_t = std_t.view(-1, *([1] * (tensor.ndim - 2)))
        return tensor.sub(mean_t).div_(std_t)


class Resize:
    """Resize transform class."""

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        interpolation: str = InterpolationMode.BICUBIC,
        antialias: bool = True,
    ):
        self.size = size
        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, img):
        return resize(img, self.size, self.interpolation, self.antialias)


class CenterCrop:
    """CenterCrop transform class."""

    def __init__(self, size: Union[int, Sequence[int]]):
        self.size = size

    def __call__(self, img):
        return center_crop(img, self.size)
