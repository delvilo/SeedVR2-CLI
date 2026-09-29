"""Invocation-local tensor ownership, separate from immutable application settings."""

from collections.abc import Callable
from dataclasses import dataclass, field
import torch


@dataclass(slots=True)
class GenerationContext:
    dit_device: torch.device = field(default_factory=lambda: torch.device('cpu'))
    vae_device: torch.device = field(default_factory=lambda: torch.device('cpu'))
    dit_offload_device: torch.device | None = None
    vae_offload_device: torch.device | None = None
    tensor_offload_device: torch.device | None = None
    compute_dtype: torch.dtype = torch.bfloat16
    postprocess_device: torch.device | None = None
    video_transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    text_embeds: dict[str, list[torch.Tensor]] | None = None
    input_images: torch.Tensor | None = None
    final_video: torch.Tensor | None = None
    all_latents: list[torch.Tensor | None] | None = None
    all_upscaled_latents: list[torch.Tensor | None] | None = None
    all_alpha_channels: list[torch.Tensor | None] | None = None
    all_input_rgb: list[torch.Tensor | None] | None = None
    all_ori_lengths: list[int | None] | None = None
    batch_metadata: list[tuple[int, int, int] | None] | None = None
    decode_batch_info: list[tuple[int, int, int, int]] | None = None
    true_target_dims: tuple[int, int] | None = None
    total_frames: int | None = None
    total_padding_removed: int | None = None
    actual_temporal_overlap: int | None = None
    postprocess_skip_head: int | None = None
    is_rgba: bool = False

    def clear(self) -> None:
        """Drop this invocation's references, without mutating aliased tensor storage."""
        self.video_transform = None
        self.text_embeds = None
        self.input_images = self.final_video = None
        self.all_latents = self.all_upscaled_latents = None
        self.all_alpha_channels = self.all_input_rgb = None
        self.all_ori_lengths = None
        self.batch_metadata = None
        self.decode_batch_info = None
        self.true_target_dims = None
        self.total_frames = self.total_padding_removed = None
        self.actual_temporal_overlap = self.postprocess_skip_head = None
