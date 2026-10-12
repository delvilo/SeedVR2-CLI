"""Immutable application settings. This module must not import torch or codecs."""

from dataclasses import dataclass, field
from typing import Literal

from .utils.model_registry import DEFAULT_DIT


@dataclass(frozen=True, slots=True)
class ModelConfig:
    dit_model: str = DEFAULT_DIT
    model_dir: str | None = None


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    resolution: int = 1080
    max_resolution: int = 0
    batch_size: int = 5
    uniform_batch_size: bool = False
    seed: int = 42
    prepend_frames: int = 0
    temporal_overlap: int = 0
    color_correction: str = "lab"
    input_noise_scale: float = 0.0
    latent_noise_scale: float = 0.0


@dataclass(frozen=True, slots=True)
class OffloadConfig:
    dit: str = "cpu"
    vae: str = "cpu"
    tensor: str = "cpu"


@dataclass(frozen=True, slots=True)
class TileConfig:
    enabled: bool = True
    size: int = 1024
    overlap: int = 128


@dataclass(frozen=True, slots=True)
class VAEConfig:
    encode: TileConfig = field(default_factory=TileConfig)
    decode: TileConfig = field(default_factory=TileConfig)
    debug: Literal["false", "encode", "decode"] = "false"


@dataclass(frozen=True, slots=True)
class CompileConfig:
    dit: bool = False
    vae: bool = False
    backend: str = "inductor"
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = False
    dynamo_cache_size_limit: int = 64
    dynamo_recompile_limit: int = 128


@dataclass(frozen=True, slots=True)
class CacheConfig:
    dit: bool = False
    vae: bool = False
    ram_gb: float | None = None
    vram_gb: float = 0.0


@dataclass(frozen=True, slots=True)
class BlockSwapConfig:
    blocks_to_swap: int = 0
    swap_io_components: bool = False
    offload_device: str = "cpu"


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    swap: BlockSwapConfig = field(default_factory=BlockSwapConfig)
    attention_mode: str = "sageattn_2"
    keep_on_vram: bool = False

@dataclass(frozen=True, slots=True)
class VideoConfig:
    chunk_size: int = 0
    skip_first_frames: int = 0
    load_cap: int = 0


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    ram_budget_gb: float = 0.0
    vram_budget_gb: float = 0.0
    oom_retries: int = 3
    report: str | None = None


@dataclass(frozen=True, slots=True)
class OutputConfig:
    path: str | None = None
    format: Literal["mp4", "png"] | None = None
    video_backend: Literal["opencv", "ffmpeg"] = "ffmpeg"
    use_10bit: bool = False


@dataclass(frozen=True, slots=True)
class AppConfig:
    input: str
    output: OutputConfig = field(default_factory=OutputConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    cuda_device: int = 0
    debug: bool = False
