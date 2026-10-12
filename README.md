# SeedVR2-cli — standalone CLI

SeedVR2 image and video upscaling from a Linux command line. The inference pipeline uses one NVIDIA GPU and supports CPU parallelism for offload and preprocessing, with directory processing, bounded video streaming, model reuse, VAE tiling, BlockSwap, and optional torch.compile.
**Key Improvements & Fixes**:
- **Zero OOM on Colab Free T4 GPU / CPU**: Resolves upstream OOM crashes through automatic multi-level VAE tiling (down to 128px), causal 3D convolution slicing, and transactional OOM recovery with dynamic BlockSwap.
- **Pre-flight VRAM Capacity Check**: Validates available GPU memory before downloading weights or allocating buffers; terminates early with helpful recommendations instead of crashing mid-run.
- **TorchCodec Native Video Decoding**: Completely removes `torchvision` and `torchaudio` dependencies; adopts high-throughput PyTorch `TorchCodec` for direct tensor decoding and pure PyTorch functional image transforms.
- **GPU-Adaptive Attention**: Automatically selects SageAttention 2 on SM ≥ 8.0 (Ampere/Ada/Hopper/L4) and falls back to PyTorch SDPA on SM 7.5 (T4) where the SageAttention Triton kernel is unsupported.
- **Tested on Colab T4 (15 GB) and L4 (22 GB)**: See benchmark results below.

[English](README.md) | [中文說明檔](README_zh.md)

This fork targets the current Google Colab runtime: Ubuntu 24.04, Python 3.13, PyTorch 2.11.0+cu130, and the CUDA 13.0 toolkit. The "CUDA Version 13.0" reported by nvidia-smi describes the driver capability; it is not the CUDA build used by PyTorch.

## Start in Colab

Open [the Colab notebook](colab/SeedVR2Lite_Colab.ipynb), select a GPU runtime, and run its cells in order. Configure `DRIVE_ROOT`, `RAM_BUDGET_GB`, `VRAM_BUDGET_GB` and optionally `INSTALL_FLASH_ATTN` in the paths cell. A budget of `0` uses 80% of available RAM or 90% of free VRAM; explicit GiB allowances are capped at the free amount. Weights, memory reports, results and `output/colab_diagnostics.json` are stored on Drive; the repository runs from local Colab storage. The diagnostics include the checked-out commit, paths, GPU and driver, available memory, effective budgets, and installed versions.

The runtime supplies PyTorch. **requirements.txt** contains only the remaining application dependencies (`torchcodec`, `diffusers`, `einops`, etc.). `torchvision` and `torchaudio` are **not required**. The notebook validates PyTorch before and after dependency installation. PyTorch SDPA is the default attention backend; the optional FlashAttention 2 wheel is checked against Python, CUDA, PyTorch, Linux architecture and GPU capability before installation, then tested with a small CUDA kernel. This wheel targets Ampere, Ada and Hopper GPUs (SM 8.x/9.x); T4 is unsupported (uses SDPA or SageAttention). Every install command stops on failure. The notebook installs FFmpeg from the Ubuntu 24.04 FFmpeg 9 PPA and verifies the actual `ffmpeg`/`ffprobe` major versions and x264/x265 encoders. MP4 output uses FFmpeg; 10-bit encoding is available with `--10bit`.

Model downloads use the selected --model_dir and store a validation cache alongside the weights. After a Colab restart, point --model_dir to the same Drive folder to reuse them. Some model hosting repository names contain "comfyUI"; those are weight download locations, not a runtime dependency.

## CLI examples

From the cloned repository in a compatible Linux/CUDA environment:

~~~bash
python -m pip install -r requirements.txt
python inference_cli.py --help

# Image: input and output are explicit paths.
python inference_cli.py examples/inputs/Sadhu_320x478.png \
  --resolution 960 \
  --model_dir /content/drive/MyDrive/SeedVR2/models \
  --output /content/drive/MyDrive/SeedVR2/output/image.png

# Long video: limit memory use and write MP4 with FFmpeg.
python inference_cli.py /content/drive/MyDrive/SeedVR2/input/video.mp4 \
  --resolution 1080 --batch_size 5 --chunk_size 33 \
  --temporal_overlap 3 --video_backend ffmpeg \
  --memory_report /content/drive/MyDrive/SeedVR2/output/memory.jsonl \
  --model_dir /content/drive/MyDrive/SeedVR2/models \
  --output /content/drive/MyDrive/SeedVR2/output/video.mp4

# Directory processing can reuse loaded models between files.
python inference_cli.py /content/drive/MyDrive/SeedVR2/input \
  --output /content/drive/MyDrive/SeedVR2/output \
  --model_dir /content/drive/MyDrive/SeedVR2/models \
  --cache_dit --cache_vae --dit_offload_device cpu --vae_offload_device cpu

~~~

When --model_dir is omitted, models are stored under **models/SEEDVR2** in the repository. The SEEDVR2_MODEL_DIR environment variable can set another default. --output controls the result path; omit it to use the CLI's automatically generated filename.

## Common options

| Option | Purpose |
| --- | --- |
| --dit_model | Choose a 3B or 7B FP16, FP8, or GGUF model; the default is 3B FP8. |
| --resolution | Target short edge in pixels. |
| --batch_size | Frames per model batch; use 1, 5, 9, 13, 17, and so on (4n+1). |
| --chunk_size | Maximum input window; 0 automatically chooses a finite window of at most 33 frames. |
| --vae_encode_tiled / --vae_decode_tiled | Tiling is enabled by default; use --no-vae_encode_tiled / --no-vae_decode_tiled to opt out initially. |
| --blocks_to_swap / --dit_offload_device cpu | Offload DiT blocks to system RAM when VRAM is limited. |
| --cache_dit / --cache_vae | Request model reuse within the RAM/VRAM cache caps. |
| --ram_budget_gb / --vram_budget_gb | Set memory planning allowances (GiB); 0 uses available-memory defaults. |
| --cache_ram_gb / --cache_vram_gb | Limit retained model storage; 0 disables that cache tier. |
| --memory_report / --oom_retries | Record phase peaks to JSONL; retry caught OOMs with smaller workloads. |
| --attention_mode | PyTorch SDPA is the default. Optional Flash/Sage backends require separate compatible installations. |
| --compile_dit / --compile_vae | Enable torch.compile; first-run compilation can take time. |
| --video_backend ffmpeg / --10bit | Encode MP4 using FFmpeg, optionally with 10-bit x265 output. |
| --debug | Show environment, timing, VRAM, and processing details. |
| --cuda_device | Select one visible CUDA device index; defaults to device 0. |

### Model VRAM Requirements (Pre-flight Check)

| Model Name | Precision | Weight Size | Min VRAM (No BlockSwap) | Colab T4 (15 GB) | Colab L4 (22 GB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `seedvr2_ema_3b-Q4_K_M.gguf` | Q4_K_M | ~1.86 GB | **5.0 GB** | ✅ Native | ✅ Native (**Recommended**) |
| `seedvr2_ema_3b-Q8_0.gguf` | Q8_0 | ~3.4 GB | **7.0 GB** | ✅ Native | ✅ Native |
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | FP8 | ~3.4 GB | **7.0 GB** | ✅ Native | ✅ Native |
| `seedvr2_ema_3b_fp16.safetensors` | FP16 | ~6.8 GB | **9.5 GB** | ✅ Native | ✅ Native |
| `seedvr2_ema_7b-Q4_K_M.gguf` | Q4_K_M | ~4.3 GB | **8.5 GB** | ✅ Native | ✅ Native |
| `seedvr2_ema_7b-Q8_0.gguf` | Q8_0 | ~7.8 GB | **12.0 GB** | ✅ Native | ✅ Native |
| `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors` | Mixed FP8/FP16 | ~7.5 GB | **11.5 GB** | ✅ Native | ✅ Native |
| `seedvr2_ema_7b_fp16.safetensors` | FP16 | ~14.8 GB | **18.0 GB** | ⚠️ `--blocks_to_swap 32` | ✅ Native |

Use --help for the complete option list. Larger models, batches, and output resolutions require more VRAM. Start with a short sample at a modest resolution before processing a long video.

### L4 GPU 1080p Video Benchmark (NVIDIA L4 · 22 GB VRAM · SM 8.9 Ada Lovelace)

Measured on Google Colab L4 upscaling video from 640×360 to **1080p (1920×1080)**. SageAttention 2 is the default attention backend on L4 (SM 8.9).

| Configuration | Model | Batch Size | `--keep_on_vram` | Peak VRAM | DiT Time | Total Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **3B Q4 Baseline** | `seedvr2_ema_3b-Q4_K_M.gguf` | 5 | ❌ (Offload) | 5.01 GB | 7.95 s | 62.88 s |
| **3B Q4 Fast** | `seedvr2_ema_3b-Q4_K_M.gguf` | 5 | ✅ | 6.94 GB | 6.59 s | **55.84 s** |
| **3B Q4 Batch 9** | `seedvr2_ema_3b-Q4_K_M.gguf` | 9 | ✅ | 8.39 GB | 9.05 s (1.0s/f) | 89.20 s |
| **3B Q4 Batch 13** | `seedvr2_ema_3b-Q4_K_M.gguf` | 13 | ✅ | 8.46 GB | 11.48 s (**0.88s/f**) | 122.86 s |
| **7B Q4 Balance** | `seedvr2_ema_7b-Q4_K_M.gguf` | 5 | ✅ | 9.49 GB | 9.94 s | 72.66 s |
| **7B Q4 Batch 9** | `seedvr2_ema_7b-Q4_K_M.gguf` | 9 | ✅ | 10.95 GB | 13.61 s (1.5s/f) | 95.80 s |
| **7B Q8 High-Q** | `seedvr2_ema_7b-Q8_0.gguf` | 5 | ✅ | 13.29 GB | 9.60 s | 84.79 s |
| **7B Q8 Batch 9** | `seedvr2_ema_7b-Q8_0.gguf` | 9 | ✅ | 14.75 GB | 13.20 s (1.46s/f) | 96.11 s |
| **7B FP16 Offload** | `seedvr2_ema_7b_fp16.safetensors` | 5 | ❌ (Offload) | 13.42 GB | 29.37 s | 101.83 s |
| **7B FP16 Resident** | `seedvr2_ema_7b_fp16.safetensors` | 5 | ✅ | 20.41 GB | 8.90 s | **63.67 s** |
| **3B Q4 Compiled** | `seedvr2_ema_3b-Q4_K_M.gguf` | 5 | ❌ (`--compile_dit`) | 5.01 GB | 113.44 s | 172.62 s (17× slower) |

#### Large Models Analysis (7B Q8_0 & 7B FP16 on L4 GPU)
- **`seedvr2_ema_7b-Q8_0.gguf` (~8.3 GB)**: Fully compatible with `--keep_on_vram` up to `--batch_size 9` (14.75 GB peak VRAM). Delivers visual fidelity virtually indistinguishable from FP16 while maintaining a comfortable 7 GB safety margin on L4's 22 GB.
- **`seedvr2_ema_7b_fp16.safetensors` (~15.8 GB)**:
  - **With `--keep_on_vram`**: Works smoothly at `--batch_size 5` and 1080p (20.41 GB peak VRAM). Do **not** use `--batch_size 9` with `--keep_on_vram`, as the pre-flight check will detect that ~22.6 GB is needed and safely abort to prevent OOM.
  - **Without `--keep_on_vram`**: Runs with standard CPU offload across all batch sizes.

#### 4 Optimal Parameter Configurations for L4 GPU

##### 1. 最佳畫質 (Best Quality) — 7B Q8-0, Batch 9, 1080p
```bash
python inference_cli.py input.mp4 \
  --dit_model seedvr2_ema_7b-Q8_0.gguf \
  --resolution 1080 --batch_size 9 \
  --attention_mode sageattn_2 --keep_on_vram \
  --video_backend ffmpeg --output output_best_quality.mp4
```

##### 2. 最快時間 (Fastest Time / Max Throughput) — 3B Q4, Batch 13, 1080p
```bash
python inference_cli.py input.mp4 \
  --dit_model seedvr2_ema_3b-Q4_K_M.gguf \
  --resolution 1080 --batch_size 13 \
  --attention_mode sageattn_2 --keep_on_vram \
  --video_backend ffmpeg --output output_fastest.mp4
```

##### 3. 長影音 (Long Video / Movies / Multi-minute) — 7B Q4, Chunk 33, Overlap 3
```bash
python inference_cli.py input.mp4 \
  --dit_model seedvr2_ema_7b-Q4_K_M.gguf \
  --resolution 1080 --batch_size 9 \
  --chunk_size 33 --temporal_overlap 3 \
  --attention_mode sageattn_2 --keep_on_vram \
  --video_backend ffmpeg --output output_long_video.mp4
```

##### 4. 短影音 (Short Video / High Precision Clips) — 7B FP16, Batch 5
```bash
python inference_cli.py input.mp4 \
  --dit_model seedvr2_ema_7b_fp16.safetensors \
  --resolution 1080 --batch_size 5 \
  --attention_mode sageattn_2 --keep_on_vram \
  --video_backend ffmpeg --output output_short_video.mp4
```

CPU model offload and VAE tiling are enabled by default. Finalized output batches are postprocessed on CPU and written immediately. OOM retries restart the file in temporary output, preserving resolution and seed while reducing execution granularity. Changes to batch/tile boundaries can affect pixels.

See [memory behavior, measured buffer baseline and Colab validation](docs/memory.md) for the budgeting policy, cache limits, precision path and remaining GPU checks.

## Development checks

The CLI, media I/O, scheduler and inference session have separate modules and immutable typed settings. See [architecture and ownership](docs/architecture.md) for the module boundaries and failure lifecycle.

Install `requirements-dev.txt`, then run `python -m compileall -q inference_cli.py src tests tools`, `python -m mypy`, `python -m pytest -q tests`, and `python tools/benchmark_memory.py`. The CPU CI workflow uses Python 3.13 / PyTorch 2.11 and checks that streaming output buffers remain bounded as clip length grows. The Colab notebook's short video sample checks the full inference path on the target GPU runtime.

## Models and attribution

DiT and VAE checkpoints are downloaded on first use from the repositories listed in **src/utils/model_registry.py**, checked against the listed SHA-256 hashes, and reused from the selected model directory. The small text embeddings included with this repository remain part of inference.

This implementation is derived from [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler), which is based on [ByteDance SeedVR](https://github.com/ByteDance-Seed/SeedVR). Original contributors and their credits are preserved in the source history. See [LICENSE](LICENSE) for the Apache 2.0 license.
