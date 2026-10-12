# Memory-first CLI pipeline

This refactor implements items 1–7 on top of `23d3845` for the supported Colab runtime: Ubuntu 24.04, Python 3.13, PyTorch 2.11.0+cu128, and CUDA toolkit 12.8. The inference baseline remains `adc0480`; the intervening upstream commits change documentation and model registry entries.

The new upstream Q8 entries are retained with corrected download identities: the 3B and 7B hashes are matched to their actual files, and the trailing space in the sharp filename is removed. Publisher SHA-256 metadata was checked on 2026-09-28 for [3B Q8](https://huggingface.co/AInVFX/SeedVR2_comfyUI/blob/main/seedvr2_ema_3b-Q8_0.gguf), [7B Q8](https://huggingface.co/cmeka/SeedVR2-GGUF/blob/main/seedvr2_ema_7b-Q8_0.gguf), and [7B sharp Q8](https://huggingface.co/cmeka/SeedVR2-GGUF/blob/main/seedvr2_ema_7b_sharp-Q8_0.gguf). These metadata checks do not download or run the model weights.

## Changes and ownership

| Item | Implementation |
| --- | --- |
| 1. Measurement and regression baseline | Append-only JSONL measurements of RSS and CUDA allocation peaks; CPU regression tests; isolated buffer benchmark; Python 3.13 CI. |
| 2. Media copies | Decode input into one contiguous uint8 window. Normalize only the batch being encoded. Quantize output one frame at a time. FFmpeg receives RGB directly. |
| 3. Streaming output | Encode/upscale the bounded input window, then decode, blend, postprocess and write finalized batches. The single-GPU CLI never constructs a whole-window `final_video`. Retain only the unfinalized overlap tail, including overlaps spanning multiple batches. |
| 4. VAE peak memory | Normalize spatial tiles on their accumulation device. Copy temporal decode slices into one output buffer on the offload device. Rearrangement uses views, without stacking an extra decoded batch. |
| 5. Resource lifetime | Captures, generators, writers, temporary output and sessions have explicit cleanup. Drop tensor owners instead of mutating `.data` storage. Failed, uncached models are released without first copying them to RAM. |
| 6. Model cache | One `InferenceSession` owns the runner, keyed by checkpoint path/size/mtime, devices and compute dtype. No global cache or node IDs. Count distinct parameter/buffer storage and evict models when the configured cache allowance is exceeded. Generation contexts and outputs never enter the cache. |
| 7. Planning and OOM recovery | Choose finite chunk, 4n+1 batch, tile and BlockSwap settings from available RAM/VRAM and checkpoint estimates. On a caught OOM, discard temporary output and restart the file with a smaller workload and CPU offload. |

The output buffer bound applies to the CLI's supported execution mode: one GPU streams finalized batches to the writer.

## Defaults and controls

The input window is finite by default: `--chunk_size 0` now means automatic planning, with at most 33 new input frames per window. A positive value is a ceiling that the planner may reduce. Input context and optional prepended frames are included in the estimate.

VAE tiling and memory scheduling are handled automatically by the planner.

| Option | Meaning |
| --- | --- |
| `--memory_policy` | Memory scheduling: `auto` (default), `staged` (CPU offload between phases), or `resident` (models stay on GPU). |
| `--memory_report path.jsonl` | Append setup, read, encode, upscale and combined decode/postprocess/write measurements, plans, attempts, retries and completion events. |
| `--resume` | Resume long video processing across sessions. Persists chunk checkpoints and manifest; losslessly stitches chunks into the final MP4 with FFmpeg concat. |

Budgets are **admission estimates and model-retention caps**, not allocator-enforced process limits. The planner reserves checkpoint/cache headroom and estimates pixel, latent, overlap and postprocess workspaces. Quantization, compiled graphs, CUDA kernels and codecs can use additional memory. Cache accounting measures parameter/buffer storage, not compiler caches. Cache caps are enforced between invocations; active inference can exceed them. CPU model offload trades VRAM for RAM. Explicit CUDA tensor offload can raise VRAM use beyond the default CPU-storage estimate.

OOM retries reduce chunk and batch sizes, decrease tile sizes (minimum 128), enable tiling, increase BlockSwap, use CPU offload, and disable retained models. The **requested resolution and seed are preserved**. Changing batch boundaries or tile sizes can still change numerical results and temporal continuity; retries do not promise identical pixels. Non-memory errors are propagated. A kernel/OS kill cannot be caught as a Python OOM.

Each attempt writes to a temporary destination beside the final output. MP4 and single-image output are published with one replacement after successful encoder close. PNG sequence replacement backs up matching numbered frames, removes stale frames from a longer prior run, and preserves unrelated files; it rolls back Python/filesystem exceptions during publication. Sequence replacement is not atomic across a machine crash. Retries never append to a prior attempt.

`--10bit` requires `--video_backend ffmpeg`. The raw pipe uses 16-bit RGB (`rgb48le`) before x265 encodes `yuv420p10le`, avoiding the former 8-bit transport bottleneck. The source/model precision still limits recoverable detail. Input video decoding remains OpenCV's 8-bit path. Audio copying is unchanged and is not implemented by this CLI.

## Reproduce the CPU baseline

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python tools/benchmark_memory.py --frames 33 330 --output memory-baseline.json
```

On a CPU-only validation machine, install PyTorch 2.11.0+cpu and torchvision 0.26.0+cpu from the PyTorch CPU index first. These commands do not need model weights. The CI workflow runs on Ubuntu 24.04 / Python 3.13 and uploads the baseline JSON.

[Recorded raw results](memory-baseline-cpu.json): Python 3.13.15, PyTorch 2.11.0+cpu, 320×180 RGB, batch 5, separate processes. The legacy case reproduces the previous whole-output BF16 → FP32 → uint8 allocation pattern; the stream case uses the actual streaming assembler and per-frame conversion. Both emit the same frame counts and synthetic pixel checksum.

| Output buffer mode | Frames | Process RSS peak | Increase above initialized runtime |
| --- | ---: | ---: | ---: |
| Legacy allocation pattern | 33 | 455.0 MiB | 63.6 MiB |
| Legacy allocation pattern | 330 | 997.4 MiB | 605.9 MiB |
| Streaming | 33 | 399.9 MiB | 8.4 MiB |
| Streaming | 330 | 399.3 MiB | 7.8 MiB |

This is a buffer regression baseline, **not a full inference benchmark**. CI allows a fixed 64 MiB variation in streaming RSS between lengths to accommodate allocator/import noise. The 50 ms sampler in real phase reports can miss short RSS spikes; the synthetic benchmark uses Linux process peak RSS. CUDA reserved-memory statistics can be uninformative with `cudaMallocAsync`; use peak allocated bytes and external GPU telemetry as well.

## Colab GPU acceptance

GPU inference was not available in the implementation workspace. Before merging, run the notebook's sample on the target runtime, then use the same checkpoint, input, resolution, seed, attention backend and workload settings to compare this branch with `adc0480`. For baseline runs on the old commit, use external RSS/GPU monitoring because it does not have `--memory_report`.

```bash
python inference_cli.py sample.mp4 --resolution 1080 --batch_size 5 \
  --chunk_size 33 --temporal_overlap 3 --seed 42 \
  --video_backend ffmpeg --10bit --memory_report memory.jsonl \
  --output output.mp4
python tools/benchmark_memory.py --profile memory.jsonl --output memory-summary.json
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,pix_fmt,nb_frames -of json output.mp4
```

Use a fixed sample set: RGB video, RGBA still, short final batch, `prepend_frames`, temporal overlap, all color modes, and at least one 1080p and 4K target. Record frames, dimensions, PSNR/SSIM plus visual comparisons, RSS peak, CUDA peak allocated bytes, runtime and any retries. Compare 33/330/3300-frame clips at the **same effective chunk/batch/tile plan**: peak memory should stabilize rather than scale with duration. Repeat directory processing with cache on/off and verify no result/context remains retained. Exercise a deliberately oversized initial batch, confirm the retry event, and verify final frame count and the absence of partial outputs. Also interrupt a running encode/decode/write and verify capture, FFmpeg and model cleanup.

The CPU tests cover blend equivalence (including overlap larger than batch stride), all six color modes against the prior four-phase assembly, RGBA behavior, temporal/spatial VAE assembly, alias safety, cache eviction, generator closure, failed writers, OOM rollback, and real FFmpeg 8/10-bit frame counts. They do not substitute for the GPU quality and peak-memory checks above.

## Typed settings and invocation state

The [CLI architecture](architecture.md) keeps per-file plans immutable and uses an
explicit `GenerationContext` for tensor ownership. Streaming still writes each
finalized batch immediately; context views are shallow and never serialize or
deep-copy tensors. A session retains model state only.
