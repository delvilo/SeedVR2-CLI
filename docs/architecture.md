# CLI and inference architecture

`python inference_cli.py ...` remains the supported command. The entry point only
calls `src.cli.app.main`; it does not parse arguments or change CUDA visibility
when imported.

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `src/cli/arguments.py` | Parse and validate flags; translate the local argparse namespace into `AppConfig` once. |
| `src/cli/bootstrap.py` | Select one visible GPU before importing PyTorch. Preserve CPU thread and allocator settings supplied by the user. |
| `src/cli/app.py` | Runtime validation, model download, CLI logging and exit status. |
| `src/config.py` | Frozen, slotted settings for input/output, sampling, models, offload, tiling, compilation, cache, video windows and memory. No tensor or codec imports. |
| `src/media/reader.py` | Discover media, read images, own video captures and decode bounded uint8 windows. |
| `src/media/output.py` | Output naming, per-frame quantization, MP4/PNG sinks, writer close/abort and transactional publication. |
| `src/core/scheduler.py` | Resolve each file's format/path, schedule windows, restore raw overlap and restart failed OOM attempts. |
| `src/core/planning.py` | Derive a per-file execution configuration from available memory and session cache budgets. |
| `src/core/session.py` | Own one runner and bounded model caches; expose `iter_batches` independently of the CLI. |
| `src/core/pipeline.py` | Run encode, upscale and streamed decode/postprocess for one input window. |
| `src/core/streaming.py` | Finalize decoded overlap and yield bounded output batches. |
| `src/core/context.py` | Explicit mutable `GenerationContext` fields for one invocation's tensors and metadata. Supports GPU resident pipelines. |
| `src/core/resumable.py` | Manage chunk checkpoints, manifest state, and seamless FFmpeg concat demuxer assembly for resumable long video processing across sessions. |

Dependencies flow from the CLI to scheduling, media and sessions. Core inference
never imports the CLI or reads `sys.argv`. Model class imports are deferred until
runner construction, so parsing arguments and `--help` do not require PyTorch,
OpenCV or model dependencies.

## Settings and tensor ownership

`AppConfig` contains smaller immutable configuration objects. File format
selection, memory planning, chunk prepending and OOM fallback use
`dataclasses.replace`, so processing one file cannot modify the next file's
requested settings. `InferenceSession.iter_batches` receives only
`InferenceConfig`, an input tensor, a device and an optional memory-report path.
It has no dependency on output naming or argparse.

`GenerationContext` is mutable and invocation-local. Its slots make the supported
fields explicit instead of allowing arbitrary string keys. It owns input
references, latent lists, transforms, embeddings and temporary output. Closing
an inference generator clears these references; no context is stored in the
session. Streaming creates a shallow context view for each finalized batch and
copies only the small alpha-reference lists when necessary. It does not copy
video tensors or accumulate whole-video output.

BlockSwap settings remain typed down to their adapters. The model's OmegaConf configuration remains its native format.

## Failure and resource lifetime

1. A scheduling run owns one `InferenceSession`, reused across files.
2. Each file receives an independently resolved plan and output transaction.
3. Each attempt owns its video reader, output sink and inference generators.
4. A write failure or cancellation closes generators, releases the capture and
   aborts or closes the writer. The transaction removes partial output.
5. A caught OOM exits the exception scope, closes the session's cached runner,
   derives a smaller immutable plan and restarts the entire file.
6. Only a successful, closed output is committed to its destination.

The existing pixel processing, single-GPU execution, CPU parallelism, cache caps,
10-bit FFmpeg transport and finite input windows are retained.

## Verification

Run the existing CPU regression workflow:

```bash
python -m compileall -q inference_cli.py src tests tools
python inference_cli.py --help
python -m mypy
python -m pytest -q tests
python tools/benchmark_memory.py --frames 33 330
```

Architecture tests check import isolation, CUDA selection, immutable settings,
per-file auto-format isolation, planner isolation, typed runner configuration,
actual encode/upscale/decode/postprocess plumbing with tiny substitute models,
and a real video range decoded and streamed to PNG. The memory suite covers OOM
transaction rollback, generator cancellation, cache eviction, temporal overlap,
alpha, color correction and 8/10-bit FFmpeg round trips.

The synthetic memory benchmark measures output buffers only. Full model quality,
CUDA peak usage and optional compilation backends still require the target Colab
GPU runtime.
