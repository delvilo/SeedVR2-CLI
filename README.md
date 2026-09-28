# SeedVR2 Lite — standalone CLI

SeedVR2 image and video upscaling from a Linux command line. The existing inference pipeline supports one or several NVIDIA GPUs, directory processing, bounded-memory video chunks, model reuse, VAE tiling, BlockSwap, and optional torch.compile.

This fork targets the current Google Colab runtime: Ubuntu 24.04, Python 3.13, PyTorch 2.11.0+cu128, and the CUDA 12.8 toolkit. The "CUDA Version 13.0" reported by nvidia-smi describes the driver capability; it is not the CUDA build used by PyTorch.

## Start in Colab

Open [the Colab notebook](colab/SeedVR2Lite_Colab.ipynb). It mounts Google Drive, checks the GPU runtime, installs application dependencies, and runs a small sample before your own input. Configure the Drive folder in its path cell. Weights and results are stored on Drive; the repository runs from local Colab storage.

The runtime already supplies PyTorch and torchvision. **requirements.txt** contains only the remaining application dependencies. The notebook checks the installed PyTorch version before installing anything and does not replace the CUDA wheel. The notebook uses FFmpeg for MP4 output; 10-bit encoding is available with --10bit.

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
  --resolution 1080 --batch_size 33 --chunk_size 330 \
  --temporal_overlap 3 --video_backend ffmpeg \
  --model_dir /content/drive/MyDrive/SeedVR2/models \
  --output /content/drive/MyDrive/SeedVR2/output/video.mp4

# Directory processing can reuse loaded models between files.
python inference_cli.py /content/drive/MyDrive/SeedVR2/input \
  --output /content/drive/MyDrive/SeedVR2/output \
  --model_dir /content/drive/MyDrive/SeedVR2/models \
  --cache_dit --cache_vae --dit_offload_device cpu --vae_offload_device cpu

# Multi-GPU processing, when the runtime exposes two CUDA GPUs.
python inference_cli.py video.mp4 --cuda_device 0,1 \
  --batch_size 33 --temporal_overlap 3
~~~

When --model_dir is omitted, models are stored under **models/SEEDVR2** in the repository. The SEEDVR2_MODEL_DIR environment variable can set another default. --output controls the result path; omit it to use the CLI's automatically generated filename.

## Common options

| Option | Purpose |
| --- | --- |
| --dit_model | Choose a 3B or 7B FP16, FP8, or GGUF model; the default is 3B FP8. |
| --resolution | Target short edge in pixels. |
| --batch_size | Frames per model batch; use 1, 5, 9, 13, 17, and so on (4n+1). |
| --chunk_size | Process long videos in chunks instead of keeping all frames in RAM. |
| --vae_encode_tiled / --vae_decode_tiled | Reduce peak VRAM for large frames. |
| --blocks_to_swap / --dit_offload_device cpu | Offload DiT blocks to system RAM when VRAM is limited. |
| --cache_dit / --cache_vae | Reuse loaded models for directory or chunk processing. |
| --attention_mode | PyTorch SDPA is the default. Optional Flash/Sage backends require separate compatible installations. |
| --compile_dit / --compile_vae | Enable torch.compile; first-run compilation can take time. |
| --video_backend ffmpeg / --10bit | Encode MP4 using FFmpeg, optionally with 10-bit x265 output. |
| --debug | Show environment, timing, VRAM, and processing details. |

Use --help for the complete option list. Larger models, batches, and output resolutions require more VRAM. Start with a short sample at a modest resolution before processing a long video.

## Development checks

Run `python -m compileall -q inference_cli.py src tests` and `python -m unittest discover -s tests -v` before proposing changes. The Colab notebook's short video sample checks the full inference path on the target GPU runtime.

## Models and attribution

DiT and VAE checkpoints are downloaded on first use from the repositories listed in **src/utils/model_registry.py**, checked against the listed SHA-256 hashes, and reused from the selected model directory. The small text embeddings included with this repository remain part of inference.

This implementation is derived from [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler), which is based on [ByteDance SeedVR](https://github.com/ByteDance-Seed/SeedVR). Original contributors and their credits are preserved in the source history. See [LICENSE](LICENSE) for the Apache 2.0 license.
