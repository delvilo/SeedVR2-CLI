"""Checked installation and diagnostics for the supported Colab runtime."""

import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

FLASH_WHEEL_URL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/"
    "flash_attn-2.8.3+cu128torch2.11-cp313-cp313-linux_x86_64.whl"
)
FFMPEG_PPA = "ppa:ubuntuhandbook1/ffmpeg9"
GIB = 1024 ** 3


def run_step(label: str, command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    """Show the failing stage and abort the notebook on any nonzero exit status."""
    print(f"[{label}] {' '.join(command)}", flush=True)
    try:
        return subprocess.run(command, check=True, text=True, capture_output=capture)
    except subprocess.CalledProcessError as error:
        if capture:
            print(error.stdout or "", error.stderr or "", file=sys.stderr)
        raise RuntimeError(f"{label} failed (exit {error.returncode})") from error


def check_flash_wheel(url: str, torch, *, machine: str | None = None,
                      python: tuple[int, int] | None = None) -> dict:
    """Check this prebuilt wheel's Python, CUDA, torch and Linux architecture tags."""
    filename = Path(urlparse(url).path).name
    match = re.fullmatch(
        r"flash_attn-[\w.]+\+cu(\d+)torch(\d+\.\d+)-"
        r"cp(\d+)-cp(\d+)-linux_(x86_64)\.whl", filename
    )
    if not match:
        raise RuntimeError(f"Unexpected FlashAttention wheel filename: {filename}")
    cuda_tag, torch_tag, py_tag, abi_tag, arch_tag = match.groups()
    python = python or sys.version_info[:2]
    machine = machine or platform.machine()
    expected_python = f"{python[0]}{python[1]}"
    expected_cuda = (torch.version.cuda or "").replace(".", "")
    expected_torch = ".".join(torch.__version__.split("+", 1)[0].split(".")[:2])
    if (py_tag != expected_python or abi_tag != expected_python
            or cuda_tag != expected_cuda or torch_tag != expected_torch
            or arch_tag != machine or platform.system() != "Linux"):
        raise RuntimeError(
            f"FlashAttention wheel {filename} is incompatible with "
            f"Python {python[0]}.{python[1]}, torch {torch.__version__}, "
            f"CUDA {torch.version.cuda}, {platform.system()}/{machine}"
        )
    return {"wheel": filename, "python_tag": f"cp{py_tag}", "cuda_tag": f"cu{cuda_tag}",
            "torch_tag": torch_tag, "platform_tag": f"linux_{arch_tag}"}


def inspect_ffmpeg() -> dict:
    """Verify the installed binaries, actual major version and MP4 encoders."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and ffprobe must both be on PATH")
    version = run_step("FFmpeg version", [ffmpeg, "-version"], capture=True).stdout.splitlines()[0]
    probe_version = run_step("ffprobe version", [ffprobe, "-version"], capture=True).stdout.splitlines()[0]
    for label, line in (("ffmpeg", version), ("ffprobe", probe_version)):
        match = re.match(rf"^{label} version (\d+)(?:\.|\s|$)", line)
        if not match or int(match.group(1)) != 9:
            raise RuntimeError(f"Expected {label} 9 from {FFMPEG_PPA}; found {line}")
    encoders = run_step("FFmpeg encoders", [ffmpeg, "-hide_banner", "-encoders"], capture=True).stdout
    for encoder in ("libx264", "libx265"):
        if not re.search(rf"^\s*V\S*\s+{encoder}\s", encoders, re.MULTILINE):
            raise RuntimeError(f"FFmpeg 9 is missing required {encoder} video encoder")
    return {"ffmpeg_path": ffmpeg, "ffmpeg_version": version,
            "ffprobe_path": ffprobe, "ffprobe_version": probe_version,
            "video_encoders": ["libx264", "libx265"]}


def install_dependencies(repo_dir: Path, *, install_flash_attn: bool = False) -> dict:
    """Install the application and FFmpeg 9; optional FA2 only on a matching GPU."""
    import torch
    from src.utils.runtime import validate_runtime

    validate_runtime()
    wheel = check_flash_wheel(FLASH_WHEEL_URL, torch)
    run_step("Python requirements", [sys.executable, "-m", "pip", "install", "-r",
                                     str(repo_dir / "requirements.txt")])
    validate_runtime()  # Catch an accidental replacement of Colab's CUDA torch.
    import torchvision  # type: ignore[import-untyped]
    from torchvision.transforms import InterpolationMode  # type: ignore[import-untyped] # noqa: F401

    flash_status = "skipped (default SDPA attention)"
    if install_flash_attn:
        if torch.cuda.get_device_capability(0)[0] not in (8, 9):
            raise RuntimeError("This FlashAttention 2 wheel targets Ampere/Ada/Hopper GPUs (SM 8.x/9.x)")
        run_step("FlashAttention wheel", [sys.executable, "-m", "pip", "install", "--no-deps",
                                           FLASH_WHEEL_URL])
        import flash_attn  # type: ignore[import-not-found]
        import flash_attn_2_cuda  # type: ignore[import-not-found] # noqa: F401
        from flash_attn import flash_attn_func  # type: ignore[import-not-found]
        q = torch.randn((1, 16, 2, 64), dtype=torch.float16, device="cuda:0")
        flash_attn_func(q, q, q)
        torch.cuda.synchronize(0)
        del q
        flash_status = f"installed {flash_attn.__version__} (CUDA kernel smoke test passed)"

    run_step("APT package index", ["apt-get", "update"])
    run_step("PPA prerequisites", ["apt-get", "install", "-y", "software-properties-common"])
    run_step("FFmpeg 9 PPA", ["add-apt-repository", "-y", FFMPEG_PPA])
    run_step("APT package index", ["apt-get", "update"])
    run_step("FFmpeg 9 packages", ["apt-get", "install", "-y", "ffmpeg"])
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "torchvision": torchvision.__version__, "flash_attention": flash_status,
            "flash_wheel_tags": wheel, **inspect_ffmpeg()}


def write_diagnostics(repo_dir: Path, input_dir: Path, model_dir: Path, output_dir: Path,
                      install_result: dict, *, ram_budget_gb: float = 0,
                      vram_budget_gb: float = 0) -> Path:
    """Record paths, GPU/driver, available memory and the CLI's budget allowances."""
    import torch
    from src.optimization.memory_budget import available_ram_bytes

    if ram_budget_gb < 0 or vram_budget_gb < 0:
        raise ValueError("RAM and VRAM budgets must be nonnegative")
    available_ram = available_ram_bytes()
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    ram_allowance = min(int(ram_budget_gb * GIB), available_ram) if ram_budget_gb else int(available_ram * .8)
    vram_allowance = min(int(vram_budget_gb * GIB), free_vram) if vram_budget_gb else int(free_vram * .9)
    gpu = run_step("NVIDIA GPU", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                   "--format=csv,noheader"], capture=True).stdout.strip()
    revision = run_step("Source revision", ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                        capture=True).stdout.strip()
    report = {
        "source_revision": revision, "os": platform.platform(), "python": sys.version.split()[0],
        "gpu": gpu, "available_ram_gib": round(available_ram / GIB, 2),
        "free_vram_gib": round(free_vram / GIB, 2), "total_vram_gib": round(total_vram / GIB, 2),
        "ram_budget_gb": ram_budget_gb, "vram_budget_gb": vram_budget_gb,
        "effective_ram_budget_gib": round(ram_allowance / GIB, 2),
        "effective_vram_budget_gib": round(vram_allowance / GIB, 2),
        "paths": {"repository": str(repo_dir), "input": str(input_dir),
                  "models": str(model_dir), "output": str(output_dir)}, **install_result,
    }
    path = output_dir / "colab_diagnostics.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Diagnostics saved:", path)
    return path
