"""Colab setup checks run without package installation or a GPU."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from colab import workflow


def test_all_setup_commands_fail_with_the_step_name(monkeypatch):
    def fail(command, **kwargs):
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(workflow.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match=r"Python requirements failed \(exit 7\)"):
        workflow.run_step("Python requirements", ["pip", "install"])


def test_flash_wheel_tags_match_colab_but_reject_other_builds():
    torch = SimpleNamespace(__version__="2.11.0+cu128", version=SimpleNamespace(cuda="12.8"))
    assert workflow.check_flash_wheel(workflow.FLASH_WHEEL_URL, torch,
                                      machine="x86_64", python=(3, 13))["python_tag"] == "cp313"
    with pytest.raises(RuntimeError, match="incompatible"):
        workflow.check_flash_wheel(workflow.FLASH_WHEEL_URL, torch,
                                   machine="aarch64", python=(3, 13))
    with pytest.raises(RuntimeError, match="incompatible"):
        workflow.check_flash_wheel(workflow.FLASH_WHEEL_URL, torch,
                                   machine="x86_64", python=(3, 12))
    torch.version.cuda = "13.0"
    with pytest.raises(RuntimeError, match="incompatible"):
        workflow.check_flash_wheel(workflow.FLASH_WHEEL_URL, torch,
                                   machine="x86_64", python=(3, 13))


def test_ffmpeg_checks_actual_version_and_encoder_set(monkeypatch):
    monkeypatch.setattr(workflow.shutil, "which", lambda name: f"/usr/bin/{name}")
    versions = {"ffmpeg": "ffmpeg version 9.1 Copyright ...\n",
                "ffprobe": "ffprobe version 9.1 Copyright ...\n"}
    encoders = "Encoders:\n V....D libx264 H.264\n V....D libx265 HEVC\n"

    def run(label, command, *, capture):
        assert capture
        return SimpleNamespace(stdout=encoders if "-encoders" in command else
                               versions[Path(command[0]).name])

    monkeypatch.setattr(workflow, "run_step", run)
    assert workflow.inspect_ffmpeg()["video_encoders"] == ["libx264", "libx265"]
    versions["ffprobe"] = "ffprobe version 6.1 Copyright ...\n"
    with pytest.raises(RuntimeError, match="Expected ffprobe 9"):
        workflow.inspect_ffmpeg()
    versions["ffprobe"] = "ffprobe version 9.1 Copyright ...\n"
    encoders = " V....D libx264 H.264\n"
    with pytest.raises(RuntimeError, match="missing required libx265"):
        workflow.inspect_ffmpeg()


def test_notebook_cells_compile_and_use_checked_processes():
    notebook = json.loads((Path(__file__).parents[1] / "colab/SeedVR2Lite_Colab.ipynb").read_text())
    assert notebook["nbformat"] == 4
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    all_code = "\n".join(cells)
    assert "ppa:ubuntuhandbook1/ffmpeg9" in all_code
    assert "L4 GPU" in "\n".join(["".join(cell["source"]) for cell in notebook["cells"]])
    assert "T4 GPU" in "\n".join(["".join(cell["source"]) for cell in notebook["cells"]])


def test_diagnostics_records_effective_budgets_and_paths(monkeypatch, tmp_path):
    import torch
    from src.optimization import memory_budget

    monkeypatch.setattr(memory_budget, "available_ram_bytes", lambda: 10 * workflow.GIB)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 * workflow.GIB, 8 * workflow.GIB))

    def run(label, command, *, capture):
        return SimpleNamespace(stdout="abc123\n" if label == "Source revision" else "L4, 570.0, 8192 MiB\n")

    monkeypatch.setattr(workflow, "run_step", run)
    path = workflow.write_diagnostics(tmp_path / "repo", tmp_path / "input",
                                      tmp_path / "models", tmp_path, {"ffmpeg_version": "9.1"},
                                      ram_budget_gb=20, vram_budget_gb=0)
    report = json.loads(path.read_text())
    assert report["effective_ram_budget_gib"] == 10
    assert report["effective_vram_budget_gib"] == 3.6
    assert report["paths"]["models"] == str(tmp_path / "models")
    assert report["source_revision"] == "abc123"
