"""Resumable video processing using chunk checkpoints and lossless FFmpeg assembly.

Enables processing of arbitrarily long videos across sessions on resource-limited
environments like Google Colab free tier.
"""

import json
import os
import shutil
import subprocess
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from ..media.output import OutputSink
from ..media.reader import VideoReader, probe_dimensions
from ..utils.debug import Debug
from .planning import plan_file

if TYPE_CHECKING:
    from .scheduler import FileJob
    from .session import InferenceSession


def get_resume_dir(destination: str) -> Path:
    """Return the dedicated hidden checkpoint directory for a destination video."""
    dest = Path(destination).resolve()
    return dest.parent / f".{dest.name}_chunks"


def load_resume_manifest(chunks_dir: Path) -> Optional[dict]:
    """Read the resume manifest JSON file if it exists and is valid."""
    manifest_file = chunks_dir / "resume_manifest.json"
    if not manifest_file.is_file():
        return None
    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "chunks" in data:
            return data
    except Exception:
        pass
    return None


def save_resume_manifest(chunks_dir: Path, manifest: dict) -> None:
    """Save the resume manifest JSON file atomically."""
    manifest_file = chunks_dir / "resume_manifest.json"
    temp_file = chunks_dir / "resume_manifest.json.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, manifest_file)


def assemble_chunks(
    chunk_paths: list[str],
    source_video: str,
    destination: str,
    fps: float,
    debug: Debug
) -> bool:
    """Concatenate chunk MP4 files losslessly using FFmpeg concat demuxer and mux source audio."""
    if not chunk_paths:
        return False

    dest_path = Path(destination).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest_path.parent / f".{dest_path.name}.assembling.mp4"

    concat_list = dest_path.parent / f".{dest_path.name}.concat_list.txt"
    try:
        with open(concat_list, "w", encoding="utf-8") as f:
            for cp in chunk_paths:
                abs_cp = str(Path(cp).resolve()).replace("'", "'\\''")
                f.write(f"file '{abs_cp}'\n")

        debug.log(f"[RESUME] Assembling {len(chunk_paths)} chunks with FFmpeg concat...", category="resume", force=True)

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-i", str(Path(source_video).resolve()),
            "-c:v", "copy",
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a?",
            "-shortest",
            str(temp_dest)
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not temp_dest.is_file() or temp_dest.stat().st_size == 0:
            # Fallback without audio mapping if audio mux failed
            debug.log(f"[RESUME] Audio remux fallback (video-only concat): {res.stderr.strip()}", category="resume")
            cmd_fallback = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-c:v", "copy",
                str(temp_dest)
            ]
            res_fb = subprocess.run(cmd_fallback, capture_output=True, text=True)
            if res_fb.returncode != 0 or not temp_dest.is_file() or temp_dest.stat().st_size == 0:
                raise RuntimeError(f"FFmpeg chunk assembly failed: {res_fb.stderr.strip()}")

        os.replace(temp_dest, dest_path)
        return True

    finally:
        if concat_list.is_file():
            concat_list.unlink()
        if temp_dest.is_file():
            temp_dest.unlink()


def process_resumable_video(
    job: 'FileJob',
    device: str,
    session: 'InferenceSession',
    debug: Debug
) -> int:
    """Process a video job with chunk-level persistence and resumption support."""
    config = job.config
    dest_path = Path(job.destination).resolve()

    # 1. Quick check: if target file already exists and is non-empty
    if dest_path.is_file() and dest_path.stat().st_size > 0:
        debug.log(f"[RESUME] Target output '{job.destination}' already exists. Checking completeness...", category="resume", force=True)
        try:
            with VideoReader(str(dest_path)) as existing_reader:
                with VideoReader(job.source, config.video.skip_first_frames, config.video.load_cap) as src_reader:
                    if existing_reader.count >= src_reader.count and src_reader.count > 0:
                        debug.log(f"[RESUME] Output video is already complete ({existing_reader.count}/{src_reader.count} frames). Skipping.", category="resume", force=True)
                        return existing_reader.count
        except Exception:
            pass

    width, height = probe_dimensions(job.source, job.kind, debug)
    planned = plan_file(config, width, height, device, session, debug)

    chunks_dir = get_resume_dir(job.destination)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_resume_manifest(chunks_dir)
    if manifest is None:
        manifest = {
            "source": job.source,
            "destination": job.destination,
            "chunk_size": planned.video.chunk_size,
            "chunks": [],
            "completed_frames": 0,
        }

    # Verify existing chunks
    valid_chunks = []
    frames_already_done = 0
    for chunk in manifest.get("chunks", []):
        cfile = chunks_dir / chunk["file"]
        if cfile.is_file() and cfile.stat().st_size > 0:
            valid_chunks.append(chunk)
            frames_already_done += chunk.get("frames", 0)
        else:
            break  # Stop at first broken chunk

    manifest["chunks"] = valid_chunks
    manifest["completed_frames"] = frames_already_done
    save_resume_manifest(chunks_dir, manifest)

    with VideoReader(job.source, planned.video.skip_first_frames, planned.video.load_cap) as full_source:
        total_frames = full_source.count
        fps = full_source.fps

    if frames_already_done >= total_frames and total_frames > 0:
        # All chunks completed, assemble
        chunk_files = [str(chunks_dir / c["file"]) for c in valid_chunks]
        assemble_chunks(chunk_files, job.source, job.destination, fps, debug)
        shutil.rmtree(chunks_dir, ignore_errors=True)
        debug.log(f"[RESUME] Assembly complete: {job.destination} ({total_frames} frames)", category="success", force=True)
        return total_frames

    chunk_size = planned.video.chunk_size
    overlap = planned.inference.sampling.temporal_overlap
    total_chunks = (total_frames + chunk_size - 1) // chunk_size

    if frames_already_done > 0:
        debug.log(
            f"[RESUME] Resuming from frame {frames_already_done}/{total_frames} "
            f"({len(valid_chunks)}/{total_chunks} chunks already completed)",
            category="resume", force=True
        )

    # Prepare raw overlap context if resuming mid-video
    prev_raw_tail: Optional[torch.Tensor] = None
    if frames_already_done > 0 and overlap > 0:
        tail_start = max(0, planned.video.skip_first_frames + frames_already_done - overlap)
        with VideoReader(job.source, skip=tail_start, limit=overlap) as tail_reader:
            prev_raw_tail = tail_reader.read(overlap)
            if prev_raw_tail is not None and len(prev_raw_tail) > 0:
                debug.log(f"[RESUME] Restored {len(prev_raw_tail)} raw context frames for seamless temporal overlap blending", category="resume")

    chunk_idx = len(valid_chunks)
    frames_read = frames_already_done

    with VideoReader(job.source, planned.video.skip_first_frames + frames_already_done, total_frames - frames_already_done) as stream_source:
        while frames_read < total_frames:
            read_count = min(chunk_size, total_frames - frames_read)
            new_frames = stream_source.read(read_count)
            if new_frames is None or len(new_frames) == 0:
                break

            actual_new_count = len(new_frames)
            chunk_idx += 1

            chunk_filename = f"chunk_{chunk_idx:05d}.mp4"
            chunk_output_path = str(chunks_dir / chunk_filename)
            chunk_temp_path = str(chunks_dir / f".tmp_{chunk_filename}")

            context_count = min(overlap, len(prev_raw_tail)) if prev_raw_tail is not None else 0
            frames = torch.cat((prev_raw_tail[-context_count:], new_frames)) if context_count and prev_raw_tail is not None else new_frames
            prev_raw_tail = frames[-overlap:].clone() if overlap else None
            del new_frames

            debug.log(
                f"[RESUME] Processing Chunk {chunk_idx}/{total_chunks}: "
                f"{actual_new_count} new frames (progress: {frames_read + actual_new_count}/{total_frames})",
                category="generation", force=True
            )

            drop = context_count
            chunk_cfg = planned.inference
            if chunk_idx > 1:
                chunk_cfg = replace(chunk_cfg, sampling=replace(chunk_cfg.sampling, prepend_frames=0))

            written_in_chunk = 0
            with OutputSink(chunk_temp_path, planned.output, job.source, job.kind, fps, debug) as sink:
                with closing(session.iter_batches(frames, chunk_cfg, device, planned.memory.report)) as outputs:
                    for result in outputs:
                        skip = min(drop, len(result))
                        drop -= skip
                        if skip < len(result):
                            batch_slice = result[skip:]
                            sink.write(batch_slice)
                            written_in_chunk += len(batch_slice)
                            del batch_slice
                        del result
            del frames

            if os.path.exists(chunk_temp_path) and os.path.getsize(chunk_temp_path) > 0:
                os.replace(chunk_temp_path, chunk_output_path)
            else:
                raise RuntimeError(f"Failed to produce video chunk: {chunk_output_path}")

            frames_read += actual_new_count
            valid_chunks.append({
                "chunk_idx": chunk_idx,
                "frames": actual_new_count,
                "file": chunk_filename
            })
            manifest["chunks"] = valid_chunks
            manifest["completed_frames"] = frames_read
            save_resume_manifest(chunks_dir, manifest)

            debug.log(f"[RESUME] Saved checkpoint chunk {chunk_idx}/{total_chunks} ({actual_new_count} frames)", category="resume", force=True)

    # All chunks complete! Assemble with FFmpeg concat
    chunk_files = [str(chunks_dir / c["file"]) for c in valid_chunks]
    assemble_chunks(chunk_files, job.source, job.destination, fps, debug)
    shutil.rmtree(chunks_dir, ignore_errors=True)
    debug.log(f"[RESUME] Output successfully assembled into: {job.destination} ({frames_read} frames)", category="file", force=True)

    return frames_read
