"""Audio format handling and chunking, backed by ffmpeg.

Design decision: rather than trusting each downstream component (or each
speech-to-text backend) to understand every possible input container/codec,
we normalize every input up front to a single canonical format - 16kHz
mono 16-bit PCM WAV. ffmpeg already understands virtually every audio
format (mp3, m4a/aac, flac, ogg, wav, video containers, ...), so this one
conversion step is what actually answers "how do you handle different
audio formats": everything funnels through the same decoder and the rest
of the pipeline only ever has to deal with one format.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


class FFmpegNotFoundError(RuntimeError):
    pass


class AudioProbeError(RuntimeError):
    pass


@dataclasses.dataclass
class AudioInfo:
    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    codec: str


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegNotFoundError(
            f"'{name}' was not found on PATH. Install ffmpeg (which bundles ffprobe) "
            "to enable audio format conversion, e.g. `brew install ffmpeg`."
        )
    return path


def probe_audio(path: Path) -> AudioInfo:
    """Inspect a media file's audio stream without decoding it."""
    ffprobe = _require_binary("ffprobe")
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,codec_name",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioProbeError(f"ffprobe failed for {path}: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise AudioProbeError(f"No audio stream found in {path}")
    stream = streams[0]
    duration = float(data.get("format", {}).get("duration", 0.0))
    return AudioInfo(
        path=path,
        duration_seconds=duration,
        sample_rate=int(stream.get("sample_rate", 0)),
        channels=int(stream.get("channels", 0)),
        codec=str(stream.get("codec_name", "unknown")),
    )


def normalize_to_wav(src_path: Path, dst_path: Path) -> Path:
    """Decode any ffmpeg-supported input into mono 16kHz PCM WAV.

    This is the single place format differences get absorbed, so every
    later stage (chunking, the transcription backend) can assume one
    consistent format regardless of whether the caller uploaded mp3, m4a,
    flac, ogg, or wav.
    """
    ffmpeg = _require_binary("ffmpeg")
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(src_path),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        "-f", "wav",
        str(dst_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioProbeError(f"ffmpeg conversion failed for {src_path}: {result.stderr.strip()}")
    return dst_path


@dataclasses.dataclass
class AudioChunk:
    path: Path
    start_offset: float
    duration: float


def iter_chunks(
    wav_path: Path,
    total_duration: float,
    chunk_seconds: float,
    overlap_seconds: float,
    work_dir: Path,
) -> Iterator[AudioChunk]:
    """Slice a normalized WAV into overlapping chunks for long-audio processing.

    Long files are handled by never holding more than one chunk of decoded
    audio at a time: we cut the (already-normalized) WAV into fixed-size
    windows with a small overlap, transcribe them one at a time, and delete
    each chunk as soon as it has been consumed. A small overlap
    (`overlap_seconds`) protects against words being cut off exactly at a
    chunk boundary; the caller de-duplicates the overlap region using
    segment timestamps when merging results.
    """
    if total_duration <= chunk_seconds:
        yield AudioChunk(path=wav_path, start_offset=0.0, duration=total_duration)
        return

    ffmpeg = _require_binary("ffmpeg")
    start = 0.0
    index = 0
    step = chunk_seconds - overlap_seconds
    if step <= 0:
        raise ValueError("chunk_seconds must be greater than overlap_seconds")

    while start < total_duration:
        duration = min(chunk_seconds, total_duration - start)
        chunk_path = work_dir / f"chunk_{index:04d}.wav"
        cmd = [
            ffmpeg,
            "-y",
            "-ss", str(start),
            "-t", str(duration),
            "-i", str(wav_path),
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", str(TARGET_CHANNELS),
            "-f", "wav",
            str(chunk_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise AudioProbeError(f"ffmpeg chunking failed at offset {start}: {result.stderr.strip()}")

        yield AudioChunk(path=chunk_path, start_offset=start, duration=duration)

        index += 1
        start += step


def make_temp_dir(prefix: str = "transcription_pipeline_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))
