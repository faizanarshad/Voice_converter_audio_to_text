"""Orchestrates the transcription pipeline end to end.

Flow: probe -> normalize format -> (chunk if long) -> transcribe each
chunk -> offset & de-duplicate segments -> postprocess -> write outputs.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from . import audio_utils
from .backends import RawSegment, TranscriptionBackend
from .postprocess import Segment, TranscriptResult, clean_text

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SECONDS = 300.0  # 5 minutes
DEFAULT_OVERLAP_SECONDS = 5.0
LONG_AUDIO_THRESHOLD_SECONDS = DEFAULT_CHUNK_SECONDS


class TranscriptionPipeline:
    def __init__(
        self,
        backend: TranscriptionBackend,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    ):
        self.backend = backend
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds

    def run(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None,
        language: Optional[str] = None,
    ) -> TranscriptResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        info = audio_utils.probe_audio(audio_path)
        logger.info(
            "Probed %s: duration=%.2fs sample_rate=%d channels=%d codec=%s",
            audio_path, info.duration_seconds, info.sample_rate, info.channels, info.codec,
        )

        work_dir = audio_utils.make_temp_dir()
        try:
            normalized_path = work_dir / "normalized.wav"
            audio_utils.normalize_to_wav(audio_path, normalized_path)

            raw_segments: list[RawSegment] = []
            detected_language: Optional[str] = language

            for chunk in audio_utils.iter_chunks(
                normalized_path,
                total_duration=info.duration_seconds,
                chunk_seconds=self.chunk_seconds,
                overlap_seconds=self.overlap_seconds,
                work_dir=work_dir,
            ):
                logger.info(
                    "Transcribing chunk offset=%.2fs duration=%.2fs", chunk.start_offset, chunk.duration
                )
                output = self.backend.transcribe(chunk.path, language=language)
                if detected_language is None:
                    detected_language = output.language

                for seg in output.segments:
                    absolute_start = seg.start + chunk.start_offset
                    absolute_end = seg.end + chunk.start_offset
                    raw_segments.append(
                        RawSegment(start=absolute_start, end=absolute_end, text=seg.text)
                    )

                # Free the chunk file as soon as it's been consumed so disk
                # usage for long files stays bounded to ~1 chunk at a time.
                if chunk.path != normalized_path:
                    chunk.path.unlink(missing_ok=True)

            merged = _dedupe_overlapping_segments(raw_segments)
            segments = [
                Segment(id=i, start=s.start, end=s.end, text=clean_text(s.text))
                for i, s in enumerate(merged)
            ]

            result = TranscriptResult(
                source_file=str(audio_path),
                duration_seconds=info.duration_seconds,
                language=detected_language,
                backend=type(self.backend).__name__,
                segments=segments,
            )

            if output_dir is not None:
                result.write(Path(output_dir), stem=audio_path.stem)

            return result
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


def _dedupe_overlapping_segments(raw_segments: list[RawSegment]) -> list[RawSegment]:
    """Drop segments that fall entirely inside the overlap already covered
    by the previous chunk's segments.

    Chunks are produced with a small time overlap so words aren't cut off
    at a boundary; that means the tail of chunk N and the head of chunk
    N+1 can both transcribe the same few seconds of audio. We keep
    segments in start-time order and skip any new segment that starts
    before the furthest point already covered.
    """
    merged: list[RawSegment] = []
    covered_until = -1.0
    for seg in sorted(raw_segments, key=lambda s: s.start):
        if seg.start < covered_until - 0.05:
            continue
        merged.append(seg)
        covered_until = max(covered_until, seg.end)
    return merged
