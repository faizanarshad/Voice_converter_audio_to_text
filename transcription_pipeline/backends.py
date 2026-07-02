"""Speech-to-text backends.

`TranscriptionBackend` is a small abstraction so the pipeline itself never
depends on a specific speech-to-text engine. Two implementations are
provided:

- `FasterWhisperBackend`: real transcription using faster-whisper
  (CTranslate2 Whisper). This is what actually runs in production.
- `MockBackend`: deterministic, dependency-free fake transcription. This
  is what the problem statement's "mock data where needed" is for - it
  lets the rest of the pipeline (chunk merging, postprocessing, CLI, file
  I/O) be exercised in tests without downloading model weights or
  depending on network access / GPU.

Both yield the same `RawSegment` shape so the pipeline code doesn't need
to know which one produced them.
"""
from __future__ import annotations

import abc
import dataclasses
import math
from pathlib import Path
from typing import Iterator, Optional


@dataclasses.dataclass
class RawSegment:
    start: float
    end: float
    text: str


@dataclasses.dataclass
class TranscriptionOutput:
    segments: list[RawSegment]
    language: Optional[str]


class TranscriptionBackend(abc.ABC):
    """Interface every speech-to-text backend must implement."""

    @abc.abstractmethod
    def transcribe(self, wav_path: Path, language: Optional[str] = None) -> TranscriptionOutput:
        """Transcribe a mono 16kHz WAV file and return segments with timestamps."""
        raise NotImplementedError


class FasterWhisperBackend(TranscriptionBackend):
    """Real transcription via faster-whisper.

    Model loading is lazy and cached on the instance: constructing this
    class is cheap, the (potentially multi-hundred-MB) model download and
    load only happens the first time `transcribe` is actually called.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, wav_path: Path, language: Optional[str] = None) -> TranscriptionOutput:
        model = self._load_model()
        segments_iter, info = model.transcribe(
            str(wav_path),
            language=language,
            vad_filter=True,
        )
        segments = [
            RawSegment(start=seg.start, end=seg.end, text=seg.text.strip())
            for seg in segments_iter
        ]
        return TranscriptionOutput(segments=segments, language=info.language)


class MockBackend(TranscriptionBackend):
    """Deterministic fake backend for tests and offline/no-GPU environments.

    Produces one fixed-length segment per `segment_seconds` of audio, with
    placeholder text. It reads only the duration of the WAV (via the
    standard-library `wave` module) so it has zero third-party
    dependencies and never touches the network.
    """

    def __init__(self, segment_seconds: float = 5.0, language: str = "en"):
        self.segment_seconds = segment_seconds
        self.language = language

    def transcribe(self, wav_path: Path, language: Optional[str] = None) -> TranscriptionOutput:
        import wave

        with wave.open(str(wav_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate) if rate else 0.0

        segments: list[RawSegment] = []
        n_segments = max(1, math.ceil(duration / self.segment_seconds))
        for i in range(n_segments):
            start = i * self.segment_seconds
            end = min(duration, start + self.segment_seconds)
            if start >= duration:
                break
            segments.append(
                RawSegment(
                    start=start,
                    end=end,
                    text=f"This is mock transcription segment {i + 1}.",
                )
            )
        return TranscriptionOutput(segments=segments, language=language or self.language)
