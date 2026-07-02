"""Turn raw transcription segments into structured output for downstream use.

"Downstream use" is treated concretely here: other systems will want the
transcript as JSON (for programmatic consumption / storage), as SRT
(subtitles, keyed off the per-segment timestamps), and as plain text
(search indexing, display). A tiny keyword extraction pass is included as
an example of a lightweight downstream NLP step (e.g. tagging or search
facets) that consumes the cleaned transcript.
"""
from __future__ import annotations

import dataclasses
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from .backends import RawSegment

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-zA-Z']+")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "by",
    "this", "that", "it", "as", "i", "you", "we", "they", "he", "she",
    "have", "has", "had", "do", "does", "did", "not", "so", "if", "then",
}


def clean_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip())


@dataclasses.dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {"id": self.id, "start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}


@dataclasses.dataclass
class TranscriptResult:
    source_file: str
    duration_seconds: float
    language: Optional[str]
    backend: str
    segments: list[Segment]

    @property
    def full_text(self) -> str:
        return clean_text(" ".join(seg.text for seg in self.segments))

    def top_keywords(self, n: int = 10) -> list[tuple[str, int]]:
        words = (w.lower() for w in _WORD_RE.findall(self.full_text))
        words = (w for w in words if len(w) > 2 and w not in _STOPWORDS)
        return Counter(words).most_common(n)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "duration_seconds": round(self.duration_seconds, 3),
            "language": self.language,
            "backend": self.backend,
            "full_text": self.full_text,
            "segments": [seg.to_dict() for seg in self.segments],
            "keywords": [{"word": w, "count": c} for w, c in self.top_keywords()],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_srt(self) -> str:
        lines = []
        for i, seg in enumerate(self.segments, start=1):
            lines.append(str(i))
            lines.append(f"{_srt_timestamp(seg.start)} --> {_srt_timestamp(seg.end)}")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)

    def write(self, output_dir: Path, stem: str) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": output_dir / f"{stem}.json",
            "srt": output_dir / f"{stem}.srt",
            "txt": output_dir / f"{stem}.txt",
        }
        paths["json"].write_text(self.to_json(), encoding="utf-8")
        paths["srt"].write_text(self.to_srt(), encoding="utf-8")
        paths["txt"].write_text(self.full_text, encoding="utf-8")
        return paths


def _srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms_total = round(seconds * 1000)
    hours, ms_total = divmod(ms_total, 3_600_000)
    minutes, ms_total = divmod(ms_total, 60_000)
    secs, millis = divmod(ms_total, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def merge_raw_segments(raw_segments: list[RawSegment]) -> list[Segment]:
    return [
        Segment(id=i, start=r.start, end=r.end, text=clean_text(r.text))
        for i, r in enumerate(raw_segments)
    ]
