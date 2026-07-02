#!/usr/bin/env python3
"""CLI entrypoint for the transcription pipeline.

Examples:
    python cli.py sample_audio/hello.mp3
    python cli.py sample_audio/hello.wav --backend mock
    python cli.py long_meeting.wav --model-size small --chunk-seconds 600
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from transcription_pipeline.backends import FasterWhisperBackend, MockBackend
from transcription_pipeline.pipeline import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_OVERLAP_SECONDS,
    TranscriptionPipeline,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe an audio file to text with timestamps.")
    parser.add_argument("audio_path", type=Path, help="Path to a WAV/MP3/etc audio file")
    parser.add_argument(
        "--backend", choices=["whisper", "mock"], default="whisper",
        help="Transcription backend to use (default: whisper)",
    )
    parser.add_argument("--model-size", default="base", help="faster-whisper model size (default: base)")
    parser.add_argument("--device", default="cpu", help="Device for the whisper model (default: cpu)")
    parser.add_argument("--language", default=None, help="Force a language code (default: auto-detect)")
    parser.add_argument(
        "--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS,
        help="Max seconds of audio processed per chunk for long files",
    )
    parser.add_argument(
        "--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS,
        help="Overlap between consecutive chunks, in seconds",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Directory to write <stem>.json/.srt/.txt into (default: ./output)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO-level logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.backend == "mock":
        backend = MockBackend()
    else:
        backend = FasterWhisperBackend(model_size=args.model_size, device=args.device)

    pipeline = TranscriptionPipeline(
        backend=backend,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
    )

    result = pipeline.run(args.audio_path, output_dir=args.output_dir, language=args.language)

    print(f"Language: {result.language}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Segments: {len(result.segments)}")
    print()
    for seg in result.segments:
        print(f"[{seg.start:7.2f} - {seg.end:7.2f}] {seg.text}")
    print()
    print(f"Wrote outputs to {args.output_dir}/{args.audio_path.stem}.{{json,srt,txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
