import json

from transcription_pipeline.backends import MockBackend
from transcription_pipeline.pipeline import TranscriptionPipeline


def test_pipeline_end_to_end_short_audio(short_wav, tmp_path):
    pipeline = TranscriptionPipeline(backend=MockBackend(segment_seconds=1.0))
    output_dir = tmp_path / "out"

    result = pipeline.run(short_wav, output_dir=output_dir)

    assert result.duration_seconds > 0
    assert len(result.segments) == 3
    assert result.language == "en"

    json_path = output_dir / f"{short_wav.stem}.json"
    srt_path = output_dir / f"{short_wav.stem}.srt"
    txt_path = output_dir / f"{short_wav.stem}.txt"
    assert json_path.exists() and srt_path.exists() and txt_path.exists()

    data = json.loads(json_path.read_text())
    assert data["backend"] == "MockBackend"
    assert len(data["segments"]) == 3


def test_pipeline_chunks_long_audio_and_merges_segments(long_wav, tmp_path):
    pipeline = TranscriptionPipeline(
        backend=MockBackend(segment_seconds=1.0),
        chunk_seconds=5.0,
        overlap_seconds=1.0,
    )
    result = pipeline.run(long_wav)

    # Segments must be contiguous/non-overlapping and cover the full duration.
    assert result.segments[0].start == 0.0
    for prev, curr in zip(result.segments, result.segments[1:]):
        assert curr.start >= prev.start
    assert result.segments[-1].end <= result.duration_seconds + 0.5


def test_pipeline_raises_for_missing_file(tmp_path):
    pipeline = TranscriptionPipeline(backend=MockBackend())
    try:
        pipeline.run(tmp_path / "does_not_exist.wav")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
