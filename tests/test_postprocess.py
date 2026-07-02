from transcription_pipeline.backends import RawSegment
from transcription_pipeline.postprocess import Segment, TranscriptResult, clean_text
from transcription_pipeline.pipeline import _dedupe_overlapping_segments


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello   world \n") == "hello world"


def test_dedupe_overlapping_segments_drops_reprocessed_overlap():
    raw = [
        RawSegment(start=0.0, end=4.0, text="first"),
        RawSegment(start=3.5, end=7.5, text="overlap-reprocessed"),  # covered by chunk 1 overlap
        RawSegment(start=8.0, end=12.0, text="third"),
    ]
    merged = _dedupe_overlapping_segments(raw)
    assert [s.text for s in merged] == ["first", "third"]


def test_transcript_result_full_text_and_srt():
    segments = [
        Segment(id=0, start=0.0, end=1.5, text="Hello there"),
        Segment(id=1, start=1.5, end=3.0, text="General Kenobi"),
    ]
    result = TranscriptResult(
        source_file="test.wav", duration_seconds=3.0, language="en",
        backend="MockBackend", segments=segments,
    )
    assert result.full_text == "Hello there General Kenobi"

    srt = result.to_srt()
    assert "00:00:00,000 --> 00:00:01,500" in srt
    assert "Hello there" in srt

    data = result.to_dict()
    assert data["segments"][0]["text"] == "Hello there"
    assert data["duration_seconds"] == 3.0


def test_top_keywords_ignores_stopwords_and_short_words():
    segments = [Segment(id=0, start=0.0, end=1.0, text="the cat sat on the mat and the cat ran")]
    result = TranscriptResult(
        source_file="t.wav", duration_seconds=1.0, language="en", backend="Mock", segments=segments,
    )
    keywords = dict(result.top_keywords())
    assert keywords.get("cat") == 2
    assert "the" not in keywords
    assert "on" not in keywords
