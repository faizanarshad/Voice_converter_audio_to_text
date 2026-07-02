from transcription_pipeline.backends import MockBackend


def test_mock_backend_produces_expected_segment_count(short_wav):
    backend = MockBackend(segment_seconds=1.0)
    output = backend.transcribe(short_wav)
    # 3 second clip / 1 second segments -> 3 segments
    assert len(output.segments) == 3
    assert output.language == "en"
    assert output.segments[0].start == 0.0
    assert output.segments[-1].end <= 3.01


def test_mock_backend_respects_forced_language(short_wav):
    backend = MockBackend()
    output = backend.transcribe(short_wav, language="fr")
    assert output.language == "fr"
