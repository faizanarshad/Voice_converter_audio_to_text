from pathlib import Path

import pytest

from transcription_pipeline import audio_utils


def test_probe_audio_reports_duration_and_format(short_wav):
    info = audio_utils.probe_audio(short_wav)
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert 2.9 <= info.duration_seconds <= 3.1


def test_normalize_converts_stereo_high_rate_to_mono_16k(stereo_wav, tmp_path):
    info_before = audio_utils.probe_audio(stereo_wav)
    assert info_before.channels == 2
    assert info_before.sample_rate == 44100

    normalized = audio_utils.normalize_to_wav(stereo_wav, tmp_path / "normalized.wav")
    info_after = audio_utils.probe_audio(normalized)
    assert info_after.channels == 1
    assert info_after.sample_rate == 16000


def test_iter_chunks_single_chunk_when_short(short_wav, tmp_path):
    chunks = list(
        audio_utils.iter_chunks(
            short_wav, total_duration=3.0, chunk_seconds=300, overlap_seconds=5, work_dir=tmp_path
        )
    )
    assert len(chunks) == 1
    assert chunks[0].start_offset == 0.0
    assert chunks[0].path == short_wav


def test_iter_chunks_splits_long_audio_with_overlap(long_wav, tmp_path):
    chunks = list(
        audio_utils.iter_chunks(
            long_wav, total_duration=12.0, chunk_seconds=5.0, overlap_seconds=1.0, work_dir=tmp_path
        )
    )
    # step = 4s, so offsets should be 0, 4, 8 (covering up to 12s)
    offsets = [round(c.start_offset, 2) for c in chunks]
    assert offsets == [0.0, 4.0, 8.0]
    for chunk in chunks:
        assert chunk.path.exists()


def test_iter_chunks_rejects_overlap_ge_chunk_seconds(long_wav, tmp_path):
    with pytest.raises(ValueError):
        list(
            audio_utils.iter_chunks(
                long_wav, total_duration=12.0, chunk_seconds=5.0, overlap_seconds=5.0, work_dir=tmp_path
            )
        )
