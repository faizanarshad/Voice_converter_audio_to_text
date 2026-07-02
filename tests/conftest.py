import math
import struct
import wave
from pathlib import Path

import pytest


def make_wav(path: Path, duration_seconds: float, sample_rate: int = 16000, channels: int = 1) -> Path:
    """Write a synthetic sine-wave WAV file, used as test audio (mock data)."""
    n_frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_frames):
            value = int(3000 * math.sin(2 * math.pi * 440 * (i / sample_rate)))
            sample = struct.pack("<h", value)
            frames.extend(sample * channels)
        wav_file.writeframes(bytes(frames))
    return path


@pytest.fixture
def short_wav(tmp_path) -> Path:
    return make_wav(tmp_path / "short.wav", duration_seconds=3.0)


@pytest.fixture
def long_wav(tmp_path) -> Path:
    # Long enough to exceed a small chunk_seconds threshold used in tests.
    return make_wav(tmp_path / "long.wav", duration_seconds=12.0)


@pytest.fixture
def stereo_wav(tmp_path) -> Path:
    return make_wav(tmp_path / "stereo.wav", duration_seconds=2.0, sample_rate=44100, channels=2)
