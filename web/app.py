"""Web UI for the transcription pipeline.

A thin FastAPI layer over the existing `TranscriptionPipeline` - all the
actual audio handling, transcription, and postprocessing logic lives in
`transcription_pipeline/`; this module just exposes it over HTTP and
serves the static frontend.

Run with:
    uvicorn web.app:app --reload
"""
from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transcription_pipeline.backends import FasterWhisperBackend, MockBackend, TranscriptionBackend
from transcription_pipeline.pipeline import TranscriptionPipeline

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Voice Transcription Pipeline")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")

# faster-whisper model loads are expensive (weight download + decode setup);
# cache backend instances per (model_size, device) so repeated requests in
# the same server process reuse the already-loaded model instead of paying
# that cost on every upload.
_whisper_backend_cache: dict[tuple[str, str], FasterWhisperBackend] = {}


def _get_backend(name: str, model_size: str, device: str = "cpu") -> TranscriptionBackend:
    if name == "mock":
        return MockBackend()
    if name == "whisper":
        key = (model_size, device)
        if key not in _whisper_backend_cache:
            _whisper_backend_cache[key] = FasterWhisperBackend(model_size=model_size, device=device)
        return _whisper_backend_cache[key]
    raise HTTPException(status_code=400, detail=f"Unknown backend '{name}', expected 'mock' or 'whisper'")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/transcribe")
def transcribe(
    file: UploadFile = File(...),
    backend: str = Form("mock"),
    model_size: str = Form("base"),
    language: str = Form(""),
):
    """Accept an uploaded audio file, run it through the pipeline, and
    return the structured transcript plus links to the audio and the
    generated JSON/SRT/TXT files.

    Defined as a plain `def` (not `async def`) so FastAPI runs the
    blocking transcription work in a worker thread instead of the event
    loop, keeping the server responsive to other requests while a
    transcription is in progress.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    tb = _get_backend(backend, model_size)

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True)
    audio_path = job_dir / file.filename
    with audio_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    pipeline = TranscriptionPipeline(backend=tb)
    try:
        result = pipeline.run(audio_path, output_dir=job_dir, language=language or None)
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the client
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"Transcription failed: {exc}") from exc

    data = result.to_dict()
    stem = audio_path.stem
    data["job_id"] = job_id
    data["audio_url"] = f"/files/{job_id}/{audio_path.name}"
    data["downloads"] = {
        "json": f"/files/{job_id}/{stem}.json",
        "srt": f"/files/{job_id}/{stem}.srt",
        "txt": f"/files/{job_id}/{stem}.txt",
    }
    return data
