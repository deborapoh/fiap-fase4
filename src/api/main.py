"""FastAPI — endpoint de análise multimodal."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.fusion.orchestrator import run_full_analysis
from src.utils import project_root

load_dotenv(project_root() / ".env")

app = FastAPI(
    title="FIAP Fase 4 — Monitoramento Multimodal",
    description="Análise de vídeo (OpenPose+YOLOv8), áudio (Azure+librosa), vitais e texto.",
    version="1.0.0",
)


class AnalyzeResponse(BaseModel):
    severity: str
    message: str
    risk_score: float
    evidence: list[dict[str, Any]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    video: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    use_sample_data: bool = True,
    run_yolo: bool = True,
):
    video_path = None
    audio_paths: list[Path] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        if video and video.filename:
            video_path = tmp_path / video.filename
            video_path.write_bytes(await video.read())
        elif use_sample_data:
            root = project_root()
            video_path = root / "data" / "keraal" / "videos" / "G2A-Anon-RTK-S1-Roscoff-005.mp4"

        if audio and audio.filename:
            ap = tmp_path / audio.filename
            ap.write_bytes(await audio.read())
            audio_paths.append(ap)
        elif use_sample_data:
            audio_paths = sorted((project_root() / "data" / "audio").glob("consulta_*.wav"))

        result = run_full_analysis(
            video_path=video_path,
            audio_paths=audio_paths or None,
            run_yolo=run_yolo,
        )

    return JSONResponse(content=result)


@app.get("/analyze/demo")
def analyze_demo(run_yolo: bool = False):
    result = run_full_analysis(run_yolo=run_yolo)
    return result
