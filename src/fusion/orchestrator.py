from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv
from src.anomaly.clinical_notes import analyze_clinical_notes
from src.anomaly.movement import analyze_movement
from src.anomaly.prescriptions import analyze_prescriptions
from src.anomaly.vitals import analyze_vitals
from src.audio.analyzer import analyze_audio
from src.fusion.risk_scorer import compute_risk_score
from src.utils import project_root
from src.video.analyzer import analyze_video

def run_full_analysis(video_path=None, audio_paths=None, use_azure=None, run_yolo=True):
    root = project_root()
    load_dotenv(root / ".env")
    video_path = Path(video_path) if video_path else root / "data" / "keraal" / "videos" / "G2A-Anon-RTK-S1-Roscoff-005.mp4"
    if audio_paths is None:
        audio_paths = sorted((root / "data" / "audio").glob("consulta_*.wav"))
    video_result = analyze_video(video_path, run_yolo=run_yolo)
    audio_results = [analyze_audio(p, use_azure=use_azure) for p in audio_paths]
    audio_agg = max(audio_results, key=lambda x: x.get("score", 0)) if audio_results else {"modality": "audio", "score": 0, "alerts": ["Nenhum áudio"]}
    if audio_results:
        audio_agg["all_consultations"] = len(audio_results)
    modality_results = {
        "video": video_result, "audio": audio_agg, "vitals": analyze_vitals(),
        "prescriptions": analyze_prescriptions(), "clinical_notes": analyze_clinical_notes(),
        "movement": analyze_movement(video_path=video_path),
    }
    risk = compute_risk_score(modality_results)
    return {"alert": {"severity": risk["severity"], "message": risk["message"], "risk_score": risk["risk_score"], "evidence": risk["evidence"]},
            "modalities": modality_results, "inputs": {"video": str(video_path), "audios": [str(p) for p in audio_paths]}}
