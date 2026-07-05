"""Pipeline de áudio."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
from src.utils import load_critical_terms
from .azure_stt import transcribe_audio, transcribe_offline_fallback
from .azure_text import analyze_text, analyze_text_offline_fallback
from .vocal_features import extract_vocal_features

def check_critical_terms(text, terms):
    text_lower = text.lower()
    return [t for t in terms if t in text_lower]

def analyze_audio(audio_path, use_azure=None):
    audio_path = Path(audio_path)
    terms = load_critical_terms()
    azure_available = bool(os.getenv("AZURE_SPEECH_KEY") and os.getenv("AZURE_LANGUAGE_KEY"))
    if use_azure is None:
        use_azure = azure_available
    if use_azure and os.getenv("AZURE_SPEECH_KEY"):
        try:
            transcript = transcribe_audio(audio_path)
        except Exception as e:
            transcript = transcribe_offline_fallback(audio_path)
            transcript["azure_error"] = str(e)
    else:
        transcript = transcribe_offline_fallback(audio_path)
    text = transcript.get("text", "")
    cache_text = audio_path.with_suffix(".textanalytics.json")
    if use_azure and os.getenv("AZURE_LANGUAGE_KEY") and text:
        try:
            text_analysis = analyze_text(text, cache_key=str(cache_text))
        except Exception as e:
            text_analysis = analyze_text_offline_fallback(text)
            text_analysis["azure_error"] = str(e)
    else:
        text_analysis = analyze_text_offline_fallback(text)
    vocal = extract_vocal_features(audio_path)
    critical_hits = check_critical_terms(text, terms)
    alerts = []
    if critical_hits:
        alerts.append(f"Termos críticos detectados: {', '.join(critical_hits)}")
    if text_analysis.get("sentiment") == "negative":
        alerts.append("Sentimento negativo na fala do paciente")
    alerts.extend(vocal.get("alerts", []))
    score = min(1.0, vocal.get("score", 0) * 0.4 + (0.3 if critical_hits else 0) + (0.3 if text_analysis.get("sentiment") == "negative" else 0))
    return {"modality": "audio", "source": str(audio_path), "transcript": transcript, "text_analysis": text_analysis,
            "vocal_features": vocal, "critical_terms": critical_hits, "alerts": alerts, "score": round(score, 3),
            "azure_used": use_azure and azure_available}
