"""Azure Speech-to-Text."""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any

def transcribe_audio(audio_path, use_cache=True):
    import azure.cognitiveservices.speech as speechsdk
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION", "brazilsouth")
    if not key:
        raise ValueError("AZURE_SPEECH_KEY não configurada")
    audio_path = Path(audio_path)
    cache_path = audio_path.with_suffix(".transcript.json")
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_recognition_language = "pt-BR"
    audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        payload = {"text": result.text, "status": "success", "source": str(audio_path)}
    elif result.reason == speechsdk.ResultReason.NoMatch:
        payload = {"text": "", "status": "no_match", "source": str(audio_path)}
    else:
        payload = {"text": "", "status": "error", "error": str(result.reason), "source": str(audio_path)}
    if use_cache and payload.get("text"):
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

def transcribe_offline_fallback(audio_path):
    audio_path = Path(audio_path)
    cache_path = audio_path.with_suffix(".transcript.json")
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    placeholders = {
        "consulta_01_cansaco": "Doutor, estou muito cansada, não consigo fazer as tarefas do dia.",
        "consulta_02_falta_ar": "Estou com falta de ar, principalmente quando subo escada.",
        "consulta_03_dor_peito": "Sinto uma dor no peito que vem e vai, estou preocupada.",
    }
    for key, text in placeholders.items():
        if key in audio_path.stem.lower():
            return {"text": text, "status": "offline_placeholder", "source": str(audio_path)}
    return {"text": "", "status": "offline_no_cache", "source": str(audio_path)}
