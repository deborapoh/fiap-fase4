"""Azure Text Analytics."""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any

def analyze_text(text, use_cache=True, cache_key=None):
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
    if not text.strip():
        return {"sentiment": "neutral", "confidence_scores": {}, "key_phrases": [], "status": "empty"}
    cache_path = Path(cache_key) if cache_key else None
    if cache_path and use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    key = os.getenv("AZURE_LANGUAGE_KEY")
    endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
    if not key or not endpoint:
        raise ValueError("Azure Language não configurado")
    client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    doc = client.analyze_sentiment([text], language="pt")[0]
    if doc.is_error:
        return {"status": "error", "error": doc.error.message}
    kp = client.extract_key_phrases([text], language="pt")[0]
    key_phrases = list(kp.key_phrases) if not kp.is_error else []
    result = {
        "sentiment": doc.sentiment,
        "confidence_scores": {"positive": doc.confidence_scores.positive, "neutral": doc.confidence_scores.neutral, "negative": doc.confidence_scores.negative},
        "key_phrases": key_phrases, "status": "success",
    }
    if cache_path:
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

def analyze_text_offline_fallback(text):
    text_lower = text.lower()
    negative_terms = ["dor", "cansad", "falta de ar", "preocupad", "pior"]
    score = sum(1 for t in negative_terms if t in text_lower)
    sentiment = "negative" if score >= 2 else ("neutral" if score == 1 else "positive")
    return {"sentiment": sentiment, "confidence_scores": {"positive": 0.1, "neutral": 0.3, "negative": 0.6},
            "key_phrases": [t for t in negative_terms if t in text_lower], "status": "offline_heuristic"}
