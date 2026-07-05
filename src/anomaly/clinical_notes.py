from __future__ import annotations
from pathlib import Path
from src.utils import load_critical_terms

def analyze_clinical_notes(notes_path=None):
    root = Path(__file__).resolve().parents[2]
    path = Path(notes_path) if notes_path else root / "data" / "synthetic" / "clinical_notes.txt"
    terms = load_critical_terms()
    alerts, anomalies, score = [], [], 0.0
    if not path.exists():
        return {"modality": "clinical_notes", "source": str(path), "anomalies": [], "alerts": [f"Arquivo não encontrado: {path}"], "score": 0.0}
    previous_hits = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            day, text = line.split(":", 1)
            day = day.strip()
        else:
            day, text = "?", line
        text_lower = text.lower()
        hits = [t for t in terms if t in text_lower]
        new_hits = [h for h in hits if h not in previous_hits]
        if new_hits:
            anomalies.append({"day": day, "new_critical_terms": new_hits, "text_excerpt": text.strip()[:120], "type": "new_critical_term"})
            alerts.append(f"{day}: novos termos críticos — {', '.join(new_hits)}")
            score = min(1.0, score + 0.3)
        if any(w in text_lower for w in ("piora súbita", "piora subita", "agudizou", "piorou muito")):
            anomalies.append({"day": day, "text_excerpt": text.strip()[:120], "type": "sudden_deterioration"})
            alerts.append(f"{day}: piora súbita registrada")
            score = min(1.0, score + 0.5)
        previous_hits.update(hits)
    return {"modality": "clinical_notes", "source": str(path), "anomalies": anomalies, "alerts": alerts, "score": round(score, 3)}
