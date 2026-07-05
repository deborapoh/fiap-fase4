from __future__ import annotations
import json
from pathlib import Path
OPIOIDS = {"morfina", "fentanil", "tramadol", "codeína", "codeina", "oxycodona"}

def analyze_prescriptions(prescriptions_path=None):
    root = Path(__file__).resolve().parents[2]
    path = Path(prescriptions_path) if prescriptions_path else root / "data" / "synthetic" / "prescriptions.json"
    alerts, anomalies, score = [], [], 0.0
    if not path.exists():
        return {"modality": "prescriptions", "source": str(path), "anomalies": [], "alerts": [f"Arquivo não encontrado: {path}"], "score": 0.0}
    timeline = json.loads(path.read_text(encoding="utf-8"))
    known_drugs, last_doses = set(), {}
    for entry in timeline:
        drug = entry.get("drug", "").lower()
        dose = float(entry.get("dose_mg", 0))
        day = entry.get("day", 0)
        if drug in OPIOIDS and drug not in known_drugs:
            anomalies.append({"day": day, "drug": drug, "dose_mg": dose, "type": "new_opioid", "message": f"Novo opioide: {drug}"})
            alerts.append(f"Dia {day}: novo opioide {drug}")
            score = min(1.0, score + 0.5)
        if drug in last_doses and dose >= last_doses[drug] * 2:
            anomalies.append({"day": day, "drug": drug, "dose_mg": dose, "previous_dose_mg": last_doses[drug], "type": "dose_doubled"})
            alerts.append(f"Dia {day}: dose de {drug} dobrou")
            score = min(1.0, score + 0.4)
        known_drugs.add(drug)
        last_doses[drug] = dose
    return {"modality": "prescriptions", "source": str(path), "anomalies": anomalies, "alerts": alerts, "score": round(score, 3)}
