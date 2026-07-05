from __future__ import annotations
from typing import Any
WEIGHTS = {"video": 0.30, "audio": 0.30, "vitals": 0.20, "prescriptions": 0.10, "clinical_notes": 0.05, "movement": 0.05}

def compute_risk_score(modality_results):
    weighted_sum, weight_used, evidence = 0.0, 0.0, []
    for modality, weight in WEIGHTS.items():
        result = modality_results.get(modality)
        if not result:
            continue
        score = float(result.get("score", 0))
        weighted_sum += score * weight
        weight_used += weight
        if result.get("alerts"):
            evidence.append({"modality": modality, "score": score, "alerts": result["alerts"][:5]})
    final_score = weighted_sum / weight_used if weight_used > 0 else 0.0
    if final_score >= 0.7:
        severity, message = "critical", "Alerta crítico: múltiplos indicadores de risco. Acionar equipe médica."
    elif final_score >= 0.4:
        severity, message = "warning", "Alerta de atenção: revisar paciente e dados clínicos."
    else:
        severity, message = "info", "Monitoramento dentro do esperado."
    return {"risk_score": round(final_score, 3), "severity": severity, "message": message, "evidence": evidence, "weights": WEIGHTS}
