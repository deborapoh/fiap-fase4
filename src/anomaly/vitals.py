from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def _load_shiraz_bpm(shiraz_excel):
    if not shiraz_excel.exists():
        return None
    df = pd.read_excel(shiraz_excel)
    for col in ("BPM", "bpm", "FHR", "fhr", "Heart Rate"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").dropna()
    numeric = df.select_dtypes(include=[np.number])
    return numeric.iloc[:, 0].dropna() if not numeric.empty else None

def analyze_vitals(vitals_csv=None, shiraz_excel=None, contamination=0.1):
    root = Path(__file__).resolve().parents[2]
    vitals_path = Path(vitals_csv) if vitals_csv else root / "data" / "synthetic" / "vitals.csv"
    shiraz_path = Path(shiraz_excel) if shiraz_excel else root / "data" / "shiraz-university-fetal-heart-sounds-database-1.0.1" / "FetalPCGSpreadsheet.xlsx"
    alerts, anomalies, score = [], [], 0.0
    if vitals_path.exists():
        df = pd.read_csv(vitals_path)
        feature_cols = [c for c in ("hr", "spo2", "systolic_bp", "diastolic_bp") if c in df.columns]
        if feature_cols:
            X = df[feature_cols].values
            preds = IsolationForest(contamination=contamination, random_state=42).fit_predict(X)
            for idx, pred in enumerate(preds):
                if pred == -1:
                    row = df.iloc[idx]
                    anomalies.append({"timestamp": str(row.get("timestamp", idx)), "values": {c: float(row[c]) for c in feature_cols}, "type": "vitals_isolation_forest"})
            if anomalies:
                alerts.append(f"{len(anomalies)} leituras vitais anômalas detectadas")
                score = min(1.0, len(anomalies) / max(len(df), 1) * 3)
    else:
        alerts.append(f"Arquivo de vitais não encontrado: {vitals_path}")
    bpm_series = _load_shiraz_bpm(shiraz_path)
    fetal_anomalies = []
    if bpm_series is not None and len(bpm_series) > 5:
        preds_bpm = IsolationForest(contamination=contamination, random_state=42).fit_predict(bpm_series.values.reshape(-1, 1))
        for idx, pred in enumerate(preds_bpm):
            if pred == -1:
                fetal_anomalies.append({"index": int(idx), "bpm": float(bpm_series.iloc[idx])})
        if fetal_anomalies:
            alerts.append(f"{len(fetal_anomalies)} leituras de BPM fetal anômalas (Shiraz)")
            score = min(1.0, score + len(fetal_anomalies) / len(bpm_series))
    return {"modality": "vitals", "source": str(vitals_path), "shiraz_source": str(shiraz_path),
            "anomalies": anomalies[:20], "fetal_bpm_anomalies": fetal_anomalies[:20],
            "total_vital_anomalies": len(anomalies), "total_fetal_anomalies": len(fetal_anomalies),
            "alerts": alerts, "score": round(score, 3)}
