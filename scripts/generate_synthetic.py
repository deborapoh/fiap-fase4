"""Gera dados sintéticos para vitais, prescrições e notas clínicas."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def generate_vitals(n_rows: int = 200, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    hr = rng.normal(75, 8, n_rows)
    spo2 = rng.normal(97, 1.5, n_rows)
    sys_bp = rng.normal(120, 12, n_rows)
    dia_bp = rng.normal(80, 8, n_rows)

    anomaly_idx = rng.choice(n_rows, size=int(n_rows * 0.1), replace=False)
    hr[anomaly_idx] = rng.uniform(130, 160, len(anomaly_idx))
    spo2[anomaly_idx[: len(anomaly_idx) // 2]] = rng.uniform(85, 90, len(anomaly_idx) // 2)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="15min"),
            "hr": hr.round(1),
            "spo2": spo2.round(1),
            "systolic_bp": sys_bp.round(0),
            "diastolic_bp": dia_bp.round(0),
        }
    )

    path = out_dir / "vitals.csv"
    df.to_csv(path, index=False)
    return path


def generate_prescriptions() -> Path:
    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline = [
        {"day": 1, "drug": "dipirona", "dose_mg": 500},
        {"day": 2, "drug": "dipirona", "dose_mg": 500},
        {"day": 3, "drug": "dipirona", "dose_mg": 1000},
        {"day": 4, "drug": "dipirona", "dose_mg": 1000},
        {"day": 5, "drug": "morfina", "dose_mg": 5},
        {"day": 6, "drug": "morfina", "dose_mg": 10},
    ]

    path = out_dir / "prescriptions.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)
    return path


def generate_clinical_notes() -> Path:
    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    notes = """# Evolução clínica sintética
Dia 1: Paciente estável, sem queixas.
Dia 2: Leve desconforto, sem sinais de alarme.
Dia 3: Relata cansaço ao caminhar.
Dia 4: Apresentou falta de ar ao subir escada.
Dia 5: Piora súbita — dor no peito e tontura. Acionar equipe.
"""

    path = out_dir / "clinical_notes.txt"
    path.write_text(notes, encoding="utf-8")
    return path


def main():
    v = generate_vitals()
    p = generate_prescriptions()
    n = generate_clinical_notes()
    print(f"Gerado: {v}")
    print(f"Gerado: {p}")
    print(f"Gerado: {n}")


if __name__ == "__main__":
    main()
