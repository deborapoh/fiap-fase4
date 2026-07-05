"""Features vocais com librosa."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np

def extract_vocal_features(audio_path):
    import librosa
    audio_path = Path(audio_path)
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))
    energy_cv = energy_std / energy_mean if energy_mean > 0 else 0.0
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
    f0_mean = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    f0_std = float(np.nanstd(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    alerts, flags = [], []
    if 0 < f0_mean < 120:
        alerts.append(f"Pitch médio baixo ({f0_mean:.1f} Hz) — possível fadiga vocal")
        flags.append("fatigue_low_pitch")
    if energy_cv > 0.8:
        alerts.append(f"Energia vocal irregular (CV={energy_cv:.2f}) — possível disartria")
        flags.append("dysarthria_irregular_energy")
    if f0_std > 50 and f0_mean > 0:
        alerts.append(f"Alta variabilidade de pitch (std={f0_std:.1f} Hz)")
        flags.append("pitch_instability")
    return {"duration_sec": round(float(len(y)/sr), 2), "f0_mean_hz": round(f0_mean, 2), "f0_std_hz": round(f0_std, 2),
            "energy_mean": round(energy_mean, 6), "energy_cv": round(energy_cv, 4), "flags": flags,
            "alerts": alerts, "score": min(1.0, len(flags) * 0.35)}
