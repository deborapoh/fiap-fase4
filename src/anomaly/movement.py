from __future__ import annotations
from pathlib import Path
import numpy as np
from src.video.openpose_loader import joint_velocity, load_openpose
from src.video.analyzer import resolve_openpose_for_video

def analyze_movement(openpose_path=None, video_path=None, velocity_threshold=None):
    root = Path(__file__).resolve().parents[2]
    op_dir = root / "data" / "keraal" / "openpose"
    if openpose_path:
        op_file = Path(openpose_path)
    elif video_path:
        op_file = resolve_openpose_for_video(Path(video_path), op_dir)
    else:
        op_file = op_dir / "G2A-OP-RTK-S1-Roscoff-005.json"
    alerts, anomalies, score = [], [], 0.0
    if not op_file or not Path(op_file).exists():
        return {"modality": "movement", "source": str(op_file), "anomalies": [], "alerts": ["OpenPose não encontrado"], "score": 0.0}
    positions = load_openpose(op_file)
    all_velocities = []
    for joint in ("rWrist", "lWrist", "rAnkle", "lAnkle"):
        for v in joint_velocity(positions, joint=joint):
            all_velocities.append(v["velocity"])
    if not all_velocities:
        return {"modality": "movement", "source": str(op_file), "anomalies": [], "alerts": ["Sem dados de velocidade"], "score": 0.0}
    mean_v, std_v = float(np.mean(all_velocities)), float(np.std(all_velocities))
    threshold = velocity_threshold if velocity_threshold is not None else mean_v + 2.5 * std_v
    for joint in ("rWrist", "lWrist", "rAnkle", "lAnkle"):
        for v in joint_velocity(positions, joint=joint):
            if v["velocity"] > threshold:
                anomalies.append({**v, "threshold": round(threshold, 6), "type": "high_joint_velocity"})
    if anomalies:
        alerts.append(f"{len(anomalies)} frames com velocidade acima do baseline (OpenPose)")
        score = min(1.0, len(anomalies) / 50)
    return {"modality": "movement", "source": str(op_file), "baseline_mean_velocity": round(mean_v, 6),
            "baseline_std_velocity": round(std_v, 6), "threshold": round(threshold, 6),
            "anomalies": anomalies[:30], "total_anomalies": len(anomalies), "alerts": alerts, "score": round(score, 3)}
