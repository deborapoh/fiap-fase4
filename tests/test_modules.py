from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_openpose_loader():
    from src.video.openpose_loader import load_openpose, compute_joint_angles
    op_file = ROOT / "data" / "keraal" / "openpose" / "G2A-OP-RTK-S1-Roscoff-005.json"
    if not op_file.exists():
        return
    positions = load_openpose(op_file)
    first_frame = positions[sorted(positions.keys(), key=float)[0]]
    angles = compute_joint_angles(first_frame)
    assert "right_elbow" in angles or "left_elbow" in angles


def test_vocal_features():
    from src.audio.vocal_features import extract_vocal_features
    audio = ROOT / "data" / "audio" / "consulta_01_cansaco.wav"
    if not audio.exists():
        return
    result = extract_vocal_features(audio)
    assert "f0_mean_hz" in result
    assert "score" in result


def test_prescriptions_rules():
    from src.anomaly.prescriptions import analyze_prescriptions
    result = analyze_prescriptions()
    assert result["score"] > 0
    assert len(result["anomalies"]) >= 2


def test_risk_scorer():
    from src.fusion.risk_scorer import compute_risk_score
    result = compute_risk_score({
        "video": {"score": 0.5, "alerts": ["test"]},
        "audio": {"score": 0.8, "alerts": ["test"]},
    })
    assert 0 <= result["risk_score"] <= 1
    assert result["severity"] in ("info", "warning", "critical")
