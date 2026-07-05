"""Análise de vídeo: OpenPose KERAAL + YOLOv8."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .openpose_loader import detect_angle_deviations, load_openpose
from .yolo_detect import detect_objects_in_video, summarize_detections

def resolve_openpose_for_video(video_path, openpose_dir):
    stem = video_path.stem.replace("G2A-Anon-", "G2A-OP-")
    candidate = openpose_dir / f"{stem}.json"
    if candidate.exists():
        return candidate
    suffix = stem.split("-", 2)[-1] if "-" in stem else stem
    matches = list(openpose_dir.glob(f"*{suffix}*.json"))
    return matches[0] if matches else None

def analyze_video(video_path, openpose_path=None, openpose_dir=None, run_yolo=True, yolo_sample_every=30):
    video_path = Path(video_path)
    project_root = Path(__file__).resolve().parents[2]
    op_dir = Path(openpose_dir) if openpose_dir else project_root / "data" / "keraal" / "openpose"
    op_file = Path(openpose_path) if openpose_path else resolve_openpose_for_video(video_path, op_dir)
    report = {
        "video": str(video_path), "openpose_file": str(op_file) if op_file else None,
        "modality": "video", "deviations": [], "yolo": {}, "alerts": [], "score": 0.0,
    }
    if op_file and op_file.exists():
        positions = load_openpose(op_file)
        deviations = detect_angle_deviations(positions)
        report["deviations"] = deviations[:50]
        report["total_deviation_frames"] = len(deviations)
        if deviations:
            report["alerts"].append(f"{len(deviations)} frames com ângulos fora do padrão (OpenPose)")
    else:
        report["alerts"].append("Arquivo OpenPose não encontrado")
    if run_yolo and video_path.exists():
        detections = detect_objects_in_video(video_path, sample_every_n=yolo_sample_every)
        summary = summarize_detections(detections)
        report["yolo"] = {"summary": summary, "sample_detections": detections[:20]}
        report["alerts"].extend(summary.get("alerts", []))
    n_dev = report.get("total_deviation_frames", 0)
    report["score"] = min(1.0, n_dev / 100.0) if n_dev else (0.3 if report["alerts"] else 0.0)
    return report
