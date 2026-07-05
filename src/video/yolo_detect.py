"""Detecção de objetos com YOLOv8 em frames de vídeo."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import cv2

def detect_objects_in_video(video_path, sample_every_n=30, model_name="yolov8n.pt", conf_threshold=0.4):
    from ultralytics import YOLO
    video_path = Path(video_path)
    if not video_path.exists():
        return [{"error": f"Vídeo não encontrado: {video_path}"}]
    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [{"error": f"Não foi possível abrir: {video_path}"}]
    detections = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every_n == 0:
            results = model(frame, verbose=False, conf=conf_threshold)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    detections.append({
                        "frame": frame_idx,
                        "label": model.names[cls_id],
                        "confidence": round(float(box.conf[0]), 3),
                        "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
                    })
        frame_idx += 1
    cap.release()
    return detections

def summarize_detections(detections):
    if not detections:
        return {"total": 0, "by_label": {}, "alerts": []}
    by_label = {}
    for d in detections:
        if "label" not in d:
            continue
        by_label[d["label"]] = by_label.get(d["label"], 0) + 1
    alerts = []
    if by_label.get("person", 0) == 0:
        alerts.append("Nenhuma pessoa detectada no vídeo amostrado")
    if by_label.get("person", 0) > 1:
        alerts.append("Múltiplas pessoas detectadas")
    return {"total": len(detections), "by_label": by_label, "alerts": alerts}
