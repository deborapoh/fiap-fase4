"""Parser de keypoints OpenPose do dataset KERAAL (formato JSON)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_openpose(path: str | Path) -> dict[str, dict[str, list[float]]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("positions", data)


def _angle(a: list[float], b: list[float], c: list[float]) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def compute_joint_angles(frame: dict[str, list[float]]) -> dict[str, float]:
    angles: dict[str, float] = {}

    def get(joint: str) -> list[float] | None:
        return frame.get(joint)

    if all(get(j) for j in ("rShoulder", "rElbow", "rWrist")):
        angles["right_elbow"] = _angle(get("rShoulder"), get("rElbow"), get("rWrist"))
    if all(get(j) for j in ("lShoulder", "lElbow", "lWrist")):
        angles["left_elbow"] = _angle(get("lShoulder"), get("lElbow"), get("lWrist"))
    if all(get(j) for j in ("rHip", "rKnee", "rAnkle")):
        angles["right_knee"] = _angle(get("rHip"), get("rKnee"), get("rAnkle"))
    if all(get(j) for j in ("lHip", "lKnee", "lAnkle")):
        angles["left_knee"] = _angle(get("lHip"), get("lKnee"), get("lAnkle"))
    if all(get(j) for j in ("mShoulder", "rShoulder", "rElbow")):
        angles["right_shoulder"] = _angle(get("mShoulder"), get("rShoulder"), get("rElbow"))
    if all(get(j) for j in ("mShoulder", "lShoulder", "lElbow")):
        angles["left_shoulder"] = _angle(get("mShoulder"), get("lShoulder"), get("lElbow"))
    return angles


NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "right_elbow": (30.0, 170.0),
    "left_elbow": (30.0, 170.0),
    "right_knee": (20.0, 175.0),
    "left_knee": (20.0, 175.0),
    "right_shoulder": (20.0, 160.0),
    "left_shoulder": (20.0, 160.0),
}


def detect_angle_deviations(
    positions: dict[str, dict[str, list[float]]],
    normal_ranges: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    ranges = normal_ranges or NORMAL_RANGES
    deviations: list[dict[str, Any]] = []
    for frame_key, joints in positions.items():
        angles = compute_joint_angles(joints)
        for joint_name, angle in angles.items():
            if joint_name not in ranges:
                continue
            low, high = ranges[joint_name]
            if angle < low or angle > high:
                deviations.append({
                    "frame": frame_key,
                    "joint": joint_name,
                    "angle_deg": round(angle, 2),
                    "expected_range": [low, high],
                    "type": "postural_deviation",
                })
    return deviations


def joint_velocity(positions: dict[str, dict[str, list[float]]], joint: str = "rWrist") -> list[dict[str, Any]]:
    frames = sorted(positions.keys(), key=lambda x: float(x))
    velocities: list[dict[str, Any]] = []
    for i in range(1, len(frames)):
        prev_frame = positions[frames[i - 1]]
        curr_frame = positions[frames[i]]
        if joint not in prev_frame or joint not in curr_frame:
            continue
        p0, p1 = prev_frame[joint], curr_frame[joint]
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        dt = float(frames[i]) - float(frames[i - 1])
        if dt <= 0:
            continue
        velocities.append({"frame": frames[i], "joint": joint, "velocity": round(dist / dt, 6)})
    return velocities
