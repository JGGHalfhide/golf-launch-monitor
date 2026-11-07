"""Debug overlay drawing for the vision pipeline."""

import cv2
import numpy as np


def draw_overlay(
    frame,
    detections: list,
    shot_result: dict | None = None,
) -> np.ndarray:
    """
    Draw: detected ball circles (green), trajectory line (yellow),
    text overlay of speed + VLA (top-left corner).
    detections: list of (cx, cy, radius) or None per frame.
    """
    out = frame.copy()

    # Draw each detection circle
    valid_pts = []
    for det in detections:
        if det is None:
            continue
        cx, cy, r = int(det[0]), int(det[1]), int(det[2])
        cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)       # green circle
        cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1)      # centre dot
        valid_pts.append((cx, cy))

    # Trajectory line through all valid detections
    if len(valid_pts) >= 2:
        for i in range(len(valid_pts) - 1):
            cv2.line(out, valid_pts[i], valid_pts[i + 1], (0, 255, 255), 2)  # yellow

    # Speed + VLA text
    if shot_result:
        speed = shot_result.get("ball_speed_mph", 0.0)
        vla = shot_result.get("vertical_launch_angle_deg", 0.0)
        lines = [
            f"Speed: {speed:.1f} mph",
            f"VLA:   {vla:.1f} deg",
        ]
        warnings = shot_result.get("warnings", [])
        for w in warnings:
            lines.append(f"WARN: {w}")

        y = 30
        for line in lines:
            # Dark shadow for readability
            cv2.putText(out, line, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            y += 28

    return out
