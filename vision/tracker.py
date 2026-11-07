"""Impact frame detection and multi-frame ball tracking."""

import numpy as np
import cv2

from .detector import find_ball
from .kinematics import positions_to_velocity, velocity_to_shot


# Search radius for large inter-frame displacements (~279 px/frame at 150 mph)
_SEARCH_RADIUS = 300


def find_impact_frame(frames: list[np.ndarray], start: int = 0) -> int:
    """Return index of impact frame using frame-differencing. Looks for first large motion burst."""
    if len(frames) < 2:
        return start

    diffs = []
    for i in range(start, len(frames) - 1):
        d = cv2.absdiff(
            cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY),
        )
        diffs.append((i, float(d.mean())))

    if not diffs:
        return start

    # Threshold: mean + 2 std of all diff values
    values = [v for _, v in diffs]
    threshold = float(np.mean(values)) + 2.0 * float(np.std(values))

    for idx, val in diffs:
        if val >= threshold:
            return idx

    # Fallback: frame with the largest diff
    return max(diffs, key=lambda x: x[1])[0]


def track_ball(
    frames: list[np.ndarray],
    impact_idx: int,
    px_per_mm: float,
    max_frames: int = 8,
) -> dict:
    """
    Track from impact_idx, return:
    {
      "positions": [(frame_idx, cx_px, cy_px), ...],
      "ball_speed_mph": float,
      "vertical_launch_angle_deg": float,
      "px_per_mm": float,
      "impact_frame": int,
      "warnings": [str, ...]
    }
    """
    warnings: list[str] = []
    positions: list[tuple[int, int, int]] = []  # (frame_idx, cx, cy)

    last_cx: int | None = None
    last_cy: int | None = None

    end_idx = min(impact_idx + max_frames + 1, len(frames))

    for fi in range(impact_idx, end_idx):
        frame = frames[fi]
        h, w = frame.shape[:2]

        if last_cx is not None:
            # Crop a search window around last known position
            x0 = max(0, last_cx - _SEARCH_RADIUS)
            y0 = max(0, last_cy - _SEARCH_RADIUS)
            x1 = min(w, last_cx + _SEARCH_RADIUS)
            y1 = min(h, last_cy + _SEARCH_RADIUS)
            roi = frame[y0:y1, x0:x1]
            det = find_ball(roi)
            if det is not None:
                cx = det[0] + x0
                cy = det[1] + y0
                positions.append((fi, cx, cy))
                last_cx, last_cy = cx, cy
                continue

        # Full-frame detection
        det = find_ball(frame)
        if det is not None:
            last_cx, last_cy = det[0], det[1]
            positions.append((fi, det[0], det[1]))
        else:
            if fi == impact_idx:
                warnings.append(f"Ball not found at impact frame {fi}")
            else:
                # Stop tracking on first miss after impact
                break

    if len(positions) < 2:
        warnings.append("Fewer than 2 ball detections — cannot compute velocity")
        return {
            "positions": positions,
            "ball_speed_mph": 0.0,
            "vertical_launch_angle_deg": 0.0,
            "px_per_mm": px_per_mm,
            "impact_frame": impact_idx,
            "warnings": warnings,
        }

    # Convert pixel positions to mm; flip y (image y-down → physics y-up)
    positions_mm = [
        (cx / px_per_mm, -(cy / px_per_mm))  # negate image y → positive = upward
        for _, cx, cy in positions
    ]

    fps = 240.0
    vx, vy = positions_to_velocity(positions_mm, fps)
    speed_mph, vla_deg = velocity_to_shot(vx, vy)

    if len(positions) < 3:
        warnings.append("Only 2 detections — velocity estimate may be noisy")

    return {
        "positions": positions,
        "ball_speed_mph": round(speed_mph, 2),
        "vertical_launch_angle_deg": round(vla_deg, 2),
        "px_per_mm": px_per_mm,
        "impact_frame": impact_idx,
        "warnings": warnings,
    }
