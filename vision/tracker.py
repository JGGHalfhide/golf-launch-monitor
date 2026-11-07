"""Impact frame detection and multi-frame ball tracking."""

import numpy as np
import cv2

from .detector import find_ball, find_ball_by_diff
from .kinematics import positions_to_velocity, velocity_to_shot


BALL_DIAMETER_MM = 42.67


def find_impact_frame(frames: list[np.ndarray], start: int = 0) -> int:
    """Return index of impact frame using frame-differencing."""
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

    values = [v for _, v in diffs]
    threshold = float(np.mean(values)) + 2.0 * float(np.std(values))

    for idx, val in diffs:
        if val >= threshold:
            return idx

    return max(diffs, key=lambda x: x[1])[0]


def _find_ball_post_impact(
    frame,
    background,
    expected_r: int,
    min_x: int,
    max_y: int,
) -> tuple[int, int, int] | None:
    """
    Find ball in a post-impact frame by background subtraction.
    Only accepts candidates that are:
      - To the RIGHT of min_x  (ball moves toward target)
      - ABOVE max_y            (ball launches upward)
    Among candidates, pick the brightest (whitest) one.
    """
    diff = cv2.absdiff(frame, background)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    min_area = np.pi * (expected_r * 0.3) ** 2
    max_area = np.pi * (expected_r * 5.0) ** 2

    best = None
    best_brightness = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Directional constraint: ball must be right of min_x and above max_y
        if cx < min_x or cy > max_y:
            continue

        # Brightness of this blob in the original frame (ball is white)
        mask = np.zeros(gray_frame.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        brightness = float(cv2.mean(gray_frame, mask=mask)[0])

        if brightness > best_brightness:
            best_brightness = brightness
            radius = max(1, int(np.sqrt(area / np.pi)))
            best = (cx, cy, radius)

    return best


def track_ball(
    frames: list[np.ndarray],
    impact_idx: int,
    px_per_mm: float,
    max_frames: int = 8,
    timestamps_ms: list[float] | None = None,
    address_pos: tuple[int, int] | None = None,
) -> dict:
    """
    Track from impact_idx.
    Uses background subtraction with directional constraints post-impact.

    Returns dict with positions, ball_speed_mph, vertical_launch_angle_deg, etc.
    """
    warnings: list[str] = []
    positions: list[tuple[int, int, int]] = []  # (frame_idx, cx, cy)

    background = frames[max(0, impact_idx - 1)]
    expected_r = max(10, int((BALL_DIAMETER_MM / 2) * px_per_mm))

    # Determine ball address position (used as direction anchor; NOT included in velocity regression)
    if address_pos is not None:
        addr_x, addr_y = address_pos
    else:
        ball_at_address = find_ball(
            frames[impact_idx],
            min_radius_px=max(5, expected_r // 2),
            max_radius_px=expected_r * 2,
        )
        if ball_at_address is not None:
            addr_x, addr_y = ball_at_address[0], ball_at_address[1]
        else:
            h, w = frames[impact_idx].shape[:2]
            addr_x, addr_y = w // 2, int(h * 0.6)
            warnings.append("Ball not detected at address; using frame center for direction constraint")

    min_x_post = addr_x - expected_r * 2
    max_y_post = addr_y + expected_r

    end_idx = min(impact_idx + max_frames + 1, len(frames))

    # Only track post-impact frames for velocity (ball is stationary at impact)
    for fi in range(impact_idx + 1, end_idx):
        frame = frames[fi]

        # Update directional bounds based on last known position
        if positions:
            last_fi, last_cx, last_cy = positions[-1]
            # Ball must keep moving right; allow significant upward range
            search_min_x = last_cx - expected_r     # small backward tolerance
            search_max_y = last_cy + expected_r * 2  # allow some downward (rolling/low shots)
        else:
            search_min_x = min_x_post
            search_max_y = max_y_post

        det = _find_ball_post_impact(
            frame, background,
            expected_r=expected_r,
            min_x=search_min_x,
            max_y=search_max_y,
        )

        if det is not None:
            positions.append((fi, det[0], det[1]))
        else:
            break  # lost the ball

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
        (cx / px_per_mm, -(cy / px_per_mm))
        for _, cx, cy in positions
    ]

    times_s: list[float] | None = None
    if timestamps_ms is not None:
        times_s = [timestamps_ms[fi] / 1000.0 for fi, _, _ in positions]

    vx, vy = positions_to_velocity(positions_mm, times_s=times_s)
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
