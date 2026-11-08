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
    predicted: tuple[float, float] | None = None,
    gate: float | None = None,
    address: tuple[float, float] | None = None,
    min_roundness: float = 0.4,
) -> tuple[int, int, int] | None:
    """
    Find the ball in a post-impact frame.

    Candidates are regions that changed from the pre-impact background, are roughly ball-sized,
    lie right of min_x (toward target) and above max_y (launching up), are round (area /
    enclosing-circle area ≥ min_roundness: ball ≈ 0.55–0.8, shaft ≈ 0.03–0.1), and are not the
    empty address spot (which also "changed": mat where the ball was). Among them the ball is
    the whitest and roundest. If a predicted position is given, only candidates within `gate`
    px of it are considered.
    """
    diff = cv2.absdiff(frame, background)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    min_area = np.pi * (expected_r * 0.4) ** 2
    max_area = np.pi * (expected_r * 3.0) ** 2

    best = None
    best_score = -1e9
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        if cx < min_x or cy > max_y:
            continue
        if predicted is not None and gate is not None and np.hypot(cx - predicted[0], cy - predicted[1]) > gate:
            continue
        if address is not None and np.hypot(cx - address[0], cy - address[1]) < 1.5 * expected_r:
            continue  # the hole the ball left behind
        (_, _), enc_r = cv2.minEnclosingCircle(cnt)
        roundness = area / (np.pi * enc_r * enc_r) if enc_r > 0 else 0.0
        if roundness < min_roundness:
            continue

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        h_mean, s_mean, v_mean, _ = cv2.mean(hsv, mask=mask)
        if v_mean < 120:
            continue  # the ball is bright; dark changes are shadows/shoes
        score = v_mean - 1.5 * s_mean + 100 * roundness
        if score > best_score:
            best_score = score
            # Motion blur stretches the ball only along its path: the ellipse's short axis is
            # the true diameter (less ~1 px of mask dilation).
            minor_r = (min(cv2.fitEllipse(cnt)[1]) / 2 - 1) if len(cnt) >= 5 else np.sqrt(area / np.pi)
            best = (int(round(cx)), int(round(cy)), float(minor_r))
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
    flight_radii: list[float] = []              # ball radius (px) measured in each detection

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
    max_misses = 2          # once tracking: ball can be hidden behind the shaft for a frame or two
    max_initial_search = 5  # frames after impact to find the ball's first clean appearance

    # Only track post-impact frames for velocity (ball is stationary at impact)
    misses = 0
    for fi in range(impact_idx + 1, end_idx):
        frame = frames[fi]
        predicted = gate = None
        if len(positions) >= 2:
            (f0, x0, y0), (f1, x1, y1) = positions[-2], positions[-1]
            n = (fi - f1) / (f1 - f0)
            predicted = (x1 + (x1 - x0) * n, y1 + (y1 - y0) * n)
            step = np.hypot(x1 - x0, y1 - y0) / (f1 - f0)
            gate = max(3 * expected_r, 0.35 * step * (fi - f1))
            search_min_x = x1 - expected_r
            search_max_y = y1 + expected_r * 2
        elif positions:
            _, last_cx, last_cy = positions[-1]
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
            predicted=predicted,
            gate=gate,
            address=(addr_x, addr_y),
        )

        if det is not None:
            positions.append((fi, det[0], det[1]))
            flight_radii.append(det[2])
            misses = 0
        elif not positions:
            # Right after impact the ball is often tangled with the clubhead or passing in front
            # of something white; keep looking for its first clean appearance.
            if fi - impact_idx >= max_initial_search:
                break
        else:
            misses += 1
            if misses > max_misses or len(positions) >= 3:
                break  # lost the ball (or it left the frame)

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
        "flight_radii": flight_radii,
        "warnings": warnings,
    }
