"""Pixel-to-mm calibration using the golf ball's known diameter."""

import cv2
import numpy as np


BALL_DIAMETER_MM = 42.67


def calibrate_from_ball(frame, expected_diameter_mm: float = BALL_DIAMETER_MM) -> tuple[float, int, int] | None:
    """
    Find the stationary ball and return (px_per_mm, cx, cy), or None.
    Use this when you also need the ball's pixel position for tracking.
    """
    h, w = frame.shape[:2]
    y0 = int(h * 0.35)
    x0 = int(w * 0.10)
    x1 = int(w * 0.90)
    roi = frame[y0:h, x0:x1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    roi_w = x1 - x0
    min_r = max(12, int(roi_w * 0.012))
    max_r = int(roi_w * 0.055)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 3,
        param1=60, param2=25, minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    best_cx, best_cy, best_r = None, None, None
    best_brightness = -1.0
    for cx, cy, r in circles:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        mean_val = float(cv2.mean(gray, mask=mask)[0])
        if mean_val > best_brightness:
            best_brightness = mean_val
            best_cx, best_cy, best_r = cx, cy, r

    if best_r is None:
        return None

    ppm = px_per_mm_from_known_width(float(best_r * 2), expected_diameter_mm)
    # Translate ROI-relative coords back to full-frame coords
    return (ppm, best_cx + x0, best_cy + y0)


def px_per_mm_from_ball(frame, expected_diameter_mm: float = BALL_DIAMETER_MM) -> float | None:
    """
    Find the stationary ball in a frame and return px/mm from its diameter.

    Searches only the lower 60% / center 80% of the frame — where a ball at address
    sits on the mat. This avoids false positives from background objects.
    Picks the brightest (whitest) candidate circle, since golf balls are very white.
    """
    h, w = frame.shape[:2]

    # Restrict to the region where the ball sits on the mat
    y0 = int(h * 0.35)
    x0 = int(w * 0.10)
    x1 = int(w * 0.90)
    roi = frame[y0:h, x0:x1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    roi_w = x1 - x0
    # Golf ball at 1.5–3 m typically appears 25–70 px in diameter on 1920-wide frame
    min_r = max(12, int(roi_w * 0.012))
    max_r = int(roi_w * 0.055)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_r * 3,
        param1=60,
        param2=25,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # Among all candidates, pick the one with the highest mean brightness
    # (golf ball is white; fence posts, decorations etc. are darker on average)
    best_r = None
    best_brightness = -1.0
    for cx, cy, r in circles:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        mean_val = float(cv2.mean(gray, mask=mask)[0])
        if mean_val > best_brightness:
            best_brightness = mean_val
            best_r = r * 2  # diameter in px

    if best_r is None:
        return None

    return px_per_mm_from_known_width(float(best_r), expected_diameter_mm)


def px_per_mm_from_known_width(px_width: float, real_mm: float) -> float:
    """Return px/mm given a measured pixel width and the real-world size in mm."""
    if real_mm <= 0:
        raise ValueError("real_mm must be > 0")
    return px_width / real_mm
