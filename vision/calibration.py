"""Pixel-to-mm calibration using the golf ball's known diameter."""

import cv2
import numpy as np


BALL_DIAMETER_MM = 42.67


def px_per_mm_from_ball(frame, expected_diameter_mm: float = BALL_DIAMETER_MM) -> float | None:
    """Find the stationary ball in a frame and return px/mm from its diameter."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = gray.shape
    min_r = max(5, int(w * 0.01))
    max_r = int(w * 0.15)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_r * 2,
        param1=60,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    # Pick the circle with the highest accumulator response (first returned by HoughCircles)
    best = circles[0]
    diameter_px = best[2] * 2
    return px_per_mm_from_known_width(float(diameter_px), expected_diameter_mm)


def px_per_mm_from_known_width(px_width: float, real_mm: float) -> float:
    """Return px/mm given a measured pixel width and the real-world size in mm."""
    if real_mm <= 0:
        raise ValueError("real_mm must be > 0")
    return px_width / real_mm
