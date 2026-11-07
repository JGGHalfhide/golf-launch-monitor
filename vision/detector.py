"""Per-frame ball detection. No tracking state — pure detection."""

import cv2
import numpy as np


def find_ball(frame, min_radius_px: int = 5, max_radius_px: int = 80) -> tuple[int, int, int] | None:
    """Return (cx, cy, radius) of best ball candidate, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Primary: HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_radius_px * 2,
        param1=60,
        param2=25,
        minRadius=min_radius_px,
        maxRadius=max_radius_px,
    )

    if circles is not None:
        best = np.round(circles[0, 0]).astype(int)
        return (int(best[0]), int(best[1]), int(best[2]))

    # Fallback: threshold + contour circularity
    return _blob_fallback(gray, min_radius_px, max_radius_px)


def find_ball_by_diff(
    frame,
    background,
    expected_radius_px: int = 30,
) -> tuple[int, int, int] | None:
    """
    Find the ball by subtracting a pre-impact background frame.
    Much more robust than HoughCircles on busy scenes.

    Returns (cx, cy, estimated_radius) or None.
    expected_radius_px: rough ball radius at this distance, used for size filtering.
    """
    diff = cv2.absdiff(frame, background)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold: keep only pixels that changed significantly
    _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate to fill gaps from motion blur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = np.pi * (expected_radius_px * 0.4) ** 2
    max_area = np.pi * (expected_radius_px * 4.0) ** 2

    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        # Score: circularity × area (prefer rounder, larger blobs)
        score = circularity * area
        if score > best_score:
            best_score = score
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            radius = max(1, int(np.sqrt(area / np.pi)))
            best = (cx, cy, radius)

    return best


def _blob_fallback(gray, min_radius_px: int, max_radius_px: int) -> tuple[int, int, int] | None:
    """Find largest white circular blob via thresholding."""
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    best_circularity = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * min_radius_px ** 2:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        radius = int(np.sqrt(area / np.pi))
        if radius < min_radius_px or radius > max_radius_px:
            continue
        if circularity > best_circularity:
            best_circularity = circularity
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            best_candidate = (cx, cy, radius)

    if best_circularity > 0.6:
        return best_candidate
    return None
