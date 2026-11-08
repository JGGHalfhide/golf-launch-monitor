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


def white_blob_candidates(
    frame_bgr: np.ndarray,
    top_fraction: float = 0.3,
    min_area: int = 20,
    min_circularity: float = 0.3,
    min_radius: float = 5.0,
    window: int = 101,
    thresholds: tuple[float, ...] = (30.0, 50.0, 70.0),
) -> list[tuple[float, float, float, float]]:
    """
    Cheap ball candidates: compact, roughly round blobs that are much whiter than their
    surroundings.

    Whiteness = V − S (bright and colourless). A ball's whiteness relative to the local mean
    (box blur over `window` px) stands out in any light: in shade the ball picks up a blue tint
    (S ≈ 70, V ≈ 180) and fails fixed thresholds, while in sun a pale mat is itself fairly
    "white" — so only the local contrast is reliable.

    The contrast mask is cut at several thresholds. In shade the ball's contrast is modest and
    only the low cut finds it; on a sunlit, frosty mat the low cut joins the ball to bright
    mat specks in one irregular blob (IMG_0320/0322) and only a higher cut separates it.
    Round, ball-sized blobs from any cut are kept; overlapping ones are merged (roundest wins).

    Returns (cx, cy, r, circularity) in this frame's pixel coords. Deliberately permissive —
    background white objects come through too; the caller decides which candidate is the ball
    by its behaviour (rests, then vanishes at impact, then flies).
    """
    h = frame_bgr.shape[0]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    whiteness = np.clip(hsv[..., 2].astype(np.int16) - hsv[..., 1], 0, 255).astype(np.float32)
    contrast = whiteness - cv2.blur(whiteness, (window, window))
    contrast[: int(h * top_fraction)] = 0  # sky
    kernel = np.ones((3, 3), np.uint8)
    found: list[tuple[float, float, float, float]] = []
    for thr in thresholds:
        mask = cv2.morphologyEx((contrast > thr).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            circ = area / (np.pi * r * r) if r > 0 else 0.0
            if circ < min_circularity or r < min_radius:
                continue
            dup = next((k for k, f in enumerate(found) if np.hypot(f[0] - x, f[1] - y) < max(f[2], r) * 0.5), None)
            if dup is None:
                found.append((float(x), float(y), float(r), float(circ)))
            elif circ > found[dup][3]:
                found[dup] = (float(x), float(y), float(r), float(circ))
    return found


def refine_ball_radius(frames_bgr: list[np.ndarray], cx: float, cy: float, r_est: float) -> float | None:
    """
    Full-res ball radius (px) at a known resting position, averaged over several frames.

    Runs HoughCircles only on a small crop around the ball with a narrow radius range, so it is
    fast and can't latch onto background objects. r_est comes from the local-contrast blob
    (within ~5% of the true radius on test clips), so the search range is ±25–30%.
    """
    radii = []
    for frame in frames_bgr:
        H, W = frame.shape[:2]
        half = int(r_est * 3) + 4
        x0, y0 = max(0, int(cx) - half), max(0, int(cy) - half)
        x1, y1 = min(W, int(cx) + half), min(H, int(cy) + half)
        crop = cv2.GaussianBlur(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
        circles = cv2.HoughCircles(
            crop, cv2.HOUGH_GRADIENT, dp=1, minDist=max(4, int(r_est)),
            param1=60, param2=18,
            minRadius=max(3, int(r_est * 0.75)), maxRadius=int(r_est * 1.3) + 1,
        )
        if circles is None:
            continue
        # The circle centred closest to the expected position
        best = min(circles[0], key=lambda c: (c[0] + x0 - cx) ** 2 + (c[1] + y0 - cy) ** 2)
        if np.hypot(best[0] + x0 - cx, best[1] + y0 - cy) <= r_est * 1.5:
            radii.append(float(best[2]))
    return float(np.median(radii)) if radii else None


def vanished_ball_radius(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    cx: float,
    cy: float,
    search: int = 140,
    diff_threshold: int = 25,
    min_circularity: float = 0.5,
) -> float | None:
    """
    Ball radius (px) from the hole it leaves: the region that changed at its resting spot
    between a frame just before impact and one ~8 frames after (club gone, bare mat showing).

    Only the ball changed there, so this outlines the ball itself regardless of lighting,
    shading or markings — which break measurements made on the ball at address (alignment
    line splitting it, shaded underside merging into the mat, sunlit mat specks joining it).
    Radius from the blob's area (≈ true radius within ~3% on clips with known scale).
    Returns None if no round changed blob sits at the spot.
    """
    H, W = before_bgr.shape[:2]
    x0, y0 = max(0, int(cx) - search), max(0, int(cy) - search)
    x1, y1 = min(W, int(cx) + search), min(H, int(cy) + search)
    a = cv2.GaussianBlur(cv2.cvtColor(before_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (5, 5), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(after_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (5, 5), 0)
    mask = (cv2.absdiff(a, b) > diff_threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, _, centroids = cv2.connectedComponentsWithStats(mask)
    px, py = cx - x0, cy - y0
    best = None
    for i in range(1, n):
        dist = float(np.hypot(centroids[i][0] - px, centroids[i][1] - py))
        if dist < 60 and (best is None or dist < best[0]):
            best = (dist, i)
    if best is None:
        return None
    contours, _ = cv2.findContours((labels == best[1]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    (_, _), enc_r = cv2.minEnclosingCircle(cnt)
    if enc_r <= 0 or area / (np.pi * enc_r * enc_r) < min_circularity:
        return None
    return float(np.sqrt(area / np.pi))
