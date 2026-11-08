"""
Impact detection by watching the resting ball's spot.

Impact is defined as the last frame of the final continuous run in which the ball is
still sitting at its address position. Whole-frame motion (waggles, legs, pressing the
Watch, the follow-through) doesn't matter — only whether the ball is still there.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RestCluster:
    """A spot where a ball candidate was seen repeatedly (sampled frames)."""
    cx: float
    cy: float
    r: float
    circularity: float
    frames: list[int]


def resting_clusters(
    detections: list[tuple[int, float, float, float, float]],
    min_support: int = 3,
    cell: float = 8.0,
) -> list[RestCluster]:
    """
    Group per-frame candidates (frame_idx, cx, cy, r, circularity) that sit at the same spot.

    Votes each detection's centre into a coarse grid (cell px), smooths the vote map over
    3×3 cells, and takes local maxima with ≥ min_support votes as spots. Each spot collects
    the detections within max(r, cell) px of the peak's vote-weighted centre. Linear in the
    number of detections (the sensitive candidate finder produces thousands per clip).
    Returns clusters largest first; positions/radius/circularity are means over members.
    """
    if not detections:
        return []
    d = np.asarray([(f, cx, cy, r, c) for f, cx, cy, r, c in detections], dtype=float)
    gx = np.floor(d[:, 1] / cell).astype(int)
    gy = np.floor(d[:, 2] / cell).astype(int)
    gx0, gy0 = gx.min(), gy.min()
    votes = np.zeros((gy.max() - gy0 + 3, gx.max() - gx0 + 3), dtype=np.float32)
    np.add.at(votes, (gy - gy0 + 1, gx - gx0 + 1), 1)
    smooth = votes.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                smooth += np.roll(np.roll(votes, dy, axis=0), dx, axis=1)
    # local maxima of the smoothed map
    peaks = []
    H, W = smooth.shape
    for y, x in zip(*np.nonzero(smooth >= min_support)):
        v = smooth[y, x]
        win = smooth[max(0, y - 1): y + 2, max(0, x - 1): x + 2]
        if v >= win.max():
            peaks.append((v, y, x))
    peaks.sort(reverse=True)

    taken = np.zeros(len(d), dtype=bool)
    clusters: list[RestCluster] = []
    for _, y, x in peaks:
        near = (~taken) & (np.abs(gy - gy0 + 1 - y) <= 1) & (np.abs(gx - gx0 + 1 - x) <= 1)
        if near.sum() < min_support:
            continue
        cx, cy = d[near, 1].mean(), d[near, 2].mean()
        r = float(np.median(d[near, 3]))
        members = (~taken) & (np.hypot(d[:, 1] - cx, d[:, 2] - cy) <= max(r, cell))
        if members.sum() < min_support:
            continue
        taken |= members
        m = d[members]
        clusters.append(RestCluster(float(m[:, 1].mean()), float(m[:, 2].mean()), float(m[:, 3].mean()),
                                    float(m[:, 4].mean()), sorted(int(f) for f in m[:, 0])))
    clusters.sort(key=lambda c: -len(c.frames))
    return clusters


def ball_patch_signal(
    frames_gray: list[np.ndarray],
    cx: float,
    cy: float,
    r: float,
    ref_indices: list[int],
) -> np.ndarray:
    """
    Per-frame mean absolute difference between the ball's spot and the ball at rest.

    frames_gray: grayscale frames (any scale — pass cx, cy, r in the same scale).
    ref_indices: frames known to show the ball at rest; their median patch is the reference.
    Low values = ball present; high values = ball gone (or covered).
    """
    half = max(1, int(round(0.6 * r)))
    h, w = frames_gray[0].shape[:2]
    x0, x1 = max(0, int(round(cx)) - half), min(w, int(round(cx)) + half + 1)
    y0, y1 = max(0, int(round(cy)) - half), min(h, int(round(cy)) + half + 1)
    patches = np.stack([f[y0:y1, x0:x1] for f in frames_gray]).astype(np.float32)
    ref = np.median(patches[ref_indices], axis=0)
    return np.abs(patches - ref).mean(axis=(1, 2))


def impact_from_signal(
    signal: np.ndarray,
    min_run: int = 48,
    margin: float = 15.0,
    min_gone: int = 12,
) -> int | None:
    """
    Impact frame from a ball-patch signal (see ball_patch_signal).

    "Present" = within `margin` grey levels of the resting-ball level (10th percentile).
    Anything else — bare mat, a shoe, the club — counts as gone; how different it is doesn't
    matter. Impact is the last frame of the final run of ≥ min_run consecutive present frames
    (default 0.2 s at 240 fps — a ball at address rests far longer), so something briefly
    resembling the ball on the empty spot can't be mistaken for it.
    Returns None if the ball never clearly leaves (fewer than min_gone non-present frames after).
    """
    s = np.asarray(signal, dtype=float)
    if s.size < min_run + min_gone:
        return None
    present = s < float(np.percentile(s, 10)) + margin

    run = 0
    last_end = None
    for i, p in enumerate(present):
        run = run + 1 if p else 0
        if run >= min_run:
            last_end = i
    if last_end is None or np.count_nonzero(~present[last_end + 1:]) < min_gone:
        return None
    return last_end


def vanish_score(signal: np.ndarray, impact: int, pre: int = 12, post: int = 12) -> float:
    """
    How much a spot's disappearance looks like a struck ball (higher = more ball-like; 0 = not).

    A struck ball leaves its spot within a single 240 fps frame and leaves steady bare mat
    behind. Feet, shadows and body parts change over several frames or keep flickering.
      jump  = median level after (skipping 2 frames) − median level before
      sharp = largest single-frame step around impact / jump   (ball ≈ 0.85–0.95)
      wobble = std after / jump                                  (ball ≈ 0.01–0.03)
    Score = jump × min(sharp, 1) × (1 − wobble); a spike that comes straight back
    (sharp ≫ 1, tiny jump) scores 0.
    """
    s = np.asarray(signal, dtype=float)
    before = s[max(0, impact - pre): impact + 1]
    after = s[impact + 3: impact + 3 + post]
    if len(after) < post // 2 or len(before) == 0:
        return 0.0
    jump = float(np.median(after) - np.median(before))
    if jump <= 0:
        return 0.0
    sharp = float(np.diff(s[max(0, impact - 2): impact + 4]).max()) / jump
    wobble = float(np.std(after)) / jump
    if sharp > 1.5:
        return 0.0
    return jump * min(sharp, 1.0) * max(0.0, 1.0 - wobble)


MAX_BALL_SPEED_MPH = 220.0  # comfortably above any real shot (tour driver ≈ 190 mph)


def track_is_plausible(
    positions: list[tuple[int, int, int]],
    px_per_mm: float,
    min_steps: int = 1,
    fps: float = 240.0,
) -> bool:
    """
    A real launch moves the ball a large but bounded distance every frame.

    Lower bound: even a 20 mph chip moves ~37 mm ≈ 0.9 ball diameters per 240 fps frame, so a
    median step under half a diameter is a static blob (shoe logo, junk). Upper bound: no step
    may exceed MAX_BALL_SPEED_MPH — two unrelated blobs far apart once produced "1111 mph".
    """
    if len(positions) < min_steps + 1:
        return False
    steps = [np.hypot(x1 - x0, y1 - y0) / max(1, f1 - f0)
             for (f0, x0, y0), (f1, x1, y1) in zip(positions, positions[1:])]
    ball_diameter_px = 42.67 * px_per_mm
    max_step_px = MAX_BALL_SPEED_MPH * 447.04 / fps * px_per_mm  # mph → mm/s is ×447.04
    return float(np.median(steps)) >= 0.5 * ball_diameter_px and max(steps) <= max_step_px


def rank_spots(
    spots: list[tuple["RestCluster", int, float]],
    window: int = 2,
    crowd: int = 2,
    full_rest_samples: int = 40,
) -> list[tuple["RestCluster", int, float]]:
    """
    Order (cluster, impact, vanish_score) candidates, most ball-like first.

    - A struck ball vanishes alone. When the feet pivot in the follow-through, many spots
      change in the same frame; a spot whose impact is shared (±window frames) by more than
      `crowd` other spots has its score divided by the number sharing it.
    - The ball rests at address for a second or more; spots seen in fewer than
      full_rest_samples sampled frames are scaled down proportionally.
    Returns (cluster, impact, adjusted_score), best first.
    """
    impacts = np.asarray([imp for _, imp, _ in spots])
    ranked = []
    for c, imp, score in spots:
        sharing = int(np.count_nonzero(np.abs(impacts - imp) <= window)) - 1
        adj = score / (1 + sharing) if sharing > crowd else score
        adj *= min(1.0, len(c.frames) / full_rest_samples)
        ranked.append((c, imp, adj))
    ranked.sort(key=lambda t: -t[2])
    return ranked
