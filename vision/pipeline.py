"""Full offline vision pipeline: load clip → detect impact → track → compute shot."""

import cv2
import numpy as np

from .calibration import BALL_DIAMETER_MM, refine_ball_radius, vanished_ball_radius, white_blob_candidates
from .impact import ball_patch_signal, hole_shape, impact_from_signal, is_ball_hole, rank_spots, rest_frames, resting_clusters, track_is_plausible, vanish_score
from .tracker import track_ball
from .overlay import draw_overlay
from .clipinfo import measured_fps


SCALE = 4           # impact search runs on 1/SCALE-size grayscale frames
DETECT_SCALE = 2    # ball candidates are found on 1/DETECT_SCALE-size colour frames
SAMPLE_EVERY = 6    # ... every N frames (25 ms at 240 fps)
WINDOW_PRE = 15     # full-res frames kept before impact (radius refinement, tracker background)
WINDOW_POST = 15    # ... and after (tracker uses up to 8)
HOLE_AFTER = 8      # frames after impact for measuring the hole the ball left (club gone by then)
MAX_SPOTS = 4       # resting spots to try tracking from, best first
SPOT_TOLERANCE = 3  # with a calibrated ball spot: max distance (in ball radii) of the resting ball
TRACK_RETRIES = 2   # re-track skipping a bad first detection (clubhead) when a track is implausible


def run(
    clip_path: str,
    px_per_mm: float | None = None,
    output_path: str | None = None,
    fps: float = 240.0,
    ball_spot: tuple[float, float, float] | None = None,
) -> dict:
    """
    Full pipeline. Returns the same dict as track_ball plus clip metadata.

    ball_spot: (x, y, r) of the calibrated tee spot in full-res pixels (tools/set_ball_spot.py).
    When given, only resting spots within SPOT_TOLERANCE ball radii of it are considered —
    no searching the whole frame, which is where most misidentifications came from.

    Pass 1 streams the clip once, keeping only small grayscale frames, timestamps and
    white-blob ball candidates from sampled frames (bounded memory: ~250 MB for an 8 s clip).
    Candidates that sit still are grouped into resting spots; the ball is a spot that later
    vanishes for good (vision/impact.py) and can then be tracked flying away. Background white
    objects never vanish; anything else that does but can't be tracked is rejected.
    Pass 2 decodes only a short full-res window around impact.
    If px_per_mm is None, it comes from the ball's radius refined at full res over pre-impact frames.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open clip: {clip_path}")

    file_fps = cap.get(cv2.CAP_PROP_FPS)
    timestamps_ms: list[float] = []
    small: list[np.ndarray] = []
    candidates: list[tuple[int, float, float, float, float]] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small.append(cv2.resize(gray, (w // SCALE, h // SCALE), interpolation=cv2.INTER_AREA))
        if i % SAMPLE_EVERY == 0:
            half = cv2.resize(frame, (w // DETECT_SCALE, h // DETECT_SCALE), interpolation=cv2.INTER_AREA)
            for cx, cy, r, circ in white_blob_candidates(half):
                candidates.append((i, cx * DETECT_SCALE, cy * DETECT_SCALE, r * DETECT_SCALE, circ))
        i += 1
    cap.release()

    if not timestamps_ms:
        raise ValueError("Clip contains no frames")
    clip_fps = file_fps if (file_fps and file_fps > 1) else fps

    # Resting spots that later vanish for good, most ball-like vanish first (a struck ball
    # leaves its spot within one frame; feet, shadows and body parts change gradually).
    spots = []
    for c in resting_clusters(candidates):
        sig = ball_patch_signal(small, c.cx / SCALE, c.cy / SCALE, c.r / SCALE, c.frames)
        impact = impact_from_signal(sig)
        if impact is not None and (score := vanish_score(sig, impact)) > 0:
            spots.append((c, impact, score, rest_frames(sig, impact)))
    if ball_spot is not None:
        sx, sy, sr = ball_spot
        spots = [sp for sp in spots if np.hypot(sp[0].cx - sx, sp[0].cy - sy) <= SPOT_TOLERANCE * sr]
        if not spots:
            raise RuntimeError("No shot detected at the calibrated ball spot — ball moved off the tee mark?")
    if not spots:
        raise RuntimeError("No shot detected — no resting ball that later left its spot")
    ranked = rank_spots(spots)
    # Spots that leave a round, ball-sized hole at impact go first (roundest first) — only the
    # struck ball does that. The heuristic ranking orders the rest as a fallback.
    holes = {id(c): hole_shape(small, imp, c.cx / SCALE, c.cy / SCALE) for c, imp, _ in ranked}
    with_hole = sorted((sp for sp in ranked if is_ball_hole(holes[id(sp[0])], SCALE)),
                       key=lambda sp: -holes[id(sp[0])][1])
    spots = with_hole + [sp for sp in ranked if not is_ball_hole(holes[id(sp[0])], SCALE)]
    del small

    result, result_ok = None, False
    for c, impact_idx, _ in spots[:MAX_SPOTS]:
        lo = max(0, impact_idx - WINDOW_PRE)
        hi = min(len(timestamps_ms), impact_idx + WINDOW_POST + 1)
        frames = _read_window(clip_path, timestamps_ms, lo, hi)

        ppm, scale_source = px_per_mm, "given"
        if ppm is None:
            k = impact_idx - lo
            hole_r = None
            if k - 3 >= 0 and k + HOLE_AFTER < len(frames):
                hole_r = vanished_ball_radius(frames[k - 3], frames[k + HOLE_AFTER], c.cx, c.cy)
            if hole_r is not None:
                r_px, scale_source = hole_r, "hole"
            else:
                # Fallback: ball at address, pre-impact frames stopping short of impact
                r_px = refine_ball_radius(frames[: max(1, k - 3)], c.cx, c.cy, c.r)
                r_px, scale_source = (r_px, "address") if r_px is not None else (c.r, "candidate")
            ppm = 2 * r_px / BALL_DIAMETER_MM

        local_impact = impact_idx - lo
        start_after = None
        for _ in range(1 + TRACK_RETRIES):
            attempt = track_ball(frames, local_impact, ppm, max_frames=8,
                                 timestamps_ms=timestamps_ms[lo:hi],
                                 address_pos=(int(round(c.cx)), int(round(c.cy))),
                                 start_after=start_after)
            ok = track_is_plausible(attempt["positions"], ppm, fps=measured_fps(timestamps_ms) or fps,
                                    address=(c.cx, c.cy), impact=local_impact)
            if ok or not attempt["positions"]:
                break
            # First detection was likely the clubhead (it leaves with the ball and can look
            # like a blurred ball); retry with the ball's first appearance after it.
            start_after = attempt["positions"][0][0]
        if result is None or (ok and not result_ok):
            result, result_ok, best = attempt, ok, (c, impact_idx, lo, frames)
        attempt["scale_source"] = scale_source
        if ok:
            break

    if not result_ok:
        result["warnings"].append("No plausible ball flight found after impact")
        result["ball_speed_mph"] = 0.0

    c, impact_idx, lo, frames = best
    if result_ok and result.get("scale_source") in ("address", "candidate"):
        _rescale_from_flight(result)  # address measurements can be short; the hole one isn't
    result["positions"] = [(fi + lo, cx, cy) for fi, cx, cy in result["positions"]]
    result["impact_frame"] = impact_idx
    result["ball_address"] = (round(c.cx, 1), round(c.cy, 1))
    result["clip_path"] = clip_path
    result["total_frames"] = len(timestamps_ms)
    result["clip_fps"] = clip_fps
    result["measured_fps"] = measured_fps(timestamps_ms)

    if output_path:
        _write_overlay_video(frames, result, output_path, clip_fps, frame_offset=lo)

    return result


def _rescale_from_flight(result: dict, max_disagreement: float = 0.8, max_growth: float = 2.0) -> None:
    """
    Cross-check the address scale against the ball's size in flight.

    At address the ball can be measured short: its shaded underside merges into the mat, dark
    markings split it, or a white background hides an edge (IMG_0294: radius 20 measured vs
    ~31 real → 124 mph instead of ~75). In flight it is a clean disc against sky or fence, and
    the short axis of its blur ellipse is its true diameter. Flight detections can be partial
    (smaller) but rarely larger, so the largest plausible one (≤ max_growth × address) is used. If the address radius is under
    max_disagreement × that, rescale: speed ∝ 1/px_per_mm, VLA is unaffected.
    """
    addr_r = result["px_per_mm"] * BALL_DIAMETER_MM / 2
    # A flight blob far bigger than the ball (merged with the club or a passing object —
    # IMG_0316 read 230 px vs 32 at address and sent 14 mph) is not the ball's size.
    radii = [r for r in (result.get("flight_radii") or []) if r <= max_growth * addr_r]
    if not radii:
        return
    flight_r = max(radii)
    if addr_r >= max_disagreement * flight_r:
        return
    new_ppm = 2 * flight_r / BALL_DIAMETER_MM
    result["ball_speed_mph"] = round(result["ball_speed_mph"] * result["px_per_mm"] / new_ppm, 2)
    result["warnings"].append(
        f"Ball measured {addr_r:.0f} px radius at address but {flight_r:.0f} px in flight — "
        f"scale taken from flight"
    )
    result["px_per_mm"] = new_ppm


def _read_window(clip_path: str, timestamps_ms: list[float], lo: int, hi: int) -> list[np.ndarray]:
    """
    Decode frames [lo, hi) at full res. Seeks by timestamp (frame-index seeking is unreliable
    on variable-frame-rate slo-mo files), then reads forward and matches each frame by its
    timestamp. Falls back to a sequential read if the seek lands past the window.
    """
    cap = cv2.VideoCapture(clip_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamps_ms[lo] - 1000.0))
    frames: list[np.ndarray] = []
    first = True
    while len(frames) < hi - lo:
        ok, frame = cap.read()
        if not ok:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        if first and ts > timestamps_ms[lo] + 0.5:
            cap.release()  # seek overshot: restart from the beginning
            cap = cv2.VideoCapture(clip_path)
            first = False
            continue
        first = False
        if ts >= timestamps_ms[lo] - 0.5:
            frames.append(frame)
    cap.release()
    if len(frames) != hi - lo:
        raise RuntimeError(f"Could not re-read frames {lo}–{hi} around impact")
    return frames


def _write_overlay_video(
    frames: list[np.ndarray],
    result: dict,
    output_path: str,
    fps: float,
    frame_offset: int = 0,
) -> None:
    """Write a debug overlay video with ball detections and shot data.
    frames[k] is clip frame k + frame_offset (result indices are clip-global)."""
    if not frames:
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Build a lookup: frame_idx → (cx, cy, radius)
    det_map: dict[int, tuple] = {}
    for fi, cx, cy in result.get("positions", []):
        # radius not stored in result — estimate from px_per_mm
        r = max(5, int(result["px_per_mm"] * 42.67 / 2))
        det_map[fi] = (cx, cy, r)

    for fi, frame in enumerate(frames, start=frame_offset):
        # Pass all detections up to this frame for trail
        trail = [det_map.get(j) for j in range(max(0, fi - 5), fi + 1)]
        shot_data = result if fi >= result["impact_frame"] else None
        out = draw_overlay(frame, trail, shot_data)
        writer.write(out)

    writer.release()
