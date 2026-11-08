"""Full offline vision pipeline: load clip → detect impact → track → compute shot."""

import cv2
import numpy as np

from .calibration import BALL_DIAMETER_MM, refine_ball_radius, white_blob_candidates
from .impact import ball_patch_signal, impact_from_signal, rank_spots, resting_clusters, track_is_plausible, vanish_score
from .tracker import track_ball
from .overlay import draw_overlay
from .clipinfo import measured_fps


SCALE = 4           # impact search runs on 1/SCALE-size grayscale frames
DETECT_SCALE = 2    # ball candidates are found on 1/DETECT_SCALE-size colour frames
SAMPLE_EVERY = 6    # ... every N frames (25 ms at 240 fps)
WINDOW_PRE = 15     # full-res frames kept before impact (radius refinement, tracker background)
WINDOW_POST = 15    # ... and after (tracker uses up to 8)
MAX_SPOTS = 4       # resting spots to try tracking from, best first


def run(
    clip_path: str,
    px_per_mm: float | None = None,
    output_path: str | None = None,
    fps: float = 240.0,
) -> dict:
    """
    Full pipeline. Returns the same dict as track_ball plus clip metadata.

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
            spots.append((c, impact, score))
    del small
    if not spots:
        raise RuntimeError("No shot detected — no resting ball that later left its spot")
    spots = rank_spots(spots)

    result, result_ok = None, False
    for c, impact_idx, _ in spots[:MAX_SPOTS]:
        lo = max(0, impact_idx - WINDOW_PRE)
        hi = min(len(timestamps_ms), impact_idx + WINDOW_POST + 1)
        frames = _read_window(clip_path, timestamps_ms, lo, hi)

        ppm = px_per_mm
        if ppm is None:
            # Pre-impact frames, stopping short of impact (club may be arriving)
            r_px = refine_ball_radius(frames[: max(1, impact_idx - lo - 3)], c.cx, c.cy, c.r)
            ppm = 2 * (r_px if r_px is not None else c.r) / BALL_DIAMETER_MM

        attempt = track_ball(frames, impact_idx - lo, ppm, max_frames=8,
                             timestamps_ms=timestamps_ms[lo:hi],
                             address_pos=(int(round(c.cx)), int(round(c.cy))))
        ok = track_is_plausible(attempt["positions"], ppm, fps=measured_fps(timestamps_ms) or fps)
        if result is None or (ok and not result_ok):
            result, result_ok, best = attempt, ok, (c, impact_idx, lo, frames)
        if ok:
            break

    if not result_ok:
        result["warnings"].append("No plausible ball flight found after impact")
        result["ball_speed_mph"] = 0.0

    c, impact_idx, lo, frames = best
    if px_per_mm is None and result_ok:
        _rescale_from_flight(result)
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


def _rescale_from_flight(result: dict, max_disagreement: float = 0.8) -> None:
    """
    Cross-check the address scale against the ball's size in flight.

    At address the ball can be measured short: its shaded underside merges into the mat, dark
    markings split it, or a white background hides an edge (IMG_0294: radius 20 measured vs
    ~31 real → 124 mph instead of ~75). In flight it is a clean disc against sky or fence, and
    the short axis of its blur ellipse is its true diameter. Flight detections can be partial
    (smaller) but rarely larger, so the largest is used. If the address radius is under
    max_disagreement × that, rescale: speed ∝ 1/px_per_mm, VLA is unaffected.
    """
    radii = result.get("flight_radii") or []
    if not radii:
        return
    addr_r = result["px_per_mm"] * BALL_DIAMETER_MM / 2
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
