"""Full offline vision pipeline: load clip → detect impact → track → compute shot."""

import cv2
import numpy as np

from .calibration import px_per_mm_from_ball
from .tracker import find_impact_frame, track_ball
from .overlay import draw_overlay


def run(
    clip_path: str,
    px_per_mm: float | None = None,
    output_path: str | None = None,
    fps: float = 240.0,
) -> dict:
    """
    Full pipeline. Returns the same dict as track_ball plus clip metadata.
    If px_per_mm is None, auto-calibrate from the frame just before impact.
    If output_path given, write debug overlay video.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open clip: {clip_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("Clip contains no frames")

    clip_fps = fps  # trust the caller / iPhone metadata; don't rely on cap.get
    impact_idx = find_impact_frame(frames)

    # Auto-calibrate from the frame just before impact (ball is stationary)
    if px_per_mm is None:
        calib_frame_idx = max(0, impact_idx - 1)
        px_per_mm = px_per_mm_from_ball(frames[calib_frame_idx])
        if px_per_mm is None:
            # Try a few frames before impact
            for offset in range(2, min(10, calib_frame_idx + 1)):
                px_per_mm = px_per_mm_from_ball(frames[calib_frame_idx - offset])
                if px_per_mm is not None:
                    break
        if px_per_mm is None:
            raise RuntimeError(
                "Auto-calibration failed: could not detect stationary ball. "
                "Supply --calib <px_per_mm> manually."
            )

    result = track_ball(frames, impact_idx, px_per_mm, max_frames=8)
    result["clip_path"] = clip_path
    result["total_frames"] = len(frames)
    result["clip_fps"] = clip_fps

    if output_path:
        _write_overlay_video(frames, result, output_path, clip_fps)

    return result


def _write_overlay_video(
    frames: list[np.ndarray],
    result: dict,
    output_path: str,
    fps: float,
) -> None:
    """Write a debug overlay video with ball detections and shot data."""
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

    for fi, frame in enumerate(frames):
        # Pass all detections up to this frame for trail
        trail = [det_map.get(j) for j in range(max(0, fi - 5), fi + 1)]
        shot_data = result if fi >= result["impact_frame"] else None
        out = draw_overlay(frame, trail, shot_data)
        writer.write(out)

    writer.release()
