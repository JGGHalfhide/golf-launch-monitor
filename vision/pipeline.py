"""Full offline vision pipeline: load clip → detect impact → track → compute shot."""

import cv2
import numpy as np

from .calibration import px_per_mm_from_ball, calibrate_from_ball
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

    file_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    timestamps_ms: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("Clip contains no frames")

    # Prefer fps from file metadata; fall back to caller-supplied value
    clip_fps = file_fps if (file_fps and file_fps > 1) else fps
    impact_idx = find_impact_frame(frames)

    # Auto-calibrate from frames before impact; also capture the ball's address position
    address_pos: tuple[int, int] | None = None
    if px_per_mm is None:
        calib_frame_idx = max(0, impact_idx - 1)
        for offset in range(0, min(15, calib_frame_idx + 1)):
            calib = calibrate_from_ball(frames[calib_frame_idx - offset])
            if calib is not None:
                px_per_mm, addr_cx, addr_cy = calib
                address_pos = (addr_cx, addr_cy)
                break
        if px_per_mm is None:
            raise RuntimeError(
                "Auto-calibration failed: could not detect stationary ball. "
                "Supply --calib <px_per_mm> manually."
            )

    result = track_ball(frames, impact_idx, px_per_mm, max_frames=8,
                        timestamps_ms=timestamps_ms, address_pos=address_pos)
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
