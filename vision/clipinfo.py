"""Clip metadata helpers (pure functions)."""

import numpy as np


def measured_fps(timestamps_ms: list[float]) -> float | None:
    """
    Frame rate from the median per-frame timestamp interval.

    CAP_PROP_FPS is an average over variable-frame-rate files and can't be trusted
    to tell a real 240 fps original from a transcode; per-frame intervals can.
    Returns None if there are fewer than two distinct timestamps.
    """
    ts = np.asarray(timestamps_ms, dtype=float)
    if ts.size < 2:
        return None
    dt = np.diff(ts)
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    return float(1000.0 / np.median(dt))
