"""Pure math for converting ball positions to speed and launch angle. No OpenCV."""

import math
from typing import Sequence


def positions_to_velocity(
    positions_mm: Sequence[tuple[float, float]],
    fps: float = 240.0,
    times_s: Sequence[float] | None = None,
) -> tuple[float, float]:
    """
    Linear regression fit over (x_mm, y_mm) time series.
    Returns (vx_mps, vy_mps). y is already sign-corrected (positive = up).

    Pass times_s (seconds) for variable frame rate clips; otherwise uniform 1/fps spacing.
    """
    n = len(positions_mm)
    if n < 2:
        raise ValueError(f"Need at least 2 positions, got {n}")

    if times_s is not None:
        times = list(times_s)
        # Normalise so first timestamp = 0
        t0 = times[0]
        times = [t - t0 for t in times]
    else:
        dt = 1.0 / fps
        times = [i * dt for i in range(n)]

    # Linear regression for x(t) and y(t) independently
    vx_mps = _linear_slope(times, [p[0] for p in positions_mm]) / 1000.0  # mm/s → m/s
    vy_mps = _linear_slope(times, [p[1] for p in positions_mm]) / 1000.0

    return (vx_mps, vy_mps)


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    """Return slope of least-squares linear fit (mm/s when xs are seconds, ys are mm)."""
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


def velocity_to_shot(vx_mps: float, vy_mps: float) -> tuple[float, float]:
    """Returns (ball_speed_mph, vertical_launch_angle_deg)."""
    speed_mps = math.sqrt(vx_mps ** 2 + vy_mps ** 2)
    speed_mph = speed_mps * 2.23694
    vla_deg = math.degrees(math.atan2(vy_mps, vx_mps))
    return (speed_mph, vla_deg)
