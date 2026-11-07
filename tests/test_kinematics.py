"""Unit tests for vision.kinematics — pure math, no OpenCV."""

import math
import pytest

from vision.kinematics import positions_to_velocity, velocity_to_shot


FPS = 240.0


def _make_positions(vx_mps: float, vy_mps: float, n: int = 6, fps: float = FPS):
    """Generate synthetic ball positions (mm) for a given velocity."""
    dt = 1.0 / fps
    # vx_mps and vy_mps are in m/s; convert to mm/s for position generation
    vx_mms = vx_mps * 1000.0
    vy_mms = vy_mps * 1000.0
    return [(i * dt * vx_mms, i * dt * vy_mms) for i in range(n)]


# ── Test 1: horizontal shot at 150 mph ────────────────────────────────────────

def test_horizontal_shot_speed():
    target_mph = 150.0
    target_mps = target_mph / 2.23694
    positions = _make_positions(vx_mps=target_mps, vy_mps=0.0)
    vx, vy = positions_to_velocity(positions, FPS)
    speed_mph, _ = velocity_to_shot(vx, vy)
    assert abs(speed_mph - target_mph) < 0.5, f"Expected ~{target_mph} mph, got {speed_mph:.2f}"


def test_horizontal_shot_vla():
    target_mps = 150.0 / 2.23694
    positions = _make_positions(vx_mps=target_mps, vy_mps=0.0)
    vx, vy = positions_to_velocity(positions, FPS)
    _, vla = velocity_to_shot(vx, vy)
    assert abs(vla) < 0.5, f"Expected ~0° VLA, got {vla:.2f}°"


# ── Test 2: 45° launch ────────────────────────────────────────────────────────

def test_45_degree_vla():
    speed_mps = 50.0
    component = speed_mps / math.sqrt(2)
    positions = _make_positions(vx_mps=component, vy_mps=component)
    vx, vy = positions_to_velocity(positions, FPS)
    _, vla = velocity_to_shot(vx, vy)
    assert abs(vla - 45.0) < 0.5, f"Expected ~45°, got {vla:.2f}°"


def test_45_degree_speed():
    speed_mps = 50.0
    component = speed_mps / math.sqrt(2)
    positions = _make_positions(vx_mps=component, vy_mps=component)
    vx, vy = positions_to_velocity(positions, FPS)
    speed_mph, _ = velocity_to_shot(vx, vy)
    expected_mph = speed_mps * 2.23694
    assert abs(speed_mph - expected_mph) < 0.5, f"Expected ~{expected_mph:.1f} mph, got {speed_mph:.2f}"


# ── Test 3: image y-flip (vy positive = upward) ───────────────────────────────

def test_upward_ball_positive_vla():
    # Ball moving right and upward → vy should be positive → VLA > 0
    positions = _make_positions(vx_mps=50.0, vy_mps=10.0)
    vx, vy = positions_to_velocity(positions, FPS)
    assert vy > 0, f"Expected vy > 0 (upward), got {vy}"
    _, vla = velocity_to_shot(vx, vy)
    assert vla > 0, f"Expected positive VLA for upward shot, got {vla:.2f}°"


def test_downward_ball_negative_vla():
    # Ball moving right and downward → vy negative → VLA < 0
    positions = _make_positions(vx_mps=50.0, vy_mps=-10.0)
    vx, vy = positions_to_velocity(positions, FPS)
    assert vy < 0, f"Expected vy < 0 (downward), got {vy}"
    _, vla = velocity_to_shot(vx, vy)
    assert vla < 0, f"Expected negative VLA for downward shot, got {vla:.2f}°"


# ── Test 4: regression is better than a single pair ──────────────────────────

def test_regression_uses_all_points():
    # With noisy data, regression result should be close to true velocity
    # even with 6 points, not just 2
    target_mps = 67.0  # ~150 mph
    rng = [(i * target_mps * 1000 / FPS + (i % 2) * 0.5, 0.0) for i in range(6)]
    # Slight noise on alternating frames
    vx, vy = positions_to_velocity(rng, FPS)
    speed_mph, _ = velocity_to_shot(vx, vy)
    expected = target_mps * 2.23694
    assert abs(speed_mph - expected) < 2.0, f"Expected ~{expected:.1f} mph, got {speed_mph:.2f}"


# ── Test 5: velocity_to_shot formula check ───────────────────────────────────

def test_velocity_to_shot_formula():
    vx, vy = 60.0, 0.0
    speed_mph, vla = velocity_to_shot(vx, vy)
    assert abs(speed_mph - 60.0 * 2.23694) < 0.001
    assert abs(vla) < 0.001


# ── Test 6: minimum positions requirement ────────────────────────────────────

def test_positions_to_velocity_requires_two_points():
    with pytest.raises(ValueError, match="at least 2"):
        positions_to_velocity([(0.0, 0.0)])
