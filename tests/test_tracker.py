import numpy as np


# --- blurred-ball shape (IMG_0338: dull light smears the ball into a capsule) ---------

from vision.tracker import is_blurred_ball_shape  # noqa: E402


def test_blurred_ball_capsule_accepted():
    # IMG_0338 frame 1028: short half-axis 23, long 66, ellipse fill 0.98; ball r ≈ 24
    assert is_blurred_ball_shape(23, 66, 0.98 * np.pi * 23 * 66, expected_r=24)


def test_shaft_rejected():
    # short half-axis of a few px
    assert not is_blurred_ball_shape(5, 90, 0.9 * np.pi * 5 * 90, expected_r=24)


def test_ragged_blob_rejected():
    # merged club/leg change: ball-ish width but poor fill
    assert not is_blurred_ball_shape(22, 83, 0.75 * np.pi * 22 * 83, expected_r=24)


def test_overlong_streak_rejected():
    assert not is_blurred_ball_shape(24, 130, 0.95 * np.pi * 24 * 130, expected_r=24)
