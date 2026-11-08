import cv2
import numpy as np
import pytest

from vision.calibration import white_blob_candidates


def scene(mat_bgr, ball_bgr, r=15, size=(540, 960)):
    """Flat mat with a ball at (480, 400) and some grass-like noise."""
    rng = np.random.default_rng(0)
    img = np.empty((*size, 3), dtype=np.uint8)
    img[:] = mat_bgr
    img = np.clip(img.astype(np.int16) + rng.integers(-12, 13, img.shape), 0, 255).astype(np.uint8)
    cv2.circle(img, (480, 400), r, ball_bgr, -1)
    return img


@pytest.mark.parametrize("label,mat,ball", [
    # IMG_0291: ball in shade picks up a blue tint (S≈70, V≈180), dark mat — failed fixed thresholds
    ("shade", (90, 110, 70), (180, 160, 140)),
    # IMG_0278: pale sunlit mat that is itself fairly "white" (S≈60, V≈160)
    ("pale sunny mat", (130, 160, 135), (235, 235, 232)),
    # IMG_0285: bright saturated mat in full sun
    ("bright sun", (60, 220, 120), (250, 250, 250)),
])
def test_ball_found_in_any_light(label, mat, ball):
    cands = white_blob_candidates(scene(mat, ball))
    near = [c for c in cands if np.hypot(c[0] - 480, c[1] - 400) < 5]
    assert near, f"ball not found ({label})"
    cx, cy, r, circ = near[0]
    assert r == pytest.approx(15, abs=2.5)
    assert circ > 0.6


def test_plain_mat_gives_no_candidates():
    img = scene((90, 110, 70), (90, 110, 70))
    assert white_blob_candidates(img) == []
