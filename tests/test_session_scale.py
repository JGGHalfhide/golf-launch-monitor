import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from clip_to_sim import load_session_scale  # noqa: E402
from set_ball_spot import agreed_radius  # noqa: E402


def test_agreed_radius_ignores_oversized_fits():
    # 2026-11-13 IMG_0384: guesses near 16 px land on the ball, bigger guesses fit ball + shadow
    r = agreed_radius([None, 16.4, 15.9, 16.1, 23.5, 33.8, 43.8])
    assert r == pytest.approx(16.1, abs=0.3)


def test_agreed_radius_tie_prefers_smaller():
    assert agreed_radius([15.0, 15.2, 30.0, 30.5]) == pytest.approx(15.1)


def test_agreed_radius_none_when_nothing_fits():
    assert agreed_radius([None, None]) is None


def test_session_scale_loaded_from_spot_file(tmp_path):
    p = tmp_path / "ball_spot.json"
    p.write_text(json.dumps({"x": 1078, "y": 605, "r": 15.9, "px_per_mm": 0.745}))
    assert load_session_scale(p) == pytest.approx(0.745)


def test_session_scale_absent(tmp_path):
    p = tmp_path / "ball_spot.json"
    p.write_text(json.dumps({"x": 1078, "y": 605, "r": 30.0}))  # older spot files have no scale
    assert load_session_scale(p) is None
    assert load_session_scale(tmp_path / "missing.json") is None
