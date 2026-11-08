"""Unit tests for shotmodel — interpolation, spin, contact quality, lateral generation."""

import numpy as np
import pytest

from shotmodel import SpeedHistory, estimate_shot, load_config, resolve_club, rolling_median
from shotmodel.core import (
    base_sd, club_baseline, estimate_spin, generate_lateral, interp_anchors,
    quality_score, range_at_handicap, speed_z, vla_z,
)

ANCHORS = [0, 5, 14, 24]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ── Interpolation ─────────────────────────────────────────────────────────────

def test_interp_hits_anchors_exactly():
    vals = [10, 20, 30, 40]
    for a, v in zip(ANCHORS, vals):
        assert interp_anchors(a, ANCHORS, vals) == v


def test_interp_between_anchors():
    assert interp_anchors(9.5, ANCHORS, [10, 20, 30, 40]) == pytest.approx(25.0)


def test_interp_clamps_below_and_above():
    vals = [10, 20, 30, 40]
    assert interp_anchors(-4, ANCHORS, vals) == 10
    assert interp_anchors(36, ANCHORS, vals) == 40


def test_range_at_handicap_mid_and_half():
    driver_vla = [[10.5, 12.0], [11.5, 13.5], [12.5, 14.5], [13.5, 16.5]]
    assert range_at_handicap(driver_vla, 14, ANCHORS) == pytest.approx((13.5, 1.0))
    assert range_at_handicap(driver_vla, 30, ANCHORS) == pytest.approx((15.0, 1.5))


def test_club_baseline_driver_hcp14(cfg):
    b = club_baseline("driver", 14, cfg)
    assert b.vla_mid == pytest.approx(13.5)
    assert b.spin_mid == pytest.approx(3000)
    assert b.disp_mid == pytest.approx(51.5)


def test_base_sd_anchors_and_interpolation(cfg):
    anchors = cfg["dispersion"]["sd_anchors"]
    assert base_sd("driver", cfg) == pytest.approx(tuple(anchors["driver"]))
    assert base_sd("7-iron", cfg) == pytest.approx(tuple(anchors["7-iron"]))
    hla5, axis5 = base_sd("5-iron", cfg)
    assert anchors["7-iron"][0] < hla5 < anchors["driver"][0]
    assert anchors["7-iron"][1] < axis5 < anchors["driver"][1]


def test_sd_scales_with_handicap(cfg):
    assert club_baseline("7-iron", 24, cfg).sd_hla > club_baseline("7-iron", 0, cfg).sd_hla


@pytest.mark.parametrize("name,expected", [
    ("DR", "driver"), ("Driver", "driver"), ("7i", "7-iron"), ("7 Iron", "7-iron"),
    ("PW", "pw"), ("Pitching Wedge", "pw"), ("LW", "sw"), ("3I", "4-iron"), ("Putter", None),
])
def test_resolve_club(cfg, name, expected):
    assert resolve_club(name, cfg) == expected


# ── Spin ──────────────────────────────────────────────────────────────────────

def test_spin_at_mid_vla_is_mid_spin():
    assert estimate_spin(18.0, 18.0, 6000, 3.0) == 6000


def test_spin_rises_with_vla_by_percentage():
    assert estimate_spin(20.0, 18.0, 6000, 3.0) == 6360  # +2° × 3% = +6%
    assert estimate_spin(16.0, 18.0, 6000, 3.0) == 5640


def test_spin_clamped():
    assert estimate_spin(45, 10, 11000, 5.0, max_rpm=12000) == 12000


# ── Contact quality ───────────────────────────────────────────────────────────

def test_perfect_shot_q_near_one():
    assert quality_score(vla_z(18, 18, 1.5), speed_z(100, 100, 0.05)) == pytest.approx(1.0)


def test_big_speed_drop_gives_low_q():
    zs = speed_z(85, 100, 0.05)  # 15% slow = 3 sigma
    assert quality_score(0.0, zs) < 0.1


def test_high_speed_not_penalised():
    assert speed_z(110, 100, 0.05) == 0.0
    assert quality_score(0.0, speed_z(110, 100, 0.05)) == 1.0


def test_vla_deviation_two_sided():
    assert vla_z(21, 18, 1.5) == pytest.approx(2.0)
    assert vla_z(15, 18, 1.5) == pytest.approx(-2.0)
    assert quality_score(2.0, 0) == quality_score(-2.0, 0) < 1.0


def test_speed_weighted_more_than_vla():
    assert quality_score(0.0, 1.0) < quality_score(1.0, 0.0)


def test_no_median_skips_speed_term():
    assert speed_z(60, None, 0.05) == 0.0


def test_rolling_median_warmup_and_window():
    assert rolling_median([100.0] * 9, warmup=10, window=30) is None
    speeds = [50.0] * 20 + [100.0] * 30
    assert rolling_median(speeds, warmup=10, window=30) == 100.0


def test_speed_history_persists(tmp_path):
    path = tmp_path / "h.json"
    h = SpeedHistory(path)
    h.add("7-iron", 101.234)
    h.save()
    assert SpeedHistory(path).speeds("7-iron") == [101.23]


# ── Lateral generation ────────────────────────────────────────────────────────

def _draws(q, n=4000, seed=0, **kw):
    rng = np.random.default_rng(seed)
    params = dict(sd_hla=1.5, sd_axis=6.0, bias_hla=0.0, bias_axis=0.0,
                  k=1.25, rho=0.3, tail_prob=0.1, tail_mult=2.0) | kw
    return np.array([generate_lateral(q, rng=rng, **params)[:2] for _ in range(n)])


def test_generation_is_reproducible_with_seed():
    assert np.array_equal(_draws(0.8, n=50, seed=7), _draws(0.8, n=50, seed=7))


def test_spread_widens_as_q_drops():
    sd_hi = _draws(1.0).std(axis=0)
    sd_mid = _draws(0.6).std(axis=0)
    sd_lo = _draws(0.1).std(axis=0)
    assert (sd_lo > sd_mid).all() and (sd_mid > sd_hi).all()


def test_bias_shifts_centre():
    d = _draws(1.0, bias_hla=1.0, bias_axis=3.0, tail_prob=0.0)
    assert d[:, 0].mean() == pytest.approx(1.0, abs=0.1)
    assert d[:, 1].mean() == pytest.approx(3.0, abs=0.3)


def test_hla_axis_correlated():
    d = _draws(1.0, rho=0.3, tail_prob=0.0, n=8000)
    assert np.corrcoef(d[:, 0], d[:, 1])[0, 1] == pytest.approx(0.3, abs=0.05)


def test_tail_probability():
    rng = np.random.default_rng(3)
    tails = [generate_lateral(1.0, 1, 1, 0, 0, 1.25, 0.3, 0.1, 2.0, rng)[2] for _ in range(5000)]
    assert np.mean(tails) == pytest.approx(0.1, abs=0.015)


def test_estimate_shot_end_to_end(cfg):
    rng = np.random.default_rng(1)
    est = estimate_shot("7-iron", vla=19.5, speed=100.0, cfg=cfg, speed_median=100.0, rng=rng, handicap=14)
    assert est.q == pytest.approx(1.0)
    assert est.spin == 5400
    shot = est.sim_shot()
    assert set(shot) == {"ballSpeed", "verticalLaunchAngle", "horizontalLaunchAngle", "spinSpeed", "spinAxis"}


def test_left_handed_mirrors_bias(cfg):
    lefty = cfg | {"player": cfg["player"] | {"handedness": "left"},
                   "dispersion": cfg["dispersion"] | {"tail_prob": 0.0, "k": 0.0}}
    axes = [estimate_shot("7-iron", 19.5, 100, lefty, 100, np.random.default_rng(i)).axis for i in range(2000)]
    assert np.mean(axes) == pytest.approx(-cfg["bias"]["axis_deg"], abs=0.4)
