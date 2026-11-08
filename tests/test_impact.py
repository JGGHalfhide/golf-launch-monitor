import numpy as np
import pytest

from vision.impact import (
    impact_from_signal,
    resting_clusters,
    track_is_plausible,
    vanish_score,
)


def ball_signal(n=600, impact=400, rest=3.0, mat=38.0, noise=0.5, seed=0):
    """Ball at rest (low diff) until `impact`, then bare mat (steady high diff)."""
    rng = np.random.default_rng(seed)
    s = np.full(n, mat)
    s[: impact + 1] = rest
    return s + rng.normal(0, noise, n)


# --- impact_from_signal ------------------------------------------------------

def test_impact_is_last_resting_frame():
    assert impact_from_signal(ball_signal(impact=400)) == 400


def test_shoe_covering_spot_late_does_not_count_as_ball():
    # IMG_0285 regression: bare mat (~38) after impact, then a shoe (~95) at the end.
    # The end of the clip is not "ball gone" baseline; mat must still count as gone.
    s = ball_signal(n=1000, impact=400)
    s[900:] = 95.0
    assert impact_from_signal(s) == 400


def test_brief_lookalike_after_impact_ignored():
    s = ball_signal(impact=400)
    s[500:510] = 3.0  # something ball-like passes for 10 frames (< 0.2 s)
    assert impact_from_signal(s) == 400


def test_waggle_before_impact_ignored():
    s = ball_signal(impact=400)
    s[100:104] = 40.0  # club briefly occludes the ball at address
    assert impact_from_signal(s) == 400


def test_ball_never_leaves_returns_none():
    assert impact_from_signal(np.full(600, 3.0)) is None


def test_ball_leaves_too_close_to_end_returns_none():
    assert impact_from_signal(ball_signal(n=600, impact=595)) is None


# --- vanish_score ------------------------------------------------------------

def test_struck_ball_scores_far_above_gradual_change():
    ball = ball_signal(impact=400)
    gradual = np.concatenate([np.full(400, 3.0), np.linspace(3, 55, 30), np.full(170, 55.0)])
    assert vanish_score(ball, 400) > 5 * vanish_score(gradual, 400)


def test_spike_that_returns_scores_zero():
    s = np.full(600, 4.0)
    s[401] = 90.0  # one-frame occlusion, then the spot looks the same again
    s[402:] = 8.0
    assert vanish_score(s, 400) == 0.0


def test_no_change_scores_zero():
    assert vanish_score(np.full(600, 4.0), 400) == 0.0


# --- resting_clusters --------------------------------------------------------

def test_clusters_group_same_spot_and_split_others():
    dets = [(i, 845 + (i % 3), 578, 24, 0.4) for i in range(0, 120, 6)]   # ball, 20 samples
    dets += [(i, 1075, 411, 18, 0.5) for i in range(0, 60, 6)]            # background, 10
    dets += [(30, 300, 300, 10, 0.4)]                                      # noise, 1
    cl = resting_clusters(dets)
    assert len(cl) == 2
    assert cl[0].cx == pytest.approx(846, abs=1) and len(cl[0].frames) == 20
    assert cl[1].cx == pytest.approx(1075) and len(cl[1].frames) == 10


def test_clusters_need_min_support():
    assert resting_clusters([(0, 10, 10, 5, 0.5), (6, 10, 10, 5, 0.5)]) == []


# --- track_is_plausible ------------------------------------------------------

def test_real_flight_is_plausible():
    # IMG_0285: ~205 px/frame, ball ≈ 59 px across at 1.378 px/mm
    assert track_is_plausible([(927, 1077, 427), (929, 1423, 209), (930, 1593, 100)], 1.378)


def test_static_junk_track_rejected():
    # IMG_0285 shoe-logo "track": barely moves after the first jump
    pos = [(1531, 1258, 6), (1532, 1256, 8), (1533, 1256, 8), (1534, 1256, 9)]
    assert not track_is_plausible(pos, 0.415)


def test_single_detection_not_plausible():
    assert not track_is_plausible([(10, 100, 100)], 1.5)


def test_impossible_speed_rejected():
    # IMG_0292 regression: two unrelated blobs 1100 px apart in one frame read as 1111 mph
    assert not track_is_plausible([(993, 277, 150), (994, 1406, 91)], 0.546)


def test_fast_real_shot_still_plausible():
    # ~170 mph driver at 1.0 px/mm: 170 * 447.04 / 240 ≈ 317 px/frame — under the 220 mph cap
    assert track_is_plausible([(0, 100, 500), (1, 417, 480), (2, 734, 460)], 1.0)


# --- rank_spots --------------------------------------------------------------

from vision.impact import RestCluster, rank_spots  # noqa: E402


def _c(n):
    return RestCluster(0, 0, 10, 0.5, list(range(0, 6 * n, 6)))


def test_spots_vanishing_together_are_marked_down():
    # The follow-through moves feet: many spots change in the same frame; the ball vanishes alone
    ball = (_c(150), 945, 20.0)
    feet = [(_c(150), 1377 + d, 40.0) for d in (0, 0, 1, -1, 2)]
    ranked = rank_spots([ball] + feet)
    assert ranked[0][1] == 945


def test_briefly_resting_spots_are_marked_down():
    ball = (_c(150), 945, 20.0)
    blip = (_c(8), 400, 60.0)   # seen in only 8 samples (0.2 s)
    assert rank_spots([blip, ball])[0][1] == 945
