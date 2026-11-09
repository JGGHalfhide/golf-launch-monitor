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


def test_low_contrast_mat_drifting_toward_ball_level_still_gone():
    # IMG_0338 regression (dusk, dewy mat): ball ~1, bare mat only ~23 after impact, drifting
    # down to ~11 as the light fades. A fixed 15-level margin read the tail as "ball back".
    n, impact = 1966, 1021
    s = ball_signal(n=n, impact=impact, rest=1.0, mat=23.0, noise=0.3)
    s[1600:] = np.linspace(21.0, 11.0, n - 1600)
    assert impact_from_signal(s) == impact


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


def test_clubhead_overshoot_still_counts_as_ball():
    # IMG_0311/0324: a big dark clubhead covers the spot for a frame at impact (spike to ~100),
    # then bare mat (~45). This was scored 0 and the shot was lost.
    s = ball_signal(impact=400, rest=2.0, mat=45.0)
    s[401:403] = 100.0
    assert vanish_score(s, 400) > 30


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


def test_steep_launch_rejected():
    # Plausible speed, but ~79° up: no full shot launches that steeply (IMG_0292 misread)
    assert not track_is_plausible([(0, 500, 500), (1, 520, 400), (2, 540, 300)], 1.2)


def test_backward_launch_rejected():
    assert not track_is_plausible([(0, 500, 500), (1, 400, 450), (2, 300, 400)], 1.2)


# --- rank_spots --------------------------------------------------------------

from vision.impact import RestCluster, rank_spots  # noqa: E402


def _c(n):
    return RestCluster(0, 0, 10, 0.5, list(range(0, 6 * n, 6)))


def test_spots_vanishing_together_are_marked_down():
    # The follow-through moves feet: many spots change in the same frame; the ball vanishes alone
    ball = (_c(150), 945, 20.0, 900)
    feet = [(_c(150), 1377 + d, 40.0, 900) for d in (0, 0, 1, -1, 2)]
    ranked = rank_spots([ball] + feet)
    assert ranked[0][1] == 945


def test_briefly_resting_spots_are_marked_down():
    ball = (_c(150), 945, 20.0, 900)
    blip = (_c(150), 400, 60.0, 48)   # rested only 0.2 s
    assert rank_spots([blip, ball])[0][1] == 945




# --- hole_shape / is_ball_hole -------------------------------------------------

import cv2  # noqa: E402

from vision.impact import hole_shape, is_ball_hole  # noqa: E402


def _frames_with_change(draw_before, n=30, impact=10):
    """Small grayscale frames: mat (100) with something drawn until `impact`, gone after."""
    frames = []
    for i in range(n):
        f = np.full((270, 480), 100, np.uint8)
        if i <= impact:
            draw_before(f)
        frames.append(f)
    return frames


def test_struck_ball_leaves_round_ball_sized_hole():
    frames = _frames_with_change(lambda f: cv2.circle(f, (200, 150), 8, 230, -1))
    shape = hole_shape(frames, 10, 200, 150)
    assert shape is not None and shape[0] == pytest.approx(8, abs=1.5)
    assert is_ball_hole(shape, scale=4)          # radius 32 px at full res


def test_irregular_change_is_not_a_ball_hole():
    # e.g. a shoe turning: long thin change
    frames = _frames_with_change(lambda f: cv2.rectangle(f, (170, 145), (230, 152), 30, -1))
    assert not is_ball_hole(hole_shape(frames, 10, 200, 148), scale=4)


def test_huge_change_is_not_a_ball_hole():
    # IMG_0310: a shadow/body moving over the mat changes a region ~130 px in radius
    frames = _frames_with_change(lambda f: cv2.circle(f, (200, 150), 33, 30, -1))
    assert not is_ball_hole(hole_shape(frames, 10, 200, 150), scale=4)


def test_no_change_no_hole():
    frames = [np.full((270, 480), 100, np.uint8) for _ in range(30)]
    assert hole_shape(frames, 10, 200, 150) is None


def test_track_must_start_at_the_resting_spot():
    # IMG_0318: junk near the top of the frame, nowhere near the ball's spot (898, 772)
    junk = [(1513, 1289, 215), (1516, 1336, 45)]
    assert not track_is_plausible(junk, 0.505, address=(898, 772), impact=1509)


def test_real_tracks_start_at_the_resting_spot():
    # IMG_0278 and IMG_0311 (low skimmer) extrapolate back to their spots
    assert track_is_plausible([(508, 1316, 497), (509, 1724, 265)], 1.644, address=(1017, 661), impact=507)
    assert track_is_plausible([(897, 961, 768), (898, 1325, 758), (899, 1659, 733)], 1.404,
                              address=(786, 770), impact=896)
