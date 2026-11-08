import pytest

from vision.clipinfo import measured_fps


def test_regular_240fps():
    ts = [i * 1000 / 240 for i in range(100)]
    assert measured_fps(ts) == pytest.approx(240.0)


def test_30fps_transcode_detected():
    ts = [i * 1000 / 30 for i in range(30)]
    assert measured_fps(ts) == pytest.approx(30.0)


def test_median_ignores_jitter_and_duplicates():
    step = 1000 / 240
    ts = [0, step, 2 * step, 2 * step, 3 * step, 3 * step + 40, 3 * step + 40 + step]
    assert measured_fps(ts) == pytest.approx(240.0)


@pytest.mark.parametrize("ts", [[], [5.0], [3.0, 3.0, 3.0]])
def test_degenerate_returns_none(ts):
    assert measured_fps(ts) is None


from vision.pipeline import _rescale_from_flight  # noqa: E402


def test_scale_taken_from_flight_when_address_measured_short():
    # IMG_0294: radius 20 at address (shaded underside lost), 35 in flight
    r = {"px_per_mm": 2 * 19.9 / 42.67, "ball_speed_mph": 123.89, "flight_radii": [16.7, 35.1], "warnings": []}
    _rescale_from_flight(r)
    assert r["ball_speed_mph"] == pytest.approx(123.89 * 19.9 / 35.1, rel=1e-3)
    assert r["warnings"]


def test_scale_kept_when_address_and_flight_agree():
    # IMG_0285: 28.8 at address, partial flight detections read smaller — leave it alone
    r = {"px_per_mm": 1.35, "ball_speed_mph": 81.0, "flight_radii": [22.8, 27.9, 26.5], "warnings": []}
    _rescale_from_flight(r)
    assert r["ball_speed_mph"] == 81.0 and not r["warnings"]


def test_huge_flight_blob_ignored_for_scale():
    # IMG_0316: a 230 px "ball" in flight vs 32 at address rescaled 7x and sent 14 mph
    r = {"px_per_mm": 2 * 32 / 42.67, "ball_speed_mph": 98.0, "flight_radii": [230.0, 30.0], "warnings": []}
    _rescale_from_flight(r)
    assert r["ball_speed_mph"] == 98.0 and not r["warnings"]
