import pytest
from relay.validation import validate_shot


VALID_SHOT = {
    "ballSpeed": 150.0,
    "verticalLaunchAngle": 12.0,
    "horizontalLaunchAngle": 0.0,
    "spinSpeed": 2800.0,
    "spinAxis": 0.0,
}


def test_valid_shot_passes_through_unchanged():
    shot, warnings = validate_shot(VALID_SHOT)
    assert shot["ballSpeed"] == 150.0
    assert shot["verticalLaunchAngle"] == 12.0
    assert shot["horizontalLaunchAngle"] == 0.0
    assert shot["spinSpeed"] == 2800.0
    assert shot["spinAxis"] == 0.0
    assert warnings == []


def test_vertical_launch_angle_clamped_low():
    raw = {**VALID_SHOT, "verticalLaunchAngle": -5.0}
    shot, warnings = validate_shot(raw)
    assert shot["verticalLaunchAngle"] == 0.0
    assert any("verticalLaunchAngle" in w for w in warnings)


def test_vertical_launch_angle_clamped_high():
    raw = {**VALID_SHOT, "verticalLaunchAngle": 90.0}
    shot, warnings = validate_shot(raw)
    assert shot["verticalLaunchAngle"] == 45.0
    assert any("verticalLaunchAngle" in w for w in warnings)


def test_horizontal_launch_angle_clamped():
    raw = {**VALID_SHOT, "horizontalLaunchAngle": -60.0}
    shot, warnings = validate_shot(raw)
    assert shot["horizontalLaunchAngle"] == -45.0
    assert any("horizontalLaunchAngle" in w for w in warnings)


def test_spin_speed_clamped_high():
    raw = {**VALID_SHOT, "spinSpeed": 25000.0}
    shot, warnings = validate_shot(raw)
    assert shot["spinSpeed"] == 20000.0
    assert any("spinSpeed" in w for w in warnings)


def test_spin_speed_clamped_low():
    raw = {**VALID_SHOT, "spinSpeed": -100.0}
    shot, warnings = validate_shot(raw)
    assert shot["spinSpeed"] == 0.0
    assert any("spinSpeed" in w for w in warnings)


def test_spin_axis_clamped():
    raw = {**VALID_SHOT, "spinAxis": 50.0}
    shot, warnings = validate_shot(raw)
    assert shot["spinAxis"] == 45.0
    assert any("spinAxis" in w for w in warnings)


def test_zero_ball_speed_raises():
    raw = {**VALID_SHOT, "ballSpeed": 0.0}
    with pytest.raises(ValueError, match="ballSpeed must be > 0"):
        validate_shot(raw)


def test_negative_ball_speed_raises():
    raw = {**VALID_SHOT, "ballSpeed": -10.0}
    with pytest.raises(ValueError, match="ballSpeed must be > 0"):
        validate_shot(raw)


def test_missing_ball_speed_raises():
    raw = {k: v for k, v in VALID_SHOT.items() if k != "ballSpeed"}
    with pytest.raises(ValueError, match="Missing required field: ballSpeed"):
        validate_shot(raw)


def test_missing_other_field_raises():
    raw = {k: v for k, v in VALID_SHOT.items() if k != "spinAxis"}
    with pytest.raises(ValueError, match="Missing required field: spinAxis"):
        validate_shot(raw)


def test_multiple_clamps_produce_multiple_warnings():
    raw = {
        "ballSpeed": 100.0,
        "verticalLaunchAngle": -1.0,
        "horizontalLaunchAngle": 90.0,
        "spinSpeed": 30000.0,
        "spinAxis": -90.0,
    }
    shot, warnings = validate_shot(raw)
    assert shot["verticalLaunchAngle"] == 0.0
    assert shot["horizontalLaunchAngle"] == 45.0
    assert shot["spinSpeed"] == 20000.0
    assert shot["spinAxis"] == -45.0
    assert len(warnings) == 4


def test_boundary_values_not_clamped():
    raw = {
        "ballSpeed": 0.001,
        "verticalLaunchAngle": 0.0,
        "horizontalLaunchAngle": -45.0,
        "spinSpeed": 20000.0,
        "spinAxis": 45.0,
    }
    shot, warnings = validate_shot(raw)
    assert warnings == []
    assert shot["verticalLaunchAngle"] == 0.0
    assert shot["horizontalLaunchAngle"] == -45.0
    assert shot["spinSpeed"] == 20000.0
    assert shot["spinAxis"] == 45.0
