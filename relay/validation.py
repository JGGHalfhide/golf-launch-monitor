"""
Shot validation and clamping logic.
Importable by the vision pipeline and any other source.
"""

REQUIRED_FIELDS = ("ballSpeed", "verticalLaunchAngle", "horizontalLaunchAngle", "spinSpeed", "spinAxis")


def validate_shot(raw: dict) -> tuple[dict, list[str]]:
    """
    Validate and clamp shot fields.

    Returns (cleaned_shot, warnings).
    Raises ValueError for unrecoverable errors (missing or invalid ballSpeed).
    """
    warnings: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in raw:
            raise ValueError(f"Missing required field: {field}")

    try:
        ball_speed = float(raw["ballSpeed"])
    except (TypeError, ValueError):
        raise ValueError("ballSpeed must be a number")

    if ball_speed <= 0:
        raise ValueError(f"ballSpeed must be > 0, got {ball_speed}")

    def clamp(value, lo, hi, name):
        v = float(value)
        if v < lo:
            warnings.append(f"{name} clamped from {v} to {lo}")
            return lo
        if v > hi:
            warnings.append(f"{name} clamped from {v} to {hi}")
            return hi
        return v

    shot = {
        "ballSpeed": ball_speed,
        "verticalLaunchAngle": clamp(raw["verticalLaunchAngle"], 0, 45, "verticalLaunchAngle"),
        "horizontalLaunchAngle": clamp(raw["horizontalLaunchAngle"], -45, 45, "horizontalLaunchAngle"),
        "spinSpeed": clamp(raw["spinSpeed"], 0, 20000, "spinSpeed"),
        "spinAxis": clamp(raw["spinAxis"], -45, 45, "spinAxis"),
    }

    return shot, warnings
