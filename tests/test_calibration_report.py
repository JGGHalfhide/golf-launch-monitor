import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from calibration_report import summarise, target_key  # noqa: E402

TARGETS = {"7-iron": {"carry": 172.6}, "sw": {"carry": 98.7}}


def shot(id_, club, speed, carry):
    return {"id": id_, "club": club, "speed": speed, "vla": 16.0, "spin": 4500, "carry": carry, "lateral": 0.0}


def test_excluded_misread_does_not_count():
    # 2026-11-08: 14.1 mph on a full, clean 7i — must not drag the average
    results = [shot("a", "7-iron", 107.3, 132), shot("b", "7-iron", 14.1, 2), shot("c", "7-iron", 106.1, 128)]
    row = summarise(results, {}, TARGETS, excluded={"b": "misread"})[0]
    assert row["n"] == 2 and row["excl"] == 1
    assert row["speed"] == pytest.approx(106.7)
    assert row["delta_pct"] == pytest.approx(100 * (130 / 172.6 - 1))


def test_failures_and_clubs_without_valid_shots_still_listed():
    rows = summarise([shot("a", "7-iron", 100, 120)], {"sw": 2}, TARGETS)
    assert [r["club"] for r in rows] == ["7-iron", "sw"]
    assert rows[1]["n"] == 0 and rows[1]["fail"] == 2


@pytest.mark.parametrize("sim,key", [("7i", "7-iron"), ("7I", "7-iron"), ("3W", "3-wood"), ("DR", "driver"),
                                     ("SW", "sw"), (None, None)])
def test_sim_club_names_map_to_targets(sim, key):
    assert target_key(sim) == key
