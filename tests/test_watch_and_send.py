import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from watch_and_send import WatchState, find_new_clips, spoken_club, spoken_result  # noqa: E402


@pytest.mark.parametrize("club,expected", [
    ("pw", "P W"), ("sw", "S W"), ("7-iron", "7 iron"), ("3-wood", "3 wood"),
    ("driver", "driver"), (None, "unknown club"),
])
def test_spoken_club(club, expected):
    assert spoken_club(club) == expected


def test_spoken_result_rounds():
    assert spoken_result("pw", 76.34, 30.6) == "P W, 76 miles per hour, 31 degrees"


def test_find_new_clips_filters_and_orders(tmp_path):
    for i, name in enumerate(["b.MOV", "a.mp4", "notes.txt", ".part.mov", "seen.mov"]):
        p = tmp_path / name
        p.write_bytes(b"x")
        os.utime(p, (1000 + i, 1000 + i))
    (tmp_path / "done").mkdir()
    (tmp_path / "done" / "old.mov").write_bytes(b"x")
    assert [p.name for p in find_new_clips(tmp_path, {"seen.mov"})] == ["b.MOV", "a.mp4"]


def test_watch_state_persists(tmp_path):
    path = tmp_path / "state.json"
    s = WatchState(path)
    s.mark("clip1.mov")
    assert WatchState(path).handled == {"clip1.mov"}


from afc_source import is_original_video, newest_folders, stable_new_files  # noqa: E402


@pytest.mark.parametrize("name,ok", [
    ("IMG_0285.MOV", True), ("IMG_0285.mov", True), ("IMG_0285.MP4", True),
    ("IMG_E0285.MOV", False), ("IMG_0285.AAE", False), ("IMG_0285.JPG", False),
    (".IMG_0285.MOV", False),
])
def test_is_original_video(name, ok):
    assert is_original_video(name) is ok


def test_newest_folders_ignores_non_apple_entries():
    entries = ["100APPLE", ".MISC", "140APPLE", "ispRegDump.bin", "139APPLE", "99APPLE"]
    assert newest_folders(entries) == ["139APPLE", "140APPLE"]


def test_stable_new_files_waits_for_size_to_settle():
    known = {"/DCIM/140APPLE/IMG_0284.MOV"}
    prev = {"/DCIM/140APPLE/IMG_0284.MOV": 10, "/DCIM/140APPLE/IMG_0285.MOV": 500}
    now = {"/DCIM/140APPLE/IMG_0284.MOV": 10, "/DCIM/140APPLE/IMG_0285.MOV": 900,
           "/DCIM/140APPLE/IMG_0286.MOV": 700}
    assert stable_new_files(now, prev, known) == []          # 0285 still growing, 0286 just seen
    later = dict(now)
    assert stable_new_files(later, now, known) == ["/DCIM/140APPLE/IMG_0285.MOV", "/DCIM/140APPLE/IMG_0286.MOV"]


def test_stable_new_files_skips_empty():
    assert stable_new_files({"/a/IMG_0001.MOV": 0}, {"/a/IMG_0001.MOV": 0}, set()) == []
