"""Per-club ball-speed history, persisted to JSON so the rolling median survives restarts."""

import json
from pathlib import Path


class SpeedHistory:
    def __init__(self, path: str | Path, max_len: int = 200):
        self._path = Path(path)
        self._max_len = max_len
        self._data: dict[str, list[float]] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text(encoding="utf-8"))

    def speeds(self, club: str) -> list[float]:
        return list(self._data.get(club, []))

    def add(self, club: str, speed: float) -> None:
        self._data.setdefault(club, []).append(round(float(speed), 2))
        self._data[club] = self._data[club][-self._max_len:]

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1), encoding="utf-8")
        tmp.replace(self._path)
