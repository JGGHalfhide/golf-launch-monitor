import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class ShotLogger:
    def __init__(self, path: str):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        line = json.dumps(entry) + "\n"
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)
