import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

from relay.validation import validate_shot


class RelayHandler(BaseHTTPRequestHandler):
    # Injected by _make_handler_class
    sim_client = None
    shot_logger = None
    shots_sent = None   # threading.Value-like; use _counter_lock
    _counter_lock = None
    _events: deque = None
    _events_lock = None
    _current_club = None   # list[dict|None] — mutable container shared with on_event closure
    _club_lock = None

    # ------------------------------------------------------------------
    def do_POST(self):
        if self.path == "/shot":
            self._handle_shot()
        else:
            self._send_json(404, {"status": "error", "message": "Not found"})

    def do_GET(self):
        if self.path == "/status":
            self._handle_status()
        elif self.path == "/events":
            self._handle_events()
        else:
            self._send_json(404, {"status": "error", "message": "Not found"})

    # ------------------------------------------------------------------
    def _handle_shot(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            raw = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Invalid JSON body"})
            return

        try:
            shot, warnings = validate_shot(raw)
        except ValueError as e:
            self._send_json(400, {"status": "error", "message": str(e)})
            return

        try:
            self.sim_client.send_shot(shot)
        except RuntimeError as e:
            self._send_json(503, {"status": "error", "message": str(e)})
            return

        with self._counter_lock:
            self.__class__.shots_sent[0] += 1
            shot_id = self.__class__.shots_sent[0]

        record = {"event": "shot_sent", "shot_id": shot_id, "shot": shot}
        if warnings:
            record["warnings"] = warnings
        self.shot_logger.log(record)

        self._send_json(200, {"status": "sent", "shot_id": shot_id})

    def _handle_status(self):
        with self._counter_lock:
            count = self.__class__.shots_sent[0]
        with self._club_lock:
            club = self.__class__._current_club[0]
        self._send_json(200, {
            "sim_connected": self.sim_client.connected,
            "shots_sent": count,
            "current_club": club,
        })

    def _handle_events(self):
        with self._events_lock:
            snapshot = list(self._events)
        self._send_json(200, snapshot)

    # ------------------------------------------------------------------
    def _send_json(self, code: int, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # silence default access log
        pass


def _make_handler_class(sim_client, shot_logger):
    events: deque = deque(maxlen=20)
    events_lock = threading.Lock()
    counter_lock = threading.Lock()
    current_club: list = [None]   # mutable container so closure and Handler share the same ref
    club_lock = threading.Lock()

    def on_event(event: dict):
        shot_logger.log({"event": "sim_event", "data": event})
        with events_lock:
            events.append(event)
        if event.get("type") == "player":
            club = event.get("data", {}).get("club")
            if club:
                with club_lock:
                    current_club[0] = club

    sim_client.on_event = on_event

    class Handler(RelayHandler):
        pass

    Handler.sim_client = sim_client
    Handler.shot_logger = shot_logger
    Handler.shots_sent = [0]       # mutable int in list so class attr is shared
    Handler._counter_lock = counter_lock
    Handler._events = events
    Handler._events_lock = events_lock
    Handler._current_club = current_club
    Handler._club_lock = club_lock

    return Handler


def make_server(host: str, port: int, sim_client, shot_logger) -> HTTPServer:
    handler = _make_handler_class(sim_client, shot_logger)
    return HTTPServer((host, port), handler)
