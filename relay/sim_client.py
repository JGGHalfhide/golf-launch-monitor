import json
import socket
import threading
import time
import logging

log = logging.getLogger(__name__)

RECONNECT_DELAY = 5  # seconds


class SimClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 3111):
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._connected = False
        self._running = False
        self._on_event = None
        self._reader_thread: threading.Thread | None = None
        self._connect_thread: threading.Thread | None = None

    @property
    def on_event(self):
        return self._on_event

    @on_event.setter
    def on_event(self, callback):
        self._on_event = callback

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """Start background connection + reader loop."""
        self._running = True
        self._connect_thread = threading.Thread(target=self._connection_loop, daemon=True, name="sim-connect")
        self._connect_thread.start()

    def disconnect(self) -> None:
        self._running = False
        self._close_socket()

    def send_shot(self, shot: dict) -> None:
        """Send device-ready then the shot. Raises RuntimeError if not connected."""
        self._send_msg({"type": "device", "status": "ready"})
        time.sleep(0.2)
        self._send_msg({"type": "shot", "unit": "imperial", "shot": shot})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_msg(self, msg: dict) -> None:
        with self._lock:
            if not self._connected or self._sock is None:
                raise RuntimeError("Not connected to sim")
            data = (json.dumps(msg) + "\n").encode("utf-8")
            self._sock.sendall(data)

    def _close_socket(self) -> None:
        with self._lock:
            self._connected = False
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def _connection_loop(self) -> None:
        """Keeps trying to connect; on success, starts reader; on drop, retries."""
        while self._running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((self._host, self._port))
                sock.settimeout(None)  # blocking reads in reader thread
                with self._lock:
                    self._sock = sock
                    self._connected = True
                log.info("Connected to sim at %s:%s", self._host, self._port)
                self._read_loop(sock)  # blocks until connection drops
            except OSError as e:
                log.warning("Sim connection failed: %s — retrying in %ss", e, RECONNECT_DELAY)
            finally:
                self._close_socket()

            if self._running:
                time.sleep(RECONNECT_DELAY)

    def _read_loop(self, sock: socket.socket) -> None:
        """Read newline-terminated JSON lines until the socket closes."""
        buf = b""
        try:
            while self._running:
                chunk = sock.recv(4096)
                if not chunk:
                    log.info("Sim closed the connection")
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._dispatch(line.decode("utf-8", errors="replace"))
        except OSError as e:
            if self._running:
                log.warning("Sim read error: %s", e)

    def _dispatch(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            log.warning("Non-JSON from sim: %r", line)
            return
        log.debug("Sim event: %s", event)
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                log.exception("on_event callback raised")
