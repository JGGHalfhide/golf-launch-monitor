#!/usr/bin/env python3
"""
Send a hardcoded test shot to OpenGolfSim and print any events received.

Correct flow (per observed sim behaviour):
  1. Connect
  2. Send device "ready"
  3. Wait for a "player" event from the sim (sim is standing at tee, expecting a shot)
  4. Send the shot
  5. Wait for a "result" event

Usage:
    python tools/send_fake_shot.py [--host HOST] [--port PORT] [--no-wait]

Options:
    --no-wait   Send shot immediately without waiting for a "player" event
                (useful for diagnosing what the sim does on connection)

Defaults: host=127.0.0.1, port=3111
"""
import argparse
import json
import queue
import socket
import threading
import time


DEVICE_READY = {"type": "device", "status": "ready"}

SHOT = {
    "type": "shot",
    "unit": "imperial",
    "shot": {
        "ballSpeed": 150,
        "verticalLaunchAngle": 12,
        "horizontalLaunchAngle": 0,
        "spinSpeed": 2800,
        "spinAxis": 0,
    },
}


def send(sock: socket.socket, msg: dict) -> None:
    data = (json.dumps(msg) + "\n").encode("utf-8")
    sock.sendall(data)
    print(f"→ {json.dumps(msg)}")


def recv_lines(sock: socket.socket, event_q: queue.Queue, stop: threading.Event) -> None:
    """Read newline-delimited JSON from the socket and push parsed dicts onto event_q."""
    buf = ""
    sock.settimeout(0.5)
    while not stop.is_set():
        try:
            chunk = sock.recv(4096).decode("utf-8")
            if not chunk:
                print("[connection closed by sim]")
                break
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if line:
                    try:
                        msg = json.loads(line)
                        print(f"← {json.dumps(msg, indent=2)}")
                        event_q.put(msg)
                    except json.JSONDecodeError:
                        print(f"← [non-JSON] {line!r}")
        except socket.timeout:
            continue
        except OSError:
            break


def wait_for(event_q: queue.Queue, msg_type: str, timeout: float) -> dict | None:
    """Block until a message of the given type appears, or timeout expires."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            msg = event_q.get(timeout=min(remaining, 0.5))
            if msg.get("type") == msg_type:
                return msg
        except queue.Empty:
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a fake shot to OpenGolfSim")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3111)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Send shot immediately, don't wait for 'player' event",
    )
    args = parser.parse_args()

    print(f"Connecting to {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=5) as sock:
        print("Connected.")

        event_q: queue.Queue = queue.Queue()
        stop = threading.Event()
        reader = threading.Thread(target=recv_lines, args=(sock, event_q, stop), daemon=True)
        reader.start()

        time.sleep(0.1)
        send(sock, DEVICE_READY)

        if args.no_wait:
            time.sleep(0.3)
            send(sock, SHOT)
        else:
            print('\nWaiting up to 60 s for "player" event from sim ...')
            print("(In OpenGolfSim: make sure a round is active and the player is at the tee.)\n")
            player_event = wait_for(event_q, "player", timeout=60)
            if player_event is None:
                print(
                    '\n[timeout] No "player" event received. The sim may not be in an active round.\n'
                    "Re-run with --no-wait to send the shot anyway, or start a round in the sim first."
                )
                stop.set()
                reader.join(timeout=1)
                return
            print(f'\n"player" event received — club: {player_event.get("data", {}).get("club", "?")}')
            print("Sending shot ...\n")
            send(sock, SHOT)

        print("\nWaiting 20 s for result event ...")
        result = wait_for(event_q, "result", timeout=20)
        if result is None:
            print("[timeout] No result event received within 20 s.")
        else:
            print("\nShot result:")
            r = result.get("data", {}).get("result", {})
            for k, v in r.items():
                print(f"  {k}: {v} m")

        # Drain any remaining events
        time.sleep(1)
        stop.set()
        reader.join(timeout=1)

    print("Done.")


if __name__ == "__main__":
    main()
