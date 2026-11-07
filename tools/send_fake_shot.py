#!/usr/bin/env python3
"""
Send a hardcoded test shot to OpenGolfSim and print any events received.

Usage:
    python tools/send_fake_shot.py [--host HOST] [--port PORT]

Defaults: host=127.0.0.1, port=3111
"""
import argparse
import json
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


def listen(sock: socket.socket, stop_event: threading.Event) -> None:
    buf = ""
    sock.settimeout(0.5)
    while not stop_event.is_set():
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
                    except json.JSONDecodeError:
                        print(f"← [non-JSON] {line!r}")
        except socket.timeout:
            continue
        except OSError:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a fake shot to OpenGolfSim")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3111)
    args = parser.parse_args()

    print(f"Connecting to {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=5) as sock:
        print("Connected.")

        stop = threading.Event()
        listener = threading.Thread(target=listen, args=(sock, stop), daemon=True)
        listener.start()

        # Small delay so listener is ready before we send
        time.sleep(0.1)

        send(sock, DEVICE_READY)
        time.sleep(0.2)
        send(sock, SHOT)

        print("\nWaiting 15 s for result event (watch the sim) ...")
        time.sleep(15)

        stop.set()
        listener.join(timeout=1)

    print("Done.")


if __name__ == "__main__":
    main()
