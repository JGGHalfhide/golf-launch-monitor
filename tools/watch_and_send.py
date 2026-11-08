#!/usr/bin/env python3
"""
Continuous shot pipeline for a round: new clip → vision → shotmodel → relay → sim → spoken result.

Usage:
    python tools/watch_and_send.py                   # USB: iPhone camera roll over AFC (tools/afc_source.py)
    python tools/watch_and_send.py --source dir      # just watch the inbox (iCloud/osxphotos export, manual drop)
    python tools/watch_and_send.py --dry-run --no-speak

Sources all land clips in --inbox (default inbox/). Each clip is processed exactly once:
handled names persist in logs/watch_state.json, and clips already in the inbox when the
watcher starts are ignored. Processed clips move to inbox/done/, failures to inbox/failed/.
The club comes from the relay's tracked sim club. Every result and failure is spoken.
"""
import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from afc_source import AfcSource  # noqa: E402
from clip_to_sim import ShotFailed, process_clip  # noqa: E402
from shotmodel import load_config  # noqa: E402

VIDEO_EXTS = {".mov", ".mp4", ".m4v"}
FAIL_SOUND = "/System/Library/Sounds/Basso.aiff"
READY_SOUND = "/System/Library/Sounds/Glass.aiff"


# ---------------------------------------------------------------------------
# Pure helpers (tested in tests/test_watch_and_send.py)
# ---------------------------------------------------------------------------

def spoken_club(club: str | None) -> str:
    """Config club key → something `say` pronounces well: 'pw' → 'P W', '7-iron' → '7 iron'."""
    if not club:
        return "unknown club"
    words = club.replace("-", " ").replace("_", " ").split()
    if len(words) == 1 and len(words[0]) <= 2 and words[0].isalpha():
        return " ".join(words[0].upper())
    return " ".join(words)


def spoken_result(club: str | None, speed_mph: float, vla_deg: float) -> str:
    return f"{spoken_club(club)}, {speed_mph:.0f} miles per hour, {vla_deg:.0f} degrees"


def find_new_clips(inbox: Path, handled: set[str]) -> list[Path]:
    """Video files directly in inbox (not hidden, not in subdirs) that haven't been handled, oldest first."""
    clips = [
        p for p in inbox.iterdir()
        if p.is_file() and not p.name.startswith(".")
        and p.suffix.lower() in VIDEO_EXTS and p.name not in handled
    ]
    return sorted(clips, key=lambda p: (p.stat().st_mtime, p.name))


class WatchState:
    """Persisted set of handled clip names — guarantees exactly-once processing across restarts."""

    def __init__(self, path: Path):
        self.path = path
        self.handled: set[str] = set()
        if path.exists():
            self.handled = set(json.loads(path.read_text()).get("handled", []))

    def mark(self, name: str) -> None:
        self.handled.add(name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"handled": sorted(self.handled)}, indent=1))
        tmp.replace(self.path)


# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------

class Speaker:
    """Serialises `say` calls on a background thread so speech never blocks processing."""

    def __init__(self, enabled: bool, voice: str | None):
        self.enabled = enabled
        self.voice = voice
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text: str, sound: str | None = None) -> None:
        print(f"🔊 {text}")
        if self.enabled:
            self.q.put((text, sound))

    def drain(self, timeout: float = 15) -> None:
        end = time.time() + timeout
        while not self.q.empty() and time.time() < end:
            time.sleep(0.1)
        time.sleep(0.5)

    def _run(self) -> None:
        while True:
            text, sound = self.q.get()
            if sound:
                subprocess.run(["afplay", sound], check=False)
            cmd = ["say"] + (["-v", self.voice] if self.voice else []) + [text]
            subprocess.run(cmd, check=False)


# ---------------------------------------------------------------------------
# USB source: AFC (tools/afc_source.py)
# ---------------------------------------------------------------------------

def start_usb(inbox: Path, speaker: Speaker) -> AfcSource:
    def on_event(kind: str, **ev) -> None:
        name = ev.get("name", "")
        if kind == "connected":
            print(f"[usb] {name} connected ({ev.get('existing')} existing clips ignored)")
            speaker.say("iPhone connected. Ready.", READY_SOUND)
        elif kind == "waiting":
            print(f"[usb] waiting for iPhone ({ev.get('message')}) — plug in, unlock, Trust")
            speaker.say("Plug in and unlock the iPhone.")
        elif kind == "disconnected":
            print(f"[usb] disconnected: {ev.get('message')}")
            speaker.say("iPhone disconnected.", FAIL_SOUND)
        elif kind == "downloading":
            print(f"[usb] new clip {name} ({ev.get('bytes', 0) / 1e6:.1f} MB) — downloading")
        elif kind == "downloaded":
            print(f"[usb] downloaded {name} in {ev.get('seconds', 0):.1f}s")
        elif kind == "error":
            print(f"[usb] ERROR {ev.get('stage')} {name}: {ev.get('message')}")
            speaker.say(f"Transfer failed for {name}. Re-hit or play it manually.", FAIL_SOUND)

    src = AfcSource(inbox, on_event)
    src.start()
    return src


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def wait_until_stable(path: Path, settle: float) -> bool:
    """For sources that write in place: wait until size stops changing. Returns False if the file vanished."""
    last = -1
    while True:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last and size > 0:
            return True
        last = size
        time.sleep(settle)


def move_to(path: Path, sub: str) -> Path:
    dest_dir = path.parent / sub
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / path.name
    shutil.move(str(path), dest)
    return dest


def short_reason(err: ShotFailed) -> str:
    msg = str(err).lower()
    if "relay" in msg:
        return "Relay not reachable."
    if "fps" in msg:
        return "Clip is not slow motion. Set the camera to slow mo."
    return "Ball not tracked."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["usb", "dir"], default="usb")
    parser.add_argument("--inbox", default=str(ROOT / "inbox"))
    parser.add_argument("--state", default=str(ROOT / "logs" / "watch_state.json"))
    parser.add_argument("--relay", default="http://localhost:3001")
    parser.add_argument("--config", default=None)
    parser.add_argument("--handicap", type=float, default=None)
    parser.add_argument("--club", default=None, help="Force club (default: tracked from sim via relay)")
    parser.add_argument("--min-fps", type=float, default=200.0, help="Reject clips below this measured fps")
    parser.add_argument("--poll", type=float, default=0.5, help="Inbox poll interval (s)")
    parser.add_argument("--voice", default=None, help="macOS `say` voice")
    parser.add_argument("--no-speak", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Process but don't send to relay")
    args = parser.parse_args()

    inbox = Path(args.inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    state = WatchState(Path(args.state))
    speaker = Speaker(enabled=not args.no_speak, voice=args.voice)

    # Anything already sitting in the inbox predates this session: skip it.
    preexisting = {p.name for p in find_new_clips(inbox, state.handled)}
    if preexisting:
        print(f"Ignoring {len(preexisting)} clip(s) already in {inbox}")
    skip = state.handled | preexisting

    usb = start_usb(inbox, speaker) if args.source == "usb" else None
    print(f"Watching {inbox} (source: {args.source}{', DRY RUN' if args.dry_run else ''}). Ctrl-C to stop.")
    if args.source == "dir":
        speaker.say("Watching for shots.", READY_SOUND)

    shots = 0
    try:
        while True:
            if usb is not None and not usb.is_alive():
                speaker.say("Phone watcher stopped. Restart the shot pipeline.", FAIL_SOUND)
                speaker.drain()
                return 1

            for clip in find_new_clips(inbox, skip):
                skip.add(clip.name)
                if args.source == "dir" and not wait_until_stable(clip, settle=1.0):
                    continue
                state.mark(clip.name)  # mark before processing: a crash mid-shot must not re-send it
                print(f"\n=== {clip.name} ===")
                t0 = time.time()
                try:
                    s = process_clip(str(clip), cfg, relay_url=args.relay, club_name=args.club,
                                     handicap=args.handicap, dry_run=args.dry_run, min_fps=args.min_fps)
                except ShotFailed as e:
                    print(f"SHOT FAILED: {e}", file=sys.stderr)
                    speaker.say(f"Shot failed. {short_reason(e)} Re-hit or play it manually.", FAIL_SOUND)
                    move_to(clip, "failed")
                    continue
                except Exception as e:  # never let one bad clip kill the round
                    print(f"SHOT FAILED (unexpected): {e!r}", file=sys.stderr)
                    speaker.say("Shot failed. Unexpected error. Re-hit or play it manually.", FAIL_SOUND)
                    move_to(clip, "failed")
                    continue
                shots += 1
                print(f"Processed in {time.time() - t0:.1f}s")
                speaker.say(spoken_result(s["club"], s["speed"], s["vla"]))
                move_to(clip, "done")
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print(f"\nStopped. {shots} shot(s) sent this session.")
        return 0
    finally:
        if usb is not None:
            usb.stop()


if __name__ == "__main__":
    sys.exit(main())
