#!/usr/bin/env python3
"""
Calibrate the ball's tee spot for a session: click the ball once, and every shot afterwards
only looks for the ball there.

Usage:
    python tools/set_ball_spot.py                      # newest clip in inbox/ (incl. done/, failed/)
    python tools/set_ball_spot.py path/to/clip.mov
    python tools/set_ball_spot.py --xy 830,675         # no window: give the ball centre in pixels
    python tools/set_ball_spot.py --clear              # back to searching the whole frame

Record any short slo-mo clip with the ball on the tee mark (a practice swing that misses is
fine — the frame shown is from early in the clip). Click the ball; it snaps to the ball's
outline. Enter/Space saves, R retries, Esc cancels. Saved to logs/ball_spot.json and picked up
by tools/watch_and_send.py on the next shot. Re-run whenever the tripod or tee mark moves.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vision.calibration import white_blob_candidates  # noqa: E402

SPOT_FILE = ROOT / "logs" / "ball_spot.json"
VIDEO_EXTS = {".mov", ".mp4", ".m4v"}
DEFAULT_R = 30.0


def newest_clip(inbox: Path) -> Path | None:
    clips = [p for p in inbox.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
             and ".incoming" not in p.parts]
    return max(clips, key=lambda p: p.stat().st_mtime) if clips else None


def read_frame(clip: Path, fraction: float = 0.15) -> np.ndarray:
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    target = int(n * fraction)
    frame = None
    for _ in range(target + 1):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
    cap.release()
    if frame is None:
        raise SystemExit(f"Could not read frames from {clip}")
    return frame


def snap_to_ball(frame: np.ndarray, x: float, y: float) -> tuple[float, float, float, bool]:
    """
    Snap the click to the ball blob it landed on → (x, y, r, snapped).

    Only snaps to a blob that contains the click — never moves the click onto something else
    nearby (the clubhead at address is right next to the ball). Otherwise the click is used as
    given with a generous radius; the pipeline only needs the spot to within a few ball radii.
    """
    h, w = frame.shape[:2]
    half = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    cands = [(cx * 2, cy * 2, r * 2) for cx, cy, r, _ in white_blob_candidates(half, top_fraction=0)]
    hits = [(r, cx, cy) for cx, cy, r in cands if np.hypot(cx - x, cy - y) <= 1.2 * r and r >= 12]
    if not hits:
        return x, y, DEFAULT_R, False
    r, cx, cy = max(hits)  # the largest blob under the click: the whole ball, not a fragment of it
    return float(cx), float(cy), float(max(r, DEFAULT_R * 0.8)), True


def save(x: float, y: float, r: float, clip: Path | None, frame_shape) -> None:
    SPOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPOT_FILE.write_text(json.dumps({
        "x": round(x, 1), "y": round(y, 1), "r": round(r, 1),
        "frame_w": frame_shape[1], "frame_h": frame_shape[0],
        "clip": str(clip) if clip else None, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1))
    print(f"Saved ball spot ({x:.0f}, {y:.0f}), r≈{r:.0f} px → {SPOT_FILE}")


def pick_interactively(frame: np.ndarray) -> tuple[float, float, float] | None:
    h, w = frame.shape[:2]
    scale = min(1.0, 1400 / w)
    shown = cv2.resize(frame, (int(w * scale), int(h * scale)))
    state: dict = {"pick": None}

    def on_mouse(event, mx, my, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pick"] = snap_to_ball(frame, mx / scale, my / scale)

    win = "Click the ball - Enter/Space save, R retry, Esc cancel"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = shown.copy()
        if state["pick"]:
            x, y, r, snapped = state["pick"]
            cv2.circle(view, (int(x * scale), int(y * scale)), max(3, int(r * scale)),
                       (0, 255, 0) if snapped else (0, 165, 255), 2)
            cv2.putText(view, "snapped to ball" if snapped else "no ball outline found - using click",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if snapped else (0, 165, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32) and state["pick"]:
            cv2.destroyAllWindows()
            x, y, r, _ = state["pick"]
            return x, y, r
        if key in (ord("r"), ord("R")):
            state["pick"] = None
        if key == 27:
            cv2.destroyAllWindows()
            return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", nargs="?", help="Clip with the ball on the tee mark (default: newest in inbox/)")
    ap.add_argument("--xy", help="Ball centre as x,y in full-res pixels (skip the window)")
    ap.add_argument("--clear", action="store_true", help="Remove the calibrated spot")
    args = ap.parse_args()

    if args.clear:
        SPOT_FILE.unlink(missing_ok=True)
        print("Ball spot cleared — the pipeline will search the whole frame.")
        return 0

    clip = Path(args.clip) if args.clip else newest_clip(ROOT / "inbox")
    if clip is None:
        print("No clip found. Record a short slo-mo clip with the ball on the tee mark first.", file=sys.stderr)
        return 1
    print(f"Using {clip}")
    frame = read_frame(clip)

    if args.xy:
        x, y = (float(v) for v in args.xy.split(","))
        x, y, r, snapped = snap_to_ball(frame, x, y)
        print("Snapped to ball outline." if snapped else "No ball outline near that point — using it as given.")
    else:
        pick = pick_interactively(frame)
        if pick is None:
            print("Cancelled — nothing saved.")
            return 1
        x, y, r = pick
    save(x, y, r, clip, frame.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
