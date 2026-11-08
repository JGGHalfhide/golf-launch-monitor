#!/usr/bin/env python3
"""
Run the vision pipeline on a clip and send the result to the relay → sim.

Usage:
    python tools/clip_to_sim.py <clip.mov> [--relay http://localhost:3001] [--calib <px_per_mm>]
                                [--club "7 Iron"] [--handicap 14] [--seed N] [--dry-run]

VLA and ball speed are measured. Spin, contact quality, and the lateral launch
(HLA + spin axis) come from shotmodel, driven by config/shot_model.toml.
If --club is not supplied, the current club is auto-fetched from the relay /status endpoint.
Each shot's model output is logged to logs/shot_model.jsonl.
"""
import argparse
import json
import sys
import urllib.request
import urllib.error

import numpy as np

sys.path.insert(0, ".")
from relay.logger import ShotLogger
from shotmodel import SpeedHistory, estimate_shot, load_config, resolve_club, rolling_median
from vision.pipeline import run as run_pipeline


def _fallback_spin(vla_deg: float) -> int:
    """Unknown club: linear across full VLA range (0° → 2000, 45° → 10000)."""
    return max(1000, min(12000, int(2000 + (vla_deg / 45.0) * 8000)))


# ---------------------------------------------------------------------------
# Relay helpers
# ---------------------------------------------------------------------------

def fetch_current_club(relay_url: str) -> str | None:
    """Ask the relay what club the sim currently has selected. Returns None on any failure."""
    try:
        with urllib.request.urlopen(f"{relay_url}/status", timeout=3) as resp:
            data = json.loads(resp.read())
        club = data.get("current_club")
        if isinstance(club, dict):
            return club.get("name")
        return None
    except Exception:
        return None


def post_shot(relay_url: str, shot: dict) -> dict:
    body = json.dumps(shot).encode()
    req = urllib.request.Request(
        f"{relay_url}/shot",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", help="Path to .mov / .mp4 clip")
    parser.add_argument("--relay", default="http://localhost:3001")
    parser.add_argument("--calib", type=float, default=None, metavar="px_per_mm")
    parser.add_argument("--club", default=None, help='Override club name, e.g. "7 Iron"')
    parser.add_argument("--handicap", type=float, default=None, help="Override config handicap")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (reproducible dispersion)")
    parser.add_argument("--config", default=None, help="Shot model config (default config/shot_model.toml)")
    parser.add_argument("--log", default="logs/shot_model.jsonl", help="Shot model log path")
    parser.add_argument("--dry-run", action="store_true", help="Print shot but don't send (history not updated)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    print(f"Running vision pipeline on {args.clip} ...")
    try:
        result = run_pipeline(args.clip, px_per_mm=args.calib)
    except Exception as e:
        print(f"ERROR: Vision pipeline failed: {e}", file=sys.stderr)
        return 1

    speed = result["ball_speed_mph"]
    vla   = result["vertical_launch_angle_deg"]

    # Determine club: CLI override > relay auto-detect > None (fallback formula)
    club_name = args.club
    if club_name is None and not args.dry_run:
        club_name = fetch_current_club(args.relay)

    club = resolve_club(club_name, cfg)

    print(f"\nVision result:")
    print(f"  Impact frame : {result['impact_frame']}")
    print(f"  Detections   : {len(result['positions'])} frames")
    print(f"  Ball speed   : {speed:.1f} mph")
    print(f"  VLA          : {vla:.1f} deg")

    if result.get("warnings"):
        for w in result["warnings"]:
            print(f"  WARNING: {w}")

    if speed <= 0:
        print("ERROR: Ball speed is 0 — detection failed, not sending.", file=sys.stderr)
        return 1

    history = SpeedHistory(cfg["quality"]["history_path"])
    est = None
    if club is None:
        print(f"  WARNING: club {club_name!r} not recognised — VLA-only spin, straight shot. Use --club.")
        shot = {
            "ballSpeed":             round(speed, 1),
            "verticalLaunchAngle":   round(vla, 1),
            "horizontalLaunchAngle": 0,
            "spinSpeed":             _fallback_spin(vla),
            "spinAxis":              0,
        }
    else:
        qc = cfg["quality"]
        med = rolling_median(history.speeds(club), qc["speed_warmup_shots"], qc["speed_window"])
        est = estimate_shot(club, vla, speed, cfg, med, np.random.default_rng(args.seed), args.handicap)
        shot = est.sim_shot()
        n = len(history.speeds(club))
        med_txt = f"{med:.1f} mph" if med else f"warming up ({n}/{qc['speed_warmup_shots']} shots)"
        print(f"\nShot model ({club}, hcp {args.handicap if args.handicap is not None else cfg['player']['handicap']}):")
        print(f"  Speed median : {med_txt}")
        print(f"  Quality q    : {est.q:.2f}  (z_vla {est.z_vla:+.2f}, z_speed {est.z_speed:.2f})")
        print(f"  Spin (est.)  : {est.spin} rpm")
        print(f"  HLA (sim.)   : {est.hla:+.2f} deg{'  [tail]' if est.tail else ''}")
        print(f"  Axis (sim.)  : {est.axis:+.2f} deg")

    print(f"\nShot to send: {json.dumps(shot)}")

    if args.dry_run:
        print("(dry-run — not sending)")
        return 0

    print(f"\nSending to relay at {args.relay} ...")
    try:
        resp = post_shot(args.relay, shot)
        print(f"Relay response: {resp}")
    except urllib.error.URLError as e:
        print(f"ERROR: Could not reach relay: {e}", file=sys.stderr)
        print("Is the relay running?  python -m relay", file=sys.stderr)
        return 1

    if est is not None:
        history.add(est.club, est.speed)
        history.save()
        hcp = args.handicap if args.handicap is not None else cfg["player"]["handicap"]
        ShotLogger(args.log).log({"event": "shot_model", "clip": args.clip, "handicap": hcp, **est.as_dict()})

    print("\nDone — watch the sim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
