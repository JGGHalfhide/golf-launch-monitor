#!/usr/bin/env python3
"""
Run the vision pipeline on a clip and send the result to the relay → sim.

Usage:
    python tools/clip_to_sim.py <clip.mov> [--relay http://localhost:3001] [--calib <px_per_mm>] [--club "7 Iron"] [--dry-run]

Spin is estimated from club identity + VLA deviation from that club's expected launch.
If --club is not supplied, the current club is auto-fetched from the relay /status endpoint.
Horizontal angle is 0 until direction tracking is implemented.
"""
import argparse
import json
import sys
import urllib.request
import urllib.error

sys.path.insert(0, ".")
from vision.pipeline import run as run_pipeline


# ---------------------------------------------------------------------------
# Club spin model
# ---------------------------------------------------------------------------

# (expected_vla_mid_deg, spin_base_rpm)
# Mid-points of the amateur average ranges from the launch/spin chart.
_CLUB_DATA: dict[str, tuple[float, int]] = {
    "driver":        (12.5,  3250),
    "3 wood":        (11.75, 4100),
    "5 wood":        (12.75, 4700),
    "3 hybrid":      (13.75, 4800),
    "4 hybrid":      (14.75, 5100),
    "4 iron":        (13.5,  4400),
    "5 iron":        (14.5,  5050),
    "6 iron":        (15.75, 5675),
    "7 iron":        (17.25, 5950),
    "8 iron":        (18.75, 6500),
    "9 iron":        (20.5,  6850),
    "pitching wedge": (23.0, 7000),
    "pw":            (23.0,  7000),
    "gap wedge":     (25.0,  7300),
    "aw":            (25.0,  7300),
    "sand wedge":    (27.5,  7500),
    "sw":            (27.5,  7500),
    "lob wedge":     (32.0,  8000),
    "lw":            (32.0,  8000),
}

# How much spin changes per degree VLA deviation from the club's expected VLA.
# A shot launched higher than expected = more spin (steeper effective attack angle).
_SPIN_PER_DEG = 200  # rpm / degree


def _match_club(name: str) -> tuple[float, int] | None:
    """Return (expected_vla, base_spin) for a club name, or None if unrecognised."""
    key = name.lower().strip()
    # Exact match first
    if key in _CLUB_DATA:
        return _CLUB_DATA[key]
    # Substring match (e.g. "7 Iron" → "7 iron", "PW" → "pw")
    for k, v in _CLUB_DATA.items():
        if k in key or key in k:
            return v
    return None


def estimate_spin(vla_deg: float, club_name: str | None = None) -> int:
    """
    Estimate spin from VLA + club identity.

    If club_name is provided and recognised, use the club's base spin and
    adjust by how far the actual VLA deviates from that club's expected VLA.
    Falls back to a simple linear interpolation across the full VLA range.
    """
    if club_name:
        match = _match_club(club_name)
        if match:
            expected_vla, base_spin = match
            spin = base_spin + int((vla_deg - expected_vla) * _SPIN_PER_DEG)
            return max(1000, min(12000, spin))

    # Fallback: linear across full VLA range (0° → 2000, 45° → 10000)
    rpm = int(2000 + (vla_deg / 45.0) * 8000)
    return max(1000, min(12000, rpm))


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
    parser.add_argument("--dry-run", action="store_true", help="Print shot but don't send")
    args = parser.parse_args()

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

    spin = estimate_spin(vla, club_name)
    spin_source = f"club-aware ({club_name})" if club_name and _match_club(club_name) else "VLA-only fallback"

    print(f"\nVision result:")
    print(f"  Impact frame : {result['impact_frame']}")
    print(f"  Detections   : {len(result['positions'])} frames")
    print(f"  Ball speed   : {speed:.1f} mph")
    print(f"  VLA          : {vla:.1f} deg")
    print(f"  Spin (est.)  : {spin} rpm  [{spin_source}]")

    if result.get("warnings"):
        for w in result["warnings"]:
            print(f"  WARNING: {w}")

    if speed <= 0:
        print("ERROR: Ball speed is 0 — detection failed, not sending.", file=sys.stderr)
        return 1

    shot = {
        "ballSpeed":             round(speed, 1),
        "verticalLaunchAngle":   round(vla, 1),
        "horizontalLaunchAngle": 0,
        "spinSpeed":             spin,
        "spinAxis":              0,
    }

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

    print("\nDone — watch the sim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
