#!/usr/bin/env python3
"""
Per-club calibration report: what the sim produced vs the player's known averages.

Usage:
    python tools/calibration_report.py                 # shots from the last 4 hours
    python tools/calibration_report.py --hours 1
    python tools/calibration_report.py --since 2026-11-08T16:00
    python tools/calibration_report.py --list                         # every shot with its id
    python tools/calibration_report.py --exclude 2026-11-08T16:11:03 --reason "misread"

Reads sim results from logs/shots.jsonl (written by the relay) and clip outcomes from
logs/watch_outcomes.jsonl (written by tools/watch_and_send.py), groups by the sim's club,
and compares with config/player_targets.toml.

How to read it:
  - speed/launch/spin are what we sent; carry/lateral are what the sim returned.
  - carry Δ% is mean sim carry vs target. If Δ% is roughly the same for every club, suspect
    the speed measurement (one scale factor). If speeds look normal but carry is short only
    for some clubs, suspect the shot model's spin/launch for those clubs.
  - "fail" counts clips that never became a shot (ball not tracked etc.).
  - "excl" counts confirmed misreads left out (logs/excluded_shots.json, via --exclude).
    Use it for readings known to be wrong (e.g. 14 mph on a full, clean 7i) — not for shots
    that were merely bad swings, which are part of the player's real averages.
"""
import argparse
import json
import sys
import tomllib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
M_TO_YD = 1.0936133

# Sim club short names → keys in config/player_targets.toml
SIM_TO_TARGET = {
    "DR": "driver", "3W": "3-wood", "5W": "5-wood",
    "3I": "3-iron", "4I": "4-iron", "5I": "5-iron", "6I": "6-iron", "7I": "7-iron",
    "8I": "8-iron", "9I": "9-iron", "PW": "pw", "SW": "sw", "LW": "lw",
}


def target_key(club_name: str | None) -> str | None:
    return SIM_TO_TARGET.get((club_name or "").upper())


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


EXCLUDE_FILE = ROOT / "logs" / "excluded_shots.json"


def load_exclusions(path: Path = EXCLUDE_FILE) -> dict[str, str]:
    """Result timestamp (to the second) → reason, for confirmed misreads."""
    return json.loads(path.read_text()) if path.exists() else {}


def shot_id(ts: str) -> str:
    return ts[:19]  # e.g. 2026-11-08T16:11:03


def load_results(path: Path, since: datetime) -> list[dict]:
    """Sim 'result' events: club, carry/total/lateral (yd), and the shot that produced them."""
    out = []
    if not path.exists():
        return out
    for line in path.open():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") != "sim_event" or parse_ts(e["ts"]) < since:
            continue
        data = e.get("data", {})
        if data.get("type") != "result":
            continue
        d = data.get("data", {})
        r, shot, club = d.get("result", {}), d.get("shot", {}), d.get("club", {})
        out.append({
            "id": shot_id(e["ts"]),
            "club": target_key(club.get("name")) or club.get("name"),
            "carry": r.get("carry", 0) * M_TO_YD, "total": r.get("total", 0) * M_TO_YD,
            "lateral": r.get("lateral", 0) * M_TO_YD,
            "speed": shot.get("ballSpeed"), "vla": shot.get("verticalLaunchAngle"),
            "spin": shot.get("spinSpeed"),
        })
    return out


def load_failures(path: Path, since: datetime) -> dict[str, int]:
    fails: dict[str, int] = defaultdict(int)
    if not path.exists():
        return fails
    for line in path.open():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "failed" and parse_ts(e["ts"]) >= since:
            fails[target_key(e.get("club")) or str(e.get("club"))] += 1
    return fails


def summarise(results: list[dict], fails: dict[str, int], targets: dict,
              excluded: dict[str, str] | None = None) -> list[dict]:
    by_club: dict[str, list[dict]] = defaultdict(list)
    excl: dict[str, int] = defaultdict(int)
    for r in results:
        if excluded and r.get("id") in excluded:
            excl[r["club"]] += 1
            continue
        by_club[r["club"]].append(r)
    order = list(targets)
    rows = []
    for club in sorted(set(by_club) | set(fails) | set(excl), key=lambda c: order.index(c) if c in order else 99):
        shots = by_club.get(club, [])
        t = targets.get(club, {})
        row = {"club": club, "n": len(shots), "fail": fails.get(club, 0), "excl": excl.get(club, 0),
               "target": t.get("carry")}
        if shots:
            carry = np.array([s["carry"] for s in shots])
            row.update({
                "speed": float(np.mean([s["speed"] for s in shots])),
                "vla": float(np.mean([s["vla"] for s in shots])),
                "spin": float(np.mean([s["spin"] for s in shots])),
                "carry": float(carry.mean()), "carry_sd": float(carry.std()) if len(carry) > 1 else 0.0,
                "lateral": float(np.mean([s["lateral"] for s in shots])),
            })
            if row["target"]:
                row["delta_pct"] = 100 * (row["carry"] / row["target"] - 1)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--since", help="ISO time (UTC if no offset), overrides --hours")
    ap.add_argument("--shots", default=str(ROOT / "logs" / "shots.jsonl"))
    ap.add_argument("--outcomes", default=str(ROOT / "logs" / "watch_outcomes.jsonl"))
    ap.add_argument("--targets", default=str(ROOT / "config" / "player_targets.toml"))
    ap.add_argument("--list", action="store_true", help="List every shot with its id instead of the summary")
    ap.add_argument("--exclude", metavar="ID", help="Mark a shot (id from --list) as a confirmed misread")
    ap.add_argument("--include", metavar="ID", help="Undo --exclude")
    ap.add_argument("--reason", default="confirmed misread")
    args = ap.parse_args()

    excluded = load_exclusions()
    if args.exclude or args.include:
        if args.exclude:
            excluded[shot_id(args.exclude)] = args.reason
            print(f"Excluded {shot_id(args.exclude)}: {args.reason}")
        if args.include:
            excluded.pop(shot_id(args.include), None)
            print(f"Re-included {shot_id(args.include)}")
        EXCLUDE_FILE.parent.mkdir(parents=True, exist_ok=True)
        EXCLUDE_FILE.write_text(json.dumps(excluded, indent=1))
        return 0

    if args.since:
        since = datetime.fromisoformat(args.since)
        since = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
    else:
        since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    targets = tomllib.loads(Path(args.targets).read_text())["clubs"]
    results = load_results(Path(args.shots), since)
    if args.list:
        for r in results:
            mark = f"  EXCLUDED ({excluded[r['id']]})" if r["id"] in excluded else ""
            print(f"{r['id']}  {r['club']:8s} {r['speed']:6.1f} mph {r['vla']:5.1f}° {r['spin']:5.0f} rpm"
                  f"  carry {r['carry']:4.0f} yd{mark}")
        return 0
    rows = summarise(results, load_failures(Path(args.outcomes), since), targets, excluded)
    if not rows:
        print(f"No shots since {since.isoformat(timespec='minutes')}.")
        return 1

    print(f"Shots since {since.astimezone().strftime('%Y-%m-%d %H:%M')}   (yd; lateral − = left)\n")
    print(f"{'club':8s} {'n':>3s} {'fail':>4s} {'excl':>4s} {'speed':>6s} {'launch':>6s} {'spin':>6s} "
          f"{'carry':>6s} {'±sd':>5s} {'target':>6s} {'Δ%':>6s} {'lat':>6s}")
    for r in rows:
        if r["n"]:
            tgt = f"{r['target']:6.0f}" if r["target"] else "     -"
            dp = f"{r['delta_pct']:+6.0f}" if "delta_pct" in r else "     -"
            print(f"{r['club']:8s} {r['n']:3d} {r['fail']:4d} {r['excl']:4d} {r['speed']:6.1f} {r['vla']:6.1f} {r['spin']:6.0f} "
                  f"{r['carry']:6.0f} {r['carry_sd']:5.0f} {tgt} {dp} {r['lateral']:+6.0f}")
        else:
            print(f"{r['club']:8s} {0:3d} {r['fail']:4d} {r['excl']:4d}   (no valid shots)")
    deltas = [r["delta_pct"] for r in rows if "delta_pct" in r and r["n"] >= 3]
    if len(deltas) >= 2:
        print(f"\nCarry Δ% across clubs with ≥3 shots: mean {np.mean(deltas):+.0f}%, spread {np.std(deltas):.0f}%"
              "  — similar Δ% everywhere points at speed measurement; uneven points at the shot model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
