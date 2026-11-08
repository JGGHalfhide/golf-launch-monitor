#!/usr/bin/env python3
"""
Per-club calibration report: what the sim produced vs the player's known averages.

Usage:
    python tools/calibration_report.py                 # shots from the last 4 hours
    python tools/calibration_report.py --hours 1
    python tools/calibration_report.py --since 2026-11-08T16:00

Reads sim results from logs/shots.jsonl (written by the relay) and clip outcomes from
logs/watch_outcomes.jsonl (written by tools/watch_and_send.py), groups by the sim's club,
and compares with config/player_targets.toml.

How to read it:
  - speed/launch/spin are what we sent; carry/lateral are what the sim returned.
  - carry Δ% is mean sim carry vs target. If Δ% is roughly the same for every club, suspect
    the speed measurement (one scale factor). If speeds look normal but carry is short only
    for some clubs, suspect the shot model's spin/launch for those clubs.
  - "fail" counts clips that never became a shot (ball not tracked etc.).
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


def summarise(results: list[dict], fails: dict[str, int], targets: dict) -> list[dict]:
    by_club: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_club[r["club"]].append(r)
    order = list(targets)
    rows = []
    for club in sorted(set(by_club) | set(fails), key=lambda c: order.index(c) if c in order else 99):
        shots = by_club.get(club, [])
        t = targets.get(club, {})
        row = {"club": club, "n": len(shots), "fail": fails.get(club, 0), "target": t.get("carry")}
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
    args = ap.parse_args()

    if args.since:
        since = datetime.fromisoformat(args.since)
        since = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
    else:
        since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    targets = tomllib.loads(Path(args.targets).read_text())["clubs"]
    rows = summarise(load_results(Path(args.shots), since), load_failures(Path(args.outcomes), since), targets)
    if not rows:
        print(f"No shots since {since.isoformat(timespec='minutes')}.")
        return 1

    print(f"Shots since {since.astimezone().strftime('%Y-%m-%d %H:%M')}   (yd; lateral − = left)\n")
    print(f"{'club':8s} {'n':>3s} {'fail':>4s} {'speed':>6s} {'launch':>6s} {'spin':>6s} "
          f"{'carry':>6s} {'±sd':>5s} {'target':>6s} {'Δ%':>6s} {'lat':>6s}")
    for r in rows:
        if r["n"]:
            tgt = f"{r['target']:6.0f}" if r["target"] else "     -"
            dp = f"{r['delta_pct']:+6.0f}" if "delta_pct" in r else "     -"
            print(f"{r['club']:8s} {r['n']:3d} {r['fail']:4d} {r['speed']:6.1f} {r['vla']:6.1f} {r['spin']:6.0f} "
                  f"{r['carry']:6.0f} {r['carry_sd']:5.0f} {tgt} {dp} {r['lateral']:+6.0f}")
        else:
            print(f"{r['club']:8s} {0:3d} {r['fail']:4d}   (no successful shots)")
    deltas = [r["delta_pct"] for r in rows if "delta_pct" in r and r["n"] >= 3]
    if len(deltas) >= 2:
        print(f"\nCarry Δ% across clubs with ≥3 shots: mean {np.mean(deltas):+.0f}%, spread {np.std(deltas):.0f}%"
              "  — similar Δ% everywhere points at speed measurement; uneven points at the shot model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
