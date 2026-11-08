#!/usr/bin/env python3
"""
Calibrate simulated dispersion (sd_hla / sd_axis) against the handicap table.

Runs the HLA/axis generator N times for one club and handicap, converts each
draw to a lateral landing offset with a simple ballistic approximation, and
compares the 95% lateral width to the table's dispersion range.

    lateral_yd = carry * sin(hla) + curve_factor * carry * sin(axis)

curve_factor is per club in the config (measured in the sim). The sim is NOT used: each shot there takes several seconds of ball flight and
would land in your play session, so 1000+ shots isn't practical.

Usage:
    python tools/calibrate_dispersion.py --club "7 iron"
    python tools/calibrate_dispersion.py --club driver --handicap 5 -n 5000 --carry 240
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from shotmodel import load_config, resolve_club
from shotmodel.core import (
    base_sd, club_baseline, generate_lateral, handedness_sign, quality_score, range_at_handicap,
)


def lateral_yd(hla: np.ndarray, axis: np.ndarray, carry: float, curve_factor: float) -> np.ndarray:
    return carry * np.sin(np.radians(hla)) + curve_factor * carry * np.sin(np.radians(axis))


def width95(x: np.ndarray) -> float:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(hi - lo)


def realistic_q(n: int, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """q from synthetic shots: VLA z ~ N(0,1) (range half-width = 1 sd), speed z one-sided half-normal."""
    qc = cfg["quality"]
    zv = rng.standard_normal(n) / qc["vla_spread_mult"]
    zs = np.maximum(0.0, rng.standard_normal(n))
    return np.array([quality_score(a, b, qc["w_vla"], qc["w_speed"]) for a, b in zip(zv, zs)])


def simulate(q: np.ndarray, sd_hla: float, sd_axis: float, cfg: dict, carry: float,
             curve_factor: float, rng: np.random.Generator) -> np.ndarray:
    d, hand = cfg["dispersion"], handedness_sign(cfg["player"]["handedness"])
    bh, ba = hand * cfg["bias"]["hla_deg"], hand * cfg["bias"]["axis_deg"]
    draws = np.array([
        generate_lateral(qi, sd_hla, sd_axis, bh, ba, d["k"], d["rho"], d["tail_prob"], d["tail_mult"], rng)[:2]
        for qi in q
    ])
    return lateral_yd(draws[:, 0], draws[:, 1] * d.get("axis_sign", 1), carry, curve_factor)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--club", required=True)
    p.add_argument("--handicap", type=float, default=None)
    p.add_argument("-n", type=int, default=1000)
    p.add_argument("--carry", type=float, default=None, help="Carry yd (default: club carry_yd in config)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    club = resolve_club(args.club, cfg)
    if club is None:
        print(f"Unknown club {args.club!r}. Known: {', '.join(cfg['clubs'])}", file=sys.stderr)
        return 1
    hcp = cfg["player"]["handicap"] if args.handicap is None else args.handicap
    carry = args.carry or cfg["clubs"][club]["carry_yd"]
    cf = cfg["clubs"][club].get("curve_factor", cfg["calibration"]["curve_factor"])
    b = club_baseline(club, hcp, cfg)
    disp_mid, disp_half = range_at_handicap(cfg["clubs"][club]["dispersion"], hcp, cfg["anchors"]["handicaps"])
    rng = np.random.default_rng(args.seed)

    q_real = realistic_q(args.n, cfg, rng)
    scenarios = {
        "realistic q": q_real,
        "q = 1 (pure)": np.ones(args.n),
        f"q = {cfg['calibration']['q_low']} (poor)": np.full(args.n, cfg["calibration"]["q_low"]),
    }

    print(f"Club {club} | handicap {hcp:g} | carry {carry:g} yd | curve_factor {cf:g} | n={args.n}")
    print(f"Effective sd_hla {b.sd_hla:.2f}°  sd_axis {b.sd_axis:.2f}°")
    print(f"Table 95% width: {disp_mid - disp_half:.1f} – {disp_mid + disp_half:.1f} yd (mid {disp_mid:.1f})")
    print(f"Realistic q: mean {q_real.mean():.2f}, p10 {np.percentile(q_real, 10):.2f}, p90 {np.percentile(q_real, 90):.2f}\n")
    print(f"{'scenario':<20}{'95% width':>11}{'mean lat':>10}  vs table")
    for name, q in scenarios.items():
        lat = simulate(q, b.sd_hla, b.sd_axis, cfg, carry, cf, np.random.default_rng(args.seed))
        w = width95(lat)
        print(f"{name:<20}{w:>9.1f}yd{lat.mean():>+8.1f}yd  {w / disp_mid:5.2f}x mid")

    # Suggest sds: width is ~linear in sd, iterate a few times to absorb sin() nonlinearity.
    factor = 1.0
    for _ in range(4):
        lat = simulate(q_real, b.sd_hla * factor, b.sd_axis * factor, cfg, carry, cf, np.random.default_rng(args.seed))
        factor *= disp_mid / width95(lat)

    ref_hla, ref_axis = base_sd(club, cfg)
    print(f"\nTo match table mid with realistic q, scale sds by {factor:.2f}:")
    print(f"  effective at hcp {hcp:g}: sd_hla {b.sd_hla * factor:.2f}°, sd_axis {b.sd_axis * factor:.2f}°")
    note = " (reference_handicap " + str(cfg["dispersion"]["reference_handicap"]) + ")" \
        if cfg["dispersion"].get("scale_sd_by_handicap", True) else ""
    print(f"  config sd_anchors value{note}: {club} = [{ref_hla * factor:.2f}, {ref_axis * factor:.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
