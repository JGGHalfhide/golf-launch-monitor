"""
Pure functions for the shot model: handicap interpolation, spin estimate,
contact quality, and simulated lateral launch (HLA + spin axis).

No I/O here — config dicts in, numbers out. Randomness comes from an injected
numpy Generator so tests are reproducible.
"""

import math
import re
from dataclasses import dataclass, asdict
from statistics import median

import numpy as np


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interp_anchors(x: float, anchors: list[float], values: list[float]) -> float:
    """Piecewise-linear interpolation of values at anchor x-positions; clamps outside the anchors."""
    if len(anchors) != len(values) or not anchors:
        raise ValueError("anchors and values must be non-empty and the same length")
    if x <= anchors[0]:
        return float(values[0])
    if x >= anchors[-1]:
        return float(values[-1])
    for (x0, y0), (x1, y1) in zip(zip(anchors, values), zip(anchors[1:], values[1:])):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
    raise AssertionError("unreachable: anchors must be sorted ascending")


def range_at_handicap(ranges: list[list[float]], handicap: float, anchors: list[float]) -> tuple[float, float]:
    """Interpolate a metric's bracket [low, high] ranges → (mid, half_range) at a handicap."""
    mids = [(lo + hi) / 2 for lo, hi in ranges]
    halves = [(hi - lo) / 2 for lo, hi in ranges]
    return interp_anchors(handicap, anchors, mids), interp_anchors(handicap, anchors, halves)


@dataclass(frozen=True)
class ClubBaseline:
    club: str
    vla_mid: float
    vla_half: float
    spin_mid: float
    disp_mid: float          # table 95% lateral width (yd) at this handicap
    spin_pct_per_deg: float
    sd_hla: float            # effective sds at this handicap
    sd_axis: float


def _normalize(name: str) -> str:
    return re.sub(r"[\s\-_]", "", name.lower())


def resolve_club(name: str | None, cfg: dict) -> str | None:
    """Map a club name from the sim/CLI ("7i", "7 Iron", "PW", "Sand Wedge") to a config club key."""
    if not name:
        return None
    key = _normalize(name)
    for club, data in cfg["clubs"].items():
        if key == _normalize(club) or key in (_normalize(a) for a in data.get("aliases", [])):
            return club
    return None


def base_sd(club: str, cfg: dict) -> tuple[float, float]:
    """(sd_hla, sd_axis) at the reference handicap, interpolated by club order between sd anchors."""
    order = list(cfg["clubs"])
    anchors = cfg["dispersion"]["sd_anchors"]
    idx = [order.index(c) for c in anchors]
    pairs = sorted(zip(idx, anchors.values()))
    xs = [p[0] for p in pairs]
    i = order.index(club)
    return (interp_anchors(i, xs, [p[1][0] for p in pairs]),
            interp_anchors(i, xs, [p[1][1] for p in pairs]))


def club_baseline(club: str, handicap: float, cfg: dict) -> ClubBaseline:
    anchors = cfg["anchors"]["handicaps"]
    c = cfg["clubs"][club]
    vla_mid, vla_half = range_at_handicap(c["vla"], handicap, anchors)
    spin_mid, _ = range_at_handicap(c["spin"], handicap, anchors)
    disp_mid, _ = range_at_handicap(c["dispersion"], handicap, anchors)

    sd_hla, sd_axis = base_sd(club, cfg)
    d = cfg["dispersion"]
    if d.get("scale_sd_by_handicap", True):
        ref_disp, _ = range_at_handicap(c["dispersion"], d["reference_handicap"], anchors)
        f = disp_mid / ref_disp
        sd_hla, sd_axis = sd_hla * f, sd_axis * f

    return ClubBaseline(club, vla_mid, vla_half, spin_mid, disp_mid,
                        c["spin_pct_per_deg"], sd_hla, sd_axis)


# ---------------------------------------------------------------------------
# Spin
# ---------------------------------------------------------------------------

def estimate_spin(vla: float, vla_mid: float, spin_mid: float, pct_per_deg: float,
                  min_rpm: float = 1000, max_rpm: float = 12000) -> int:
    """Mid spin adjusted by pct_per_deg % for each degree VLA is above (more spin) or below the mid VLA."""
    spin = spin_mid * (1 + pct_per_deg / 100.0 * (vla - vla_mid))
    return int(round(max(min_rpm, min(max_rpm, spin))))


# ---------------------------------------------------------------------------
# Contact quality
# ---------------------------------------------------------------------------

def vla_z(vla: float, vla_mid: float, vla_half: float, spread_mult: float = 1.0) -> float:
    """Two-sided VLA deviation, in units of the bracket half-range."""
    return (vla - vla_mid) / (vla_half * spread_mult)


def speed_z(speed: float, median_speed: float | None, spread_frac: float) -> float:
    """One-sided: only speed BELOW the player's median is penalised. 0 if no median yet."""
    if not median_speed:
        return 0.0
    return max(0.0, (median_speed - speed) / (spread_frac * median_speed))


def quality_score(z_v: float, z_s: float, w_vla: float = 0.3, w_speed: float = 0.7) -> float:
    """q in (0, 1]: 1 = pure centre contact."""
    return math.exp(-0.5 * (w_vla * z_v ** 2 + w_speed * z_s ** 2))


def rolling_median(speeds: list[float], warmup: int, window: int) -> float | None:
    """Median of the most recent `window` speeds, or None until `warmup` shots exist."""
    if len(speeds) < warmup:
        return None
    return float(median(speeds[-window:]))


# ---------------------------------------------------------------------------
# Lateral generation
# ---------------------------------------------------------------------------

def generate_lateral(q: float, sd_hla: float, sd_axis: float, bias_hla: float, bias_axis: float,
                     k: float, rho: float, tail_prob: float, tail_mult: float,
                     rng: np.random.Generator) -> tuple[float, float, bool]:
    """
    Draw (hla, axis, is_tail). Both sds widen by scale = 1 + k*(1-q), and by
    tail_mult on a tail_prob fraction of shots. Draws are correlated by rho.
    """
    scale = 1 + k * (1 - q)
    is_tail = bool(rng.random() < tail_prob)
    if is_tail:
        scale *= tail_mult
    z1, z2 = rng.standard_normal(2)
    hla = bias_hla + sd_hla * scale * z1
    axis = bias_axis + sd_axis * scale * (rho * z1 + math.sqrt(1 - rho ** 2) * z2)
    return float(hla), float(axis), is_tail


def handedness_sign(handedness: str) -> int:
    """+1 for right-handed (fade = right), -1 for left-handed (fade = left)."""
    h = handedness.lower()
    if h not in ("right", "left"):
        raise ValueError(f"handedness must be 'right' or 'left', got {handedness!r}")
    return 1 if h == "right" else -1


# ---------------------------------------------------------------------------
# Full shot
# ---------------------------------------------------------------------------

@dataclass
class ShotEstimate:
    club: str
    vla: float
    speed: float
    q: float
    z_vla: float
    z_speed: float
    speed_median: float | None
    spin: int
    hla: float
    axis: float
    tail: bool

    def as_dict(self) -> dict:
        return asdict(self)

    def sim_shot(self) -> dict:
        return {
            "ballSpeed":             round(self.speed, 1),
            "verticalLaunchAngle":   round(self.vla, 1),
            "horizontalLaunchAngle": round(self.hla, 2),
            "spinSpeed":             self.spin,
            "spinAxis":              round(self.axis, 2),
        }


def estimate_shot(club: str, vla: float, speed: float, cfg: dict,
                  speed_median: float | None, rng: np.random.Generator,
                  handicap: float | None = None) -> ShotEstimate:
    """Measured VLA + speed → spin, q, and simulated HLA / spin axis for one shot."""
    hcp = cfg["player"]["handicap"] if handicap is None else handicap
    b = club_baseline(club, hcp, cfg)
    qc, dc, sc = cfg["quality"], cfg["dispersion"], cfg["spin"]

    spin = estimate_spin(vla, b.vla_mid, b.spin_mid, b.spin_pct_per_deg, sc["min_rpm"], sc["max_rpm"])
    zv = vla_z(vla, b.vla_mid, b.vla_half, qc["vla_spread_mult"])
    zs = speed_z(speed, speed_median, qc["speed_spread_frac"])
    q = quality_score(zv, zs, qc["w_vla"], qc["w_speed"])

    hand = handedness_sign(cfg["player"]["handedness"])
    hla, axis, tail = generate_lateral(
        q, b.sd_hla, b.sd_axis,
        hand * cfg["bias"]["hla_deg"], hand * cfg["bias"]["axis_deg"],
        dc["k"], dc["rho"], dc["tail_prob"], dc["tail_mult"], rng,
    )
    axis *= dc.get("axis_sign", 1)
    return ShotEstimate(club, vla, speed, q, zv, zs, speed_median, spin, hla, axis, tail)
