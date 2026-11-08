"""Handicap-based spin, contact quality, and simulated lateral dispersion (HLA + spin axis)."""

from shotmodel.config import load_config
from shotmodel.core import ShotEstimate, estimate_shot, resolve_club, rolling_median
from shotmodel.history import SpeedHistory

__all__ = ["load_config", "ShotEstimate", "estimate_shot", "resolve_club", "rolling_median", "SpeedHistory"]
