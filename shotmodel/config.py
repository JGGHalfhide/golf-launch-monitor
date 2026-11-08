"""Load config/shot_model.toml."""

import tomllib
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "shot_model.toml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    unknown = set(cfg["dispersion"]["sd_anchors"]) - set(cfg["clubs"])
    if unknown:
        raise ValueError(f"sd_anchors reference unknown clubs: {sorted(unknown)}")
    return cfg
