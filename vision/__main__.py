"""CLI entry point: python -m vision <clip> [--calib <px_per_mm>] [--output <path>] [--fps 240]"""

import argparse
import sys

from .pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vision",
        description="Offline golf launch monitor — analyse a 240 fps iPhone slow-mo clip.",
    )
    parser.add_argument("clip", help="Path to .mov / .mp4 clip")
    parser.add_argument("--calib", type=float, default=None, metavar="px_per_mm",
                        help="Pixels per mm (skip auto-calibration)")
    parser.add_argument("--output", default=None, metavar="path",
                        help="Write debug overlay video to this path")
    parser.add_argument("--fps", type=float, default=240.0,
                        help="Recording frame rate (default 240)")
    args = parser.parse_args()

    try:
        result = run(
            clip_path=args.clip,
            px_per_mm=args.calib,
            output_path=args.output,
            fps=args.fps,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Impact frame : {result['impact_frame']}")
    print(f"px/mm        : {result['px_per_mm']:.4f}")
    print(f"Ball speed   : {result['ball_speed_mph']:.1f} mph")
    print(f"VLA          : {result['vertical_launch_angle_deg']:.1f} deg")
    if result.get("warnings"):
        for w in result["warnings"]:
            print(f"WARNING: {w}")
    if args.output:
        print(f"Overlay video: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
