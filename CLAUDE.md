# Golf Launch Monitor

A DIY golf launch monitor that processes iPhone 240 fps slow-motion clips on a Mac to extract ball speed and launch angle, then feeds the data to OpenGolfSim over its TCP Developer API so you can play full courses at home with no subscriptions.

## Hardware

| Item | Role |
|---|---|
| iPhone 16 Pro | Camera — 240 fps slow-mo via stock Camera app |
| MacBook Air M2 16 GB | All processing, runs OpenGolfSim |
| Tripod | Holds iPhone side-on to ball at ~1–2 m distance |
| Hitting mat | Ball tee position, provides fixed reference plane |
| Net | Ball stop, ~2–3 m in front of mat |
| Lighting | Constant bright light on ball (reduce motion blur) |

## Stack

| Tool | Version |
|---|---|
| Python | 3.12 |
| OpenCV (`opencv-python`) | ≥ 4.9 |
| NumPy | ≥ 1.26 |
| pytest | ≥ 8 |
| opengolfcoach *(optional)* | latest |
| uv or venv | — |

## Directory Layout

```
relay/      Relay service: accepts shots from any source, forwards to sim, logs shots + results
vision/     Offline vision pipeline: load clip → detect impact → track ball → compute speed & angle
tools/      Dev/test scripts (send_fake_shot.py, etc.)
samples/    Local 240 fps test clips (not committed — see .gitignore)
docs/       Protocol and calibration notes
tests/      pytest suite — all math functions must have tests
```

## Commands

```bash
# Setup — must use Python 3.13 (opencv doesn't support 3.14 yet)
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Send a fake shot to OpenGolfSim (sim must be running)
python tools/send_fake_shot.py                    # localhost
python tools/send_fake_shot.py --host 192.168.x.x # remote LAN

# Run relay service
python -m relay                              # defaults: HTTP :3001, sim 127.0.0.1:3111
python -m relay --port 3001 --sim-host 127.0.0.1 --sim-port 3111

# Run vision pipeline on a clip
python -m vision <path/to/clip.mov>                        # auto-calibrate
python -m vision <path/to/clip.mov> --calib 5.2           # supply px/mm manually
python -m vision <path/to/clip.mov> --output debug.mp4    # write overlay video
python -m vision <path/to/clip.mov> --fps 240             # explicit frame rate

# Process a clip and send to sim (relay must be running)
python tools/clip_to_sim.py samples/<clip.mov>                   # auto-detects club from relay
python tools/clip_to_sim.py samples/<clip.mov> --club "7 Iron"   # override club
python tools/clip_to_sim.py samples/<clip.mov> --dry-run         # print shot, don't send
```

## Sim Protocol Summary

OpenGolfSim Developer API — TCP port **3111**, JSON messages, newline-terminated (`\n`).

| Message direction | type | Key fields |
|---|---|---|
| Client → Sim | `"device"` | `status`: `"ready"` or `"busy"` |
| Client → Sim | `"shot"` | `unit`, `shot.ballSpeed`, `shot.verticalLaunchAngle`, `shot.horizontalLaunchAngle`, `shot.spinSpeed`, `shot.spinAxis` |
| Sim → Client | `"player"` | `data.club.name`, `data.club.id`, `data.club.distance` |
| Sim → Client | `"result"` | `data.result.{carry,height,roll,total,lateral}` (meters) |

Full field ranges and confirmed framing details: **[docs/sim-protocol.md](docs/sim-protocol.md)**

## Units and Conventions

- Ball speed: **mph** (imperial mode default)
- Angles: **degrees**
- Positive horizontal angle = **right of target**
- Camera frame rate: **240 fps** → 1 frame = 4.167 ms
- Spin: estimated from club + launch angle (not measured)
- Result distances from sim: **meters**

## Current Phase

**Phase 2 — End-to-end pipeline working. Shot data flows from real iPhone clip → sim.**

| Item | Status |
|---|---|
| Project scaffold | Done |
| `docs/sim-protocol.md` | Done — protocol fully confirmed incl. 200ms delay between device-ready and shot |
| `tools/send_fake_shot.py` | Done — confirmed working (255 yd / 235 carry / 78 ft apex at 150 mph, 12° VLA) |
| Relay service (`relay/`) | Done — HTTP :3001 → TCP :3111, logs shots.jsonl, tracks current club from sim player events |
| Vision pipeline (`vision/`) | Done — background subtraction, directional constraints, linear regression over real timestamps |
| Auto-calibration | Done — px/mm from ball diameter in pre-impact frame |
| `tools/clip_to_sim.py` | Done — full end-to-end: clip → vision → relay → sim; confirmed working (76 mph / 30° VLA / 101 yd) |
| Spin estimation | Done — club-aware model: per-club base spin + VLA deviation adjustment (200 rpm/deg) |
| Horizontal launch angle | Not started — always 0; needs side-spin / direction tracking |
| Spin axis (side spin) | **Planned** — statistical model correlating club, ball speed, and VLA (see below) |
| Shot streaming for a round | **Planned** — workflow TBD; need to solve clip transfer (USB vs WiFi) and folder-watcher automation |
| Live iPhone capture | Not started |

## Planned: Spin Axis (Side Spin) Model

The sim's `spinAxis` field controls draw/fade/straight. Currently hardcoded to 0 (straight).

**Goal:** build a statistical estimator for spin axis using club, ball speed, and VLA as inputs — similar to how `spinSpeed` is estimated. Approach:

- Compile amateur dispersion data by club (e.g. average fade/draw tendency, sigma of axis distribution)
- Fit a model: `spinAxis = f(club, ballSpeed, vla)` — likely a linear regression or lookup with noise
- Higher-lofted clubs (wedges) tend toward more consistent axis; driver has highest axis variance
- Could seed with PGA/amateur Trackman datasets or published dispersion charts

This won't be measured (that requires two cameras or spin dots) but a plausible statistical distribution will make course play feel realistic rather than always hitting dead straight.

## Planned: Shot Streaming Workflow for a Round

For a full round, every shot needs to flow: record clip → transfer to Mac → pipeline → relay → sim.

**Open question:** how to minimise friction between swings. Options under consideration:

- **Folder watcher** (`tools/watch_and_send.py`): monitors a directory, auto-processes any new `.mov` dropped in
- **USB + Image Capture**: reliable 240 fps transfer but requires cable plug-in per shot
- **iPhone Shortcut → HTTP POST**: Shortcut grabs latest Photos video and POSTs to a small upload server on Mac over WiFi — no cable, preserves native file (no transcode)
- **AirDrop**: fast but transcodes to 30 fps — only viable if we accept lower frame rate

**Putting:** not trackable with this setup. Will putt manually in the sim (mouse/controller).

## Rules

- **Keep large video files out of git.** `samples/*.mov|mp4|avi|mkv|hevc` are gitignored. Use a shared folder or external drive for clips.
- **Add pytest tests for any math** (pixel-to-mm, speed, angle calculations) in `tests/`.
- **Update this file** whenever directory structure or commands change.
