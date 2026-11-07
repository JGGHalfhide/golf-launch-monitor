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
# Setup (uv)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Setup (plain venv)
python3.12 -m venv .venv && source .venv/bin/activate
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

**Phase 1 — Protocol confirmed, fake shot tool ready.**

| Item | Status |
|---|---|
| Project scaffold | Done |
| `docs/sim-protocol.md` | Done (framing confirmed; ack/errors unverified) |
| `tools/send_fake_shot.py` | Done — confirmed working, ball flies (255 yd / 235 carry / 78 ft apex at 150 mph, 12° VLA) |
| Relay service (`relay/`) | Done — HTTP :3001, forwards to sim, logs to logs/shots.jsonl |
| Vision pipeline (`vision/`) | Done (offline) |
| Calibration + sample clips | Not started |
| Horizontal direction | Not started |
| Spin measurement | Not started |
| Live iPhone capture app | Not started |

## Rules

- **Keep large video files out of git.** `samples/*.mov|mp4|avi|mkv|hevc` are gitignored. Use a shared folder or external drive for clips.
- **Add pytest tests for any math** (pixel-to-mm, speed, angle calculations) in `tests/`.
- **Update this file** whenever directory structure or commands change.
