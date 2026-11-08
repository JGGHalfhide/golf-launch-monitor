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
shotmodel/  Handicap-based spin, contact quality q, simulated HLA + spin axis (pure functions)
config/     shot_model.toml — every shot-model tunable + the handicap baseline table
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
python tools/clip_to_sim.py samples/<clip.mov> --handicap 10 --seed 1  # override hcp, reproducible draw

# Calibrate dispersion sds against the handicap table (ballistic approximation, no sim)
python tools/calibrate_dispersion.py --club "7 iron" [--handicap 14] [-n 1000] [--carry 150]
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
- Spin: estimated from club + handicap baseline + VLA deviation (not measured)
- HLA and spin axis: **simulated** from contact quality + personal bias (not measured); positive spinAxis assumed = curves right (`axis_sign` in config flips it)
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
| Spin estimation | Done — `shotmodel`: handicap-interpolated mid spin × (1 + per-club %/deg × VLA deviation) |
| Contact quality score | Done — q from VLA z (two-sided) + speed z vs rolling per-club median (one-sided); history in `logs/speed_history.json` |
| Horizontal launch angle | Simulated — correlated draw with spin axis, sd scaled by 1 + k(1−q), fat tails, personal bias |
| Spin axis (side spin) | Simulated — see above; sds need calibrating (`tools/calibrate_dispersion.py`) |
| Shot streaming for a round | **Next** — decided: Apple Watch remote + USB-tethered ImageCaptureCore watcher (see below); not built yet |
| Live iPhone capture | Not started |

## Planned: Spin Axis (Side Spin) Model

The sim's `spinAxis` field controls draw/fade/straight. Currently hardcoded to 0 (straight).

**Goal:** build a statistical estimator for spin axis using club, ball speed, and VLA as inputs — similar to how `spinSpeed` is estimated. Approach:

- Compile amateur dispersion data by club (e.g. average fade/draw tendency, sigma of axis distribution)
- Fit a model: `spinAxis = f(club, ballSpeed, vla)` — likely a linear regression or lookup with noise
- Higher-lofted clubs (wedges) tend toward more consistent axis; driver has highest axis variance
- Could seed with PGA/amateur Trackman datasets or published dispersion charts

This won't be measured (that requires two cameras or spin dots) but a plausible statistical distribution will make course play feel realistic rather than always hitting dead straight.

## Planned: Contact Quality Score → Dispersion Scaling

**Idea:** derive a contact quality score from the measured VLA and ball speed vs. what's expected for the club. A mis-hit shows up as an inconsistency between these two — not just an unusual value in isolation, but the *combination* being off.

**Physics note (needs research to confirm):** the clearest mis-hit signatures are likely:

| Pattern | What it probably means | Dispersion effect |
|---|---|---|
| Low ball speed + VLA near normal | Fat/chunked — energy lost in ground | High — unpredictable direction |
| Normal ball speed + VLA much lower than expected | Thin/skull — struck equator of ball | High — low flight, runs |
| Normal ball speed + VLA much higher than expected | Toe or heel strike (gear effect can raise launch) | Moderate — offline tendency |
| Ball speed and VLA both close to expected | Solid centre contact | Low dispersion |

A high VLA *with* high ball speed on a 7i specifically is worth researching — it may actually indicate a good strike (more compression = higher flight) rather than a mis-hit. The key signal is probably the *ratio* of VLA deviation to ball speed deviation rather than either alone.

**Implementation sketch:**

```python
# z-score of VLA deviation from club's expected VLA, normalised by ball speed
contact_score = 1.0 - clamp(abs(vla - expected_vla) / vla_sigma, 0, 1)
# 1.0 = pure centre contact, 0.0 = severe mis-hit
# spinAxis sigma for the club then scales by (1 / contact_score)
```

**Data needed:** per-club VLA standard deviation for amateurs (Trackman or similar). Research before building.

## Planned: Shot Streaming Workflow for a Round

**Decided 2026-11-07.** Every shot flows: Watch remote start/stop → clip lands on Mac → pipeline → relay → sim → spoken result.

### Capture
- iPhone on tripod, stock Camera app in **Slo-mo** mode.
- **Apple Watch Camera Remote** starts/stops recording (confirmed working in Slo-mo — `samples/IMG_0278.mov` was shot this way).
- Two taps per shot: start just before the swing, stop right after. Keep clips short (pipeline loads every frame).

### Transfer: cable (primary)
- iPhone stays **USB-tethered to the Mac** (also charges it through a round).
- A watcher uses macOS **ImageCaptureCore** (`ICCameraDevice` delegate `didAddItems`) to detect each new clip on the connected phone and download the **original** file within seconds. No iCloud, no AirDrop transcode.
- Swift CLI or PyObjC — TBD. Phone must be trusted and unlocked (Camera app open keeps it awake).

### Transfer: iCloud (fallback)
- iCloud Photos sync + `osxphotos` exporting new slo-mo originals. Needs Photos running with "Download Originals to this Mac". Lag 10–60+ s, so practice only.

### Mac side: `tools/watch_and_send.py` (to build)
- Source-agnostic: takes clips from whichever transfer route is active, so the rest of the flow is shared.
- Process each clip **exactly once** (persist handled IDs); ignore anything recorded before the watcher started.
- Run vision → shotmodel → relay; club comes from the relay's tracked sim club (no input mid-round).
- **Speak every result** (macOS `say`, e.g. "PW, 96 mph, 28 degrees") and **announce failures** loudly so you re-hit or play it manually.

### Steps
1. Throwaway test: time from Watch "stop" to clip downloaded via ImageCaptureCore, plus whether you get the original 240 fps file. Check per-frame `CAP_PROP_POS_MSEC` intervals, not `CAP_PROP_FPS` (that's an average over variable-frame-rate files).
2. Throwaway test: iCloud/osxphotos lag and frame rate (fallback only).
3. Build `tools/watch_and_send.py` on the winning source.

### Rejected / deferred
- **AirDrop**: transcodes to 30 fps.
- **iPhone Shortcut → HTTP POST**: Shortcuts can't reliably drive slo-mo; extra friction per shot.
- **Sideloaded Swift capture app** (rolling buffer + audio-spike trigger, fully hands-free): deferred. Free Apple ID needs a reinstall every 7 days. Revisit only if two Watch taps per shot proves annoying.

**Putting:** not trackable with this setup. Will putt manually in the sim (mouse/controller).

## Rules

- **Keep large video files out of git.** `samples/*.mov|mp4|avi|mkv|hevc` are gitignored. Use a shared folder or external drive for clips.
- **Add pytest tests for any math** (pixel-to-mm, speed, angle calculations) in `tests/`.
- **Update this file** whenever directory structure or commands change.
