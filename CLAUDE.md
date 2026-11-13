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
            player_targets.toml — the player's measured per-club carry/dispersion (calibration targets)
tools/      Scripts: watch_and_send.py (round loop), afc_source.py (USB transfer), clip_to_sim.py,
            set_ball_spot.py, calibration_report.py, send_fake_shot.py, calibrate_dispersion.py
inbox/      Round-time clip inbox (gitignored); processed → inbox/done/, failures → inbox/failed/
samples/    Local 240 fps test clips (not committed — see .gitignore)
docs/       Protocol and calibration notes
tests/      pytest suite — all math functions must have tests
```

## Commands

```bash
# Setup — must use Python 3.13 (opencv doesn't support 3.14 yet)
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,stream]"   # stream = pymobiledevice3, for USB clip transfer

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

# Continuous shot pipeline for a round (relay + sim running, iPhone USB-tethered & unlocked)
python tools/watch_and_send.py                    # USB: iPhone camera roll over AFC (pymobiledevice3)
python tools/watch_and_send.py --source dir       # only watch inbox/ (iCloud/osxphotos export, manual drop)
python tools/watch_and_send.py --dry-run --club PW   # test the loop without the sim

# Calibrate the tee spot once per session (and whenever the tripod or tee mark moves)
python tools/set_ball_spot.py                     # click the ball in the newest clip; Enter saves
python tools/set_ball_spot.py --xy 830,675        # no window
python tools/set_ball_spot.py --clear             # back to searching the whole frame
python tools/set_ball_spot.py --xy 830,675 --diameter-px 31   # set the speed scale by hand if auto-measure fails
# set_ball_spot also measures the ball size → session px/mm (logs/ball_spot.json), used for every shot.
# Calibrate it on an iron shot with the ball on the mat (a teed driver ball measures small).

# Per-club report: sim carry vs the player's known averages (config/player_targets.toml)
python tools/calibration_report.py                # last 4 h of shots
python tools/calibration_report.py --since 2026-11-08T16:00

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

**Phase 3 — Calibration. The full loop works (iPhone → USB → vision → shot model → relay → OpenGolfSim → spoken result); readings must now be made reliable and calibrated to the player's known averages.**

### Start here next session
1. **Irons read ~15% slow** (2026-11-13: 7i median ~102 mph vs ~120 needed for the 173-yd target; driver ~143 mph fits its 226-yd target). Investigate offline on the 2026-11-13 clips before the next bag run: is the ball's flight plane farther from the camera than the address spot (scale too small in flight)? Are the iron targets optimistic? Also look at the 2 remaining failures (IMG_0389 5i, IMG_0395 4i). All PRs (#2, #3) are merged; work from `main`.
2. **Bag run** when the user says so. Sessions are short daylight windows (~20–30 min): start early, hit ONE shot first and confirm it reads before the full set; if shots fail mid-session keep capturing (clips land in `inbox/failed/`) and debug offline afterwards. Start `python -m relay` and `python tools/watch_and_send.py` (live, no `--dry-run`); OpenGolfSim range open; user selects each club in the sim and hits 5–8 shots per club. **Run `tools/set_ball_spot.py` on the first clip** (an iron, ball on the mat): it sets the tee spot *and* the session speed scale — check it prints a sensible "Session scale" (2026-11-13: 0.755 px/mm, 32 px ball). Without it, speed scale falls back to noisy per-shot estimates.
3. Afterwards: review every clip (frames around impact, are the detections really the ball?), `--exclude` confirmed misreads, then `python tools/calibration_report.py`.
4. Tune toward `config/player_targets.toml`: similar carry Δ% across clubs → speed-measurement scale; uneven → per-club spin/launch in `config/shot_model.toml`. Target ≈ ±5% average carry per club. "Good enough", not perfect.

**Constraints (from the user):** outdoor setup is mat + net only — never propose backdrops, curtains or other equipment. Free changes (tripod position/angle, framing) are fine to suggest. Per-session steps must stay minimal.

**Known state of accuracy (2026-11-08, 16-clip live session, re-run on current code):** ~half of clips read believably; the rest fail safely ("ball not tracked") or misread. Main causes: ball leaves the frame in 2–3 frames at 3W/7i speed (camera close, ball ~70 px), cluttered background (white slides, toys, people, dog), dappled sun. One confirmed misread is excluded (14.1 mph on a full, clean 7i). Sim carry vs targets so far: 7i −30%, 3W −21%, SW −7% (few shots) — speed measurement vs sim/spin model not yet separated.

**2026-11-09 dusk session (5 PW, all failed live, then fixed offline):** low light → low-contrast ball on a dewy mat (impact detector thought the ball never left) and a motion-blurred, capsule-shaped ball in flight (tracker rejected it as not round); clubhead taken as the first detection. Fixed in `vision-low-light`: contrast-adaptive impact margin, ellipse-based blurred-ball shape test, re-track skipping a bad first detection, launch angle bounded to −10…65°. Now 2/5 read (75.6 mph/26°, 68.5 mph/31°); remaining failures: ball teed outside the calibrated spot, dog standing on the spot late in the clip. Camera is now face-on at ground level, ball ~48 px. Keep clips short (stop right after impact).

**2026-11-13 bag session (9i→driver, 30 clips, sunny, ball often in shade):** live only 15/30 read; offline after fixes 28/30. Fixes: (1) calibrated-spot area scan — the white-blob finder misses a shaded, blue-tinted ball against a sunlit frosty mat/fence, so the pipeline also watches the area around the tee spot directly; (2) the search band around the spot is wide sideways (±4 r, teed by eye) but tight vertically (0.5 r above, 1.5 r below) — spots above the mark were shoes/clubhead; (3) **session speed scale**: per-shot px/mm scattered 0.49–0.99 (the "hole" method under-measured → 155–180 mph misreads); now one scale per session from `set_ball_spot.py` (median Hough fit over ~40 address frames; 0.755 px/mm = 32 px ball today). Corrected medians: 9i ~92, 8i ~108, 7i ~102, 6i ~120, 5i ~115, 4i 122, driver ~143 mph. Driver fits the 226-yd target; irons read ~15% slow vs the 7i 173-yd target — not yet known whether that's measurement (ball depth/plane) or optimistic iron targets. Still failing: IMG_0389 (5i), IMG_0395 (4i). These shots went to the sim with the old per-shot scale, so `calibration_report.py` on 2026-11-13 is not meaningful — re-send with the session scale if needed.

**Simulator options:** staying on OpenGolfSim (free; small course library). Awesome Golf has no third-party/API input — only its partner launch monitors. GSPro has an open "Open Connect" API (JSON over TCP) and thousands of community courses; adding it as a second relay output would be a modest change if the user wants more courses.

| Item | Status |
|---|---|
| Project scaffold | Done |
| `docs/sim-protocol.md` | Done — protocol fully confirmed incl. 200ms delay between device-ready and shot |
| `tools/send_fake_shot.py` | Done — confirmed working (255 yd / 235 carry / 78 ft apex at 150 mph, 12° VLA) |
| Relay service (`relay/`) | Done — HTTP :3001 → TCP :3111, logs shots.jsonl, tracks current club from sim player events |
| Vision pipeline (`vision/`) | Working, not yet reliable — single streaming pass (bounded memory); ball = resting spot that leaves a round, ball-sized hole at impact (`vision/impact.py`); tracker picks white + round changed blobs, prediction-gated, must extrapolate back to the spot; tracks > 220 mph rejected |
| Auto-calibration | Done — px/mm from the hole the ball leaves (frame before vs 8 after impact); address/flight size are fallbacks. Optional per-session ball spot (`tools/set_ball_spot.py`) |
| Calibration tooling | Done — `config/player_targets.toml` (player's per-club averages), `tools/calibration_report.py` (sim carry vs targets, failures, `--exclude` misreads) |
| `tools/clip_to_sim.py` | Done — full end-to-end: clip → vision → relay → sim; confirmed working (76 mph / 30° VLA / 101 yd) |
| Spin estimation | Done — `shotmodel`: handicap-interpolated mid spin × (1 + per-club %/deg × VLA deviation) |
| Contact quality score | Done — q from VLA z (two-sided) + speed z vs rolling per-club median (one-sided); history in `logs/speed_history.json` |
| Horizontal launch angle | Simulated — correlated draw with spin axis, sd scaled by 1 + k(1−q), fat tails, personal bias |
| Spin axis (side spin) | Simulated — see above; sds need calibrating (`tools/calibrate_dispersion.py`) |
| Shot streaming for a round | Done — `tools/afc_source.py` (USB/AFC) + `tools/watch_and_send.py`; transfer, relay, sim and speech confirmed live 2026-11-08. Accuracy limited by vision (see above) |
| Live iPhone capture | Not started |

## Background: Spin Axis (Side Spin) Model — implemented in `shotmodel/` (original plan kept for context)

The sim's `spinAxis` field controls draw/fade/straight. Currently hardcoded to 0 (straight).

**Goal:** build a statistical estimator for spin axis using club, ball speed, and VLA as inputs — similar to how `spinSpeed` is estimated. Approach:

- Compile amateur dispersion data by club (e.g. average fade/draw tendency, sigma of axis distribution)
- Fit a model: `spinAxis = f(club, ballSpeed, vla)` — likely a linear regression or lookup with noise
- Higher-lofted clubs (wedges) tend toward more consistent axis; driver has highest axis variance
- Could seed with PGA/amateur Trackman datasets or published dispersion charts

This won't be measured (that requires two cameras or spin dots) but a plausible statistical distribution will make course play feel realistic rather than always hitting dead straight.

## Background: Contact Quality Score → Dispersion Scaling — implemented in `shotmodel/` (original plan kept for context)

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

## Shot Streaming Workflow for a Round (built)

**Decided 2026-11-07.** Every shot flows: Watch remote start/stop → clip lands on Mac → pipeline → relay → sim → spoken result.

### Capture
- iPhone on tripod, stock Camera app in **Slo-mo** mode at **1x** (Photo mode makes 30 fps Live Photo clips — the watcher rejects them with a spoken "not slow motion").
- Nothing white directly behind the ball or along its first ~1 m of flight from the camera's view (a white slide there made the ball undetectable).
- **Setup tips (free changes only — no added equipment, per the user):**
  - Tripod back far enough that the ball is ~35–40 px across, so it stays in frame ≥ 5 frames at 3W speed (at ~70 px it left in 2–3 and one bad detection ruined the reading).
  - Angle/place the tripod so white objects (slides, toys) aren't directly behind the ball's first ~1 m of flight.
  - Keep the tripod fixed during a session; optionally run `tools/set_ball_spot.py` once at the start.
- **Apple Watch Camera Remote** starts/stops recording (confirmed working in Slo-mo — `samples/IMG_0278.mov` was shot this way).
- Two taps per shot: start just before the swing, stop right after. Keep clips short (pipeline loads every frame).

### Transfer: cable (primary)
- iPhone stays **USB-tethered to the Mac** (also charges it through a round).
- `tools/afc_source.py` polls the phone's `DCIM/` folders over **AFC** (Apple File Conduit, via `pymobiledevice3`) and downloads each new original `IMG_####.MOV` once its size settles. No iCloud, no AirDrop transcode. Phone must be trusted.
- **ImageCaptureCore/PTP was tried and rejected (2026-11-08):** with Optimize iPhone Storage on, the phone exposes only a small stale subset (20 items) over PTP — new clips never appeared, even after replugging; Image Capture.app showed the same. AFC sees them immediately.
- iPhone **Settings → Apps → Photos → Transfer to Mac or PC → Keep Originals**, otherwise iOS may transcode on transfer. The watcher rejects clips under `--min-fps` (default 200, measured from per-frame timestamps).

### Transfer: iCloud (fallback)
- iCloud Photos sync + `osxphotos` exporting new slo-mo originals. Needs Photos running with "Download Originals to this Mac". Lag 10–60+ s, so practice only.

### Mac side: `tools/watch_and_send.py` (built)
- Source-agnostic: takes clips from whichever transfer route is active, so the rest of the flow is shared.
- Process each clip **exactly once** (persist handled IDs); ignore anything recorded before the watcher started.
- Run vision → shotmodel → relay; club comes from the relay's tracked sim club (no input mid-round).
- **Speak every result** (macOS `say`, e.g. "PW, 96 mph, 28 degrees") and **announce failures** loudly so you re-hit or play it manually.

### Steps
1. ~~Throwaway test: Watch "stop" → clip downloaded, original 240 fps?~~ Done via AFC: 1.8 s for 72 MB, 240.0 fps measured (`CAP_PROP_FPS` reported 177 — it's a VFR average, never trust it).
2. Throwaway test: iCloud/osxphotos lag and frame rate (fallback only).
3. ~~Build `tools/watch_and_send.py` on the winning source.~~ Done (AFC).

### Rejected / deferred
- **AirDrop**: transcodes to 30 fps.
- **iPhone Shortcut → HTTP POST**: Shortcuts can't reliably drive slo-mo; extra friction per shot.
- **Sideloaded Swift capture app** (rolling buffer + audio-spike trigger, fully hands-free): deferred. Free Apple ID needs a reinstall every 7 days. Revisit only if two Watch taps per shot proves annoying.

**Putting:** not trackable with this setup. Will putt manually in the sim (mouse/controller).

## Rules

- **Keep large video files out of git.** `samples/*.mov|mp4|avi|mkv|hevc` are gitignored. Use a shared folder or external drive for clips.
- **Add pytest tests for any math** (pixel-to-mm, speed, angle calculations) in `tests/`.
- **Update this file** whenever directory structure or commands change.
