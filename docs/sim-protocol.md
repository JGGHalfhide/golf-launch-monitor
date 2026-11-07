# OpenGolfSim Sim Protocol

Sources: https://help.opengolfsim.com/desktop/apis/ and https://help.opengolfsim.com/desktop/apis/examples/

All findings confirmed by live testing against the native desktop app on 2026-11-07.

## Connection

- TCP, port **3111**, host = machine running OpenGolfSim (`127.0.0.1` for local, LAN IP otherwise)
- No authentication or handshake
- **Native desktop app only** — the web version does not expose port 3111

## Message Framing

Each JSON message is terminated by a single newline `\n`, UTF-8 encoded. No length prefix.

```
(json.dumps(msg) + "\n").encode("utf-8")
```

## Client → Sim Messages

### Device status

```json
{"type": "device", "status": "ready"}
{"type": "device", "status": "busy"}
```

Send "ready" after connecting and before each shot.

### Shot

```json
{
  "type": "shot",
  "unit": "imperial",
  "shot": {
    "ballSpeed": 150,
    "verticalLaunchAngle": 12,
    "horizontalLaunchAngle": 0,
    "spinSpeed": 2800,
    "spinAxis": 0
  }
}
```

| Field | Range | Unit |
|---|---|---|
| ballSpeed | > 0 | mph (imperial) or m/s (metric) |
| verticalLaunchAngle | 0 – 45 | degrees |
| horizontalLaunchAngle | -45 – 45 | degrees (positive = right of target) |
| spinSpeed | > 0 | RPM |
| spinAxis | -45 – 45 | degrees |

`"unit"` is optional; defaults to `"imperial"`.

**Validation behaviour (confirmed):**
- **Out-of-range values**: silently accepted and simulated. No error returned. The relay should clamp values before forwarding.
- **Missing fields**: shot silently dropped; no echo, no result, no error.
- **Malformed JSON**: returns `{"status":400,"error":"Unable to parse payload"}` (newline-terminated), connection stays open.

## Sim → Client Events

### Shot echo

Immediately after receiving a valid shot message, the sim echoes it back:

```json
{
  "type": "shot",
  "shot": {
    "ballSpeed": 150,
    "verticalLaunchAngle": 12,
    "horizontalLaunchAngle": 0,
    "spinSpeed": 2800,
    "spinAxis": 0
  }
}
```

This is the only acknowledgement for a valid shot. No ack for device-ready.

### Player event

Sent after a shot is received (and on club/position changes). Includes the full bag.

```json
{
  "type": "player",
  "data": {
    "player": {
      "name": "Tester",
      "id": "guest-1a1b745dd32",
      "clubs": [
        {"fullName": "Driver", "name": "DR", "id": "DR", "distance": 228.6, "angle": 10.5},
        {"fullName": "3 Wood",  "name": "3W", "id": "3W", "distance": 205.74, "angle": 14},
        "..."
      ]
    },
    "currentPosition": [0, 0, 0],
    "club": {
      "fullName": "Driver",
      "name": "DR",
      "id": "DR",
      "distance": 228.6,
      "angle": 10.5
    }
  }
}
```

**Corrections to official docs:** `currentPosition` is an array `[x, y, z]`, not an object. Each club has `fullName` and `angle` fields not mentioned in the docs.

### Result event

Sent after the ball lands. All distances in **meters**.

```json
{
  "type": "result",
  "data": {
    "result": {
      "apex": 23.76,
      "lateral": 0.0,
      "carry": 214.88,
      "total": 232.78,
      "roll": 17.91
    },
    "club": {
      "fullName": "Driver",
      "name": "DR",
      "id": "DR",
      "distance": 228.6,
      "angle": 10.5
    },
    "shot": {
      "ballSpeed": 150,
      "verticalLaunchAngle": 12,
      "horizontalLaunchAngle": 0,
      "spinSpeed": 2800,
      "spinAxis": 0,
      "id": 3
    },
    "sessionId": 1
  }
}
```

**Correction to official docs:** result field is `apex`, not `height`. No `height` field exists.

## Operation Order

1. TCP connect to port 3111
2. Send `{"type":"device","status":"ready"}\n`
3. Send shot (newline-terminated)
4. Receive shot echo `{"type":"shot",...}\n`
5. Receive `"player"` event (full bag)
6. Receive `"result"` event when ball lands
7. Repeat from step 2 for next shot

**Note:** "player" events also fire independently when the user changes club or position in the sim. The relay should handle them arriving at any time.

## Live Test Results (2026-11-07)

Shot: 150 mph, 12° VLA, 0° HLA, 2800 rpm, 0° axis

| Metric | Sim value | Converted |
|---|---|---|
| Carry | 214.88 m | 235 yd |
| Total | 232.78 m | 255 yd |
| Apex | 23.76 m | 78 ft |
| Roll | 17.91 m | 20 yd |
| Lateral | 0.0 m | straight |
