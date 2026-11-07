# OpenGolfSim Sim Protocol

Sources: https://help.opengolfsim.com/desktop/apis/ and https://help.opengolfsim.com/desktop/apis/examples/

## Connection

- TCP, port **3111**, host = machine running OpenGolfSim (`127.0.0.1` for local, LAN IP otherwise)
- No authentication or handshake

## Message Framing

**CONFIRMED (from code examples):** Each JSON message is terminated by a single newline `\n`. No length prefix.

```
json_string + "\n"   →  encoded as UTF-8
```

## Client → Sim Messages

### Device status

```json
{"type": "device", "status": "ready"}
{"type": "device", "status": "busy"}
```

Send "ready" after connecting and before each shot. Send "busy" when unavailable.

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

Field ranges:
| Field | Range | Unit |
|---|---|---|
| ballSpeed | > 0 | mph (imperial) or m/s (metric) |
| verticalLaunchAngle | 0 – 45 | degrees |
| horizontalLaunchAngle | -45 – 45 | degrees (positive = right of target) |
| spinSpeed | > 0 | RPM |
| spinAxis | -45 – 45 | degrees |

`"unit"` is optional; defaults to `"imperial"`.

## Sim → Client Events

### Player event

Sent when the player or club selection changes.

```json
{
  "type": "player",
  "data": {
    "playerId": "<uuid>",
    "currentPosition": {"x": 0, "y": 0, "z": 0},
    "club": {
      "name": "Driver",
      "id": "DR",
      "distance": 230
    }
  }
}
```

### Result event

Sent after the ball lands. All distances in **meters**.

```json
{
  "type": "result",
  "data": {
    "result": {
      "carry": 180.0,
      "height": 28.0,
      "roll": 10.0,
      "total": 190.0,
      "lateral": 2.0
    },
    "club": {"name": "Driver", "id": "DR", "distance": 230},
    "shot": {
      "ballSpeed": 150,
      "verticalLaunchAngle": 12,
      "horizontalLaunchAngle": 0,
      "spinAxis": 0,
      "spinSpeed": 2800
    },
    "sessionId": 1
  }
}
```

## Ack / Error Replies

**CONFIRMED (live test 2026-11-07):** After receiving a shot message, the sim echoes it back as:

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

No error messages observed. No explicit ack for the device-ready message.

**UNVERIFIED:** Whether an error reply is sent for malformed or out-of-range values.

## Operation Order

1. TCP connect to port 3111
2. Send `{"type":"device","status":"ready"}\n`
3. Send shot message (newline-terminated)
4. Read lines from socket — expect "player" event (club info) and "result" event after ball lands
5. Repeat from step 2 for next shot
