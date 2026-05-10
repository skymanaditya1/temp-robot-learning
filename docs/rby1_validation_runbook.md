# RBY1 End-to-End Validation Runbook

Bottom-up validation of the Jetson ↔ Zima RBY1 stack. Run each step in
order, only proceed once the **Pass criteria** are met. Designed for
left-arm-only operation with the proxy actively holding the right side
(see commits `e205483`, `d931fc6`).

Hosts:
- **Jetson** — runs cameras, RBY1 SDK client, robot proxy. IPs:
  `10.45.1.14`, `192.168.55.1`, `192.168.30.2`.
- **Zima** — workstation, runs subscribers, recording, training, policy
  rollout.

> Convention: commands prefixed `[J]` run on Jetson, `[Z]` on Zima.
> Replace `<JETSON>` with the Jetson IP reachable from Zima
> (typically `10.45.1.14`).

---

## Step 1 — ZED publisher / subscriber sanity

Goal: confirm 3 cameras stream from Jetson to Zima with correct color
order and stable framerate.

### 1a. Start publisher (Jetson)

```bash
[J] cd /data/objsearch/rby1_policy_learning
[J] bash scripts/rby1/start_zed_publisher.sh
```

Wait until logs show all 3 cameras opened and frames being published.
Ports: head=5555 (stereo), right_wrist=5556 (mono), left_wrist=5557 (mono).

### 1b. Loopback verify (Jetson)

In a second Jetson shell:

```bash
[J] uv run python scripts/rby1/verify_zmq_rgb.py --port 5555 --camera-name head_cam
[J] uv run python scripts/rby1/verify_zmq_rgb.py --port 5556 --camera-name right_wrist_cam
[J] uv run python scripts/rby1/verify_zmq_rgb.py --port 5557 --camera-name left_wrist_cam
```

Each writes `temp_images/zmq_raw_bgr.png` and `zmq_fixed_rgb.png`.
Open the `_fixed_rgb.png` — colors should look correct (no blue/red swap).

### 1c. Network verify from Zima

```bash
[Z] cd <repo>
[Z] uv run python scripts/rby1/verify_zmq_rgb.py --address <JETSON> --port 5555 --camera-name head_cam
[Z] uv run python scripts/rby1/verify_zmq_rgb.py --address <JETSON> --port 5556 --camera-name right_wrist_cam
[Z] uv run python scripts/rby1/verify_zmq_rgb.py --address <JETSON> --port 5557 --camera-name left_wrist_cam
```

### 1d. Sustained-rate sanity (Zima)

Run this short loop on Zima to check fps + frame shape via the same
`ZMQCamera` class the recorder/policy uses:

```bash
[Z] uv run python - <<'PY'
import time
from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig

JETSON = "<JETSON>"
cams = {
    "head_cam":        ZMQCamera(ZMQCameraConfig(server_address=JETSON, port=5555, fps=10, width=640, height=480)),
    "right_wrist_cam": ZMQCamera(ZMQCameraConfig(server_address=JETSON, port=5556, fps=10, width=640, height=480)),
    "left_wrist_cam":  ZMQCamera(ZMQCameraConfig(server_address=JETSON, port=5557, fps=10, width=640, height=480)),
}
for c in cams.values():
    c.connect()

t0 = time.time(); n = {k: 0 for k in cams}
while time.time() - t0 < 5.0:
    for k, c in cams.items():
        f = c.async_read(timeout_ms=200)
        if f is not None:
            n[k] += 1
            shape = f.shape

dt = time.time() - t0
for k in cams:
    print(f"{k:18s}  {n[k]/dt:5.2f} Hz   shape={shape}")
PY
```

### Pass criteria
- ≥9 Hz on all 3 cameras
- shape `(480, 640, 3)`
- no `RCVTIMEO` / no `None` frames
- `_fixed_rgb.png` looks visually correct

---

## Step 2 — Robot proxy: state PUB + action PULL

Goal: verify state stream Jetson→Zima and action path Zima→Jetson without
cameras or policy. Start with read-only, then small commanded deltas, then
parquet round-trip.

### 2a. State streaming (read-only, safe)

```bash
[J] cd /data/objsearch/rby1_policy_learning
[J] bash scripts/rby1/start_robot_proxy.sh
```

Watch logs for: SDK gRPC connect to `192.168.30.1:50051`, DXL bus open on
`/dev/rby1_gripper`, gripper writer thread up, state PUB on 5560.

On Zima, read state for 10 s:

```bash
[Z] uv run python - <<'PY'
import time
from lerobot.robots.rby1_remote import RBY1Remote, RBY1RemoteConfig

robot = RBY1Remote(RBY1RemoteConfig(jetson_host="<JETSON>"))
robot.connect()

t0 = time.time(); n = 0; last_ts = None; gaps = []
while time.time() - t0 < 10.0:
    obs = robot.get_observation()
    ts = obs.get("ts") or obs.get("observation.ts")
    if last_ts is not None and ts != last_ts:
        gaps.append(ts - last_ts)
    last_ts = ts
    n += 1
    time.sleep(0.005)

print(f"Got {n} observations in 10 s -> {n/10:.1f} polls/s")
print("Joint keys:", sorted(k for k in obs if "joint" in k.lower() or "arm" in k.lower())[:8], "...")
print("Gripper keys:", [k for k in obs if "gripper" in k.lower()])
if gaps:
    import statistics
    print(f"State ts gaps: median={statistics.median(gaps)*1000:.1f}ms  max={max(gaps)*1000:.1f}ms  count={len(gaps)}")
robot.disconnect()
PY
```

**Pass:** ≥90 distinct state ts updates per second; all joints + 2 gripper
keys present; max ts gap < ~30 ms.

### 2b. Command echo test (small, safe)

Pick a safe joint and send small deltas. **Robot must be in a clear pose,
arms hanging or in cradle.**

```bash
[Z] uv run python - <<'PY'
import time
from lerobot.robots.rby1_remote import RBY1Remote, RBY1RemoteConfig

robot = RBY1Remote(RBY1RemoteConfig(jetson_host="<JETSON>"))
robot.connect()
obs0 = robot.get_observation()

# Start with a single, well-known joint. Replace 'head_pitch' with the
# actual key after inspecting obs0. Send +/-0.05 rad relative to current.
joint = "head_pitch"   # EDIT after you see the keys
target = obs0[joint] + 0.05
robot.send_action({joint: target})
time.sleep(0.5)
obs1 = robot.get_observation()
print(f"{joint}: before={obs0[joint]:.4f}  target={target:.4f}  after={obs1[joint]:.4f}")

# Return
robot.send_action({joint: obs0[joint]})
time.sleep(0.5)
robot.disconnect()
PY
```

### 2c. Gripper round-trip

Same harness, command gripper meters (open ~5 mm wider, then back). The
gripper writer thread runs the DXL bus — confirm `gripper_*` keys in PUB
state reflect the new commanded width within ~200 ms.

### 2d. Frozen-right behavior

Confirm `start_robot_proxy.sh` flags actually freeze the right side:

```bash
[J] grep -E 'freeze|frozen|right' scripts/rby1/start_robot_proxy.sh
[J] head -40 scripts/rby1/robot_proxy.py
```

Then on Zima send only **left-arm** joint commands (no `right_*` keys in
the action dict). Watch the state PUB stream — `right_arm_*` joints should
remain ~constant for 10 s.

### 2e. Parquet round-trip via lerobot-record

A 5 s no-op record using `rby1_remote` to confirm both
`observation.state` and `action` carry joint + gripper columns:

```bash
[Z] uv run lerobot-record \
      --robot.type=rby1_remote \
      --robot.jetson_host=<JETSON> \
      --dataset.repo_id=local/proxy_smoke_$(date +%s) \
      --dataset.num_episodes=1 \
      --dataset.episode_time_s=5 \
      --teleop.type=<your teleop or 'none' equivalent>
[Z] # then inspect parquet:
[Z] uv run python - <<'PY'
import pandas as pd, glob
p = sorted(glob.glob("datasets/local/proxy_smoke_*/data/chunk-*/episode_*.parquet"))[-1]
df = pd.read_parquet(p)
print("cols:", [c for c in df.columns if "gripper" in c or "state" in c or "action" in c])
print(df.filter(like="gripper").describe())
PY
```

### Pass criteria
- 2a: ≥90 Hz, all joint+gripper keys, ts gaps < 30 ms.
- 2b: commanded joint settles to target within 100–200 ms.
- 2c: gripper meters in state reflect command within ~200 ms.
- 2d: right-arm joints stay frozen when only-left actions are sent.
- 2e: parquet has non-NaN gripper + joint columns in both state and action.

---

## Step 3 — Teleop record + replay (with cameras)

### 3a. Record locally on Jetson (no network in the loop)

Prereqs: ZED publisher (Step 1) running, master-arm teleop server running
(see `record_dataset.sh` header). Then:

```bash
[J] bash scripts/rby1/record_dataset.sh
```

Stop after one short episode (~10 s). Note the dataset path printed.

### 3b. Visual replay (Rerun, no robot)

```bash
[J] bash scripts/rby1/replay_dataset_rerun.sh <dataset_path>
```

Confirm: 3 image streams, joint+action plots, no dropped frames.

### 3c. Replay on robot — Jetson local

```bash
[J] uv run python scripts/rby1/replay_dataset_rerun.py <dataset_path> \
      --replay-on-robot --robot.type=rby1
```

Confirm robot tracks recorded actions at 10 Hz with no SDK timeouts.

### 3d. Replay on robot — Zima → Jetson (the real test)

Stop the local-record processes. On Jetson run only the proxy:

```bash
[J] bash scripts/rby1/start_robot_proxy.sh
```

On Zima, replay the recorded dataset through the proxy:

```bash
[Z] uv run python scripts/rby1/replay_dataset_rerun.py <dataset_path> \
      --replay-on-robot \
      --robot.type=rby1_remote \
      --robot.jetson_host=<JETSON>
```

Watch on Jetson side:
- Proxy log: time between received actions; gaps > 300 ms indicate
  starvation.
- Robot motion smoothness vs. local replay (3c).

### Pass criteria
- 3a: dataset has 3 camera tracks + state + action.
- 3b: clean Rerun playback.
- 3c: smooth robot motion, no SDK timeouts.
- 3d: same as 3c, with action-receive period ~80–150 ms p99.

---

## Step 4 — Data collection + ACT training

Only attempt once 1–3 are green.

### 4a. Collect

~50–100 short episodes via `record_dataset.sh`, target task. After:

```bash
[J] bash scripts/rby1/analyze_dataset.sh <dataset_path>
```

Look for: episode-length distribution, action range per joint, gripper
actuation distribution (both open and closed states represented).

### 4b. Train ACT

```bash
[Z] uv run lerobot-train --config_path=<your left-only ACT config>
```

Use the config established in commit `d931fc6` as the baseline.

### 4c. Offline validation

```bash
[Z] uv run python scripts/rby1/validate_policy.py \
      --policy-path checkpoints/<step>/pretrained_model \
      --dataset-repo-id <dataset> \
      --output-dir temp_images/validation_$(date +%s)
```

Inspect per-joint MAE + the 4×N PNG plots.

### 4d. Online rollout (Zima → Jetson)

```bash
[J] bash scripts/rby1/start_robot_proxy.sh
[Z] bash scripts/rby1/rollout_policy_remote.sh
```

`rollout_policy_remote.sh` already passes
`--robot.with_right_arm=false --robot.with_right_gripper=false` so the
right side stays frozen via the proxy snapshot.

### Pass criteria
- 4a: dataset diversity reasonable; no all-zero gripper episodes.
- 4c: per-joint MAE comparable to training curves.
- 4d: one full episode completes; ZED frames ≥9 Hz on Zima during
  rollout; robot follows policy smoothly; right side does not drift.

---

## Cheat-sheet of critical files

- ZED publisher: `scripts/rby1/zed_zmq_publisher.py`
- ZMQ subscriber: `src/lerobot/cameras/zmq/camera_zmq.py:209,216`
- Verify tool: `scripts/rby1/verify_zmq_rgb.py`
- Robot proxy loops: `scripts/rby1/robot_proxy.py:68-225`
- Proxy launcher: `scripts/rby1/start_robot_proxy.sh`
- RBY1 driver: `src/lerobot/robots/rby1/robot_rby1.py`
- RBY1 config: `src/lerobot/robots/rby1/config_rby1.py`
- Remote client: `src/lerobot/robots/rby1_remote/robot_rby1_remote.py`
- Recorder: `scripts/rby1/record_dataset.sh`
- Replay: `scripts/rby1/replay_dataset_rerun.py`
- Offline validate: `scripts/rby1/validate_policy.py`
- Online rollout: `scripts/rby1/rollout_policy_remote.sh`
