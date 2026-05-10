# RBY1 End-to-End Component Test Checklist

## Context

The RBY1 policy-learning workflow spans two hosts (Jetson on the robot, Zima
workstation for training/rollout) and several streaming layers (ZED ZMQ, robot
state PUB/PULL, RBY1 gRPC SDK, Dynamixel USB for grippers). When something is
wrong end-to-end it is often hard to tell which layer is at fault. The goal of
this checklist is to walk component-by-component, bottom-up, so that each
layer is proven good before the next is exercised.

Scope for this pass:
- **Left-arm only** (right arm + right gripper held frozen by the proxy).
- All the way through to **online Zima → Jetson policy rollout**.
- Pure walkthrough — sequencing checks and pass criteria. We are not
  implementing or modifying scripts in this round; gaps are noted at the end.

Hosts referenced:
- `[J]` = Jetson (on robot). Cameras, RBY1 SDK client, gripper DXL bus, robot proxy.
- `[Z]` = Zima (workstation). Subscribers, recorder, training, policy rollout.

---

## Step 1 — ZED publisher streams correctly

**Goal:** confirm 3 ZED cameras (head stereo on port 5555, right wrist mono on
5556, left wrist mono on 5557) publish frames at the expected rate, with
correct color order, from the Jetson.

**Sub-steps:**
1. Start the ZED publisher on the Jetson.
2. Verify in the publisher logs that all 3 cameras opened and are publishing.
3. **Loopback on the Jetson** — subscribe locally (127.0.0.1) on each of 5555 /
   5556 / 5557, decode one frame, save as PNG. Confirms publisher health
   independent of the network path.

**Pass criteria:**
- Each camera shows ≥9 Hz publish rate.
- Decoded frame shape is `(480, 640, 3)`.
- Saved PNG looks visually correct (no R/B swap, not all black).

---

## Step 2 — ZED subscriber on Zima receives frames

**Goal:** confirm the Zima workstation can connect to the publisher across the
network and receive frames through the same `ZMQCamera` class the recorder and
policy use.

**Sub-steps:**
1. From Zima, run the verify tool against each of 5555 / 5556 / 5557 with the
   Jetson's IP. Confirms basic network reachability.
2. Run a short multi-camera fps probe on Zima for ~5 s using `ZMQCamera`
   directly — the same code path used at record-time and rollout-time.

**Pass criteria:**
- All 3 cameras show ≥9 Hz on Zima.
- No `RCVTIMEO` errors, no `None` frames returned.
- Frame shape `(480, 640, 3)` matches loopback test.

---

## Step 3 — Robot joint + gripper state streams Jetson → Zima

**Goal:** confirm joint state (14-DOF) and 2 gripper widths flow from the
robot, through the proxy, out via ZMQ state PUB on port 5560, and into the
`RBY1Remote` client on Zima.

**Sub-steps:**
1. Start the robot proxy on the Jetson. Verify the logs show:
   - RBY1 SDK gRPC connect succeeds (`192.168.30.1:50051`).
   - Dynamixel bus opens (`/dev/rby1_gripper`) and gripper writer thread starts.
   - State PUB socket binds on port 5560.
   - Gripper homing completes successfully and reports the homed encoder range.
2. From Zima, instantiate `RBY1Remote`, connect, poll `get_observation()` for
   ~10 s. Inspect:
   - Number of distinct timestamps received.
   - Inter-sample timestamp gaps (median, max).
   - Presence of all expected keys: 14 joint keys (right_arm_0..6,
     left_arm_0..6) + 2 gripper keys (right_gripper, left_gripper).

**Pass criteria:**
- ≥90 distinct state timestamps per second.
- Max ts gap < ~30 ms.
- All 14 joint keys + 2 gripper keys present.
- Joint values look plausible (radians, not NaN, not constant zero).
- Gripper values in meters within [0.0, 0.1].

---

## Step 4 — Action path Zima → Jetson (proxy command echo)

**Goal:** confirm the action PULL path (port 5561) actually drives the robot
and gripper, and that the resulting motion shows up in the state stream.

**Sub-steps (with robot in a clear, safe pose):**
1. Read current state on Zima.
2. Send a small joint delta on a single low-risk left-arm joint (e.g., +0.05
   rad) via `RBY1Remote.send_action()`. Read state again, confirm the joint
   moved toward target. Return to original.
3. Send a gripper width command (e.g., open ~5 mm wider, then back). Confirm
   the `gripper_*` value in the state PUB stream reflects the new width.
4. **Frozen-right check:** with only left-arm joints in the action dict, watch
   right-arm joints in the state stream for 10 s — they must not drift.

**Pass criteria:**
- Commanded joint settles toward target within 100–200 ms.
- Gripper state mirrors commanded width within ~200 ms.
- Right arm + right gripper hold their pose within sensor noise.

---

## Step 5 — Teleop record (master-arm → recorded dataset)

**Goal:** confirm the master-arm teleop framework drives the robot and that
synchronized observations (3 cameras + joints + grippers) and actions
(commanded joints + grippers) all land in the recorded dataset.

**Sub-steps:**
1. Prereqs running: ZED publisher (Step 1) + master-arm teleop server.
2. Run the record script for one short episode (~10 s).
3. Inspect the resulting parquet file:
   - Columns include 3 image refs, `observation.state.*` joint+gripper,
     `action.*` joint+gripper.
   - No NaN columns.
   - Gripper values are in meters and span both open and closed during the
     episode (if the episode included a grasp).
   - Joint values are in radians.

**Pass criteria:**
- One episode written successfully, dataset path printed.
- Parquet has all expected columns, all non-NaN.
- Episode length and per-joint action range look sane.

---

## Step 6 — Visual replay of recorded trajectory

**Goal:** confirm the recorded dataset is internally consistent before we ever
touch the robot again.

**Sub-steps:**
1. Run the Rerun-based replay against the dataset recorded in Step 5.
2. Visually inspect:
   - 3 image streams render and step in sync.
   - Joint and action time-series plots have no obvious dropouts or spikes.
   - Action trace leads the state trace by ~1 step (sanity check).

**Pass criteria:**
- Clean playback, no decode errors, all 3 cameras present.

---

## Step 7 — Replay trajectory on robot, Jetson-local

**Goal:** confirm action playback works without the network in the loop. This
isolates SDK + gripper bus from ZMQ proxy issues.

**Sub-steps:**
1. With the recorded dataset, run replay on the Jetson directly with
   `--robot.type=rby1` (no proxy).
2. Watch SDK/gripper logs for timeouts.
3. Compare resulting motion against the original episode visually.

**Pass criteria:**
- Robot tracks recorded actions at 10 Hz with no SDK timeouts.
- Smooth motion; gripper opens/closes as recorded.

---

## Step 8 — Replay trajectory on robot, Zima → Jetson

**Goal:** the same replay as Step 7, but driven from Zima through the proxy.
This is the integration test for state PUB + action PULL under realistic load.

**Sub-steps:**
1. Stop any local-record processes on the Jetson; leave only the proxy
   running.
2. From Zima, run replay with `--robot.type=rby1_remote` pointed at the Jetson.
3. Watch the proxy log for action-receive intervals.

**Pass criteria:**
- Robot tracks recorded actions at 10 Hz with similar smoothness to Step 7.
- Action-receive period p99 in the 80–150 ms range (no starvation gaps > 300 ms).
- Right side stays frozen.

---

## Step 9 — Collect a real dataset (left-arm only)

**Goal:** produce a training-grade dataset.

**Sub-steps:**
1. Record ~50–100 short episodes of the target task using the same setup
   verified in Steps 5/6.
2. Run the dataset analyzer to inspect:
   - Episode length distribution.
   - Per-joint action range.
   - Gripper actuation distribution (both open and closed states should be
     well represented; not all-zero, not stuck open).

**Pass criteria:**
- Reasonable per-joint diversity.
- Gripper distribution is bimodal-ish (i.e., the task actually involves
  grasping).
- No corrupted episodes.

---

## Step 10 — Train ACT (left-only)

**Goal:** train a left-only ACT policy with the established hyperparameters.

**Sub-steps:**
1. Launch `lerobot-train` with the left-only ACT config:
   - `policy.type=act`
   - `policy.kl_weight=10`
   - `policy.chunk_size=20`
   - `policy.n_action_steps=1`
   - `policy.temporal_ensemble_coeff=0.01` (paper-faithful — confirm vs. 0.1
     in `train_rby1_pick_act.sh` before kicking off)
2. Monitor training:
   - Reconstruction loss trending down.
   - KLD loss bounded (does not blow up).
   - Validation per-joint MAE on a held-out split if configured.

**Pass criteria:**
- Training runs to completion without OOM / NaN.
- Loss curves look healthy (no divergence, no flatlining at first step).
- Final per-joint MAE on held-out data within expected envelope.

---

## Step 11 — Offline policy validation

**Goal:** before any robot motion, confirm the trained checkpoint produces
sensible actions on recorded observations.

**Sub-steps:**
1. Run `validate_policy.py` against the trained checkpoint with the same
   dataset (or a held-out set).
2. Inspect output:
   - Per-joint MAE numbers (consistent with training-time validation).
   - 4×N PNG plots: predicted vs. ground-truth trajectories per joint.
   - Gripper prediction reasonable (does not flatline at one extreme).

**Pass criteria:**
- Per-joint MAE comparable to training curves.
- Predictions visually track ground truth, especially around gripper events.

---

## Step 12 — Online policy rollout (Zima → Jetson)

**Goal:** run the full inference loop end-to-end. Zima reads ZED frames + joint
state from the Jetson, ACT predicts an action chunk, actions go back through
the proxy and execute on the robot.

**Sub-steps:**
1. Start ZED publisher on the Jetson (Step 1 already verified).
2. Start robot proxy on the Jetson (Step 3 already verified).
3. From Zima, launch the rollout script with the trained checkpoint and the
   left-only flags (`--robot.with_right_arm=false
   --robot.with_right_gripper=false`).
4. During the rollout, monitor:
   - ZED frame rate at the subscriber (should match Step 2).
   - State PUB rate at the subscriber.
   - Inference loop period (should be steady, no drift).
   - Action-receive period on the proxy side (Step 8 envelope).
   - Robot motion smoothness — no jerks, no SDK timeouts.
   - Right-arm joints stay frozen.

**Pass criteria:**
- One full episode completes without manual intervention.
- All four streams (ZED in, state in, action out, motion) hold steady rates.
- Right side does not drift.
- Behavior is recognizably the trained task (even if not perfect).

---

## Aggregate pass / fail

The system is considered "green" when Steps 1–12 all pass on a single
contiguous run on the same day. If a later step fails, regress to the
lowest-numbered failing component first; do not patch around it at a higher
layer.

---

## Notes on gaps to revisit later (out of scope for this pass)

These are observations from existing tooling that we may want to address after
this walkthrough — explicitly **not** part of executing this checklist:

- One-shot "check-everything" smoke script (currently each component has its
  own ad-hoc snippet).
- A latency profiler that reports end-to-end inference loop budget
  (camera-grab → preprocess → forward → send action → robot motion start).
- Automated regression on per-joint MAE between training runs.
- A documented procedure to confirm `temporal_ensemble_coeff` value used at
  training matches the value used at rollout.
