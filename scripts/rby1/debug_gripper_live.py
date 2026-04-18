#!/usr/bin/env python
"""Live-tail the gripper bus for 20 seconds.

Run WITH the master-arm teleop process running so we're seeing the same bus
contention the recorder sees. Squeeze the LEFT leader gripper during the window
and we'll see whether id 1's raw encoder value actually changes.

    conda run -n policy_new python scripts/rby1/debug_gripper_live.py
"""

import time

import rby1_sdk

DURATION_S = 15.0
RATE_HZ = 10.0


def main():
    bus = rby1_sdk.DynamixelBus(rby1_sdk.upc.GripperDeviceName)
    bus.open_port()
    bus.set_baud_rate(2_000_000)
    print("Bus open. Start squeezing the LEFT leader now.\n")
    print(f"{'t[s]':>5}  {'got':>4}  {'raw0':>8}  {'raw1':>8}  retries")

    period = 1.0 / RATE_HZ
    t0 = time.time()
    id0 = id1 = None
    id0_min = id0_max = id1_min = id1_max = None

    while time.time() - t0 < DURATION_S:
        t = time.time() - t0
        retries = 0

        # First the combined read
        try:
            rv = bus.group_fast_sync_read_encoder([0, 1])
        except Exception:
            rv = None
        got: dict[int, float] = {}
        if rv is not None:
            for dev_id, enc in rv:
                got[int(dev_id)] = float(enc)

        # Retry missing ids
        for missing in {0, 1} - set(got.keys()):
            retries += 1
            try:
                rv2 = bus.group_fast_sync_read_encoder([missing])
                if rv2 is not None:
                    for dev_id, enc in rv2:
                        got[int(dev_id)] = float(enc)
            except Exception:
                pass

        r0 = got.get(0)
        r1 = got.get(1)
        if r0 is not None:
            id0 = r0
            id0_min = r0 if id0_min is None else min(id0_min, r0)
            id0_max = r0 if id0_max is None else max(id0_max, r0)
        if r1 is not None:
            id1 = r1
            id1_min = r1 if id1_min is None else min(id1_min, r1)
            id1_max = r1 if id1_max is None else max(id1_max, r1)

        ids_str = ",".join(str(k) for k in sorted(got.keys())) or "-"
        s0 = f"{id0:.4f}" if id0 is not None else "n/a"
        s1 = f"{id1:.4f}" if id1 is not None else "n/a"
        print(f"{t:5.1f}  {ids_str:>4}  {s0:>8}  {s1:>8}  retries={retries}")

        time.sleep(period)

    print()
    print("Summary:")
    if id0_min is not None:
        print(f"  id 0  raw range: [{id0_min:.4f} .. {id0_max:.4f}]  span={id0_max - id0_min:.4f}")
    if id1_min is not None:
        print(f"  id 1  raw range: [{id1_min:.4f} .. {id1_max:.4f}]  span={id1_max - id1_min:.4f}")


if __name__ == "__main__":
    main()
