"""Plot the commanded-action JSON-lines log emitted by robot_proxy.py.

Reads a file written by `robot_proxy.py --log-actions <path>`, where each
line is JSON of the form:

    {"ts": <epoch_seconds>,
     "joints":   {"<joint_name>": <float>, ...},
     "grippers": {"<gripper_name>": <float>, ...}}

Produces a 4x4 grid (14 arm joints + 2 grippers) of commanded value vs.
elapsed time since the first entry.

Usage:
    python plot_commanded_actions.py --log /tmp/policy_actions.jsonl \
                                     --out /tmp/policy_actions.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


JOINT_KEYS = [
    "right_arm_0.pos", "right_arm_1.pos", "right_arm_2.pos", "right_arm_3.pos",
    "right_arm_4.pos", "right_arm_5.pos", "right_arm_6.pos",
    "left_arm_0.pos",  "left_arm_1.pos",  "left_arm_2.pos",  "left_arm_3.pos",
    "left_arm_4.pos",  "left_arm_5.pos",  "left_arm_6.pos",
    "right_gripper.pos", "left_gripper.pos",
]


def load_log(path: Path):
    ts: list[float] = []
    rows: list[list[float]] = []
    with path.open("r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  warn: line {ln} not JSON ({e}); skipping")
                continue
            t = float(rec.get("ts", 0.0))
            joints = rec.get("joints") or {}
            grippers = rec.get("grippers") or {}
            row = []
            for k in JOINT_KEYS:
                if k in joints:
                    row.append(float(joints[k]))
                elif k in grippers:
                    row.append(float(grippers[k]))
                else:
                    row.append(np.nan)
            ts.append(t)
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows parsed from {path}")
    ts_arr = np.asarray(ts, dtype=np.float64)
    arr = np.asarray(rows, dtype=np.float64)
    return ts_arr - ts_arr[0], arr  # elapsed seconds, (T, 16)


def plot(elapsed: np.ndarray, vals: np.ndarray, out: Path, log_path: Path) -> None:
    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows),
                             sharex=True)
    for j in range(len(JOINT_KEYS)):
        ax = axes[j // ncols, j % ncols]
        finite = np.isfinite(vals[:, j])
        if finite.any():
            ax.plot(elapsed[finite], vals[finite, j],
                    color="tab:blue", linewidth=1.0)
            vmin = float(np.nanmin(vals[:, j]))
            vmax = float(np.nanmax(vals[:, j]))
            rng = vmax - vmin
            ax.set_title(f"{JOINT_KEYS[j]}  range={rng:.3f}", fontsize=8)
        else:
            ax.set_title(f"{JOINT_KEYS[j]}  (no data)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.suptitle(
        f"Commanded actions from {log_path.name}\n"
        f"({len(elapsed)} ticks over {elapsed[-1]:.1f}s, "
        f"~{len(elapsed) / max(elapsed[-1], 1e-6):.1f} Hz)",
        fontsize=10,
    )
    fig.supxlabel("elapsed time (s)", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, required=True,
                   help="JSON-lines file from robot_proxy.py --log-actions")
    p.add_argument("--out", type=Path, required=True,
                   help="Output PNG path")
    args = p.parse_args()
    elapsed, vals = load_log(args.log)
    plot(elapsed, vals, args.out, args.log)


if __name__ == "__main__":
    main()
